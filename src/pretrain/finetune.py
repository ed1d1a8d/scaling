import contextlib
import dataclasses
import enum
import os
import warnings
from typing import Callable, Generic, Optional, TypeVar, Union

import ffcv.loader
import numpy as np
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets
import wandb
from simple_parsing import ArgumentParser
from src.ax.data import cifar
from src.pretrain import classifiers
from torch import nn
from tqdm.auto import tqdm

T = TypeVar("T")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def tag_dict(
    d: dict[str, T],
    prefix: str = "",
    suffix: str = "",
) -> dict[str, T]:
    return {f"{prefix}{key}{suffix}": val for key, val in d.items()}


class ModelT(enum.Enum):
    CLIP_VIT_L14 = enum.auto()


class DatasetT(enum.Enum):
    CIFAR10 = enum.auto()
    CIFAR5m = enum.auto()


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    Adam = enum.auto()
    AdamW = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # model params
    model: ModelT = ModelT.CLIP_VIT_L14

    # dataset params
    dataset: DatasetT = DatasetT.CIFAR10
    data_augmentation: bool = False

    # optimizer params
    optimizer: OptimizerT = OptimizerT.Adam
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 1e-3
    lr: float = 1e-7
    min_lr: float = 1e-7
    lr_decay_patience_evals: int = 5

    # train params
    n_train: int = 45000
    batch_size: int = 50

    # eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 50000  # (51200) aka epoch size
    n_imgs_to_log_per_eval: int = 15

    # Testing params
    n_test: Optional[int] = None  # Used for speed up testing

    # Other params
    seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"
    data_root: str = "/home/gridsan/groups/ccg/data-scratch"
    keep_model_ckpt: bool = False

    def __post_init__(self):
        self.batch_size = min(self.batch_size, self.n_train)
        assert self.min_lr <= self.lr

    @property
    def steps_per_eval(self):
        return ceil_div(self.samples_per_eval, self.batch_size)

    @property
    def num_classes(self):
        if self.dataset in (
            DatasetT.CIFAR5m,
            DatasetT.CIFAR10,
        ):
            return 10

        raise ValueError(self.dataset)

    @property
    def n_channels(self) -> int:
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return 3

        raise ValueError(self.dataset)

    def get_loader_train(self, transform: Optional[Callable]):
        if self.dataset is DatasetT.CIFAR10:
            return torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    torchvision.datasets.CIFAR10(
                        root=self.data_root,
                        train=True,
                        download=True,
                        transform=transform,
                    ),
                    indices=range(self.n_train),
                ),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        if self.dataset is DatasetT.CIFAR5m:
            raise NotImplementedError

        raise ValueError(self.dataset)

    def get_loader_val(self, transform: Optional[Callable]):
        if self.dataset is DatasetT.CIFAR10:
            return torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    torchvision.datasets.CIFAR10(
                        root=self.data_root,
                        train=True,
                        download=True,
                        transform=transform,
                    ),
                    indices=range(self.n_train, self.n_train + 5000),
                ),
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        if self.dataset is DatasetT.CIFAR5m:
            raise NotImplementedError

        raise ValueError(self.dataset)

    def get_test_loaders(self, transform: Optional[Callable]):
        if self.dataset is DatasetT.CIFAR10:
            return {
                "test": torch.utils.data.DataLoader(
                    torchvision.datasets.CIFAR10(
                        root=self.data_root,
                        train=False,
                        download=True,
                        transform=transform,
                    ),
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            }

        if self.dataset is DatasetT.CIFAR5m:
            raise NotImplementedError

        raise ValueError(self.dataset)

    def get_net_and_preprocess(
        self,
    ) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
        if self.model is ModelT.CLIP_VIT_L14:
            net = classifiers.CLIPClassifier(
                model_name="ViT-L/14",
                num_classes=self.num_classes,
            ).float()
            net.eval()
            return net, net.preprocess

        raise ValueError(self.model)

    def get_optimizer(self, net: nn.Module):
        if self.optimizer is OptimizerT.SGD:
            return torch.optim.SGD(
                net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        if self.optimizer is OptimizerT.Adam:
            return torch.optim.Adam(
                net.parameters(),
                lr=self.lr,
                betas=(0.9,0.98), # From CLIP paper
                eps=1e-6, # From CLIP paper
                weight_decay=self.weight_decay,
            )

        if self.optimizer is OptimizerT.AdamW:
            return torch.optim.AdamW(
                net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )



        raise ValueError(self.optimizer)

    def cls_name(self, idx: int) -> str:
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return cifar.cls_name(idx)

        raise ValueError(self.dataset)


@dataclasses.dataclass
class Metric(Generic[T]):
    data: T
    summary: Optional[str] = None


WANDB_METRIC_SUMMARY_MAP: dict[str, Optional[str]] = dict()


def wandb_log(d: dict[str, Metric]):
    for name, metric in d.items():
        if name not in WANDB_METRIC_SUMMARY_MAP:
            WANDB_METRIC_SUMMARY_MAP[name] = metric.summary
            if metric.summary is not None:
                wandb.define_metric(name=name, summary=metric.summary)
        elif WANDB_METRIC_SUMMARY_MAP[name] != metric.summary:
            s1 = WANDB_METRIC_SUMMARY_MAP[name]
            s2 = metric.summary
            warnings.warn(
                f"metric {name} has different summaries: {s1}, {s2}",
                RuntimeWarning,
            )

    wandb.log({name: metric.data for name, metric in d.items()})


def save_model(model: nn.Module):
    print("Saving model checkpoint...")
    torch.save(
        model.state_dict(),
        os.path.join(wandb.run.dir, "model.ckpt"),  # type: ignore
    )
    print("Saved model checkpoint.")


def load_model(model: nn.Module):
    print("Loading model checkpoint...")
    path: str = os.path.join(wandb.run.dir, "model.ckpt")  # type: ignore
    model.load_state_dict(torch.load(path))
    print("Loaded model checkpoint.")

def delete_model_ckpt():
    path: str = os.path.join(wandb.run.dir, "model.ckpt")  # type: ignore
    os.remove(path)
    print("Deleted model checkpoint to save space.")


@dataclasses.dataclass
class BatchOutput:
    size: int

    imgs: torch.Tensor
    labs: torch.Tensor

    preds: torch.Tensor

    loss: torch.Tensor
    acc: float


def process_batch(
    imgs: torch.Tensor,
    labs: torch.Tensor,
    net: nn.Module,
    cfg: ExperimentConfig,
    eval_mode: bool = False,
):
    imgs = imgs.cuda()
    labs = labs.cuda()

    with torch.no_grad() if eval_mode else contextlib.nullcontext():
        with torch.autocast("cuda"):  # type: ignore
            logits = net(imgs)
            preds = logits.argmax(dim=-1)
            loss = F.cross_entropy(logits, labs)

    acc = (preds == labs).float().mean().item()

    return BatchOutput(
        size=imgs.shape[0],
        imgs=imgs,
        labs=labs,
        preds=preds,
        loss=loss,
        acc=acc,
    )


def get_imgs_to_log(
    bo: BatchOutput,
    cfg: ExperimentConfig,
) -> list[wandb.Image]:
    def get_caption(i: int) -> str:
        lab = cfg.cls_name(int(bo.labs[i].item()))
        pred = cfg.cls_name(int(bo.preds[i].item()))
        return f"{pred=}; {lab=}"

    return [
        wandb.Image(bo.imgs[i], caption=get_caption(i))
        for i in range(min(cfg.n_imgs_to_log_per_eval, len(bo.imgs)))
    ]


def evaluate(
    net: nn.Module,
    loader: Union[ffcv.loader.Loader, torch.utils.data.DataLoader],
    cfg: ExperimentConfig,
) -> tuple[dict[str, Metric[float]], list[wandb.Image]]:
    n: int = len(loader.indices) if isinstance(loader, ffcv.loader.Loader) else len(loader.dataset)  # type: ignore

    n_correct: float = 0
    tot_loss: float = 0

    imgs: torch.Tensor
    labs: torch.Tensor
    imgs_to_log: list[wandb.Image] = []
    with tqdm(total=len(loader)) as pbar:
        for imgs, labs in loader:

            bo = process_batch(
                imgs=imgs,
                labs=labs,
                net=net,
                cfg=cfg,
                eval_mode=True,
            )

            n_correct += bo.acc * bo.size
            tot_loss += bo.loss.item() * bo.size

            if len(imgs_to_log) == 0:
                imgs_to_log = get_imgs_to_log(bo=bo, cfg=cfg)

            pbar.update(1)

    return (
        dict(
            acc=Metric(n_correct / n, "max"),
            loss=Metric(tot_loss / n, "min"),
        ),
        imgs_to_log,
    )


def train(
    net: nn.Module,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    cfg: ExperimentConfig,
):
    loader_train = cfg.get_loader_train(transform=preprocess)
    loader_val = cfg.get_loader_val(transform=preprocess)
    print("train and val loaders created!")

    optimizer = cfg.get_optimizer(net)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        min_lr=0.01 * cfg.min_lr,
        patience=cfg.lr_decay_patience_evals,
        verbose=True,
    )
    scaler = torch.cuda.amp.GradScaler()  # type: ignore

    n_steps: int = 0
    n_epochs: int = 0
    min_val_loss: float = np.inf
    with tqdm() as pbar:
        while True:  # We handle termination manually
            imgs: torch.Tensor
            labs: torch.Tensor
            for imgs, labs in loader_train:
                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                bo = process_batch(
                    imgs=imgs,
                    labs=labs,
                    net=net,
                    cfg=cfg,
                )

                optimizer.zero_grad()
                scaler.scale(bo.loss).backward()  # type: ignore
                scaler.step(optimizer)
                scaler.update()

                n_steps += 1
                pbar.update(1)
                pbar.set_description(
                    f"epochs={n_epochs}; loss={bo.loss.item():6e}; acc={bo.acc:.4f}; lr={cur_lr:6e}"
                )

                log_dict: dict[str, Metric] = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    is_training: bool = net.training
                    net.eval()
                    val_dict, val_imgs = evaluate(net, loader_val, cfg)
                    net.train(is_training)

                    val_loss: float = val_dict["loss"].data
                    lr_scheduler.step(val_loss)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        wandb.run.summary["best_checkpoint_steps"] = n_steps  # type: ignore
                        save_model(net)

                    log_dict |= tag_dict(val_dict, prefix=f"val_")
                    log_dict["val_imgs"] = Metric(val_imgs)
                    log_dict["train_imgs"] = Metric(
                        get_imgs_to_log(bo=bo, cfg=cfg)
                    )

                wandb_log(
                    dict(
                        step=Metric(n_steps),
                        epoch=Metric(n_epochs),
                        lr=Metric(cur_lr),
                        train_loss=Metric(bo.loss.item(), "min"),
                        train_acc=Metric(bo.acc, "max"),
                    )
                    | log_dict
                )

            n_epochs += 1


def run_experiment(cfg: ExperimentConfig):
    torch.manual_seed(cfg.seed)

    net: nn.Module
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    net, preprocess = cfg.get_net_and_preprocess()

    try:
        train(
            net=net,
            preprocess=preprocess,
            cfg=cfg,
        )
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    load_model(net)
    if not cfg.keep_model_ckpt:
        delete_model_ckpt()

    test_loaders = cfg.get_test_loaders(transform=preprocess)
    net.eval()
    for split_name, loader in test_loaders.items():
        print(f"Starting evaluation of {split_name} split...")
        test_dict, test_imgs = evaluate(
            net=net,
            loader=loader,
            cfg=cfg,
        )

        wandb_log({f"{split_name}_imgs": Metric(test_imgs)})
        test_metrics = tag_dict(
            test_dict,
            prefix=f"{split_name}_",
        )
        for name, metric in test_metrics.items():
            wandb.run.summary[name] = metric.data  # type: ignore

        print(f"Finished evaluation of {split_name} split.")


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Don't upload ckpt's to wandb, since they are big and are saved on supercloud.
    os.environ["WANDB_IGNORE_GLOBS"] = "*.ckpt"

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="transfer",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
