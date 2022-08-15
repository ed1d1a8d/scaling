import dataclasses
import enum
import os
import warnings
from typing import Generic, Optional, TypeVar, Union

import ffcv.loader
import mup
import numpy as np
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils
import wandb
from simple_parsing import ArgumentParser
from src.ax.attack.FastAutoAttack import FastAutoAttack
from src.ax.attack.FastPGD import FastPGD
from src.ax.data import cifar, synthetic
from src.ax.models import vit, wrn
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
    WideResNet = enum.auto()
    VisionTransformer = enum.auto()


class DatasetT(enum.Enum):
    CIFAR10 = enum.auto()
    CIFAR5m = enum.auto()
    LightDark = enum.auto()
    HVStripe = enum.auto()
    SquareCircle = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # dataset params
    dataset: DatasetT = DatasetT.CIFAR5m
    LightDark_cfg: synthetic.LightDarkDSConfig = synthetic.LightDarkDSConfig()
    HVStripe_cfg: synthetic.HVStripeDSConfig = synthetic.HVStripeDSConfig()
    SquareCircle_cfg: synthetic.SquareCircleDSConfig = (
        synthetic.SquareCircleDSConfig()
    )

    # model params
    model: ModelT = ModelT.WideResNet
    depth: int = 28
    width: int = 10  # Only applicable for wide nets

    # train params
    n_train: int = 2048
    weight_decay: float = 5e-4
    batch_size: int = 512
    do_adv_training: bool = True
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size
    n_imgs_to_log_per_eval: int = 30

    # attack parameter only for CIFAR-10 and SVHN
    adv_eps_train: float = 8 / 255
    adv_eps_test: float = 8 / 255
    pgd_steps: int = 10
    use_autoattack: bool = True

    # Other params
    seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"

    def __post_init__(self):
        self.batch_size = min(self.batch_size, self.n_train)

    @property
    def steps_per_eval(self):
        return ceil_div(self.samples_per_eval, self.batch_size)

    @property
    def data_mean(self):
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return wrn.CIFAR10_MEAN
        return (0.0, 0.0, 0.0)

    @property
    def data_std(self):
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return wrn.CIFAR10_STD
        return (1.0, 1.0, 1.0)

    @property
    def num_classes(self):
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return 10
        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return 2

        raise ValueError(self.dataset)

    def get_synthetic_loader(self, split: str, size: int, **kwargs):
        ds_cls, ds_cfg = {
            DatasetT.LightDark: (synthetic.LightDarkDS, self.LightDark_cfg),
            DatasetT.HVStripe: (synthetic.HVStripeDS, self.HVStripe_cfg),
            DatasetT.SquareCircle: (
                synthetic.SquareCircleDS,
                self.SquareCircle_cfg,
            ),
        }[self.dataset]

        return torch.utils.data.DataLoader(
            dataset=(ds_cls(cfg=ds_cfg, split=split, size=size)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def get_loader_train(self):
        if self.dataset is DatasetT.CIFAR5m:
            return cifar.get_loader(
                split="train",
                batch_size=self.batch_size,
                indices=range(self.n_train),
                random_order=True,
                seed=self.seed,
                num_workers=self.num_workers,
            )

        if self.dataset is DatasetT.CIFAR10:
            return cifar.get_loader(
                split="train-orig",
                batch_size=self.batch_size,
                indices=range(self.n_train),
                random_order=True,
                seed=self.seed,
                num_workers=self.num_workers,
            )

        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return self.get_synthetic_loader(
                split="train", size=self.n_train, shuffle=True
            )

        raise ValueError(self.dataset)

    def get_loader_val(self):
        if self.dataset is DatasetT.CIFAR5m:
            return cifar.get_loader(
                split="val", batch_size=self.eval_batch_size
            )

        if self.dataset is DatasetT.CIFAR10:
            return cifar.get_loader(
                split="test-orig",
                batch_size=self.eval_batch_size,
                indices=range(5000),
            )

        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return self.get_synthetic_loader(split="val", size=2_000)

        raise ValueError(self.dataset)

    def get_test_loaders(self):
        if self.dataset is DatasetT.CIFAR5m:
            return {
                "test_orig": cifar.get_loader(
                    split="test-orig", batch_size=self.eval_batch_size
                ),
                "train_orig": cifar.get_loader(
                    split="train-orig", batch_size=self.eval_batch_size
                ),
                "test": cifar.get_loader(
                    split="test", batch_size=self.eval_batch_size
                ),
            }

        if self.dataset is DatasetT.CIFAR10:
            return {
                "test_orig": cifar.get_loader(
                    split="test-orig",
                    batch_size=self.eval_batch_size,
                    indices=range(5_000, 10_000),
                ),
                "test_5m": cifar.get_loader(
                    split="test", batch_size=self.eval_batch_size
                ),
            }

        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return {"test": self.get_synthetic_loader(split="test", size=5_000)}

        raise ValueError(self.dataset)

    def get_net(self) -> nn.Module:
        if self.model is ModelT.WideResNet:
            return wrn.get_mup_wrn(
                depth=self.depth,
                width=self.width,
                num_classes=self.num_classes,
                mean=self.data_mean,
                std=self.data_std,
            )

        if self.model is ModelT.VisionTransformer:
            return vit.get_mup_vit(
                img_size=32,
                num_input_channels=3,
                num_classes=self.num_classes,
                patch=8,
                dropout=0,
                num_layers=7,
                head=12,
                hidden=384,
                mlp_hidden=384,
                is_cls_token=False,
            )

        raise ValueError(self.model)


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


def get_imgs_to_log(
    imgs_nat: torch.Tensor,
    imgs_adv: torch.Tensor,
    preds_nat: torch.Tensor,
    preds_adv: torch.Tensor,
    labs: torch.Tensor,
    cfg: ExperimentConfig,
    adv_eps: float,
) -> list[wandb.Image]:
    imgs_diff = (imgs_adv - imgs_nat) / adv_eps / (2 + 1e-9) + 0.5

    def get_caption(i: int) -> str:
        lab = cifar.cls_name(int(labs[i].item()))
        pred_nat = cifar.cls_name(int(preds_nat[i].item()))
        pred_adv = cifar.cls_name(int(preds_adv[i].item()))
        return "\n".join(
            [
                f"preds: {pred_nat}, {pred_adv}; true: {lab}",
                f"img order: nat, adv, diff",
            ]
        )

    return [
        wandb.Image(
            torchvision.utils.make_grid(
                [imgs_nat[i], imgs_adv[i], imgs_diff[i]], nrow=3, ncol=1
            ),
            caption=get_caption(i),
        )
        for i in range(min(cfg.n_imgs_to_log_per_eval, len(imgs_nat)))
    ]


def evaluate(
    net: nn.Module,
    loader: Union[ffcv.loader.Loader, torch.utils.data.DataLoader],
    attack: Union[FastPGD, FastAutoAttack],
    cfg: ExperimentConfig,
) -> tuple[dict[str, Metric[float]], list[wandb.Image]]:
    n: int = len(loader.indices) if isinstance(loader, ffcv.loader.Loader) else len(loader.dataset)  # type: ignore

    n_correct_nat: float = 0
    n_correct_adv: float = 0
    tot_loss_nat: float = 0
    tot_loss_adv: float = 0

    imgs_nat: torch.Tensor
    labs: torch.Tensor
    imgs_to_log: list[wandb.Image] = []
    with tqdm(total=len(loader)) as pbar:
        for (imgs_nat, labs) in loader:
            imgs_nat = imgs_nat.cuda()
            labs = labs.cuda()

            imgs_adv: torch.Tensor = attack(imgs_nat, labs)

            with torch.autocast("cuda"):  # type: ignore
                with torch.no_grad():
                    logits_nat = net(imgs_nat)
                    logits_adv = net(imgs_adv)

                    loss_nat = F.cross_entropy(
                        logits_nat, labs, reduction="sum"
                    )
                    loss_adv = F.cross_entropy(
                        logits_adv, labs, reduction="sum"
                    )

            preds_nat = logits_nat.argmax(dim=-1)
            preds_adv = logits_adv.argmax(dim=-1)

            n_correct_nat += preds_nat.eq(labs).sum().item()
            n_correct_adv += preds_adv.eq(labs).sum().item()

            tot_loss_nat += loss_nat.item()
            tot_loss_adv += loss_adv.item()

            if len(imgs_to_log) == 0:
                imgs_to_log = get_imgs_to_log(
                    imgs_nat=imgs_nat,
                    imgs_adv=imgs_adv,
                    preds_nat=preds_nat,
                    preds_adv=preds_adv,
                    labs=labs,
                    cfg=cfg,
                    adv_eps=attack.eps,
                )

            pbar.update(1)

    return (
        dict(
            acc_nat=Metric(n_correct_nat / n, "max"),
            acc_adv=Metric(n_correct_adv / n, "max"),
            loss_nat=Metric(tot_loss_nat / n, "min"),
            loss_adv=Metric(tot_loss_adv / n, "min"),
            acc=Metric(
                n_correct_adv / n if cfg.do_adv_training else n_correct_nat / n,
                "max",
            ),
            loss=Metric(
                tot_loss_adv / n if cfg.do_adv_training else tot_loss_nat / n,
                "min",
            ),
        ),
        imgs_to_log,
    )


def train(
    net: nn.Module,
    attack_train: FastPGD,
    attack_val: FastPGD,
    cfg: ExperimentConfig,
):
    loader_train = cfg.get_loader_train()
    loader_val = cfg.get_loader_val()
    print("train and val loaders created!")

    optimizer = mup.MuAdamW(
        net.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
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
        while True:  # Termination will be handled outside
            imgs_nat: torch.Tensor
            labs: torch.Tensor
            for (imgs_nat, labs) in loader_train:
                imgs_nat = imgs_nat.cuda()
                labs = labs.cuda()

                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                imgs_adv: torch.Tensor = attack_train(imgs_nat, labs)

                with torch.autocast("cuda"):  # type: ignore
                    if cfg.do_adv_training:
                        logits_adv = net(imgs_adv)
                        loss_adv = F.cross_entropy(logits_adv, labs)
                        loss = loss_adv
                        with torch.no_grad():
                            logits_nat = net(imgs_nat)
                            loss_nat = F.cross_entropy(logits_nat, labs)
                    else:
                        logits_nat = net(imgs_nat)
                        loss_nat = F.cross_entropy(logits_nat, labs)
                        loss = loss_nat
                        with torch.no_grad():
                            logits_adv = net(imgs_adv)
                            loss_adv = F.cross_entropy(logits_adv, labs)

                preds_nat = logits_nat.argmax(dim=-1)
                preds_adv = logits_adv.argmax(dim=-1)
                acc_nat: float = preds_nat.eq(labs).float().mean().item()
                acc_adv: float = preds_adv.eq(labs).float().mean().item()
                acc = acc_adv if cfg.do_adv_training else acc_nat

                optimizer.zero_grad()
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(optimizer)
                scaler.update()

                n_steps += 1
                pbar.update(1)
                pbar.set_description(
                    f"epochs={n_epochs}; loss={loss.item():6e}; acc={acc:.4f}; lr={cur_lr:6e}"
                )

                log_dict: dict[str, Metric] = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    net.eval()
                    val_dict, val_imgs = evaluate(
                        net, loader_val, attack_val, cfg
                    )
                    net.train()

                    val_loss: float = val_dict["loss"].data
                    lr_scheduler.step(val_loss)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        wandb.run.summary["best_checkpoint_steps"] = n_steps  # type: ignore
                        save_model(net)

                    log_dict |= tag_dict(val_dict, prefix=f"val_")
                    log_dict["val_imgs"] = Metric(val_imgs)
                    log_dict["train_imgs"] = Metric(
                        get_imgs_to_log(
                            imgs_nat=imgs_nat,
                            imgs_adv=imgs_adv,
                            preds_nat=preds_nat,
                            preds_adv=preds_adv,
                            labs=labs,
                            cfg=cfg,
                            adv_eps=cfg.adv_eps_train,
                        )
                    )

                wandb_log(
                    dict(
                        step=Metric(n_steps),
                        epoch=Metric(n_epochs),
                        lr=Metric(cur_lr),
                        train_loss=Metric(loss.item(), "min"),
                        train_acc=Metric(acc, "max"),
                        train_loss_nat=Metric(loss_nat.item(), "min"),
                        train_loss_adv=Metric(loss_adv.item(), "min"),
                        train_acc_nat=Metric(acc_nat, "max"),
                        train_acc_adv=Metric(acc_adv, "max"),
                    )
                    | log_dict
                )

            n_epochs += 1


def run_experiment(cfg: ExperimentConfig):
    torch.manual_seed(cfg.seed)

    net: nn.Module = cfg.get_net().cuda()
    net = net.to(memory_format=torch.channels_last)  # type: ignore

    attack_train = FastPGD(
        model=net,
        eps=cfg.adv_eps_train,
        alpha=cfg.adv_eps_train / cfg.pgd_steps * 2.3,
        steps=cfg.pgd_steps,
        random_start=True,
    )

    attack_val = FastPGD(
        model=net,
        eps=cfg.adv_eps_test,
        alpha=cfg.adv_eps_test / cfg.pgd_steps * 2.3,
        steps=cfg.pgd_steps,
        random_start=True,
    )

    attack_test = (
        FastAutoAttack(net, eps=cfg.adv_eps_test)
        if cfg.use_autoattack
        else attack_val
    )

    try:
        train(net, attack_train, attack_val, cfg)
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    load_model(net)

    test_loaders = cfg.get_test_loaders()
    net.eval()
    for split_name, loader in test_loaders.items():
        print(f"Starting evaluation of {split_name} split...")
        test_dict, test_imgs = evaluate(net, loader, attack_test, cfg)

        wandb_log({f"{split_name}_imgs": Metric(test_imgs)})
        test_metrics = tag_dict(test_dict, prefix=f"{split_name}_")
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
        project="adv-train",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
