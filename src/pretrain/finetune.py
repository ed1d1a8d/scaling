"""
Finetune and entire model.

Sample command:
    python -m src.pretrain.finetune -h
"""

import contextlib
import dataclasses
import enum
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
from simple_parsing import ArgumentParser, subgroups
from torch import nn
from tqdm.auto import tqdm

from src.pretrain import gen_embeddings
from src.pretrain.datasets import BaseDatasetConfig, get_dataset_index
from src.pretrain.datasets.embedding import EmbeddingDataset
from src.pretrain.models import BaseEmbedderConfig, get_embedder_index
from src.pretrain.models.base import BaseEmbedder
from src.pretrain.probes import fc_probe, linear_probe
from src.utils import ceil_div
from src.wandb_utils import (
    WandBManager,
    WBMetric,
    load_model,
    save_model,
    tag_dict,
)

WANDB_MANAGER = WandBManager()


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    Adam = enum.auto()
    AdamW = enum.auto()


@dataclasses.dataclass
class Config:
    """
    Configuration for model finetuning.

    TODO: Additional features (to be implemented):
    1. Log per class metrics
    2. Specify subset of classes as task.
       e.g. via `n_classes: Optional[int] = None`
    3. Support freezing some prefix of the model.
    """

    # Model
    embedder_cfg: BaseEmbedderConfig = subgroups(get_embedder_index())
    fc_probe_cfg: fc_probe.FCProbeConfig = fc_probe.FCProbeConfig(n_layers=1)

    # Dataset
    dataset_cfg: BaseDatasetConfig = subgroups(get_dataset_index())

    # Optimizer
    optimizer: OptimizerT = OptimizerT.Adam
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 1e-3
    lr: float = 1e-5
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # Initialization
    init_with_trained_linear_probe: bool = False
    init_linear_probe_c: float = 100

    # Train config
    n_train: int = 2048
    batch_size: int = 50

    # Validation config
    # The validation set is taken out of the train set!
    val_frac: float = 0.1  # Fraction of training data to use for validation.
    n_val_override: Optional[
        int
    ] = None  # Override validation set size to a specific value.

    # Eval config
    eval_batch_size: int = 512
    samples_per_eval: int = 25000  # aka epoch size
    n_imgs_to_log_per_eval: int = 15

    # Other options
    seed: int = 0
    num_workers: int = 20
    wandb_dir: str = "/home/gridsan/groups/ccg"
    tags: tuple[str, ...] = ("test",)

    @property
    def steps_per_eval(self):
        return ceil_div(self.samples_per_eval, self.batch_size)

    @property
    def class_names(self):
        return self.dataset_cfg.class_names

    @property
    def n_val(self):
        return self.n_val_override or int(self.val_frac * self.n_train)

    def __post_init__(self):
        assert (
            self.n_val <= self.n_train
        ), "Can't have more validation samples than training samples!"

    def get_loader(
        self,
        ds: torch.utils.data.Dataset,
        eval_mode: bool = False,
    ):
        return torch.utils.data.DataLoader(
            ds,  # type: ignore
            batch_size=self.eval_batch_size if eval_mode else self.batch_size,
            shuffle=not eval_mode,
            num_workers=self.num_workers,
            pin_memory=True,
        )

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
                betas=(0.9, 0.98),  # From CLIP paper
                eps=1e-6,  # From CLIP paper
                weight_decay=self.weight_decay,
            )

        if self.optimizer is OptimizerT.AdamW:
            return torch.optim.AdamW(
                net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        raise ValueError(self.optimizer)


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
    model: nn.Module,
    eval_mode: bool = False,
):
    imgs = imgs.cuda()
    labs = labs.cuda()

    with torch.no_grad() if eval_mode else contextlib.nullcontext():
        with torch.autocast("cuda"):  # type: ignore
            logits = model(imgs)
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
    cfg: Config,
) -> list[wandb.Image]:
    def get_caption(i: int) -> str:
        lab = cfg.class_names[int(bo.labs[i].item())]  # type: ignore
        pred = cfg.class_names[int(bo.preds[i].item())]  # type: ignore
        return f"{pred=}; {lab=}"

    return [
        wandb.Image(bo.imgs[i], caption=get_caption(i))
        for i in range(min(cfg.n_imgs_to_log_per_eval, len(bo.imgs)))
    ]


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Config,
) -> tuple[dict[str, WBMetric[float]], list[wandb.Image]]:
    n: int = len(loader.dataset)  # type: ignore

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
                model=model,
                eval_mode=True,
            )

            n_correct += bo.acc * bo.size
            tot_loss += bo.loss.item() * bo.size

            if len(imgs_to_log) == 0:
                imgs_to_log = get_imgs_to_log(bo=bo, cfg=cfg)

            pbar.update(1)

    return (
        dict(
            acc=WBMetric(n_correct / n, "max"),
            loss=WBMetric(tot_loss / n, "min"),
        ),
        imgs_to_log,
    )


def train(
    model: nn.Module,
    loader_train: torch.utils.data.DataLoader,
    loader_val: torch.utils.data.DataLoader,
    cfg: Config,
):
    optimizer = cfg.get_optimizer(model)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        min_lr=0.1 * cfg.min_lr,
        patience=cfg.lr_decay_patience_evals,
        verbose=True,
    )
    scaler = torch.cuda.amp.GradScaler()  # type: ignore

    n_steps: int = 0
    n_epochs: int = 0
    min_val_loss: float = np.inf
    with tqdm() as pbar:
        model.train()
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
                    model=model,
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

                log_dict: dict[str, WBMetric] = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    is_training: bool = model.training
                    model.eval()
                    val_dict, val_imgs = evaluate(model, loader_val, cfg)
                    model.train(is_training)

                    val_loss: float = val_dict["loss"].data
                    lr_scheduler.step(val_loss)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        wandb.run.summary["best_checkpoint_steps"] = n_steps  # type: ignore
                        save_model(model)

                    log_dict |= tag_dict(val_dict, prefix=f"val_")
                    log_dict["val_imgs"] = WBMetric(val_imgs)
                    log_dict["train_imgs"] = WBMetric(
                        get_imgs_to_log(bo=bo, cfg=cfg)
                    )

                WANDB_MANAGER.log(
                    dict(
                        step=WBMetric(n_steps),
                        epoch=WBMetric(n_epochs),
                        lr=WBMetric(cur_lr),
                        train_loss=WBMetric(bo.loss.item(), "min"),
                        train_acc=WBMetric(bo.acc, "max"),
                    )
                    | log_dict
                )

            n_epochs += 1


def init_model_with_trained_linear_probe(
    model: fc_probe.FCProbe,
    ds: torch.utils.data.Dataset,
    cfg: Config,
    verbose: bool = False,
) -> tuple[EmbeddingDataset, Any]:
    """
    Procedure:
        1. Uses model.embedder (in eval mode) to embed ds.
        2. Fits a linear probe on the embeddings.
        3. Modifies model in place to have the trained linear probe as its
           readout layer.

    Returns the embeddings of ds and the trained linear probe.
    """

    # Compute embeddings of ds
    model.eval()
    xs_train, ys_train = gen_embeddings.embed_dataset(
        embedder=model.embedder,
        ds=ds,
        cfg=gen_embeddings.Config(
            embedder_cfg=cfg.embedder_cfg,
            dataset_cfg=cfg.dataset_cfg,
            batch_size=cfg.eval_batch_size,
            num_workers=cfg.num_workers,
        ),
    )

    # Construct EmbeddingDataset object
    eds = EmbeddingDataset(
        xs_train=xs_train,
        ys_train=ys_train,
        xs_test=xs_train,
        ys_test=ys_train,
        dataset_id=cfg.dataset_cfg.id + f"-finetune-n={cfg.n_train}",
        embedder_id=cfg.embedder_cfg.id,
    )

    # Compute optimal probe
    res_dict: dict[str, Any] = linear_probe.run_experiment(
        ds=eds,
        c=cfg.init_linear_probe_c,
        use_gpu=True,
        return_clf=True,
    )
    assert np.all(
        res_dict["classes"] == np.arange(len(cfg.class_names))
    )  # TODO: Get rid of this requirement
    if verbose:
        print("Linear probe results:")
        print("acc:", res_dict["acc"])
        print("xent_orig:", res_dict["xent_orig"])
        print("xent:", res_dict["xent"])

    # Set readout layer
    readout_lyr: nn.Linear
    (readout_lyr,) = model.probe  # type: ignore
    with torch.no_grad():
        readout_lyr.weight.copy_(torch.from_numpy(res_dict["clf"].coef_.T))
        readout_lyr.bias.copy_(torch.from_numpy(res_dict["clf"].intercept_))

    return eds, res_dict["clf"]


def run_experiment(cfg: Config):
    torch.manual_seed(cfg.seed)

    print(f"Loading embedder {cfg.embedder_cfg.id}...")
    embedder: BaseEmbedder = cfg.embedder_cfg.get_model().float()

    print(f"Loading dataset {cfg.dataset_cfg.id}...")
    ds_train_full = cfg.dataset_cfg.get_train_ds(embedder.preprocess)
    ds_test = cfg.dataset_cfg.get_test_ds(embedder.preprocess)

    # Create train and val datasets from ds_train_full
    ds_train, ds_val, _ = torch.utils.data.random_split(
        ds_train_full,
        (
            cfg.n_train - cfg.n_val,
            cfg.n_val,
            len(ds_train_full) - cfg.n_train,  # type: ignore
        ),
    )

    # Create dataloaders
    loader_train = cfg.get_loader(ds_train, eval_mode=False)
    loader_val = cfg.get_loader(ds_val, eval_mode=True)
    loader_test = cfg.get_loader(ds_test, eval_mode=True)

    print("Constructing finetuning model...")
    model = cfg.fc_probe_cfg.get_fc_probe(embedder)
    model.cuda()
    if cfg.init_with_trained_linear_probe:
        print("Initializing with trained linear probe...")
        init_model_with_trained_linear_probe(
            model=model,
            ds=ds_train,
            cfg=cfg,
        )

    print("Evaluating test error of model at init...")
    model.eval()
    test_dict, test_imgs = evaluate(model=model, loader=loader_test, cfg=cfg)
    WANDB_MANAGER.log({f"init_test_imgs": WBMetric(test_imgs)})
    test_metrics = tag_dict(test_dict, prefix="init_test_")
    for name, metric in test_metrics.items():
        wandb.run.summary[name] = metric.data  # type: ignore

    try:
        train(
            model=model,
            loader_train=loader_train,
            loader_val=loader_val,
            cfg=cfg,
        )
        print("Finished training!")
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    # Load best model
    load_model(model)
    print("Loaded best model")

    print("Evaluating on test set...")
    model.eval()
    test_dict, test_imgs = evaluate(model=model, loader=loader_test, cfg=cfg)
    WANDB_MANAGER.log({f"test_imgs": WBMetric(test_imgs)})
    test_metrics = tag_dict(test_dict, prefix="test_")
    for name, metric in test_metrics.items():
        wandb.run.summary[name] = metric.data  # type: ignore

    print("Finished experiment!")


if __name__ == "__main__":
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    cfg: Config = args.config

    # Don't upload ckpt's to wandb, since they are big and are saved on supercloud.
    os.environ["WANDB_IGNORE_GLOBS"] = "*.ckpt"

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="finetune",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Run main logic
    run_experiment(cfg)
