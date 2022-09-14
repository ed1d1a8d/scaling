import contextlib
import dataclasses
import enum
import tempfile
from typing import Optional, TypeVar, Union

import mup
import numpy as np
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.data
import wandb
from simple_parsing import ArgumentParser
from src.student_teacher_v2 import utils, viz
from src.student_teacher_v2.data import (
    FastTensorDataLoader,
    InfiniteTensorDataLoader,
)
from src.student_teacher_v2.fc_net import FCNet
from src.student_teacher_v2.utils import Metric
from torch import nn
from tqdm.auto import tqdm

T = TypeVar("T")
LoaderT = Union[FastTensorDataLoader, InfiniteTensorDataLoader]


class ActivationT(enum.Enum):
    ReLU = nn.ReLU
    LeakyReLU = nn.LeakyReLU
    SiLU = nn.SiLU
    Identity = nn.Identity


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    AdamW = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # Input params
    input_dim: int = 8
    input_lo: float = -1.0
    input_hi: float = 1.0

    # Network params
    teacher_widths: tuple[int, ...] = (96, 192, 1)
    activation: ActivationT = ActivationT.ReLU
    end_with_activation: bool = False
    teacher_seed: int = 101

    student_width_scale_factor: float = 1.0
    student_seed: int = 103

    # Optimizer params
    optimizer: OptimizerT = OptimizerT.AdamW
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 0
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # Dataset params
    n_train: int = -1  # -1 means infinite data
    n_val: int = 10_000
    n_test: int = 100_000

    ds_train_seed: int = -1
    ds_val_seed: int = -2
    ds_test_seed: int = -3

    data_device: str = "cuda"

    # Train params
    batch_size: int = 256

    # Eval params
    eval_batch_size: int = 2048
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size

    # Visualization params
    n_viz_dims: int = 5
    viz_side_samples: int = 256
    viz_od_value: float = 0.42

    # How much train loss has to decrease before we visualize the networks
    viz_loss_decrease_thresh: float = 0.75

    # Other params
    log_freq_in_steps: int = 1000
    half_precision: bool = False
    global_seed: int = 42
    tags: tuple[str, ...] = ("test",)
    wandb_dir: Optional[str] = None

    def __post_init__(self):
        assert self.input_lo <= self.input_hi
        assert self.teacher_widths[-1] > 0
        assert self.min_lr < self.lr

    @property
    def for_classification(self):
        return self.teacher_widths[-1] >= 2

    @property
    def steps_per_eval(self):
        return utils.ceil_div(self.samples_per_eval, self.batch_size)

    @property
    def student_widths(self):
        return tuple(
            [
                round(w * self.student_width_scale_factor)
                for w in self.teacher_widths[:-1]
            ]
            + [self.teacher_widths[-1]]
        )

    def precision_context(self, net: nn.Module):
        return (
            torch.autocast(net.device.type)  # type: ignore
            if self.half_precision
            else contextlib.nullcontext()
        )

    def _get_loader(
        self, n: int, seed: int, batch_size: int, shuffle: bool
    ) -> LoaderT:
        if n == -1:
            return InfiniteTensorDataLoader(
                gen_batch_fn=lambda bs, rng: (
                    torch.rand(size=(bs, self.input_dim), generator=rng)
                    * (self.input_hi - self.input_lo)
                    + self.input_lo,
                ),
                batch_size=batch_size,
                device=self.data_device,
                seed=seed,
            )

        assert n > 0

        xs = (
            torch.rand(
                size=(n, self.input_dim),
                generator=torch.Generator().manual_seed(seed),
            )
            * (self.input_hi - self.input_lo)
            + self.input_lo
        )
        return FastTensorDataLoader(
            xs,
            batch_size=batch_size,
            shuffle=shuffle,
            device=self.data_device,
            seed=seed,
        )

    def get_loader_train(self):
        return self._get_loader(
            n=self.n_train,
            seed=self.ds_train_seed,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_loader_val(self):
        return self._get_loader(
            n=self.n_val,
            seed=self.ds_val_seed,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def get_loader_test(self):
        return self._get_loader(
            n=self.n_test,
            seed=self.ds_test_seed,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def get_student(self) -> FCNet:
        torch.random.manual_seed(self.student_seed)
        return FCNet(
            input_dim=self.input_dim,
            layer_widths=self.student_widths,
            activation=self.activation.value,
            end_with_activation=self.end_with_activation,
        )

    def get_teacher(self) -> FCNet:
        torch.random.manual_seed(self.teacher_seed)
        teacher_net = FCNet(
            input_dim=self.input_dim,
            layer_widths=self.teacher_widths,
            activation=self.activation.value,
            end_with_activation=self.end_with_activation,
            zero_final_bias=self.for_classification,
        )

        return teacher_net

    def get_optimizer(self, net: nn.Module):
        if self.optimizer is OptimizerT.AdamW:
            return mup.MuAdamW(
                net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        if self.optimizer is OptimizerT.SGD:
            return mup.MuSGD(
                net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        raise ValueError(self.optimizer)


def get_imgs_to_log(
    student_net: FCNet,
    teacher_net: FCNet,
    cfg: ExperimentConfig,
) -> list[wandb.Image]:
    viz_fn = (
        viz.get_student_teacher_viz_clf
        if cfg.for_classification
        else viz.get_student_teacher_viz_reg
    )

    return viz_fn(
        student_net=student_net,
        teacher_net=teacher_net,
        n_viz_dims=cfg.n_viz_dims,
        viz_side_samples=cfg.viz_side_samples,
        input_lo=cfg.input_lo,
        input_hi=cfg.input_hi,
        od_value=cfg.viz_od_value,
        precision_context=cfg.precision_context(student_net),
        batch_size=cfg.eval_batch_size,
    )


@dataclasses.dataclass
class BatchOutput:
    size: int

    loss: torch.Tensor

    # For regression
    mse: Optional[float] = None
    rmse: Optional[float] = None

    # For classification
    acc: Optional[float] = None
    xent: Optional[float] = None


def process_batch(
    xs: torch.Tensor,
    net: FCNet,
    teacher_net: nn.Module,
    cfg: ExperimentConfig,
):
    xs = xs.to(net.device)
    with cfg.precision_context(net):
        preds = net(xs)

    with cfg.precision_context(net):
        with torch.no_grad():
            teacher_preds: torch.Tensor = teacher_net(xs)

    if cfg.for_classification:
        teacher_labs = teacher_preds.argmax(dim=-1)
        xent = F.cross_entropy(input=preds, target=teacher_labs)
        acc = (preds.argmax(dim=-1) == teacher_labs).float().mean()

        return BatchOutput(
            size=xs.shape[0],
            xent=xent.item(),
            acc=acc.item(),
            loss=xent,
        )
    else:
        mse = F.mse_loss(input=preds, target=teacher_preds)
        rmse = torch.sqrt(mse)

        return BatchOutput(
            size=xs.shape[0],
            mse=mse.item(),
            rmse=rmse.item(),
            loss=mse,
        )


def evaluate(
    net: FCNet,
    teacher_net: FCNet,
    loader: LoaderT,
    cfg: ExperimentConfig,
) -> dict[str, Metric[float]]:
    assert isinstance(loader, FastTensorDataLoader)
    n: int = loader.dataset_len

    tot_loss: float = 0

    tot_se: float = 0
    tot_xent: float = 0
    tot_acc: float = 0

    xs: torch.Tensor
    with tqdm(total=len(loader)) as pbar:
        for (xs,) in loader:
            with torch.no_grad():
                bo = process_batch(
                    xs=xs,
                    net=net,
                    teacher_net=teacher_net,
                    cfg=cfg,
                )

            tot_loss += bo.loss.item() * bo.size

            if cfg.for_classification:
                assert bo.xent is not None
                assert bo.acc is not None
                tot_xent += bo.xent * bo.size
                tot_acc += bo.acc * bo.size
            else:
                assert bo.mse is not None
                tot_se += bo.mse * bo.size

            pbar.update(1)

    return dict(loss=Metric(tot_loss / n, "min")) | (
        dict(
            xent=Metric(tot_xent / n, "min"),
            acc=Metric(tot_acc / n, "max"),
        )
        if cfg.for_classification
        else dict(
            mse=Metric(tot_se / n, "min"),
            rmse=Metric(np.sqrt(tot_se / n), "min"),
        )
    )


def train(
    net: FCNet,
    teacher_net: FCNet,
    cfg: ExperimentConfig,
):
    loader_train = cfg.get_loader_train()
    loader_val = cfg.get_loader_val()
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
    last_viz_train_loss: float = np.inf
    with tqdm() as pbar:
        while True:  # Termination will be handled outside
            xs: torch.Tensor
            for (xs,) in loader_train:
                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                bo = process_batch(
                    xs=xs,
                    net=net,
                    teacher_net=teacher_net,
                    cfg=cfg,
                )

                optimizer.zero_grad()
                scaler.scale(bo.loss).backward()  # type: ignore
                scaler.step(optimizer)
                scaler.update()

                n_steps += 1
                pbar.update(1)
                pbar.set_description(
                    f"epochs={n_epochs}; loss={bo.loss.item():6e}; lr={cur_lr:6e}"
                )

                log_dict: dict[str, Metric] = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    net.eval()
                    val_dict = evaluate(
                        net=net,
                        teacher_net=teacher_net,
                        loader=loader_val,
                        cfg=cfg,
                    )
                    net.train()

                    val_loss: float = val_dict["loss"].data
                    lr_scheduler.step(val_loss)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        wandb.run.summary["best_checkpoint_steps"] = n_steps  # type: ignore
                        utils.save_model(net, model_name="student")

                    log_dict |= utils.tag_dict(val_dict, prefix=f"val_")

                if (
                    bo.loss.item()
                    <= cfg.viz_loss_decrease_thresh * last_viz_train_loss
                ):
                    print("Visualizing the student and teacher!")
                    last_viz_train_loss = bo.loss.item()
                    log_dict["train_viz"] = Metric(
                        get_imgs_to_log(
                            student_net=net, teacher_net=teacher_net, cfg=cfg
                        )
                    )

                if len(log_dict) > 0 or n_steps % cfg.log_freq_in_steps == 0:
                    utils.wandb_log(
                        dict(
                            step=Metric(n_steps),
                            epoch=Metric(n_epochs),
                            lr=Metric(cur_lr),
                            train_loss=Metric(bo.loss.item(), "min"),
                        )
                        | (
                            dict(
                                train_xent=Metric(bo.xent, "min"),
                                train_acc=Metric(bo.acc, "max"),
                            )
                            if cfg.for_classification
                            else dict(
                                train_mse=Metric(bo.mse, "min"),
                                train_rmse=Metric(bo.rmse, "min"),
                            )
                        )
                        | log_dict
                    )

            n_epochs += 1


def run_experiment(cfg: ExperimentConfig):
    net: FCNet = cfg.get_student().cuda()

    teacher_net: FCNet = cfg.get_teacher().cuda()
    teacher_net.eval()
    utils.save_model(teacher_net, model_name="teacher")

    torch.random.manual_seed(cfg.global_seed)

    try:
        train(
            net=net,
            teacher_net=teacher_net,
            cfg=cfg,
        )
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    utils.load_model(net, model_name="student")
    net.eval()
    utils.wandb_log(
        {
            "final_viz": Metric(
                get_imgs_to_log(
                    student_net=net, teacher_net=teacher_net, cfg=cfg
                )
            )
        }
    )

    print(f"Starting evaluation of test split...")
    loader_test = cfg.get_loader_test()

    test_dict = evaluate(
        net=net,
        loader=loader_test,
        cfg=cfg,
        teacher_net=teacher_net,
    )
    test_metrics = utils.tag_dict(test_dict, prefix=f"test_")
    for name, metric in test_metrics.items():
        wandb.run.summary[name] = metric.data  # type: ignore
    print(f"Finished evaluation of test split.")


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    wandb_dir = cfg.wandb_dir
    with (
        tempfile.TemporaryDirectory(prefix="/var/tmp/")
        if wandb_dir is None
        else contextlib.nullcontext()
    ) as tmp_wandb_dir:
        if wandb_dir is None:
            assert tmp_wandb_dir is not None
            wandb_dir = tmp_wandb_dir
            print(f"Using temprorary wandb directory: {wandb_dir}")

        # Initialize wandb
        wandb.init(
            entity="data-frugal-learning",
            project="student-teacher-v2",
            dir=wandb_dir,
            tags=cfg.tags,
            config=dataclasses.asdict(cfg),
            save_code=True,
        )

        run_experiment(cfg)

        # Finish wandb
        wandb.finish()


if __name__ == "__main__":
    main()
