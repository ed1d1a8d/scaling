import contextlib
import dataclasses
import enum
import os
import warnings
from typing import Generic, Optional, TypeVar

import mup
import numpy as np
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils
import wandb
from simple_parsing import ArgumentParser
from src.ax.attack.FastPGD import FastPGD
from src.ax.attack.teacher_loss import TeacherCrossEntropy
from src.ax_student_teacher.fc_net import FCNet
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


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    AdamW = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    input_dim: int = 32
    data_dim: int = 8

    # Network config
    layer_widths: tuple[int, ...] = (96, 192, 2)
    teacher_seed: int = 42
    student_seed: int = 0

    teacher_use_softmax: bool = False

    # optimizer params
    optimizer: OptimizerT = OptimizerT.AdamW
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 5e-4
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # dataset sizes
    n_train: int = 2048  # -1 means infinite data
    n_val: int = 512 * 100
    n_test: int = 512 * 1000

    ds_train_seed: int = -1
    ds_val_seed: int = -2
    ds_test_seed: int = -3

    # train params
    batch_size: int = 256
    do_adv_training: bool = True

    # eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size
    viz_side_samples: int = 256

    # attack parameter only for CIFAR-10 and SVHN
    adv_eps_train: float = 8 / 255
    adv_eps_eval: float = 8 / 255
    pgd_steps: int = 10

    # Other params
    global_seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"

    def __post_init__(self):
        assert self.n_train > 0 or self.n_train == -1
        assert self.data_dim < self.input_dim
        assert self.min_lr < self.lr

    @property
    def steps_per_eval(self):
        return ceil_div(self.samples_per_eval, self.batch_size)

    def prepare_batch(self, xs_raw: torch.Tensor) -> torch.Tensor:
        return F.pad(xs_raw, pad=(0, self.input_dim - self.data_dim), value=0.5)

    def get_loader_train(self):
        xs = torch.rand(
            size=(self.n_train, self.data_dim),
            generator=torch.Generator().manual_seed(self.ds_train_seed),
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_loader_val(self):
        xs = torch.rand(
            size=(self.n_val, self.data_dim),
            generator=torch.Generator().manual_seed(self.ds_val_seed),
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_loader_test(self):
        xs = torch.rand(
            size=(self.n_test, self.data_dim),
            generator=torch.Generator().manual_seed(self.ds_test_seed),
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_student(self) -> FCNet:
        torch.random.manual_seed(self.student_seed)
        return FCNet(
            input_dim=self.input_dim,
            layer_widths=self.layer_widths,
        )

    def get_teacher(self) -> FCNet:
        torch.random.manual_seed(self.student_seed)
        teacher_net = FCNet(
            input_dim=self.input_dim,
            layer_widths=self.layer_widths,
        )

        # Set bias to zero so teacher is not a trivial function
        last_lyr: nn.Linear = teacher_net.net[-1]  # type: ignore
        last_lyr.bias.data.zero_()

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
    net: FCNet,
    teacher_net: FCNet,
    cfg: ExperimentConfig,
) -> list[wandb.Image]:
    student_viz_nat = net.viz_2d(
        side_samples=cfg.viz_side_samples,
        pad=(0, cfg.input_dim - 2),
        value=0.5,
    )[np.newaxis, ...]
    teacher_viz_nat = teacher_net.viz_2d(
        side_samples=cfg.viz_side_samples,
        pad=(0, cfg.input_dim - 2),
        value=0.5,
    )[np.newaxis, ...]

    student_viz_adv = net.viz_2d(
        side_samples=cfg.viz_side_samples,
        pad=(0, cfg.input_dim - 2),
        value=0.5 + cfg.adv_eps_eval,
    )[np.newaxis, ...]
    teacher_viz_adv = teacher_net.viz_2d(
        side_samples=cfg.viz_side_samples,
        pad=(0, cfg.input_dim - 2),
        value=0.5 + cfg.adv_eps_eval,
    )[np.newaxis, ...]

    return [
        wandb.Image(
            torchvision.utils.make_grid(
                [torch.tensor(student_viz_nat), torch.tensor(teacher_viz_nat)],
                nrow=2,
                ncol=1,
            ).float(),
            caption="Nat viz: student (left), teacher (right)",
        ),
        wandb.Image(
            torchvision.utils.make_grid(
                [torch.tensor(student_viz_adv), torch.tensor(teacher_viz_adv)],
                nrow=2,
                ncol=1,
            ).float(),
            caption="Adv viz: student (left), teacher (right)",
        ),
    ]


@dataclasses.dataclass
class BatchOutput:
    size: int

    loss: torch.Tensor
    loss_nat: float
    loss_adv: float

    acc_nat: float
    acc_adv: float
    acc: float


def process_batch(
    xs_nat_raw: torch.Tensor,
    net: nn.Module,
    teacher_net: nn.Module,
    attack: FastPGD,
    cfg: ExperimentConfig,
    eval_mode: bool = False,
):
    xs_nat_raw = xs_nat_raw.cuda()

    xs_nat = cfg.prepare_batch(xs_nat_raw)
    xs_adv: torch.Tensor = attack(xs_nat, torch.Tensor())

    with torch.autocast("cuda"):  # type: ignore
        with torch.no_grad():
            teacher_logits_nat = teacher_net(xs_nat)
            teacher_logits_adv = teacher_net(xs_adv)

    labs_nat = targets_nat = teacher_logits_nat.argmax(dim=-1)
    labs_adv = targets_adv = teacher_logits_adv.argmax(dim=-1)

    if cfg.teacher_use_softmax:
        targets_nat = teacher_logits_nat.softmax(dim=-1)
        targets_adv = teacher_logits_adv.softmax(dim=-1)

    with torch.no_grad() if eval_mode else contextlib.nullcontext():
        with torch.autocast("cuda"):  # type: ignore
            if cfg.do_adv_training:
                logits_adv = net(xs_adv)
                loss_adv = F.cross_entropy(logits_adv, targets_adv)
                loss = loss_adv
                with torch.no_grad():
                    logits_nat = net(xs_nat)
                    loss_nat = F.cross_entropy(logits_nat, targets_nat)
            else:
                logits_nat = net(xs_nat)
                loss_nat = F.cross_entropy(logits_nat, targets_nat)
                loss = loss_nat
                with torch.no_grad():
                    logits_adv = net(xs_adv)
                    loss_adv = F.cross_entropy(logits_adv, targets_adv)

    preds_nat = logits_nat.argmax(dim=-1)
    preds_adv = logits_adv.argmax(dim=-1)
    acc_nat: float = preds_nat.eq(labs_nat).float().mean().item()
    acc_adv: float = preds_adv.eq(labs_adv).float().mean().item()

    return BatchOutput(
        size=xs_nat_raw.shape[0],
        loss=loss,
        loss_nat=loss_nat.item(),
        loss_adv=loss_adv.item(),
        acc_nat=acc_nat,
        acc_adv=acc_adv,
        acc=acc_adv if cfg.do_adv_training else acc_nat,
    )


def evaluate(
    net: FCNet,
    teacher_net: FCNet,
    loader: torch.utils.data.DataLoader,
    attack: FastPGD,
    cfg: ExperimentConfig,
) -> tuple[dict[str, Metric[float]], list[wandb.Image]]:
    n: int = len(loader.dataset)  # type: ignore

    n_correct_nat: float = 0
    n_correct_adv: float = 0
    tot_loss_nat: float = 0
    tot_loss_adv: float = 0

    xs_nat_raw: torch.Tensor
    with tqdm(total=len(loader)) as pbar:
        for (xs_nat_raw,) in loader:
            bo = process_batch(
                xs_nat_raw=xs_nat_raw,
                net=net,
                teacher_net=teacher_net,
                attack=attack,
                cfg=cfg,
                eval_mode=True,
            )

            n_correct_nat += bo.acc_nat * bo.size
            n_correct_adv += bo.acc_adv * bo.size

            tot_loss_nat += bo.loss_nat * bo.size
            tot_loss_adv += bo.loss_adv * bo.size

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
        get_imgs_to_log(net=net, teacher_net=teacher_net, cfg=cfg),
    )


def train(
    net: FCNet,
    teacher_net: FCNet,
    attack_train: FastPGD,
    attack_val: FastPGD,
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
    with tqdm() as pbar:
        while True:  # Termination will be handled outside
            xs_nat_raw: torch.Tensor
            for (xs_nat_raw,) in loader_train:
                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                bo = process_batch(
                    xs_nat_raw=xs_nat_raw,
                    net=net,
                    teacher_net=teacher_net,
                    attack=attack_train,
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
                    net.eval()
                    val_dict, val_imgs = evaluate(
                        net=net,
                        teacher_net=teacher_net,
                        loader=loader_val,
                        attack=attack_val,
                        cfg=cfg,
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

                wandb_log(
                    dict(
                        step=Metric(n_steps),
                        epoch=Metric(n_epochs),
                        lr=Metric(cur_lr),
                        train_loss=Metric(bo.loss.item(), "min"),
                        train_acc=Metric(bo.acc, "max"),
                        train_loss_nat=Metric(bo.loss_nat, "min"),
                        train_loss_adv=Metric(bo.loss_adv, "min"),
                        train_acc_nat=Metric(bo.acc_nat, "max"),
                        train_acc_adv=Metric(bo.acc_adv, "max"),
                    )
                    | log_dict
                )

            n_epochs += 1


def run_experiment(cfg: ExperimentConfig):
    net: FCNet = cfg.get_student().cuda()

    teacher_net: FCNet = cfg.get_teacher().cuda()
    teacher_net.eval()

    attack_loss = TeacherCrossEntropy(teacher_net, use_softmax=False)

    attack_train = FastPGD(
        model=net,
        eps=cfg.adv_eps_train,
        alpha=cfg.adv_eps_train / cfg.pgd_steps * 2.3
        if cfg.pgd_steps > 0
        else 0,
        steps=cfg.pgd_steps,
        random_start=True,
        loss=attack_loss,
    )

    attack_val = FastPGD(
        model=net,
        eps=cfg.adv_eps_eval,
        alpha=cfg.adv_eps_eval / cfg.pgd_steps * 2.3
        if cfg.pgd_steps > 0
        else 0,
        steps=cfg.pgd_steps,
        random_start=True,
        loss=attack_loss,
    )

    attack_test = attack_val

    torch.random.manual_seed(cfg.global_seed)

    try:
        train(
            net=net,
            teacher_net=teacher_net,
            attack_train=attack_train,
            attack_val=attack_val,
            cfg=cfg,
        )
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    load_model(net)

    loader_test = cfg.get_loader_test()
    net.eval()
    print(f"Starting evaluation of test split...")
    test_dict, test_imgs = evaluate(
        net=net,
        loader=loader_test,
        attack=attack_test,
        cfg=cfg,
        teacher_net=teacher_net,
    )

    wandb_log({f"test_imgs": Metric(test_imgs)})
    test_metrics = tag_dict(test_dict, prefix=f"test_")
    for name, metric in test_metrics.items():
        wandb.run.summary[name] = metric.data  # type: ignore

    print(f"Finished evaluation of test split.")


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Uncomment the line below if you don't want to upload ckpt's to wandb
    # os.environ["WANDB_IGNORE_GLOBS"] = "*.ckpt"

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="adv-train-student-teacher",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
