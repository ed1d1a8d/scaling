import contextlib
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
from src.ax.attack import AllPairsPGD
from src.ax.attack.FastAutoAttack import FastAutoAttack
from src.ax.attack.FastPGD import FastPGD
from src.ax.attack.teacher_loss import TeacherCrossEntropy
from src.ax.data import cifar, mnist, synthetic
from src.ax.models import vit, wrn
from torch import nn
from tqdm.auto import tqdm

T = TypeVar("T")
AttackT = Union[AllPairsPGD.AllPairsPGD, FastPGD, FastAutoAttack]


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
    MNIST20m = enum.auto()
    LightDark = enum.auto()
    HVStripe = enum.auto()
    SquareCircle = enum.auto()


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    AdamW = enum.auto()


class StudentTeacherAttack(enum.Enum):
    PGD = enum.auto()
    AllPairsPGD = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # dataset params
    dataset: DatasetT = DatasetT.CIFAR5m
    LightDark_cfg: synthetic.LightDarkDSConfig = synthetic.LightDarkDSConfig()
    HVStripe_cfg: synthetic.HVStripeDSConfig = synthetic.HVStripeDSConfig()
    SquareCircle_cfg: synthetic.SquareCircleDSConfig = (
        synthetic.SquareCircleDSConfig()
    )
    data_augmentation: bool = False

    use_teacher: bool = False
    teacher_ckpt_path: Optional[str] = None
    zero_teacher_readout_bias: bool = (
        True  # Only used if teacher_ckpt_path == None
    )

    student_teacher_attack: StudentTeacherAttack = (
        StudentTeacherAttack.AllPairsPGD
    )
    student_teacher_use_softmax: bool = False

    # model params
    model: ModelT = ModelT.WideResNet
    depth: int = 28
    width: int = 10  # Only applicable for wide nets

    # optimizer params
    optimizer: OptimizerT = OptimizerT.AdamW
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 5e-4
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # train params
    n_train: int = 2048
    batch_size: int = 512
    do_adv_training: bool = True

    # eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size
    n_imgs_to_log_per_eval: int = 30

    # attack parameter only for CIFAR-10 and SVHN
    adv_eps_train: float = 8 / 255
    adv_eps_eval: float = 8 / 255
    pgd_steps: int = 10
    use_autoattack: bool = False

    # Testing params
    n_test: Optional[int] = None  # Used for speed up testing

    # Other params
    seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"

    def __post_init__(self):
        self.batch_size = min(self.batch_size, self.n_train)

        assert self.min_lr < self.lr

        # We currently don't support running student teacher with autoattack.
        assert not (self.use_autoattack and self.use_teacher)

        assert self.pgd_steps > 0

    @property
    def steps_per_eval(self):
        return ceil_div(self.samples_per_eval, self.batch_size)

    @property
    def data_mean(self):
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return wrn.CIFAR10_MEAN
        if self.dataset is DatasetT.MNIST20m:
            return (0.5,)
        return (0.0, 0.0, 0.0)

    @property
    def data_std(self):
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return wrn.CIFAR10_STD
        if self.dataset is DatasetT.MNIST20m:
            return (1.0,)
        return (1.0, 1.0, 1.0)

    @property
    def num_classes(self):
        if self.dataset in (
            DatasetT.CIFAR5m,
            DatasetT.CIFAR10,
            DatasetT.MNIST20m,
        ):
            return 10
        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return 2

        raise ValueError(self.dataset)

    @property
    def n_channels(self) -> int:
        if self.dataset in (DatasetT.CIFAR5m, DatasetT.CIFAR10):
            return 3
        if self.dataset is DatasetT.MNIST20m:
            return 1

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
                augment=self.data_augmentation,
                batch_size=self.batch_size,
                indices=range(self.n_train),
                random_order=True,
                seed=self.seed,
                num_workers=self.num_workers,
            )

        if self.dataset is DatasetT.CIFAR10:
            return cifar.get_loader(
                split="train-orig",
                augment=self.data_augmentation,
                batch_size=self.batch_size,
                indices=range(self.n_train),
                random_order=True,
                seed=self.seed,
                num_workers=self.num_workers,
            )

        if self.dataset is DatasetT.MNIST20m:
            return mnist.get_loader(
                orig=False,
                split="train",
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

        if self.dataset is DatasetT.MNIST20m:
            return mnist.get_loader(
                orig=False,
                split="val",
                batch_size=self.batch_size,
                num_workers=self.num_workers,
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
                    split="test-orig",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                ),
                "train_orig": cifar.get_loader(
                    split="train-orig",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                ),
                "test": cifar.get_loader(
                    split="test",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                ),
            }

        if self.dataset is DatasetT.CIFAR10:
            return {
                "test_orig": cifar.get_loader(
                    split="test-orig",
                    batch_size=self.eval_batch_size,
                    indices=range(5_000, 10_000)
                    if self.n_test is None
                    else range(5_000, 5_000 + self.n_test),
                ),
                "test_5m": cifar.get_loader(
                    split="test",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                ),
            }

        if self.dataset is DatasetT.MNIST20m:
            return {
                "test_orig": mnist.get_loader(
                    orig=True,
                    split="test",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                    num_workers=self.num_workers,
                ),
                "test": mnist.get_loader(
                    orig=False,
                    split="test",
                    batch_size=self.eval_batch_size,
                    indices=None if self.n_test is None else range(self.n_test),
                    num_workers=self.num_workers,
                ),
            }

        if self.dataset in (
            DatasetT.LightDark,
            DatasetT.HVStripe,
            DatasetT.SquareCircle,
        ):
            return {
                "test": self.get_synthetic_loader(
                    split="test",
                    size=5_000 if self.n_test is None else self.n_test,
                )
            }

        raise ValueError(self.dataset)

    def get_net(self) -> nn.Module:
        if self.model is ModelT.WideResNet:
            return wrn.get_mup_wrn(
                depth=self.depth,
                width=self.width,
                num_classes=self.num_classes,
                mean=self.data_mean,
                std=self.data_std,
                num_input_channels=self.n_channels,
            )

        if self.model is ModelT.VisionTransformer:
            assert self.n_channels == 3
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

    def get_attacks(self, net: nn.Module, teacher_net: Optional[nn.Module]):
        attack_train = FastPGD(
            model=net,
            eps=self.adv_eps_train,
            alpha=self.adv_eps_train / self.pgd_steps * 2.3,
            steps=self.pgd_steps,
            random_start=True,
        )

        attack_val = FastPGD(
            model=net,
            eps=self.adv_eps_eval,
            alpha=self.adv_eps_eval / self.pgd_steps * 2.3,
            steps=self.pgd_steps,
            random_start=True,
        )

        attack_test = (
            FastAutoAttack(net, eps=self.adv_eps_eval)
            if self.use_autoattack
            else attack_val
        )

        if not self.use_teacher:
            assert teacher_net is None

        elif self.student_teacher_attack is StudentTeacherAttack.PGD:
            assert teacher_net is not None
            assert isinstance(attack_test, FastPGD)

            st_loss = TeacherCrossEntropy(
                teacher_net, use_softmax=self.student_teacher_use_softmax
            )
            attack_train.loss = attack_val.loss = attack_test.loss = st_loss

        elif self.student_teacher_attack is StudentTeacherAttack.AllPairsPGD:
            assert teacher_net is not None

            attack_train = AllPairsPGD.AllPairsPGD(
                net1=net,
                net2=teacher_net,
                search_strategy=AllPairsPGD.SearchStrategy.TRAINING,
                eps=self.adv_eps_train,
                alpha=self.adv_eps_train / self.pgd_steps * 2.3,
                steps=self.pgd_steps,
            )

            attack_val = AllPairsPGD.AllPairsPGD(
                net1=net,
                net2=teacher_net,
                search_strategy=AllPairsPGD.SearchStrategy.TRAINING,
                eps=self.adv_eps_eval,
                alpha=self.adv_eps_eval / self.pgd_steps * 2.3,
                steps=self.pgd_steps,
            )

            attack_test = AllPairsPGD.AllPairsPGD(
                net1=net,
                net2=teacher_net,
                search_strategy=AllPairsPGD.SearchStrategy.ALL,
                eps=self.adv_eps_eval,
                alpha=self.adv_eps_eval / self.pgd_steps * 2.3,
                steps=self.pgd_steps,
            )

        else:
            raise ValueError(self.student_teacher_attack)

        return attack_train, attack_val, attack_test

    def cls_name(self, idx: int) -> str:
        if self.dataset is DatasetT.MNIST20m:
            return mnist.cls_name(idx)
        return cifar.cls_name(idx)


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


@dataclasses.dataclass
class BatchOutput:
    size: int

    imgs_nat: torch.Tensor
    imgs_adv: torch.Tensor

    labs_nat: torch.Tensor
    labs_adv: torch.Tensor

    preds_nat: torch.Tensor
    preds_adv: torch.Tensor

    loss: torch.Tensor
    loss_nat: float
    loss_adv: float

    acc_nat: float
    acc_adv: float
    acc: float


def process_batch(
    imgs_nat: torch.Tensor,
    orig_labs: torch.Tensor,
    net: nn.Module,
    teacher_net: Optional[nn.Module],
    attack: AttackT,
    cfg: ExperimentConfig,
    eval_mode: bool = False,
):
    imgs_nat = imgs_nat.cuda()
    orig_labs = orig_labs.cuda()

    imgs_adv: torch.Tensor = attack(imgs_nat, orig_labs)

    labs_nat = labs_adv = orig_labs
    targets_nat = targets_adv = orig_labs
    if teacher_net is not None:
        with torch.autocast("cuda"):  # type: ignore
            with torch.no_grad():
                teacher_logits_nat = teacher_net(imgs_nat)
                teacher_logits_adv = teacher_net(imgs_adv)

        labs_nat = targets_nat = teacher_logits_nat.argmax(dim=-1)
        labs_adv = targets_adv = teacher_logits_adv.argmax(dim=-1)

        if cfg.student_teacher_use_softmax:
            targets_nat = teacher_logits_nat.softmax(dim=-1)
            targets_adv = teacher_logits_adv.softmax(dim=-1)

    with torch.no_grad() if eval_mode else contextlib.nullcontext():
        with torch.autocast("cuda"):  # type: ignore
            if cfg.do_adv_training:
                logits_adv = net(imgs_adv)
                loss_adv = F.cross_entropy(logits_adv, targets_adv)
                loss = loss_adv
                with torch.no_grad():
                    logits_nat = net(imgs_nat)
                    loss_nat = F.cross_entropy(logits_nat, targets_nat)
            else:
                logits_nat = net(imgs_nat)
                loss_nat = F.cross_entropy(logits_nat, targets_nat)
                loss = loss_nat
                with torch.no_grad():
                    logits_adv = net(imgs_adv)
                    loss_adv = F.cross_entropy(logits_adv, targets_adv)

    preds_nat = logits_nat.argmax(dim=-1)
    preds_adv = logits_adv.argmax(dim=-1)
    acc_nat: float = preds_nat.eq(labs_nat).float().mean().item()
    acc_adv: float = preds_adv.eq(labs_adv).float().mean().item()

    return BatchOutput(
        size=imgs_nat.shape[0],
        imgs_nat=imgs_nat,
        imgs_adv=imgs_adv,
        preds_nat=preds_nat,
        preds_adv=preds_adv,
        labs_nat=labs_nat,
        labs_adv=labs_adv,
        loss=loss,
        loss_nat=loss_nat.item(),
        loss_adv=loss_adv.item(),
        acc_nat=acc_nat,
        acc_adv=acc_adv,
        acc=acc_adv if cfg.do_adv_training else acc_nat,
    )


def get_imgs_to_log(
    bo: BatchOutput,
    cfg: ExperimentConfig,
    adv_eps: float,
) -> list[wandb.Image]:
    imgs_diff = (bo.imgs_adv - bo.imgs_nat) / adv_eps / (2 + 1e-9) + 0.5

    def get_caption(i: int) -> str:
        lab_nat = cfg.cls_name(int(bo.labs_nat[i].item()))
        lab_adv = cfg.cls_name(int(bo.labs_adv[i].item()))
        pred_nat = cfg.cls_name(int(bo.preds_nat[i].item()))
        pred_adv = cfg.cls_name(int(bo.preds_adv[i].item()))
        return "\n".join(
            [
                f"preds: {pred_nat}, {pred_adv}; true: ({lab_nat}, {lab_adv})",
                f"img order: nat, adv, diff",
            ]
        )

    return [
        wandb.Image(
            torchvision.utils.make_grid(
                [bo.imgs_nat[i], bo.imgs_adv[i], imgs_diff[i]], nrow=3, ncol=1
            ),
            caption=get_caption(i),
        )
        for i in range(min(cfg.n_imgs_to_log_per_eval, len(bo.imgs_nat)))
    ]


def evaluate(
    net: nn.Module,
    loader: Union[ffcv.loader.Loader, torch.utils.data.DataLoader],
    attack: AttackT,
    cfg: ExperimentConfig,
    teacher_net: Optional[nn.Module],
) -> tuple[dict[str, Metric[float]], list[wandb.Image]]:
    n: int = len(loader.indices) if isinstance(loader, ffcv.loader.Loader) else len(loader.dataset)  # type: ignore

    n_correct_nat: float = 0
    n_correct_adv: float = 0
    tot_loss_nat: float = 0
    tot_loss_adv: float = 0

    imgs_nat: torch.Tensor
    orig_labs: torch.Tensor
    imgs_to_log: list[wandb.Image] = []
    with tqdm(total=len(loader)) as pbar:
        for (imgs_nat, orig_labs) in loader:
            imgs_nat = imgs_nat.cuda()
            orig_labs = orig_labs.cuda()

            bo = process_batch(
                imgs_nat=imgs_nat,
                orig_labs=orig_labs,
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

            if len(imgs_to_log) == 0:
                imgs_to_log = get_imgs_to_log(
                    bo=bo,
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
    attack_train: AttackT,
    attack_val: AttackT,
    cfg: ExperimentConfig,
    teacher_net: Optional[nn.Module],
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
            imgs_nat: torch.Tensor
            orig_labs: torch.Tensor
            for (imgs_nat, orig_labs) in loader_train:
                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                bo = process_batch(
                    imgs_nat=imgs_nat,
                    orig_labs=orig_labs,
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
                        net, loader_val, attack_val, cfg, teacher_net
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
                            bo=bo, cfg=cfg, adv_eps=cfg.adv_eps_train
                        )
                    )

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
    torch.manual_seed(cfg.seed)

    net: nn.Module = cfg.get_net().cuda()
    net = net.to(memory_format=torch.channels_last)  # type: ignore

    teacher_net: Optional[nn.Module] = None
    if cfg.use_teacher:
        teacher_net = cfg.get_net().cuda()
        teacher_net = teacher_net.to(memory_format=torch.channels_last)  # type: ignore
        if cfg.teacher_ckpt_path is not None:
            teacher_net.load_state_dict(torch.load(cfg.teacher_ckpt_path))  # type: ignore
        elif cfg.zero_teacher_readout_bias:
            if isinstance(teacher_net, wrn.WideResNet):
                teacher_net.readout.bias.data.zero_()
            else:
                raise ValueError(teacher_net)
        teacher_net.eval()  # type: ignore

    attack_train, attack_val, attack_test = cfg.get_attacks(
        net=net, teacher_net=teacher_net
    )

    try:
        train(
            net=net,
            attack_train=attack_train,
            attack_val=attack_val,
            cfg=cfg,
            teacher_net=teacher_net,
        )
    except KeyboardInterrupt:  # Catches SIGINT more generally
        print("Training interrupted!")

    load_model(net)

    test_loaders = cfg.get_test_loaders()
    net.eval()
    for split_name, loader in test_loaders.items():
        print(f"Starting evaluation of {split_name} split...")
        test_dict, test_imgs = evaluate(
            net=net,
            loader=loader,
            attack=attack_test,
            cfg=cfg,
            teacher_net=teacher_net,
        )

        wandb_log({f"{split_name}_imgs": Metric(test_imgs)})
        test_metrics = tag_dict(
            test_dict,
            prefix=f"{split_name}_",
            suffix="_autoattack" if cfg.use_autoattack else "",
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
        project="adv-train",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
