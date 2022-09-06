import contextlib
import dataclasses
import enum
from typing import TypeVar

import mup
import numpy as np
import PIL.Image
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils
import wandb
from simple_parsing import ArgumentParser
from src.ax.attack import AllPairsPGD
from src.student_teacher_v2 import utils
from src.student_teacher_v2.fc_net import FCNet
from src.student_teacher_v2.utils import Metric
from torch import nn
from tqdm.auto import tqdm

T = TypeVar("T")
AttackT = AllPairsPGD.AllPairsPGD


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
    input_dim: int = 1024

    data_dim: int = 64
    od_val: float = 0.5

    # Network params
    teacher_widths: tuple[int, ...] = (96, 192, 2)
    activation: ActivationT = ActivationT.ReLU
    teacher_seed: int = 103
    teacher_scale: float = 100.0
    teacher_shift: float = -50.0

    student_width_scale_factor: float = 1.0
    student_seed: int = 9001

    # Training objective
    use_soft_objective: bool = False

    # Optimizer params
    optimizer: OptimizerT = OptimizerT.AdamW
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 5e-4
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # Dataset params
    n_train: int = 2048  # -1 means infinite data
    n_val: int = 10_000
    n_test: int = 100_000

    ds_train_seed: int = -1
    ds_val_seed: int = -2
    ds_test_seed: int = -3

    # Train params
    batch_size: int = 256
    do_adv_training: bool = True

    # Eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size

    # Visualization params
    n_viz_dims: int = 6
    viz_side_samples: int = 256

    # Attack params
    adv_eps_train: float = 8 / 255
    adv_eps_eval: float = 8 / 255
    pgd_steps: int = 10

    # Other params
    global_seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"

    def __post_init__(self):
        assert self.data_dim <= self.input_dim
        assert self.n_train > 0 or self.n_train == -1
        assert self.min_lr < self.lr

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

    def prepare_batch(self, xs_raw: torch.Tensor) -> torch.Tensor:
        return F.pad(
            xs_raw,
            pad=(0, self.input_dim - self.data_dim),
            value=self.od_val,
        )

    def _get_loader(
        self, n: int, seed: int, batch_size: int
    ) -> torch.utils.data.DataLoader:
        xs = torch.rand(
            size=(n, self.data_dim),
            generator=torch.Generator().manual_seed(seed),
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_loader_train(self):
        return self._get_loader(
            n=self.n_train,
            seed=self.ds_train_seed,
            batch_size=self.batch_size,
        )

    def get_loader_val(self):
        return self._get_loader(
            n=self.n_val,
            seed=self.ds_val_seed,
            batch_size=self.batch_size,
        )

    def get_loader_test(self):
        return self._get_loader(
            n=self.n_test,
            seed=self.ds_test_seed,
            batch_size=self.batch_size,
        )

    def get_student(self) -> FCNet:
        torch.random.manual_seed(self.student_seed)
        return FCNet(
            input_dim=self.input_dim,
            layer_widths=self.student_widths,
            activation=self.activation.value,
            input_shift=self.teacher_shift,
            input_scale=self.teacher_scale,
        )

    def get_teacher(self) -> FCNet:
        torch.random.manual_seed(self.teacher_seed)
        teacher_net = FCNet(
            input_dim=self.input_dim,
            layer_widths=self.teacher_widths,
            activation=self.activation.value,
            input_shift=self.teacher_shift,
            input_scale=self.teacher_scale,
            zero_final_bias=True,  # Lowers chance of teacher being constant.
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

    def get_attacks(self, net: nn.Module, teacher_net: nn.Module):
        if self.use_soft_objective:
            raise NotImplementedError

        else:
            attack_train = AllPairsPGD.AllPairsPGD(
                net1=net,
                net2=teacher_net,
                search_strategy=AllPairsPGD.SearchStrategy.ALL,
                eps=self.adv_eps_train,
                alpha=self.adv_eps_train / self.pgd_steps * 2.3,
                steps=self.pgd_steps,
                random_start=True,
            )

            attack_val = AllPairsPGD.AllPairsPGD(
                net1=net,
                net2=teacher_net,
                search_strategy=AllPairsPGD.SearchStrategy.ALL,
                eps=self.adv_eps_eval,
                alpha=self.adv_eps_eval / self.pgd_steps * 2.3,
                steps=self.pgd_steps,
                random_start=True,
            )

            attack_test = attack_val

            return attack_train, attack_val, attack_test


def get_imgs_to_log(
    student_net: FCNet,
    teacher_net: FCNet,
    cfg: ExperimentConfig,
) -> list[wandb.Image]:
    assert cfg.teacher_widths[-1] == 2

    def get_render(net: FCNet, d1: int, d2: int, soft: bool) -> torch.Tensor:
        with torch.autocast("cuda"):  # type: ignore
            raw_render = net.render_2d_slice(
                d1=d1,
                d2=d2,
                side_samples=cfg.viz_side_samples,
                batch_size=cfg.eval_batch_size,
                od_value=cfg.od_val,
                output_transform=lambda x: (
                    (x[:, 1] - x[:, 0]).reshape(-1, 1)
                    if soft
                    else x.argmax(dim=-1, keepdim=True)
                ),
            )
        return torch.tensor(raw_render[np.newaxis, :, :]).float()

    imgs: list[wandb.Image] = []
    n_viz_dims = min(cfg.n_viz_dims, cfg.data_dim)
    for d1 in range(n_viz_dims):
        for d2 in range(d1 + 1, n_viz_dims):
            student_render_hard = get_render(
                student_net, d1=d1, d2=d2, soft=False
            )
            teacher_render_hard = get_render(
                teacher_net, d1=d1, d2=d2, soft=False
            )

            student_render_soft = get_render(
                student_net, d1=d1, d2=d2, soft=True
            )
            teacher_render_soft = get_render(
                teacher_net, d1=d1, d2=d2, soft=True
            )

            mean_abs = teacher_render_soft.abs().mean()

            student_render_soft = (student_render_soft / mean_abs).sigmoid()
            teacher_render_soft = (teacher_render_soft / mean_abs).sigmoid()

            combined_render: np.ndarray = (
                torchvision.utils.make_grid(
                    [
                        student_render_soft,
                        teacher_render_soft,
                        student_render_hard,
                        teacher_render_hard,
                    ],
                    nrow=2,
                    ncol=2,
                )
                .cpu()
                .numpy()
            )[0]
            assert combined_render.ndim == 2

            combined_img = PIL.Image.fromarray(np.uint8(combined_render * 255))

            imgs.append(
                wandb.Image(
                    combined_img,
                    caption="\n".join(
                        [
                            f"h={d1} w={d2}",
                            "student soft (top-left), teacher soft (top-right)",
                            "student hard (bot-left), teacher hard (bot-right)",
                        ],
                    ),
                )
            )

    return imgs


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
    net: FCNet,
    teacher_net: nn.Module,
    attack: AttackT,
    cfg: ExperimentConfig,
    eval_mode: bool = False,
):
    xs_nat_raw = xs_nat_raw.to(net.device)

    xs_nat = cfg.prepare_batch(xs_nat_raw)
    xs_adv: torch.Tensor = attack(xs_nat)

    with torch.autocast("cuda"):  # type: ignore
        with torch.no_grad():
            teacher_logits_nat = teacher_net(xs_nat)
            teacher_logits_adv = teacher_net(xs_adv)

    labs_nat = targets_nat = teacher_logits_nat.argmax(dim=-1)
    labs_adv = targets_adv = teacher_logits_adv.argmax(dim=-1)

    if cfg.use_soft_objective:
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
    attack: AttackT,
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
        get_imgs_to_log(
            student_net=net,
            teacher_net=teacher_net,
            cfg=cfg,
        ),
    )


def train(
    net: FCNet,
    teacher_net: FCNet,
    attack_train: AttackT,
    attack_val: AttackT,
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
                        utils.save_model(net, model_name="student")

                    log_dict |= utils.tag_dict(val_dict, prefix=f"val_")
                    log_dict["val_imgs"] = Metric(val_imgs)

                utils.wandb_log(
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
    utils.save_model(teacher_net, model_name="teacher")

    attack_train, attack_val, attack_test = cfg.get_attacks(
        net=net, teacher_net=teacher_net
    )

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

    utils.load_model(net, model_name="student")

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

    utils.wandb_log({f"test_imgs": Metric(test_imgs)})
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

    # Uncomment the line below if you don't want to upload ckpt's to wandb
    # os.environ["WANDB_IGNORE_GLOBS"] = "*.ckpt"

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="student-teacher-v2-ax",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
