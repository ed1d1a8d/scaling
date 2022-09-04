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
from src.student_teacher_v2 import utils
from src.student_teacher_v2.fc_net import FCNet
from src.student_teacher_v2.utils import Metric
from torch import nn
from tqdm.auto import tqdm

T = TypeVar("T")


class ActivationT(enum.Enum):
    ReLU = nn.ReLU
    LeakyReLU = nn.LeakyReLU
    SiLU = nn.SiLU


class OptimizerT(enum.Enum):
    SGD = enum.auto()
    AdamW = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # Input params
    input_dim: int = 4
    input_lo: float = 0.0
    input_hi: float = 1.0

    # Network params
    activation: ActivationT = ActivationT.ReLU
    teacher_widths: tuple[int, ...] = (2, 1)
    student_width_scale_factor: float = 1.0
    teacher_seed: int = 101
    student_seed: int = 103

    # Optimizer params
    optimizer: OptimizerT = OptimizerT.AdamW
    momentum: float = 0.9  # only for SGD
    weight_decay: float = 5e-4
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_evals: int = 5

    # Dataset params
    n_train: int = 1_000_000  # -1 means infinite data
    n_val: int = 10_000
    n_test: int = 100_000

    ds_train_seed: int = -1
    ds_val_seed: int = -2
    ds_test_seed: int = -3

    # Train params
    batch_size: int = 256

    # Eval params
    eval_batch_size: int = 2048
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size

    # Vizualization params
    n_viz_dims: int = 5
    viz_side_samples: int = 256

    # Other params
    global_seed: int = 42
    num_workers: int = 20
    tags: tuple[str, ...] = ("test",)
    wandb_dir: str = "/home/gridsan/groups/ccg"

    def __post_init__(self):
        assert self.input_lo <= self.input_hi

        # We only support scalar outputs at the moment
        assert self.teacher_widths[-1] == 1

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
            + [1]
        )

    def _get_loader(
        self, n: int, seed: int, batch_size: int
    ) -> torch.utils.data.DataLoader:
        xs = (
            torch.rand(
                size=(n, self.input_dim),
                generator=torch.Generator().manual_seed(seed),
            )
            * (self.input_hi - self.input_lo)
            + self.input_lo
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
        )

    def get_teacher(self) -> FCNet:
        torch.random.manual_seed(self.teacher_seed)
        teacher_net = FCNet(
            input_dim=self.input_dim,
            layer_widths=self.teacher_widths,
            activation=self.activation.value,
        )

        # Set bias to zero so teacher is not a trivial function
        # last_lyr: nn.Linear = teacher_net.net[-1]  # type: ignore
        # last_lyr.bias.data.zero_()

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
    def get_render(net: FCNet, d1: int, d2: int) -> torch.Tensor:
        raw_render = net.render_2d_slice(
            d1=d1,
            d2=d2,
            side_samples=cfg.viz_side_samples,
            batch_size=cfg.eval_batch_size,
            lo=cfg.input_lo,
            hi=cfg.input_hi,
        )
        return torch.tensor(raw_render[np.newaxis, :, :]).float()

    imgs: list[wandb.Image] = []
    n_viz_dims = min(cfg.n_viz_dims, cfg.input_dim)
    for d1 in range(n_viz_dims):
        for d2 in range(d1 + 1, n_viz_dims):
            student_render = get_render(student_net, d1=d1, d2=d2)
            teacher_render = get_render(teacher_net, d1=d1, d2=d2)

            mean = teacher_render.mean()
            std = torch.std(teacher_render) + 1e-6

            student_render = ((student_render - mean) / std).sigmoid()
            teacher_render = ((teacher_render - mean) / std).sigmoid()

            combined_render: np.ndarray = (
                torchvision.utils.make_grid(
                    [student_render, teacher_render], nrow=2, ncol=1
                )
                .cpu()
                .numpy()
            )[0]
            assert combined_render.ndim == 2

            combined_img = PIL.Image.fromarray(np.uint8(combined_render * 255))

            imgs.append(
                wandb.Image(
                    combined_img,
                    caption=f"Student (left), Teacher (right); h={d1} w={d2}",
                )
            )

    return imgs


@dataclasses.dataclass
class BatchOutput:
    size: int

    mse: float
    rmse: float

    loss: torch.Tensor


def process_batch(
    xs: torch.Tensor,
    net: nn.Module,
    teacher_net: nn.Module,
    cfg: ExperimentConfig,
):
    xs = xs.cuda()
    with torch.autocast("cuda"):  # type: ignore
        preds = net(xs)

    with torch.autocast("cuda"):  # type: ignore
        with torch.no_grad():
            teacher_preds = teacher_net(xs)

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
    loader: torch.utils.data.DataLoader,
    cfg: ExperimentConfig,
) -> tuple[dict[str, Metric[float]], list[wandb.Image]]:
    n: int = len(loader.dataset)  # type: ignore

    tot_loss: float = 0
    tot_se: float = 0

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
            tot_se += bo.mse * bo.size

            pbar.update(1)

    return (
        dict(
            loss=Metric(tot_loss / n, "min"),
            mse=Metric(tot_se / n, "min"),
            rmse=Metric(np.sqrt(tot_se / n), "min"),
        ),
        get_imgs_to_log(student_net=net, teacher_net=teacher_net, cfg=cfg),
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
                    f"epochs={n_epochs}; loss={bo.loss.item():6e}; rmse={bo.rmse:.6e}; lr={cur_lr:6e}"
                )

                log_dict: dict[str, Metric] = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    net.eval()
                    val_dict, val_imgs = evaluate(
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
                    log_dict["val_imgs"] = Metric(val_imgs)

                utils.wandb_log(
                    dict(
                        step=Metric(n_steps),
                        epoch=Metric(n_epochs),
                        lr=Metric(cur_lr),
                        train_loss=Metric(bo.loss.item(), "min"),
                        train_mse=Metric(bo.mse, "min"),
                        train_rmse=Metric(bo.rmse, "min"),
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

    loader_test = cfg.get_loader_test()
    net.eval()
    print(f"Starting evaluation of test split...")
    test_dict, test_imgs = evaluate(
        net=net,
        loader=loader_test,
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
        project="student-teacher-v2",
        dir=cfg.wandb_dir,
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
