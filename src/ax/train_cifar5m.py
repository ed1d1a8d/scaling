import dataclasses
import enum
import os
from typing import Any, Optional

import torch
import torch.cuda.amp
import torch.nn.functional as F
import torchattacks.attack
import wandb
from ffcv.loader import Loader
from simple_parsing import ArgumentParser
from src.ax.attack.FastPGD import FastPGD
from src.ax.data import cifar
from src.ax.models.wrn import WideResNet
from torch import nn, optim
from tqdm.auto import tqdm


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def tag_dict(
    d: dict[str, Any],
    prefix: str = "",
    suffix: str = "",
) -> dict[str, Any]:
    return {f"{prefix}{key}{suffix}": val for key, val in d.items()}


class ModelT(enum.Enum):
    WideResNet = enum.auto()


@dataclasses.dataclass
class ExperimentConfig:
    # model params
    model_type: ModelT = ModelT.WideResNet
    depth: int = 28
    width: int = 10  # Only applicable for wide nets

    # train params
    n_train: Optional[int] = 2048
    weight_decay: float = 5e-4
    batch_size: int = 512
    do_adv_training: bool = True
    lr: float = 1e-3
    min_lr: float = 1e-6
    lr_decay_patience_epochs: int = 10

    # eval params
    eval_batch_size: int = 512
    samples_per_eval: int = 512 * 100  # (51200) aka epoch size

    # attack parameter only for CIFAR-10 and SVHN
    adv_eps: float = 8 / 255
    pgd_steps: int = 10

    # Other params
    seed: int = 42
    tags: tuple[str, ...] = ("test",)

    def __post_init__(self):
        assert self.samples_per_eval % self.batch_size == 0

    @property
    def steps_per_eval(self):
        return self.samples_per_eval // self.batch_size

    def get_net(self) -> nn.Module:
        if self.model_type is ModelT.WideResNet:
            return WideResNet(
                depth=self.depth,
                width=self.width,
            )

        raise ValueError(self.model_type)


def evaluate(
    net: nn.Module,
    loader: Loader,
    attack: torchattacks.attack.Attack,
    cfg: ExperimentConfig,
) -> dict[str, float]:
    n: int = len(loader.indices)

    n_correct_nat: float = 0
    n_correct_adv: float = 0
    tot_loss_nat: float = 0
    tot_loss_adv: float = 0

    imgs_nat: torch.Tensor
    labs: torch.Tensor
    for (imgs_nat, labs) in tqdm(loader, leave=False):
        imgs_adv: torch.Tensor = attack(imgs_nat, labs)

        with torch.autocast("cuda"):  # type: ignore
            with torch.no_grad():
                logits_nat = net(imgs_nat)
                logits_adv = net(imgs_adv)

                loss_nat = F.cross_entropy(logits_nat, labs, reduction="sum")
                loss_adv = F.cross_entropy(logits_adv, labs, reduction="sum")

        preds_nat = logits_nat.amax(dim=-1)
        preds_adv = logits_adv.amax(dim=-1)

        n_correct_nat += preds_nat.eq(labs).sum().item()
        n_correct_adv += preds_adv.eq(labs).sum().item()

        tot_loss_nat += loss_nat.item()
        tot_loss_adv += loss_adv.item()

    return dict(
        acc_nat=n_correct_nat / n,
        acc_adv=n_correct_adv / n,
        loss_nat=tot_loss_nat / n,
        loss_adv=tot_loss_adv / n,
        acc=n_correct_adv / n if cfg.do_adv_training else n_correct_nat / n,
        loss=tot_loss_adv / n if cfg.do_adv_training else tot_loss_nat / n,
    )


def train(
    net: nn.Module,
    attack: torchattacks.attack.Attack,
    cfg: ExperimentConfig,
):
    loader_train = cifar.get_loader(
        split="train",
        batch_size=cfg.batch_size,
        n=cfg.n_train,
        random_order=True,
        seed=cfg.seed,
    )
    loader_val = cifar.get_loader(split="val", batch_size=cfg.eval_batch_size)
    print("train and val loaders created!")

    optimizer = optim.AdamW(
        net.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        min_lr=0.1 * cfg.min_lr,
        patience=cfg.lr_decay_patience_epochs,
        verbose=True,
    )
    scaler = torch.cuda.amp.GradScaler()  # type: ignore

    n_steps: int = 0
    n_epochs: int = 0
    with tqdm() as pbar:
        while True:  # Termination will be handled outside
            imgs_nat: torch.Tensor
            labs: torch.Tensor
            for (imgs_nat, labs) in loader_train:
                cur_lr: float = optimizer.param_groups[0]["lr"]
                if cur_lr < cfg.min_lr:
                    print(
                        "Validation loss has stopped improving. Stopping training..."
                    )
                    return

                imgs_adv: torch.Tensor = attack(imgs_nat, labs)

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

                preds_nat = logits_nat.amax(dim=-1)
                preds_adv = logits_adv.amax(dim=-1)
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

                log_dict = dict()
                if n_steps % cfg.steps_per_eval == 1:
                    net.eval()
                    log_dict |= tag_dict(
                        evaluate(net, loader_val, attack, cfg), prefix="val_"
                    )
                    net.train()
                    lr_scheduler.step(log_dict["val_loss"])

                wandb.log(
                    dict(
                        step=n_steps,
                        epoch=n_epochs,
                        lr=cur_lr,
                        loss=loss.item(),
                        acc=acc,
                        loss_nat=loss_nat.item(),
                        loss_adv=loss_adv.item(),
                        acc_nat=acc_nat,
                        acc_adv=acc_adv,
                    )
                    | log_dict
                )

            n_epochs += 1


def run_experiment(cfg: ExperimentConfig):
    torch.manual_seed(cfg.seed)

    net: nn.Module = cfg.get_net().cuda()
    net = net.to(memory_format=torch.channels_last)  # type: ignore

    attack = FastPGD(
        model=net,
        eps=cfg.adv_eps,
        alpha=cfg.adv_eps / cfg.pgd_steps * 2.3,
        steps=cfg.pgd_steps,
        random_start=True,
    )

    train(net, attack, cfg)

    test_loaders = {
        "test": cifar.get_loader(split="test", batch_size=cfg.eval_batch_size),
        "test_orig": cifar.get_loader(
            split="test-orig", batch_size=cfg.eval_batch_size
        ),
        "train_orig": cifar.get_loader(
            split="train-orig", batch_size=cfg.eval_batch_size
        ),
    }
    for name, loader in test_loaders.items():
        print(f"Starting evaluation of {name} split...")
        net.eval()
        test_metrics = tag_dict(
            evaluate(net, loader, attack, cfg), prefix=f"{name}_"
        )
        for k, v in test_metrics.items():
            wandb.run.summary[k] = v  # type: ignore
        print(f"Finished evaluation of {name} split.")

    print("Saving model...")
    torch.save(
        net.state_dict(),
        os.path.join(wandb.run.dir, "net.ckpt"),  # type: ignore
    )
    print("Model saved.")


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="adv-train",
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Save all ckpt files immediately
    wandb.save("*.ckpt")

    run_experiment(cfg)


if __name__ == "__main__":
    main()
