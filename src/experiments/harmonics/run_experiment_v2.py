import dataclasses
import logging
import math
import os
import warnings
from typing import Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import ArgumentParser
from src.experiments.harmonics.data import HypercubeDataModule
from src.experiments.harmonics.fc_net import FCNet, FCNetConfig, HFReg
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclasses.dataclass
class ExperimentConfig:
    hf_cfg: HarmonicFnConfig = HarmonicFnConfig(
        input_dim=2,
        freq_limit=2,
        num_components=16,
    )

    net_width: int = 512
    net_cfg: FCNetConfig = FCNetConfig(
        high_freq_reg=HFReg.NONE,
        layer_widths=(),
        sched_verbose=True,
        high_freq_freq_limit=2,
        high_freq_lambda=1,
        high_freq_mcls_samples=20_000,
        sched_monitor="train_loss",
    )

    n_train: int = 100
    train_data_seed: int = 0

    n_val: int = 512
    val_data_seed: int = 1

    training_seed: int = 42
    batch_size: int = 256
    patience_steps: int = 1000
    num_workers: int = 0

    viz_samples: int = 512  # Number of side samples for visualizations
    viz_pad: tuple[int, int] = (0, 0)  # Visualization padding
    viz_value: float = 0.42  # Vizualization value

    tags: tuple[str, ...] = ("test",)

    def __post_init__(self):
        assert self.hf_cfg.input_dim == self.net_cfg.input_dim

    def get_hf(self) -> HarmonicFn:
        return HarmonicFn(cfg=self.hf_cfg)

    def get_dm(self) -> HypercubeDataModule:
        return HypercubeDataModule(
            fn=self.get_hf(),
            input_dim=self.hf_cfg.input_dim,
            n_train=self.n_train,
            train_seed=self.train_data_seed,
            n_val=self.n_val,
            val_seed=self.val_data_seed,
            num_workers=self.num_workers,
        )

    def get_net(self) -> FCNet:
        patience_ub_in_epochs = math.ceil(
            self.patience_steps / (self.n_train / self.batch_size)
        )
        return FCNet(
            dataclasses.replace(
                self.net_cfg,
                layer_widths=(
                    self.net_width,
                    self.net_width,
                    self.net_width,
                    1,
                ),
                sched_patience=patience_ub_in_epochs,
            )
        )


def viz_to_wandb(
    cfg: ExperimentConfig, fn: Union[HarmonicFn, FCNet], viz_name: str
):
    fn.viz_2d(
        side_samples=cfg.viz_samples,
        pad=cfg.viz_pad,
        value=cfg.viz_value,
    )
    wandb.log({viz_name: plt.gcf()})
    plt.clf()


class CustomCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def on_train_batch_end(self, trainer: pl.Trainer, *_, **__):
        md = {k: v.item() for k, v in trainer.logged_metrics.items()} | {
            "lr": trainer.optimizers[0].param_groups[0]["lr"]
        }
        wandb.log(md)

        self.pbar.update()
        self.pbar.set_description(f"train_mse={md['train_mse']: .6e}")


def train(dm: HypercubeDataModule, net: FCNet):
    trainer = pl.Trainer(
        gpus=1,
        deterministic=True,
        logger=False,  # We do custom logging instead.
        log_every_n_steps=1,
        max_epochs=-1,
        callbacks=[
            CustomCallback(),
            EarlyStopping(
                monitor=net.cfg.sched_monitor,
                patience=max(
                    net.cfg.sched_patience + 10,
                    int(1.5 * net.cfg.sched_patience),
                ),
                mode="min",
            ),
        ],
        enable_progress_bar=False,
        weights_summary=None,
        enable_checkpointing=False,
    )

    trainer.fit(
        model=net,
        datamodule=dm,
    )

    # Save model checkpoint
    trainer.save_checkpoint(os.path.join(wandb.run.dir, "net.ckpt"))


def evaluate(dm: HypercubeDataModule, net: FCNet):
    def _get_mse(dl: DataLoader):
        return pl.Trainer(
            gpus=1,
            logger=False,
            enable_progress_bar=False,
            weights_summary=None,
            enable_checkpointing=False,
        ).test(model=net, dataloaders=dl, verbose=False,)[0]["test_mse"]

    wandb.run.summary["final_train_mse"] = _get_mse(
        dm.train_dataloader(shuffle=False)
    )
    wandb.run.summary["final_val_mse"] = _get_mse(dm.val_dataloader())


def run_experiment(cfg: ExperimentConfig):
    hf = cfg.get_hf()
    viz_to_wandb(cfg=cfg, fn=hf, viz_name="true_hf")

    dm = cfg.get_dm()
    dm.setup()

    pl.seed_everything(seed=cfg.training_seed, workers=True)
    net = cfg.get_net()

    viz_to_wandb(cfg=cfg, fn=net, viz_name="init_net")
    train(dm, net)
    viz_to_wandb(cfg=cfg, fn=net, viz_name="trained_net")

    evaluate(dm, net)


def main():
    # Parse config
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Initialize wandb
    wandb.init(
        entity="data-frugal-learning",
        project="harmonic-learning",
        tags=cfg.tags,
        config=dataclasses.asdict(cfg),
        save_code=True,
    )

    # Save all ckpt files immediately
    wandb.save("*.ckpt")

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()
