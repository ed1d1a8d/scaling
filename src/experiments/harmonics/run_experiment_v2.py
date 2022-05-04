import dataclasses
import logging
import warnings
from typing import Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import src.utils as utils
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from simple_parsing import ArgumentParser
from src.experiments.harmonics.data import HypercubeDataModule
from src.experiments.harmonics.fc_net import FCNet, FCNetConfig
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    hf_cfg: HarmonicFnConfig = HarmonicFnConfig()
    net_cfg: FCNetConfig = FCNetConfig()

    input_dim: int = 2

    freq_limit: int = 3  # For ground truth harmonic function
    num_components: int = 16  # For ground truth harmonic function
    true_hf_seed: int = 42

    layer_widths: tuple[int, ...] = (128, 128, 128, 1)

    n_train: int = 100
    train_data_seed: int = 0

    n_val: int = 1024
    val_data_seed: int = -1

    training_seed: int = 42
    batch_size: int = 256
    patience_steps: int = 200
    num_workers: int = 0

    viz_samples: int = 512  # Number of side samples for visualizations
    viz_pad: tuple[int, int] = (0, 0)  # Visualization padding
    viz_value: float = 0.42  # Vizualization value

    @property
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
        steps_in_epoch = utils.ceil_div(self.n_train, self.batch_size)
        patience_ub_in_epochs = utils.ceil_div(self.patience_steps, steps_in_epoch)
        return FCNet(
            dataclasses.replace(
                self.net_cfg,
                sched_patience=patience_ub_in_epochs,
            )
        )


def viz_to_wandb(cfg: ExperimentConfig, fn: Union[HarmonicFn, FCNet], viz_name: str):
    fn.viz_2d(
        side_samples=cfg.viz_samples,
        pad=cfg.viz_pad,
        value=cfg.viz_value,
    )
    wandb.summary({viz_name: plt.gcf()})
    plt.clf()


class CustomCallback(pl.Callback):
    def __init__(self, leave_pbar: bool):
        super().__init__()
        self.pbar = tqdm(leave=leave_pbar)

    def on_train_batch_end(self, trainer: pl.Trainer, *_, **__):
        md = {k: v.item() for k, v in trainer.logged_metrics.items()}
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
                monitor="train_loss",
                patience=max(
                    net.cfg.sched_patience + 10, int(1.5 * net.cfg.sched_patience)
                ),
                mode="min",
            ),
        ],
        enable_progress_bar=False,
        weights_summary=None,
    )

    trainer.fit(
        model=net,
        datamodule=dm,
    )


def evaluate(dm: HypercubeDataModule, net: FCNet):
    def _get_mse(dl: DataLoader):
        return pl.Trainer(
            gpus=1,
            logger=False,
            enable_progress_bar=False,
            weights_summary=None,
        ).test(model=net, dataloaders=dl, verbose=False,)[0]["test_mse"]

    wandb.summary(
        {
            "train_mse": _get_mse(dm.train_dataloader(shuffle=False)),
            "val_mse": _get_mse(dm.val_dataloader(shuffle=False)),
        }
    )


def run_experiment(cfg: ExperimentConfig):
    hf = cfg.get_true_hf()
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
        tags=["nn", "regularization"],
        config=dataclasses.asdict(cfg),
    )

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()
