import logging
import unittest
import warnings

import pytorch_lightning as pl
import torch
from src.experiments.harmonics.data import HypercubeDataModule
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig


class TestHarmonicFn(unittest.TestCase):

    hf = HarmonicFn(
        HarmonicFnConfig(
            input_dim=4,
            freq_limit=3,
            num_components=100,
        )
    )

    def setUp(self):
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*GPU available but not used.*",
        )

    def _get_mse(
        self,
        model: pl.LightningModule,
        dl: torch.utils.data.DataLoader,
    ):
        return pl.Trainer(enable_progress_bar=False, gpus=0, logger=False).test(
            model=model,
            dataloaders=dl,
            verbose=False,
        )[0]["test_mse"]

    def test_construct_via_lstsq(self):
        dm = HypercubeDataModule(
            fn=self.hf,
            input_dim=self.hf.cfg.input_dim,
            n_train=2 * (2 * self.hf.cfg.freq_limit + 1) ** self.hf.cfg.input_dim,
            n_val=1024,
            train_seed=-1,
            val_seed=-1,
        )
        dm.setup()

        hf_hat_good = HarmonicFn.construct_via_lstsq(
            xs=dm.train_ds.tensors[0].numpy(),
            ys=dm.train_ds.tensors[1].numpy(),
            freq_limit=self.hf.cfg.freq_limit,
            coeff_threshold=1e-6,
            lamb=1e-9,
        )
        self.assertAlmostEqual(
            0,
            self._get_mse(hf_hat_good, dm.val_dataloader()),
        )

        hf_hat_bad = HarmonicFn.construct_via_lstsq(
            xs=dm.train_ds.tensors[0].numpy(),
            ys=dm.train_ds.tensors[1].numpy(),
            freq_limit=self.hf.cfg.freq_limit - 1,
            coeff_threshold=1e-6,
            lamb=1e-9,
        )
        self.assertGreater(
            self._get_mse(hf_hat_bad, dm.val_dataloader()),
            0.1,
        )
