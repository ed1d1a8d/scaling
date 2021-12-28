import dataclasses
import logging
import unittest
import warnings

import pytorch_lightning as pl
import torch
from src.experiments.harmonics.data import HypercubeDataModule
from src.experiments.harmonics.poly import ChebPoly, ChebPolyConfig


class TestPoly(unittest.TestCase):
    poly = ChebPoly(
        ChebPolyConfig(
            input_dim=4,
            deg_limit=3,
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

    def test_forward(self):
        simple_poly = ChebPoly(
            dataclasses.replace(
                self.poly.cfg,
                simple_forward=True,
            )
        )

        xs = torch.rand(
            (3, 7, 5, self.poly.cfg.input_dim),
            generator=torch.Generator().manual_seed(42),
        )

        ys = self.poly.forward(xs)
        self.assertEqual(ys.shape, xs.shape[:-1])

        self.assertTrue(torch.allclose(ys, simple_poly.forward(xs)))

    def test_viz(self):
        self.poly.viz_2d(side_samples=512, pad=(self.poly.cfg.input_dim - 3, 1))

    def _get_mse(
        self,
        model: pl.LightningModule,
        dl: torch.utils.data.DataLoader,
    ):
        return pl.Trainer(enable_progress_bar=False, gpus=0).test(
            model=model,
            dataloaders=dl,
            verbose=False,
        )[0]["test_mse"]

    def test_construct_via_lstsq(self):
        dm = HypercubeDataModule(
            fn=self.poly,
            input_dim=self.poly.cfg.input_dim,
            n_train=2 * (self.poly.cfg.deg_limit + 1) ** self.poly.cfg.input_dim,
            n_val=1024,
            train_seed=-1,
            val_seed=-1,
        )
        dm.setup()

        poly_hat_good = ChebPoly.construct_via_lstsq(
            xs=dm.train_ds.tensors[0].numpy(),
            ys=dm.train_ds.tensors[1].numpy(),
            deg_limit=self.poly.cfg.deg_limit,
            freq_limit=2,
            hf_lambda=0,
        )
        self.assertAlmostEqual(
            0,
            self._get_mse(poly_hat_good, dm.val_dataloader()),
        )

        hf_hat_bad = ChebPoly.construct_via_lstsq(
            xs=dm.train_ds.tensors[0].numpy(),
            ys=dm.train_ds.tensors[1].numpy(),
            deg_limit=self.poly.cfg.deg_limit - 1,
            freq_limit=2,
            hf_lambda=0,
        )
        self.assertGreater(
            self._get_mse(hf_hat_bad, dm.val_dataloader()),
            0.05,
        )

        hf_hat_bad_2 = ChebPoly.construct_via_lstsq(
            xs=dm.train_ds.tensors[0].numpy(),
            ys=dm.train_ds.tensors[1].numpy(),
            deg_limit=self.poly.cfg.deg_limit,
            freq_limit=1,
            hf_lambda=10,
        )
        self.assertGreater(
            self._get_mse(hf_hat_bad_2, dm.val_dataloader()),
            0.05,
        )
