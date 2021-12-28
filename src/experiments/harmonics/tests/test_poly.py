import dataclasses
import unittest

import torch
from src.experiments.harmonics.poly import ChebPoly, ChebPolyConfig


class TestPoly(unittest.TestCase):
    poly = ChebPoly(
        ChebPolyConfig(
            input_dim=4,
            deg_limit=3,
            num_components=100,
        )
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
