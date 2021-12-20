"""Tests for harmonics."""

import unittest

import src.experiments.harmonics.bw_loss as bw_loss
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig
import torch


class TestHighFreqNormMC(unittest.TestCase):
    def test_harmonic_fn(self):
        with torch.no_grad():
            for freq_limit in [2, 3]:
                hf = HarmonicFn(
                    cfg=HarmonicFnConfig(
                        input_dim=4,
                        freq_limit=freq_limit,
                        num_components=100,
                        seed=-1,
                    )
                )
                self.assertAlmostEqual(
                    0,
                    float(
                        bw_loss.high_freq_norm_mc(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=freq_limit,
                            n_samples=1024,
                            device=hf.device,
                        )
                    ),
                )
                self.assertGreater(
                    float(
                        bw_loss.high_freq_norm_mc(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=freq_limit - 1,
                            n_samples=1024,
                            device=hf.device,
                        )
                    ),
                    0.1,
                )
