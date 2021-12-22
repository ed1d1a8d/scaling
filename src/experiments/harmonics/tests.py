"""Tests for harmonics."""

import unittest

import src.experiments.harmonics.bw_loss as bw_loss
import torch
from src.experiments.harmonics.harmonics import HarmonicFn, HarmonicFnConfig


class TestHighFreqNormMCLS(unittest.TestCase):
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
                        bw_loss.high_freq_norm_mcls(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=freq_limit,
                            n_samples=4 * (freq_limit + 1) ** hf.cfg.input_dim,
                            device=hf.device,
                        )
                    ),
                )
                self.assertGreater(
                    float(
                        bw_loss.high_freq_norm_mcls(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=freq_limit - 1,
                            n_samples=4 * freq_limit ** hf.cfg.input_dim,
                            device=hf.device,
                        )
                    ),
                    0.1,
                )


class TestHighFreqNormDFT(unittest.TestCase):
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
                        bw_loss.high_freq_norm_dft(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=2 * freq_limit + 2,
                            side_samples=6,
                            device=hf.device,
                        )
                    ),
                )
                self.assertGreater(
                    float(
                        bw_loss.high_freq_norm_dft(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            bandlimit=freq_limit - 1,
                            side_samples=2 * freq_limit + 2,
                            device=hf.device,
                        )
                    ),
                    0.1,
                )
