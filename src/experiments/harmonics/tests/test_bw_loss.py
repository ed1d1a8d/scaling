import unittest

import numpy as np
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
                            freq_limit=freq_limit,
                            n_samples=2
                            * (2 * freq_limit + 1) ** hf.cfg.input_dim,
                            device=hf.device,
                        )
                    ),
                )
                self.assertGreater(
                    float(
                        bw_loss.high_freq_norm_mcls(
                            fn=hf.forward,
                            input_dim=hf.cfg.input_dim,
                            freq_limit=freq_limit - 1,
                            n_samples=2
                            * (2 * freq_limit + 1) ** hf.cfg.input_dim,
                            device=hf.device,
                        )
                    ),
                    0.1,
                )

    def test_abs_x_plus_y_minus_1(self):
        torch.manual_seed(43)
        with torch.no_grad():
            self.assertAlmostEqual(
                (np.pi**4 - 63) / (18 * np.pi**4),
                bw_loss.high_freq_norm_mcls(
                    fn=lambda xs: torch.abs(xs.sum(axis=-1) - 1),
                    input_dim=2,
                    freq_limit=1,
                    n_samples=400_000,
                    device="cpu",
                    dtype=torch.float64,
                    # use_lstsq=True,
                ).item(),
                places=5,
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
                            freq_limit=2 * freq_limit + 2,
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
                            freq_limit=freq_limit - 1,
                            side_samples=2 * freq_limit + 2,
                            device=hf.device,
                        )
                    ),
                    0.1,
                )

    def test_abs_x_plus_y_minus_1(self):
        with torch.no_grad():
            self.assertAlmostEqual(
                (np.pi**4 - 63) / (18 * np.pi**4),
                bw_loss.high_freq_norm_dft(
                    fn=lambda xs: torch.abs(xs.sum(axis=-1) - 1),
                    input_dim=2,
                    freq_limit=1,
                    side_samples=2_000,
                    device="cpu",
                ).item(),
                places=5,
            )
