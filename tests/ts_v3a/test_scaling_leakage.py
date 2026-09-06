"""Scaler leakage prevention: fit only on internal-train windows."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.scaling import FoldScaler, fit_fold_scaler
from pkg.ts_v3a.split import chronological_train_val_split
from pkg.ts_v3a.windows import build_mimo_windows


class TestScalingLeakage(unittest.TestCase):
    def test_scaler_ignores_validation_and_outer_horizon(self):
        # Pre-origin history only (indices 0..35). Outer test would be 36..50.
        history = np.arange(36, dtype=float) * 10.0
        outer = np.arange(36, 51, dtype=float) * 10.0  # never used
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)

        scaler = fit_fold_scaler(split.train, method="standard")
        params_before = dict(scaler.params())

        # Mutate validation windows and outer horizon; refit must be identical
        # only if we still fit on the original train slice.
        if split.validation is not None:
            split.validation.X[:] = 1e6
            split.validation.y[:] = 1e6
        outer[:] = -1e6

        scaler2 = FoldScaler("standard").fit_on_train_windows(split.train)
        self.assertEqual(scaler2.params()["mean"], params_before["mean"])
        self.assertEqual(scaler2.params()["scale"], params_before["scale"])

        # Fitting on val instead would change params when val was mutated.
        if split.validation is not None and split.validation.n_samples > 0:
            leaked = FoldScaler("standard").fit_on_train_windows(split.validation)
            self.assertNotAlmostEqual(leaked.params()["mean"], params_before["mean"])

    def test_inverse_transform_roundtrip_y(self):
        history = np.linspace(10.0, 100.0, 40)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)
        scaler = fit_fold_scaler(split.train)
        scaled_y = scaler.transform_windows(split.train).y
        restored = scaler.inverse_transform_y(scaled_y)
        np.testing.assert_allclose(restored, split.train.y, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
