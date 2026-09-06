"""Scaler leakage and unique-observation fold scaling tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.scaling import (
    FoldScaler,
    fit_fold_scaler,
    unique_observation_indices,
)
from pkg.ts_v3a.split import chronological_train_val_split
from pkg.ts_v3a.windows import build_mimo_windows, build_recursive_windows


class TestScalingLeakage(unittest.TestCase):
    def test_scaler_ignores_validation_and_outer_horizon(self):
        history = np.arange(36, dtype=float) * 10.0
        outer = np.arange(36, 51, dtype=float) * 10.0
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)

        scaler = fit_fold_scaler(history, split.train, method="standard")
        params_before = dict(scaler.params())

        if split.validation is not None:
            split.validation.X[:] = 1e6
            split.validation.y[:] = 1e6
        history_mut = history.copy()
        # Mutate validation-covered indices that are NOT in train unique set.
        train_idxs = set(unique_observation_indices(split.train).tolist())
        val_idxs = set(unique_observation_indices(split.validation).tolist())
        val_only = sorted(val_idxs - train_idxs)
        for i in val_only:
            history_mut[i] = 1e6
        outer[:] = -1e6

        scaler2 = FoldScaler("standard").fit_on_unique_train_observations(
            history_mut, split.train
        )
        self.assertAlmostEqual(scaler2.params()["mean"], params_before["mean"])
        self.assertAlmostEqual(scaler2.params()["scale"], params_before["scale"])

        if val_only:
            leaked = FoldScaler("standard").fit_on_unique_train_observations(
                history_mut, split.validation
            )
            self.assertNotAlmostEqual(leaked.params()["mean"], params_before["mean"])

    def test_inverse_transform_roundtrip_y(self):
        history = np.linspace(10.0, 100.0, 40)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)
        scaler = fit_fold_scaler(history, split.train)
        scaled_y = scaler.transform_windows(split.train).y
        restored = scaler.inverse_transform_y(scaled_y)
        np.testing.assert_allclose(restored, split.train.y, rtol=1e-6, atol=1e-6)

    def test_fit_indices_metadata_matches_unique_train_obs(self):
        history = np.arange(40, dtype=float) * 3.0
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)
        expected = unique_observation_indices(split.train)
        scaler = fit_fold_scaler(history, split.train)
        self.assertEqual(tuple(scaler.fit_indices), tuple(expected.tolist()))
        self.assertEqual(scaler.params()["fit_indices"], list(expected.tolist()))
        # Validation-only indices must not appear in fit metadata.
        if split.validation is not None:
            val_only = set(unique_observation_indices(split.validation).tolist()) - set(
                expected.tolist()
            )
            for i in val_only:
                self.assertNotIn(i, scaler.fit_indices)

    def test_same_scaler_transforms_train_and_validation(self):
        history = np.linspace(5.0, 95.0, 48)
        ds = build_recursive_windows(history, lookback=12)
        split = chronological_train_val_split(
            ds,
            validation_fraction=0.2,
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        self.assertIsNotNone(split.validation)
        scaler = fit_fold_scaler(history, split.train)
        mean = scaler.params()["mean"]
        scale = scaler.params()["scale"]
        scaled = scaler.transform_split(split)

        np.testing.assert_allclose(
            scaled.train.X, (split.train.X - mean) / scale, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            scaled.train.y, (split.train.y - mean) / scale, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            scaled.validation.X,
            (split.validation.X - mean) / scale,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            scaled.validation.y,
            (split.validation.y - mean) / scale,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_overlapping_windows_do_not_alter_scaler_statistics(self):
        # Spike in the middle is repeated across many overlapping windows; a
        # flattened-window fit would overweight it vs unique chronological fit.
        history = np.ones(48, dtype=float)
        history[20] = 1000.0
        ds = build_recursive_windows(history, lookback=12)
        split = chronological_train_val_split(
            ds,
            validation_fraction=0.2,
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        idxs = unique_observation_indices(split.train)
        self.assertIn(20, set(idxs.tolist()))
        flat_overlap = np.concatenate(
            [split.train.X.reshape(-1), split.train.y.reshape(-1)]
        )
        self.assertGreater(flat_overlap.size, idxs.size)
        self.assertGreater(int((flat_overlap == 1000.0).sum()), 1)
        self.assertEqual(int((history[idxs] == 1000.0).sum()), 1)

        unique_scaler = fit_fold_scaler(history, split.train)
        ref = StandardScaler().fit(history[idxs].reshape(-1, 1))
        self.assertAlmostEqual(unique_scaler.params()["mean"], float(ref.mean_[0]))
        self.assertAlmostEqual(unique_scaler.params()["scale"], float(ref.scale_[0]))
        self.assertEqual(unique_scaler.params()["n_observations_fit"], int(idxs.size))

        overlap_scaler = StandardScaler().fit(flat_overlap.reshape(-1, 1))
        self.assertNotAlmostEqual(
            float(overlap_scaler.mean_[0]), unique_scaler.params()["mean"]
        )

        # Duplicating the train dataset must not change unique-observation stats.
        from pkg.ts_v3a.types import WindowDataset

        dup = WindowDataset(
            X=np.concatenate([split.train.X, split.train.X], axis=0),
            y=np.concatenate([split.train.y, split.train.y], axis=0),
            end_indices=split.train.end_indices + split.train.end_indices,
            end_dates=split.train.end_dates + split.train.end_dates,
            mode=split.train.mode,
            lookback=split.train.lookback,
            horizon=split.train.horizon,
        )
        dup_scaler = fit_fold_scaler(history, dup)
        self.assertAlmostEqual(
            dup_scaler.params()["mean"], unique_scaler.params()["mean"]
        )
        self.assertAlmostEqual(
            dup_scaler.params()["scale"], unique_scaler.params()["scale"]
        )
        self.assertEqual(
            dup_scaler.params()["n_observations_fit"],
            unique_scaler.params()["n_observations_fit"],
        )

    def test_fit_on_series_for_future_refit(self):
        history = np.arange(30, dtype=float)
        scaler = FoldScaler("standard").fit_on_series(history)
        self.assertEqual(scaler.params()["n_observations_fit"], 30)
        self.assertAlmostEqual(scaler.params()["mean"], float(history.mean()))


if __name__ == "__main__":
    unittest.main()
