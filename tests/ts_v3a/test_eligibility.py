"""Neural sample eligibility tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.eligibility import (
    IneligibleForTrainingError,
    evaluate_history_eligibility,
    evaluate_split_eligibility,
)
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.split import chronological_train_val_split
from pkg.ts_v3a.types import FoldSplit, TargetMode, WindowDataset
from pkg.ts_v3a.windows import build_mimo_windows, build_recursive_windows


def _slice_windows(dataset: WindowDataset, start: int, end: int) -> WindowDataset:
    return WindowDataset(
        X=dataset.X[start:end].copy(),
        y=dataset.y[start:end].copy(),
        end_indices=dataset.end_indices[start:end],
        end_dates=dataset.end_dates[start:end],
        mode=dataset.mode,
        lookback=dataset.lookback,
        horizon=dataset.horizon,
    )


class TestEligibility(unittest.TestCase):
    def test_constructible_but_not_eligible_when_too_few_windows(self):
        # MIMO N=27 → 1 window: constructible, not eligible (need 8+2).
        history = np.arange(27, dtype=float)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        self.assertEqual(ds.n_samples, 1)
        split = chronological_train_val_split(ds)
        elig = evaluate_split_eligibility(split, n_windows=ds.n_samples)
        self.assertTrue(elig.mathematically_constructible)
        self.assertFalse(elig.eligible_for_training)
        self.assertEqual(elig.n_validation, 0)
        self.assertIn("validation", elig.reason or "")

    def test_eligible_when_enough_recursive_windows(self):
        # Recursive N=36 → 24 windows; default mins 8/2 should pass.
        history = np.arange(36, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        cfg = NeuralExperimentConfig(
            architecture_name="small_recursive_lstm",
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        split = chronological_train_val_split(
            ds,
            validation_fraction=cfg.validation_fraction,
            min_internal_train_windows=cfg.min_internal_train_windows,
            min_internal_validation_windows=cfg.min_internal_validation_windows,
        )
        elig = evaluate_split_eligibility(split, config=cfg, n_windows=ds.n_samples)
        self.assertTrue(elig.mathematically_constructible)
        self.assertTrue(elig.eligible_for_training)
        self.assertGreaterEqual(elig.n_train, 8)
        self.assertGreaterEqual(elig.n_validation, 2)

    def test_history_eligibility_without_split(self):
        cfg = NeuralExperimentConfig(architecture_name="mimo_lstm")
        elig = evaluate_history_eligibility(
            26, mode=TargetMode.DIRECT_MIMO, config=cfg
        )
        self.assertFalse(elig.mathematically_constructible)

        elig2 = evaluate_history_eligibility(
            40, mode=TargetMode.DIRECT_MIMO, config=cfg
        )
        # N=40 → 40-26=14 windows; need 10 for 8+2 → eligible at total level
        self.assertTrue(elig2.mathematically_constructible)
        self.assertTrue(elig2.eligible_for_training)

    def test_prepare_marks_ineligible_and_require_eligible_raises(self):
        history = np.arange(27, dtype=float)
        fold, _ = prepare_neural_fold(
            history,
            forecast_origin=140501,
            config=NeuralExperimentConfig(architecture_name="mimo_lstm"),
            require_eligible=False,
        )
        self.assertFalse(fold.eligible_for_training)
        self.assertIsNotNone(fold.eligibility_reason)

        with self.assertRaises(IneligibleForTrainingError):
            prepare_neural_fold(
                history,
                forecast_origin=140501,
                config=NeuralExperimentConfig(architecture_name="mimo_lstm"),
                require_eligible=True,
            )

    def test_no_train_only_fitting_path_for_single_window(self):
        history = np.arange(13, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        split = chronological_train_val_split(ds)
        self.assertEqual(split.n_train, 1)
        self.assertEqual(split.n_validation, 0)
        elig = evaluate_split_eligibility(split, n_windows=1)
        self.assertFalse(elig.eligible_for_training)

    def test_insufficient_train_windows_rejected(self):
        # Enough validation, too few train → constructible but not eligible.
        history = np.arange(36, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        train = _slice_windows(ds, 0, 5)
        validation = _slice_windows(ds, 5, 8)
        split = FoldSplit(
            train=train,
            validation=validation,
            n_train=5,
            n_validation=3,
        )
        elig = evaluate_split_eligibility(split, n_windows=8)
        self.assertTrue(elig.mathematically_constructible)
        self.assertFalse(elig.eligible_for_training)
        self.assertEqual(elig.reason, "insufficient_internal_train_windows")

    def test_insufficient_validation_windows_rejected_recursive(self):
        history = np.arange(36, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        train = _slice_windows(ds, 0, 10)
        validation = _slice_windows(ds, 10, 11)  # only 1 val window
        split = FoldSplit(
            train=train,
            validation=validation,
            n_train=10,
            n_validation=1,
        )
        elig = evaluate_split_eligibility(split, n_windows=11)
        self.assertTrue(elig.mathematically_constructible)
        self.assertFalse(elig.eligible_for_training)
        self.assertEqual(elig.reason, "insufficient_internal_validation_windows")

    def test_eligible_when_enough_mimo_windows(self):
        # MIMO N=48 → 22 windows; default mins 8/2 should pass.
        history = np.arange(48, dtype=float)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        cfg = NeuralExperimentConfig(
            architecture_name="mimo_lstm",
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        split = chronological_train_val_split(
            ds,
            validation_fraction=cfg.validation_fraction,
            min_internal_train_windows=cfg.min_internal_train_windows,
            min_internal_validation_windows=cfg.min_internal_validation_windows,
        )
        elig = evaluate_split_eligibility(split, config=cfg, n_windows=ds.n_samples)
        self.assertTrue(elig.mathematically_constructible)
        self.assertTrue(elig.eligible_for_training)
        self.assertGreaterEqual(elig.n_train, 8)
        self.assertGreaterEqual(elig.n_validation, 2)


if __name__ == "__main__":
    unittest.main()
