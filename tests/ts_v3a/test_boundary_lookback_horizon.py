"""Boundary cases for lookback=12 and horizon=15."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.types import TargetMode
from pkg.ts_v3a.windows import (
    InsufficientHistoryError,
    build_mimo_windows,
    build_recursive_windows,
    expected_n_samples,
)


class TestBoundaryLookbackHorizon(unittest.TestCase):
    def test_mimo_exactly_27_months_one_sample(self):
        history = np.arange(27, dtype=float)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        self.assertEqual(ds.n_samples, 1)
        self.assertEqual(ds.X.shape, (1, 12, 1))
        self.assertEqual(ds.y.shape, (1, 15))
        np.testing.assert_array_equal(ds.X[0, :, 0], history[0:12])
        np.testing.assert_array_equal(ds.y[0], history[12:27])
        self.assertEqual(ds.end_indices[0], 11)

    def test_mimo_26_months_zero_samples(self):
        history = np.arange(26, dtype=float)
        self.assertEqual(expected_n_samples(26, 12, 15, TargetMode.DIRECT_MIMO), 0)
        with self.assertRaises(InsufficientHistoryError):
            build_mimo_windows(history, lookback=12, horizon=15)

    def test_recursive_exactly_13_months_one_sample(self):
        history = np.arange(13, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        self.assertEqual(ds.n_samples, 1)
        np.testing.assert_array_equal(ds.X[0, :, 0], history[0:12])
        self.assertEqual(ds.y[0, 0], history[12])

    def test_default_config_matches_protocol(self):
        cfg = DEFAULT_CONFIG
        self.assertEqual(cfg.lookback, 12)
        self.assertEqual(cfg.horizon, 15)
        self.assertEqual(cfg.hidden_units, 32)
        self.assertEqual(cfg.second_hidden_units, 16)
        self.assertEqual(cfg.dropout, 0.2)
        self.assertEqual(cfg.recurrent_dropout, 0.0)
        self.assertEqual(cfg.optimizer, "Adam")
        self.assertEqual(cfg.learning_rate, 0.001)
        self.assertEqual(cfg.loss, "Huber")
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.max_epochs, 500)
        self.assertEqual(cfg.early_stopping_patience, 30)
        self.assertEqual(cfg.random_seeds, (41, 42, 43))
        self.assertEqual(cfg.scaling_method, "standard")
        self.assertEqual(cfg.number_layers, 1)
        self.assertEqual(cfg.min_internal_train_windows, 8)
        self.assertEqual(cfg.min_internal_validation_windows, 2)

    def test_architecture_target_modes(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM),
            TargetMode.RECURSIVE,
        )
        self.assertEqual(
            target_mode_for("small_recursive_lstm"), TargetMode.RECURSIVE
        )
        self.assertEqual(target_mode_for("mimo_lstm"), TargetMode.DIRECT_MIMO)
        self.assertEqual(
            target_mode_for(ArchitectureName.A6_ATTENTION_BIDIRECTIONAL_LSTM),
            TargetMode.DIRECT_MIMO,
        )

    def test_config_rejects_unknown_architecture(self):
        with self.assertRaises(ValueError):
            NeuralExperimentConfig(architecture_name="not_a_real_arch")


if __name__ == "__main__":
    unittest.main()
