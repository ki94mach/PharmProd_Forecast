"""Tests for V3A A0 legacy_adaptive_recursive_lstm."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.dates import make_forecast_window, target_month
from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import (
    LegacyAdaptiveRecursiveLSTM,
    build_legacy_adaptive_recursive_lstm,
    config_from_legacy_spec,
    default_a0_config,
    resolve_legacy_architecture,
)
from pkg.ts_v3a.types import TargetMode


# Exact V1 thresholds: (N, l1, l2, lookback, max_epochs)
_BOUNDARY_CASES = (
    (6, 16, 4, 1, 100),
    (7, 16, 4, 3, 100),
    (12, 16, 4, 3, 100),
    (13, 128, 64, 3, 100),
    (24, 128, 64, 3, 100),
    (25, 128, 64, 6, 250),
    (36, 128, 64, 6, 250),
    (37, 512, 256, 12, 500),
)


class TestResolveLegacyArchitecture(unittest.TestCase):
    def test_boundary_table(self):
        for n, l1, l2, lookback, epochs in _BOUNDARY_CASES:
            with self.subTest(n=n):
                spec = resolve_legacy_architecture(n)
                self.assertEqual(spec.history_length, n)
                self.assertEqual(spec.l1, l1)
                self.assertEqual(spec.l2, l2)
                self.assertEqual(spec.lookback, lookback)
                self.assertEqual(spec.max_epochs, epochs)

    def test_fold_isolation_n30_vs_n60(self):
        """N=30 must never pick the N=60 (512/256/12) size."""
        mid = resolve_legacy_architecture(30)
        large = resolve_legacy_architecture(60)
        self.assertEqual((mid.l1, mid.l2, mid.lookback, mid.max_epochs), (128, 64, 6, 250))
        self.assertEqual(
            (large.l1, large.l2, large.lookback, large.max_epochs),
            (512, 256, 12, 500),
        )
        self.assertNotEqual(
            (mid.l1, mid.l2, mid.lookback),
            (large.l1, large.l2, large.lookback),
        )

    def test_config_from_legacy_spec_maps_fields(self):
        spec = resolve_legacy_architecture(30)
        cfg = config_from_legacy_spec(spec)
        self.assertEqual(
            cfg.architecture_name,
            ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
        )
        self.assertEqual(cfg.lookback, 6)
        self.assertEqual(cfg.hidden_units, 128)
        self.assertEqual(cfg.second_hidden_units, 64)
        self.assertEqual(cfg.max_epochs, 250)
        self.assertEqual(cfg.number_layers, 2)
        self.assertEqual(cfg.horizon, 15)

    def test_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM),
            TargetMode.RECURSIVE,
        )
        self.assertEqual(
            target_mode_for("legacy_adaptive_recursive_lstm"),
            TargetMode.RECURSIVE,
        )


class TestA0Builder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_two_lstms_then_dense(self):
        from keras.layers import Dense, LSTM

        spec = resolve_legacy_architecture(30)
        cfg = config_from_legacy_spec(spec)
        model = build_legacy_adaptive_recursive_lstm(cfg)
        self.assertEqual(model.input_shape, (None, 6, 1))
        self.assertEqual(model.output_shape, (None, 1))

        lstms = [layer for layer in model.layers if isinstance(layer, LSTM)]
        denses = [layer for layer in model.layers if isinstance(layer, Dense)]
        self.assertEqual(len(lstms), 2)
        self.assertEqual(len(denses), 1)
        self.assertTrue(lstms[0].return_sequences)
        self.assertFalse(lstms[1].return_sequences)
        self.assertEqual(lstms[0].units, 128)
        self.assertEqual(lstms[1].units, 64)
        self.assertEqual(denses[0].units, 1)


class TestA0FoldIsolationFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_fit_length_30_records_mid_tier_not_large(self):
        """Fit on N=30 records 128/64/lookback=6 even though N=60 is larger."""
        n = 30
        months = [target_month(140201, i + 1) for i in range(n)]
        series = pd.Series(np.linspace(10.0, 100.0, n), index=months)
        # Origin after last train month so len(train_series)==30 is the fold history.
        window = make_forecast_window(target_month(months[-1], 1))

        model = LegacyAdaptiveRecursiveLSTM(
            default_a0_config(
                early_stopping_patience=2,
                batch_size=4,
            )
        ).with_fit_overrides(max_epochs=3, early_stopping_patience=2)
        model.fit(series, window, seed=42)

        self.assertIsNotNone(model.metadata_)
        params = model.metadata_.parameters
        self.assertEqual(params["history_length"], 30)
        self.assertEqual(params["lookback"], 6)
        self.assertEqual(params["l1"], 128)
        self.assertEqual(params["l2"], 64)
        self.assertEqual(params["max_epochs"], 250)
        self.assertEqual(params.get("max_epochs_effective"), 3)
        self.assertIsNotNone(model.metadata_.parameter_count)
        self.assertGreater(model.metadata_.parameter_count, 0)

        # Sanity: separate resolve for 60 is the large tier.
        large = resolve_legacy_architecture(60)
        self.assertEqual(large.lookback, 12)
        self.assertEqual(large.l1, 512)

    def test_smoke_recursive_predict_length_15(self):
        n = 30
        months = [target_month(140201, i + 1) for i in range(n)]
        series = pd.Series(np.linspace(10.0, 100.0, n), index=months)
        window = make_forecast_window(target_month(months[-1], 1))

        model = LegacyAdaptiveRecursiveLSTM(
            default_a0_config(batch_size=4)
        ).with_fit_overrides(max_epochs=2, early_stopping_patience=1)
        model.fit(series, window, seed=41)
        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.model_name, "legacy_adaptive_recursive_lstm")
        self.assertEqual(result.metadata["history_length"], 30)
        self.assertEqual(result.metadata["lookback"], 6)
        self.assertEqual(result.metadata["l1"], 128)
        self.assertEqual(result.metadata["l2"], 64)


if __name__ == "__main__":
    unittest.main()
