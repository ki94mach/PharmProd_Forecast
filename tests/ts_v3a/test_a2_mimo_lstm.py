"""Tests for V3A A2 mimo_lstm (one-shot DIRECT/MIMO forecast)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.dates import make_screening_forecast_window, target_month
from pkg.ts_v3a.architectures import ArchitectureName, target_mode_for
from pkg.ts_v3a.models.a2_mimo_lstm import (
    MimoLSTM,
    build_mimo_lstm,
    default_a2_config,
)
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import count_trainable_parameters
from pkg.ts_v3a.types import TargetMode


class _MimoFakeModel:
    """Deterministic MIMO model: one predict returns a fixed (1, horizon) vector."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float).reshape(1, -1)
        self.calls: list[np.ndarray] = []

    def predict(self, X, verbose=0):
        self.calls.append(np.asarray(X, dtype=float).copy())
        return self.values.copy()


class TestA2BuilderAndMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_architecture_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A2_MIMO_LSTM), TargetMode.DIRECT_MIMO
        )
        self.assertEqual(target_mode_for("mimo_lstm"), TargetMode.DIRECT_MIMO)

    def test_builder_output_shape(self):
        cfg = default_a2_config()
        model = build_mimo_lstm(cfg)
        self.assertEqual(model.input_shape, (None, 12, 1))
        self.assertEqual(model.output_shape, (None, 15))
        self.assertGreater(count_trainable_parameters(model), 0)


class TestA2PredictContract(unittest.TestCase):
    def _install_fake(self, model: MimoLSTM, series: pd.Series, fake: _MimoFakeModel):
        scaler = FoldScaler("standard").fit_on_series(series.to_numpy())
        model._scaler = scaler
        model._last_lookback_raw = series.to_numpy(dtype=float)[-12:].copy()

        class _TrainerStub:
            model_ = fake

        model._trainer = _TrainerStub()  # type: ignore[assignment]
        return scaler

    def test_output_shape_horizon_ordering_and_target_dates(self):
        months = [target_month(140201, i + 1) for i in range(40)]
        series = pd.Series(np.linspace(10.0, 100.0, 40), index=months)
        window = make_screening_forecast_window(140501)
        # Identity-ish: use raw 1..15 as "scaled" outputs with a series-fitted scaler
        # so we still check ordering after inverse transform preserves order.
        scaled_out = np.arange(1, 16, dtype=float)
        fake = _MimoFakeModel(scaled_out)
        model = MimoLSTM(default_a2_config())
        scaler = self._install_fake(model, series, fake)

        result = model.predict(window)
        self.assertEqual(result.model_name, "mimo_lstm")
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.horizons, tuple(range(1, 16)))
        self.assertEqual(result.horizons, window.horizons)

        expected = scaler.inverse_transform_y(scaled_out).reshape(-1)
        np.testing.assert_allclose(result.predictions, expected)
        # Ordering: h1 < h2 < ... after inverse of strictly increasing scaled vec
        # for a positive-scale StandardScaler.
        self.assertTrue(
            all(
                result.predictions[i] < result.predictions[i + 1]
                for i in range(14)
            )
        )
        for i, (pred, date, h) in enumerate(
            zip(result.predictions, result.target_dates, result.horizons)
        ):
            self.assertEqual(h, i + 1)
            self.assertEqual(date, window.target_dates[i])
            self.assertEqual(pred, expected[i])

    def test_no_prediction_fed_back_as_input(self):
        months = [target_month(140201, i + 1) for i in range(40)]
        series = pd.Series(np.linspace(10.0, 100.0, 40), index=months)
        window = make_screening_forecast_window(140501)
        fake = _MimoFakeModel(np.arange(1, 16, dtype=float) * 0.1)
        model = MimoLSTM(default_a2_config())
        scaler = self._install_fake(model, series, fake)

        scaled_lookback = scaler.transform(model._last_lookback_raw)
        expected_x = np.asarray(scaled_lookback, dtype=float).reshape(1, 12, 1)

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(len(fake.calls), 1)
        np.testing.assert_allclose(fake.calls[0], expected_x)
        # Input was not mutated / extended with predictions.
        self.assertEqual(fake.calls[0].shape, (1, 12, 1))

    def test_all_predictions_inverse_transformed(self):
        history = np.linspace(10.0, 100.0, 40)
        months = [target_month(140201, i + 1) for i in range(40)]
        series = pd.Series(history, index=months)
        window = make_screening_forecast_window(140501)
        scaled_out = np.linspace(-1.0, 1.0, 15)
        fake = _MimoFakeModel(scaled_out)
        model = MimoLSTM(default_a2_config())
        scaler = self._install_fake(model, series, fake)

        result = model.predict(window)
        expected = scaler.inverse_transform_y(scaled_out).reshape(-1)
        np.testing.assert_allclose(result.predictions, expected)
        self.assertFalse(np.allclose(result.predictions, scaled_out))


class TestA2EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_short_train_and_parameter_count(self):
        # N=48 → 22 MIMO windows: enough for eligibility (8+2).
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        window = make_screening_forecast_window(140501)
        self.assertEqual(months[-1], window.training_end)

        model = MimoLSTM(
            default_a2_config(
                max_epochs=3,
                early_stopping_patience=2,
                batch_size=4,
            )
        )
        model.fit(series, window, seed=42)
        self.assertIsNotNone(model.metadata_)
        self.assertIsNotNone(model.metadata_.parameter_count)
        self.assertGreater(model.metadata_.parameter_count, 0)
        self.assertEqual(model.metadata_.architecture, "mimo_lstm")

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertTrue(all(np.isfinite(result.predictions)))


if __name__ == "__main__":
    unittest.main()
