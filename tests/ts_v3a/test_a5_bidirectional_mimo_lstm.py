"""Tests for V3A A5 bidirectional_mimo_lstm (historical-window-only BiLSTM)."""
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
from pkg.ts_v3a.models.a5_bidirectional_mimo_lstm import (
    BidirectionalMimoLSTM,
    build_bidirectional_mimo_lstm,
    default_a5_config,
)
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import count_trainable_parameters
from pkg.ts_v3a.types import TargetMode


class _MimoFakeModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float).reshape(1, -1)
        self.calls: list[np.ndarray] = []

    def predict(self, X, verbose=0):
        self.calls.append(np.asarray(X, dtype=float).copy())
        return self.values.copy()


class TestA5Builder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_architecture_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM),
            TargetMode.DIRECT_MIMO,
        )
        self.assertEqual(
            target_mode_for("bidirectional_mimo_lstm"), TargetMode.DIRECT_MIMO
        )

    def test_bidirectional_wraps_lstm_output_shape_15(self):
        from keras.layers import Bidirectional, LSTM

        cfg = default_a5_config()
        model = build_bidirectional_mimo_lstm(cfg)
        self.assertEqual(model.output_shape, (None, 15))
        bi_layers = [layer for layer in model.layers if isinstance(layer, Bidirectional)]
        self.assertEqual(len(bi_layers), 1)
        bi = bi_layers[0]
        # Keras 2: .layer; Keras 3: forward_layer / backward_layer
        inner = getattr(bi, "layer", None) or getattr(bi, "forward_layer", None)
        self.assertIsInstance(inner, LSTM)
        self.assertEqual(inner.units, cfg.hidden_units)
        backward = getattr(bi, "backward_layer", None)
        if backward is not None:
            self.assertIsInstance(backward, LSTM)
            self.assertEqual(backward.units, cfg.hidden_units)


class TestA5LeakageBeforeOrigin(unittest.TestCase):
    """Bidirectional scan must only see history with date < forecast_origin."""

    def test_window_end_dates_strictly_before_origin(self):
        origin = 140501
        window = make_forecast_window(origin)
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        self.assertEqual(months[-1], window.training_end)

        fold, _ = prepare_neural_fold(
            series,
            forecast_origin=origin,
            config=default_a5_config(),
            mode=TargetMode.DIRECT_MIMO,
            require_eligible=True,
        )
        outer = set(window.target_dates)
        for end_idx in fold.end_indices_train + fold.end_indices_val:
            end_date = months[end_idx]
            self.assertLess(end_date, origin)
            # Lookback span for this window: months[end_idx-11 : end_idx+1]
            lookback_dates = months[end_idx - 11 : end_idx + 1]
            self.assertEqual(len(lookback_dates), 12)
            self.assertTrue(all(d < origin for d in lookback_dates))
            self.assertTrue(set(lookback_dates).isdisjoint(outer))
            # Targets t+1..t+15 also stay before origin on prepared history
            for h in range(1, 16):
                self.assertLess(months[end_idx + h], origin)
                self.assertNotIn(months[end_idx + h], outer)

    def test_predict_lookback_latest_timestamp_before_origin(self):
        origin = 140501
        window = make_forecast_window(origin)
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)

        model = BidirectionalMimoLSTM(default_a5_config(max_epochs=1))
        # Install state without full train — same lookback selection as fit().
        lookback = 12
        model._last_lookback_raw = series.to_numpy(dtype=float)[-lookback:].copy()
        model._last_lookback_dates = tuple(int(d) for d in series.index[-lookback:])
        scaler = FoldScaler("standard").fit_on_series(series.to_numpy())
        model._scaler = scaler

        class _TrainerStub:
            model_ = _MimoFakeModel(np.arange(1, 16, dtype=float))

        model._trainer = _TrainerStub()  # type: ignore[assignment]

        latest = model._last_lookback_dates[-1]
        self.assertLess(latest, origin)
        self.assertEqual(latest, int(series.index[-1]))
        self.assertEqual(latest, window.training_end)
        self.assertTrue(all(d < origin for d in model._last_lookback_dates))
        self.assertTrue(
            set(model._last_lookback_dates).isdisjoint(set(window.target_dates))
        )

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.model_name, "bidirectional_mimo_lstm")


class TestA5EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_short_train_and_parameter_count(self):
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        window = make_forecast_window(140501)
        model = BidirectionalMimoLSTM(
            default_a5_config(
                max_epochs=3,
                early_stopping_patience=2,
                batch_size=4,
            )
        )
        model.fit(series, window, seed=42)
        self.assertIsNotNone(model.metadata_)
        self.assertGreater(model.metadata_.parameter_count, 0)
        self.assertEqual(
            model.metadata_.parameter_count,
            count_trainable_parameters(model._trainer.model_),
        )
        self.assertIsNotNone(model._last_lookback_dates)
        self.assertLess(model._last_lookback_dates[-1], window.forecast_origin)

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertTrue(all(np.isfinite(result.predictions)))


if __name__ == "__main__":
    unittest.main()
