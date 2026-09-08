"""Tests for V3A A3 stacked_mimo_lstm."""
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
from pkg.ts_v3a.models.a3_stacked_mimo_lstm import (
    StackedMimoLSTM,
    build_stacked_mimo_lstm,
    default_a3_config,
)
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import count_trainable_parameters
from pkg.ts_v3a.types import TargetMode


def _lstm_layers(model):
    from keras.layers import LSTM

    return [layer for layer in model.layers if isinstance(layer, LSTM)]


class TestA3BuilderStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_architecture_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A3_STACKED_MIMO_LSTM),
            TargetMode.DIRECT_MIMO,
        )
        self.assertEqual(target_mode_for("stacked_mimo_lstm"), TargetMode.DIRECT_MIMO)

    def test_first_layer_returns_sequences(self):
        cfg = default_a3_config()
        model = build_stacked_mimo_lstm(cfg)
        lstms = _lstm_layers(model)
        self.assertEqual(len(lstms), 2)
        self.assertTrue(lstms[0].return_sequences)
        # Intermediate shape after first LSTM: (batch, lookback, hidden_units)
        from keras import Model

        first_out = Model(inputs=model.inputs, outputs=lstms[0].output)
        self.assertEqual(first_out.output_shape, (None, cfg.lookback, cfg.hidden_units))

    def test_second_layer_returns_final_representation(self):
        cfg = default_a3_config()
        model = build_stacked_mimo_lstm(cfg)
        lstms = _lstm_layers(model)
        self.assertFalse(lstms[1].return_sequences)
        from keras import Model

        second_out = Model(inputs=model.inputs, outputs=lstms[1].output)
        self.assertEqual(second_out.output_shape, (None, cfg.second_hidden_units))

    def test_output_shape_is_fifteen(self):
        cfg = default_a3_config()
        model = build_stacked_mimo_lstm(cfg)
        self.assertEqual(model.input_shape, (None, 12, 1))
        self.assertEqual(model.output_shape, (None, 15))

    def test_serialize_and_config_reconstruct(self):
        cfg = default_a3_config()
        model = build_stacked_mimo_lstm(cfg)
        from keras import Sequential

        reconstructed = Sequential.from_config(model.get_config())
        # Build reconstructed model so shapes are available.
        reconstructed.build(input_shape=(None, cfg.lookback, 1))
        orig_lstms = _lstm_layers(model)
        recon_lstms = _lstm_layers(reconstructed)
        self.assertEqual(len(recon_lstms), 2)
        self.assertTrue(recon_lstms[0].return_sequences)
        self.assertFalse(recon_lstms[1].return_sequences)
        self.assertEqual(reconstructed.output_shape, (None, 15))
        self.assertEqual(len(reconstructed.layers), len(model.layers))
        self.assertEqual(
            count_trainable_parameters(reconstructed),
            count_trainable_parameters(model),
        )
        # Also exercise JSON round-trip.
        from keras.models import model_from_json

        via_json = model_from_json(model.to_json())
        via_json.build(input_shape=(None, cfg.lookback, 1))
        json_lstms = _lstm_layers(via_json)
        self.assertTrue(json_lstms[0].return_sequences)
        self.assertFalse(json_lstms[1].return_sequences)
        self.assertEqual(via_json.output_shape, (None, 15))


class TestA3PredictAndTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_predict_length_fifteen_with_target_dates(self):
        months = [target_month(140201, i + 1) for i in range(40)]
        series = pd.Series(np.linspace(10.0, 100.0, 40), index=months)
        window = make_screening_forecast_window(140501)
        model = StackedMimoLSTM(default_a3_config())
        scaler = FoldScaler("standard").fit_on_series(series.to_numpy())
        model._scaler = scaler
        model._last_lookback_raw = series.to_numpy(dtype=float)[-12:].copy()

        class _Fake:
            def predict(self, X, verbose=0):
                return np.arange(1, 16, dtype=float).reshape(1, 15)

        class _TrainerStub:
            model_ = _Fake()

        model._trainer = _TrainerStub()  # type: ignore[assignment]
        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.model_name, "stacked_mimo_lstm")

    def test_parameter_count_metadata_matches_model(self):
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        window = make_screening_forecast_window(140501)
        model = StackedMimoLSTM(
            default_a3_config(
                max_epochs=3,
                early_stopping_patience=2,
                batch_size=4,
            )
        )
        model.fit(series, window, seed=42)
        self.assertIsNotNone(model.metadata_)
        self.assertIsNotNone(model.metadata_.parameter_count)
        self.assertGreater(model.metadata_.parameter_count, 0)
        self.assertEqual(
            model.metadata_.parameter_count,
            count_trainable_parameters(model._trainer.model_),
        )
        self.assertEqual(model.metadata_.architecture, "stacked_mimo_lstm")
        self.assertEqual(model.config.number_layers, 2)
        self.assertEqual(model.config.second_hidden_units, 16)

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertTrue(all(np.isfinite(result.predictions)))


if __name__ == "__main__":
    unittest.main()
