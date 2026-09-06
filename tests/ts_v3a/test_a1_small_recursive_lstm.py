"""Tests for V3A A1 small_recursive_lstm and recursive rollout."""
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
from pkg.ts_v3a.models.a1_small_recursive_lstm import (
    SmallRecursiveLSTM,
    build_small_recursive_lstm,
    default_a1_config,
)
from pkg.ts_v3a.models.recursive_rollout import rollout_recursive_forecast
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import NeuralTrainer, count_trainable_parameters
from pkg.ts_v3a.types import TargetMode
from pkg.ts_v3a.prepare import prepare_neural_fold


class _CountingFakeModel:
    """Deterministic model: predict returns 1, 2, 3, ... and records inputs."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self._n = 0

    def predict(self, X, verbose=0):
        self.calls.append(np.asarray(X, dtype=float).copy())
        self._n += 1
        return np.array([[float(self._n)]], dtype=float)


class TestRecursiveRollout(unittest.TestCase):
    def test_predictions_enter_following_input_windows(self):
        lookback = 4
        initial = np.array([10.0, 20.0, 30.0, 40.0])
        fake = _CountingFakeModel()
        preds = rollout_recursive_forecast(fake, initial, horizon=5)

        np.testing.assert_array_equal(preds, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(len(fake.calls), 5)

        # First call uses the original lookback.
        np.testing.assert_array_equal(
            fake.calls[0].reshape(-1), np.array([10.0, 20.0, 30.0, 40.0])
        )
        # Each subsequent window drops oldest and appends previous prediction.
        expected = [10.0, 20.0, 30.0, 40.0]
        for i in range(1, 5):
            expected = expected[1:] + [float(i)]
            np.testing.assert_array_equal(fake.calls[i].reshape(-1), expected)
            self.assertEqual(fake.calls[i].shape, (1, lookback, 1))

    def test_exactly_fifteen_outputs_shape(self):
        fake = _CountingFakeModel()
        initial = np.zeros(12, dtype=float)
        preds = rollout_recursive_forecast(fake, initial, horizon=15)
        self.assertEqual(preds.shape, (15,))
        self.assertEqual(len(fake.calls), 15)
        np.testing.assert_array_equal(preds, np.arange(1, 16, dtype=float))


class TestA1InverseScalingAndForecastResult(unittest.TestCase):
    def test_inverse_scaling_applied_once_to_rollout(self):
        history = np.linspace(10.0, 100.0, 36)
        scaler = FoldScaler("standard").fit_on_series(history)
        # Fake model emits known scaled constants 0.5, 1.0, 1.5, ...
        class _ScaledFake:
            def __init__(self) -> None:
                self.n = 0

            def predict(self, X, verbose=0):
                self.n += 1
                return np.array([[0.5 * self.n]], dtype=float)

        scaled_lookback = scaler.transform(history[-12:])
        scaled_preds = rollout_recursive_forecast(
            _ScaledFake(), scaled_lookback, horizon=15
        )
        raw = scaler.inverse_transform_y(scaled_preds).reshape(-1)
        expected_scaled = 0.5 * np.arange(1, 16, dtype=float)
        np.testing.assert_allclose(scaled_preds, expected_scaled)
        # Raw forecasts must differ from scaled values for a non-identity scaler.
        self.assertFalse(np.allclose(raw, scaled_preds))
        np.testing.assert_allclose(
            raw, scaler.inverse_transform_y(expected_scaled).reshape(-1)
        )

    def test_forecast_result_matches_v2_window_dates(self):
        """Wire SmallRecursiveLSTM.predict path with injected fake trainer model."""
        months = [target_month(140201, i + 1) for i in range(36)]
        series = pd.Series(np.linspace(10.0, 100.0, 36), index=months)
        window = make_forecast_window(140501)
        self.assertEqual(window.training_end, months[-1])

        model = SmallRecursiveLSTM(
            default_a1_config(max_epochs=1, early_stopping_patience=1)
        )
        # Bypass full Keras train: install scaler + fake model + lookback state.
        scaler = FoldScaler("standard").fit_on_series(series.to_numpy())
        model._scaler = scaler
        model._last_lookback_raw = series.to_numpy(dtype=float)[-12:].copy()

        class _TrainerStub:
            model_ = _CountingFakeModel()

        model._trainer = _TrainerStub()  # type: ignore[assignment]
        result = model.predict(window)

        self.assertEqual(result.model_name, "small_recursive_lstm")
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.horizons, window.horizons)
        # Predictions are inverse-transformed 1..15 in scaled space.
        scaled = np.arange(1, 16, dtype=float)
        expected = scaler.inverse_transform_y(scaled).reshape(-1)
        np.testing.assert_allclose(result.predictions, expected)


class TestA1BuilderAndTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_architecture_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A1_SMALL_RECURSIVE_LSTM),
            TargetMode.RECURSIVE,
        )
        self.assertEqual(
            target_mode_for("small_recursive_lstm"), TargetMode.RECURSIVE
        )

    def test_builder_shapes_and_param_count(self):
        cfg = default_a1_config()
        keras_model = build_small_recursive_lstm(cfg)
        self.assertEqual(keras_model.input_shape, (None, 12, 1))
        self.assertEqual(keras_model.output_shape, (None, 1))
        n_params = count_trainable_parameters(keras_model)
        self.assertGreater(n_params, 0)

    def test_model_recreated_per_fold_and_parameter_count_in_metadata(self):
        history = np.linspace(10.0, 100.0, 36)
        months = [target_month(140201, i + 1) for i in range(36)]
        series = pd.Series(history, index=months)
        window = make_forecast_window(140501)

        call_count = {"n": 0}
        models: list[int] = []

        def counting_builder(config):
            call_count["n"] += 1
            m = build_small_recursive_lstm(config)
            models.append(id(m))
            return m

        cfg = default_a1_config(
            max_epochs=3,
            early_stopping_patience=2,
            batch_size=4,
        )

        # Two independent trainer fits (simulating two CV folds).
        for seed in (41, 42):
            fold, scaler = prepare_neural_fold(
                series,
                forecast_origin=window.forecast_origin,
                config=cfg,
                seed=seed,
                mode=TargetMode.RECURSIVE,
                require_eligible=True,
            )
            trainer = NeuralTrainer(
                architecture_builder=counting_builder, config=cfg
            )
            meta = trainer.fit_prepared(
                train_X=fold.train_X,
                train_y=fold.train_y,
                val_X=fold.val_X,
                val_y=fold.val_y,
                scaler=scaler,
                seed=seed,
                forecast_origin=window.forecast_origin,
            )
            self.assertIsNotNone(meta.parameter_count)
            self.assertGreater(meta.parameter_count, 0)
            self.assertEqual(meta.architecture, "small_recursive_lstm")

        self.assertEqual(call_count["n"], 2)
        self.assertEqual(len(models), 2)
        self.assertNotEqual(models[0], models[1])

    def test_small_recursive_lstm_end_to_end_short_train(self):
        months = [target_month(140201, i + 1) for i in range(36)]
        series = pd.Series(np.linspace(10.0, 100.0, 36), index=months)
        window = make_forecast_window(140501)
        model = SmallRecursiveLSTM(
            default_a1_config(
                max_epochs=3,
                early_stopping_patience=2,
                batch_size=4,
            )
        )
        model._architecture_builder = build_small_recursive_lstm
        model.fit(series, window, seed=42)
        self.assertIsNotNone(model.metadata_)
        self.assertIsNotNone(model.metadata_.parameter_count)
        self.assertGreater(model.metadata_.parameter_count, 0)

        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertTrue(all(np.isfinite(result.predictions)))


if __name__ == "__main__":
    unittest.main()
