"""NeuralTrainer tests with a tiny dummy Keras architecture."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.eligibility import IneligibleForTrainingError
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.trainer import NeuralTrainer


def _dummy_builder(config: NeuralExperimentConfig):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input

    out_dim = 1 if "recursive" in config.architecture_name else int(config.horizon)
    # Tiny MLP — not an A0–A6 LSTM; only exercises the shared trainer.
    return Sequential(
        [
            Input(shape=(config.lookback, 1)),
            Flatten(),
            Dense(4, activation="tanh"),
            Dense(out_dim),
        ]
    )


class TestNeuralTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_trainer_fit_predict_inverse_and_metadata(self):
        # Long enough recursive history for eligibility (24 windows).
        history = np.linspace(10.0, 100.0, 36)
        cfg = NeuralExperimentConfig(
            architecture_name="small_recursive_lstm",
            max_epochs=5,
            early_stopping_patience=3,
            batch_size=4,
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        fold, scaler = prepare_neural_fold(
            history,
            forecast_origin=140501,
            config=cfg,
            seed=42,
            require_eligible=True,
        )
        self.assertTrue(fold.eligible_for_training)
        self.assertIsNotNone(fold.val_X)

        trainer = NeuralTrainer(architecture_builder=_dummy_builder, config=cfg)
        meta = trainer.fit_prepared(
            train_X=fold.train_X,
            train_y=fold.train_y,
            val_X=fold.val_X,
            val_y=fold.val_y,
            scaler=scaler,
            seed=42,
            forecast_origin=140501,
        )
        self.assertEqual(meta.random_seed, 42)
        self.assertIsNotNone(meta.epochs_ran)
        self.assertGreaterEqual(meta.epochs_ran, 1)
        self.assertIsNotNone(meta.best_epoch)
        self.assertIsNotNone(meta.best_val_loss)
        self.assertGreaterEqual(meta.train_window_count, 8)
        self.assertGreaterEqual(meta.validation_window_count, 2)
        self.assertIsNotNone(meta.parameter_count)
        self.assertGreater(meta.parameter_count, 0)
        self.assertIn("mean", meta.scaler_params)
        self.assertEqual(meta.architecture, "small_recursive_lstm")

        preds = trainer.predict(fold.val_X[:2], inverse_transform=True)
        self.assertEqual(preds.shape[0], 2)
        # Inverse-transformed preds should be near raw sales scale, not z-scores.
        self.assertTrue(np.all(np.isfinite(preds)))
        rawish = trainer.predict(fold.val_X[:2], inverse_transform=False)
        self.assertFalse(np.allclose(preds, rawish))

    def test_trainer_refuses_ineligible_sample(self):
        history = np.arange(27, dtype=float)
        cfg = NeuralExperimentConfig(
            architecture_name="mimo_lstm",
            max_epochs=2,
            early_stopping_patience=1,
        )
        fold, scaler = prepare_neural_fold(
            history, forecast_origin=140501, config=cfg, require_eligible=False
        )
        self.assertFalse(fold.eligible_for_training)
        trainer = NeuralTrainer(architecture_builder=_dummy_builder, config=cfg)
        with self.assertRaises(IneligibleForTrainingError):
            trainer.fit_prepared(
                train_X=fold.train_X,
                train_y=fold.train_y,
                val_X=fold.val_X if fold.val_X is not None else np.zeros((0, 12, 1)),
                val_y=fold.val_y if fold.val_y is not None else np.zeros((0, 15)),
                scaler=scaler,
                seed=41,
                forecast_origin=140501,
            )


if __name__ == "__main__":
    unittest.main()
