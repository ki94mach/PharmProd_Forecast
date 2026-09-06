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
from pkg.ts_v3a.seeds import set_global_seeds
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


def _weight_fingerprint(model) -> tuple[float, ...]:
    parts: list[float] = []
    for w in model.get_weights():
        arr = np.asarray(w, dtype=float).ravel()
        parts.extend(arr[:8].tolist())
        parts.append(float(arr.sum()))
        parts.append(float(arr.mean()) if arr.size else 0.0)
    return tuple(parts)


def _eligible_recursive_fold(cfg: NeuralExperimentConfig, seed: int = 42):
    history = np.linspace(10.0, 100.0, 36)
    fold, scaler = prepare_neural_fold(
        history,
        forecast_origin=140501,
        config=cfg,
        seed=seed,
        require_eligible=True,
    )
    return fold, scaler


class TestNeuralTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def _cfg(self, **overrides) -> NeuralExperimentConfig:
        base = dict(
            architecture_name="small_recursive_lstm",
            max_epochs=5,
            early_stopping_patience=3,
            batch_size=4,
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
        )
        base.update(overrides)
        return NeuralExperimentConfig(**base)

    def test_trainer_fit_predict_inverse_and_metadata(self):
        cfg = self._cfg()
        fold, scaler = _eligible_recursive_fold(cfg)
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
            training_start=fold.metadata.training_start,
            training_end=fold.metadata.training_end,
            validation_start=fold.metadata.validation_start,
            validation_end=fold.metadata.validation_end,
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
        self.assertEqual(meta.validation_start, fold.metadata.validation_start)
        self.assertEqual(meta.validation_end, fold.metadata.validation_end)

        # Default predict stays in scaled space.
        scaled_default = trainer.predict(fold.val_X[:2])
        scaled_explicit = trainer.predict(fold.val_X[:2], inverse_transform=False)
        np.testing.assert_allclose(scaled_default, scaled_explicit)
        raw = trainer.predict(fold.val_X[:2], inverse_transform=True)
        self.assertEqual(raw.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(raw)))
        self.assertFalse(np.allclose(raw, scaled_default))

    def test_fresh_model_instance_per_fit(self):
        cfg = self._cfg(max_epochs=2, early_stopping_patience=1)
        fold, scaler = _eligible_recursive_fold(cfg)
        trainer = NeuralTrainer(architecture_builder=_dummy_builder, config=cfg)
        kwargs = dict(
            train_X=fold.train_X,
            train_y=fold.train_y,
            val_X=fold.val_X,
            val_y=fold.val_y,
            scaler=scaler,
            forecast_origin=140501,
        )
        trainer.fit_prepared(seed=41, **kwargs)
        id1 = id(trainer.model_)
        trainer.fit_prepared(seed=42, **kwargs)
        id2 = id(trainer.model_)
        self.assertNotEqual(id1, id2)

    def test_equal_seed_deterministic_initialization(self):
        cfg = self._cfg(max_epochs=1, early_stopping_patience=1)
        fold, scaler = _eligible_recursive_fold(cfg)
        built: list = []

        def capturing_builder(config):
            model = _dummy_builder(config)
            built.append(model)
            return model

        kwargs = dict(
            train_X=fold.train_X,
            train_y=fold.train_y,
            val_X=fold.val_X,
            val_y=fold.val_y,
            scaler=scaler,
            forecast_origin=140501,
            seed=42,
        )
        t1 = NeuralTrainer(architecture_builder=capturing_builder, config=cfg)
        t1.fit_prepared(**kwargs)
        fp1 = _weight_fingerprint(built[0])

        built.clear()
        t2 = NeuralTrainer(architecture_builder=capturing_builder, config=cfg)
        t2.fit_prepared(**kwargs)
        fp2 = _weight_fingerprint(built[0])
        # After one identical training step under the same seed, weights match.
        self.assertEqual(fp1, fp2)

    def test_different_seeds_can_differ_in_initialization(self):
        cfg = self._cfg(max_epochs=1, early_stopping_patience=1)
        fold, scaler = _eligible_recursive_fold(cfg)

        def build_and_fingerprint(seed: int):
            captured: list = []

            def builder_capture(config):
                set_global_seeds(seed)
                m = _dummy_builder(config)
                captured.append(_weight_fingerprint(m))
                return m

            trainer = NeuralTrainer(architecture_builder=builder_capture, config=cfg)
            trainer.fit_prepared(
                train_X=fold.train_X,
                train_y=fold.train_y,
                val_X=fold.val_X,
                val_y=fold.val_y,
                scaler=scaler,
                seed=seed,
                forecast_origin=140501,
            )
            return captured[0]

        fp41 = build_and_fingerprint(41)
        fp42 = build_and_fingerprint(42)
        self.assertNotEqual(fp41, fp42)

    def test_shuffle_false_passed_to_fit(self):
        cfg = self._cfg(max_epochs=1, early_stopping_patience=1)
        fold, scaler = _eligible_recursive_fold(cfg)
        fit_kwargs: dict = {}

        def spying_builder(config):
            model = _dummy_builder(config)
            original_fit = model.fit

            def wrapped_fit(*args, **kwargs):
                fit_kwargs.update(kwargs)
                return original_fit(*args, **kwargs)

            model.fit = wrapped_fit
            return model

        trainer = NeuralTrainer(architecture_builder=spying_builder, config=cfg)
        trainer.fit_prepared(
            train_X=fold.train_X,
            train_y=fold.train_y,
            val_X=fold.val_X,
            val_y=fold.val_y,
            scaler=scaler,
            seed=42,
            forecast_origin=140501,
        )
        self.assertIn("shuffle", fit_kwargs)
        self.assertIs(fit_kwargs["shuffle"], False)

    def test_early_stopping_and_parameter_count_metadata(self):
        cfg = self._cfg(max_epochs=8, early_stopping_patience=2)
        fold, scaler = _eligible_recursive_fold(cfg)
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
        self.assertIsNotNone(meta.epochs_ran)
        self.assertGreaterEqual(meta.epochs_ran, 1)
        self.assertLessEqual(meta.epochs_ran, cfg.max_epochs)
        self.assertIsNotNone(meta.best_epoch)
        self.assertGreaterEqual(meta.best_epoch, 1)
        self.assertLessEqual(meta.best_epoch, meta.epochs_ran)
        self.assertIsNotNone(meta.best_val_loss)
        self.assertTrue(np.isfinite(meta.best_val_loss))
        self.assertIsNotNone(meta.parameter_count)
        self.assertGreater(meta.parameter_count, 0)

    def test_ineligible_rejected_before_builder(self):
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
        calls = {"n": 0}

        def counting_builder(config):
            calls["n"] += 1
            return _dummy_builder(config)

        trainer = NeuralTrainer(architecture_builder=counting_builder, config=cfg)
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
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
