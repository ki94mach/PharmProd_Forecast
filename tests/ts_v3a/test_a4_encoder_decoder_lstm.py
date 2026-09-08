"""Tests for V3A A4 encoder_decoder_lstm."""
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
from pkg.ts_v3a.models.a4_encoder_decoder_lstm import (
    EncoderDecoderLSTM,
    build_encoder_decoder_lstm,
    default_a4_config,
    reshape_mimo_targets,
)
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import count_trainable_parameters
from pkg.ts_v3a.types import TargetMode


class _Seq2SeqFakeModel:
    """One-shot fake: returns fixed (1, horizon, 1) and records inputs."""

    def __init__(self, values_15: np.ndarray) -> None:
        flat = np.asarray(values_15, dtype=float).reshape(15)
        self.values = flat.reshape(1, 15, 1)
        self.calls: list[np.ndarray] = []

    def predict(self, X, verbose=0):
        self.calls.append(np.asarray(X, dtype=float).copy())
        return self.values.copy()


class TestA4BuilderAndReshape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_architecture_target_mode(self):
        self.assertEqual(
            target_mode_for(ArchitectureName.A4_ENCODER_DECODER_LSTM),
            TargetMode.DIRECT_MIMO,
        )
        self.assertEqual(target_mode_for("encoder_decoder_lstm"), TargetMode.DIRECT_MIMO)

    def test_output_shape(self):
        cfg = default_a4_config()
        model = build_encoder_decoder_lstm(cfg)
        self.assertEqual(model.input_shape, (None, 12, 1))
        self.assertEqual(model.output_shape, (None, 15, 1))

    def test_encoder_state_propagation(self):
        cfg = default_a4_config()
        model = build_encoder_decoder_lstm(cfg)
        encoder = model.get_layer("encoder_lstm")
        decoder = model.get_layer("decoder_lstm")
        self.assertTrue(encoder.return_state)
        self.assertFalse(encoder.return_sequences)
        self.assertTrue(decoder.return_sequences)
        self.assertEqual(decoder.units, cfg.hidden_units)

        # Encoder with return_state=True exposes (output, state_h, state_c).
        enc_outputs = encoder.output
        if not isinstance(enc_outputs, (list, tuple)):
            enc_outputs = [enc_outputs]
        self.assertEqual(len(enc_outputs), 3)  # output, h, c

        # Decoder inbound should include RepeatVector sequence + encoder states.
        inbound = getattr(decoder, "inbound_nodes", None) or getattr(
            decoder, "_inbound_nodes", []
        )
        self.assertGreaterEqual(len(inbound), 1)
        node = inbound[0]
        inputs = list(
            getattr(node, "input_tensors", None)
            or getattr(node, "arguments", {}).get("args", None)
            or getattr(node, "args", ())
        )
        # Flatten nested structures from Keras 3 call signatures.
        flat: list = []
        stack = list(inputs)
        while stack:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack[0:0] = list(item)
            else:
                flat.append(item)
        self.assertGreaterEqual(len(flat), 3)

        enc_state_ids = {id(t) for t in enc_outputs[1:]}
        input_ids = {id(t) for t in flat}
        self.assertTrue(
            enc_state_ids.issubset(input_ids) or len(enc_state_ids & input_ids) >= 2,
            "decoder should be initialized from encoder state_h/state_c",
        )

    def test_target_reshape_preserves_alignment(self):
        y = np.arange(30, dtype=float).reshape(2, 15)
        out = reshape_mimo_targets(y, horizon=15)
        self.assertEqual(out.shape, (2, 15, 1))
        np.testing.assert_array_equal(out[:, :, 0], y)
        # Idempotent when already 3D
        out2 = reshape_mimo_targets(out, horizon=15)
        np.testing.assert_array_equal(out2, out)


class TestA4PredictContract(unittest.TestCase):
    def test_horizon_alignment_and_no_future_access(self):
        months = [target_month(140201, i + 1) for i in range(40)]
        series = pd.Series(np.linspace(10.0, 100.0, 40), index=months)
        window = make_screening_forecast_window(140501)
        scaled_seq = np.arange(1, 16, dtype=float)
        fake = _Seq2SeqFakeModel(scaled_seq)

        model = EncoderDecoderLSTM(default_a4_config())
        scaler = FoldScaler("standard").fit_on_series(series.to_numpy())
        model._scaler = scaler
        model._last_lookback_raw = series.to_numpy(dtype=float)[-12:].copy()

        class _TrainerStub:
            model_ = fake

        model._trainer = _TrainerStub()  # type: ignore[assignment]

        expected_x = scaler.transform(model._last_lookback_raw).reshape(1, 12, 1)
        result = model.predict(window)

        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertEqual(result.horizons, tuple(range(1, 16)))
        self.assertEqual(result.model_name, "encoder_decoder_lstm")
        expected = scaler.inverse_transform_y(scaled_seq).reshape(-1)
        np.testing.assert_allclose(result.predictions, expected)
        for i, (pred, date, h) in enumerate(
            zip(result.predictions, result.target_dates, result.horizons)
        ):
            self.assertEqual(h, i + 1)
            self.assertEqual(date, window.target_dates[i])
            self.assertEqual(pred, expected[i])

        # Exactly one forward pass on lookback only — no future actuals fed in.
        self.assertEqual(len(fake.calls), 1)
        np.testing.assert_allclose(fake.calls[0], expected_x)
        self.assertEqual(fake.calls[0].shape, (1, 12, 1))

    def test_fold_targets_before_origin(self):
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        window = make_screening_forecast_window(140501)
        fold, _ = prepare_neural_fold(
            series,
            forecast_origin=window.forecast_origin,
            config=default_a4_config(),
            mode=TargetMode.DIRECT_MIMO,
            require_eligible=True,
        )
        n = len(series)
        for end_idx in fold.end_indices_train + fold.end_indices_val:
            self.assertLess(end_idx + 15, n)
            self.assertLess(months[end_idx], window.forecast_origin)


class TestA4EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_short_train_parameter_count(self):
        months = [target_month(140101, i + 1) for i in range(48)]
        series = pd.Series(np.linspace(10.0, 100.0, 48), index=months)
        window = make_screening_forecast_window(140501)
        model = EncoderDecoderLSTM(
            default_a4_config(
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
        result = model.predict(window)
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(result.target_dates, window.target_dates)
        self.assertTrue(all(np.isfinite(result.predictions)))


if __name__ == "__main__":
    unittest.main()
