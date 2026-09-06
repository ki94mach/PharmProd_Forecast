"""V3A A4: encoder_decoder_lstm.

Simple Functional seq2seq (no teacher forcing): encoder states seed a
RepeatVector + decoder LSTM + TimeDistributed Dense(1).
Trained as DIRECT/MIMO via :class:`NeuralTrainer` with y reshaped to
``(batch, horizon, 1)``.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.models.base import BaseNeuralForecastModel
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import NeuralTrainer
from pkg.ts_v3a.types import TargetMode


def reshape_mimo_targets(
    y: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Reshape MIMO targets from ``(n, horizon)`` to ``(n, horizon, 1)``."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1 and arr.shape[1] == int(horizon):
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected y shape (n, horizon), got {arr.shape}")
    if arr.shape[1] != int(horizon):
        raise ValueError(
            f"y width {arr.shape[1]} != horizon {horizon}"
        )
    return arr.reshape(arr.shape[0], int(horizon), 1)


def build_encoder_decoder_lstm(config: NeuralExperimentConfig) -> Any:
    """Functional encoder–decoder: states → RepeatVector → decoder → TD Dense(1)."""
    from keras import Model
    from keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed

    lookback = int(config.lookback)
    units = int(config.hidden_units)
    horizon = int(config.horizon)

    encoder_inputs = Input(shape=(lookback, 1), name="encoder_input")
    encoder_lstm = LSTM(units, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    decoder_inputs = RepeatVector(horizon, name="repeat_vector")(encoder_outputs)
    decoder_lstm = LSTM(
        units,
        return_sequences=True,
        name="decoder_lstm",
    )
    decoder_outputs = decoder_lstm(
        decoder_inputs, initial_state=[state_h, state_c]
    )
    outputs = TimeDistributed(Dense(1), name="td_dense")(decoder_outputs)
    return Model(encoder_inputs, outputs, name="encoder_decoder_lstm")


def default_a4_config(**overrides: Any) -> NeuralExperimentConfig:
    """Config defaults for A4 — same training knobs as A2/A3."""
    base = {
        "architecture_name": ArchitectureName.A4_ENCODER_DECODER_LSTM.value,
        "lookback": 12,
        "hidden_units": 32,
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


class EncoderDecoderLSTM(BaseNeuralForecastModel):
    """A4 encoder–decoder LSTM using the shared :class:`NeuralTrainer`."""

    name: str = ArchitectureName.A4_ENCODER_DECODER_LSTM.value

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        cfg = config or default_a4_config()
        if cfg.architecture_name != ArchitectureName.A4_ENCODER_DECODER_LSTM.value:
            cfg = cfg.with_architecture(ArchitectureName.A4_ENCODER_DECODER_LSTM)
        super().__init__(cfg)
        self._trainer: Optional[NeuralTrainer] = None
        self._scaler: Optional[FoldScaler] = None
        self._last_lookback_raw: Optional[np.ndarray] = None
        self._architecture_builder = build_encoder_decoder_lstm

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "EncoderDecoderLSTM":
        """Fit DIRECT/MIMO windows; reshape y to ``(n, H, 1)`` for seq2seq."""
        min_len = int(self.config.lookback) + int(self.config.horizon)
        if len(train_series) < min_len:
            raise ValueError(
                f"Need at least lookback+horizon={min_len} months; "
                f"got {len(train_series)}"
            )
        fold, scaler = prepare_neural_fold(
            train_series,
            forecast_origin=int(window.forecast_origin),
            config=self.config,
            seed=int(seed),
            mode=TargetMode.DIRECT_MIMO,
            require_eligible=True,
        )
        assert fold.val_X is not None and fold.val_y is not None

        horizon = int(self.config.horizon)
        train_y = reshape_mimo_targets(fold.train_y, horizon=horizon)
        val_y = reshape_mimo_targets(fold.val_y, horizon=horizon)

        trainer = NeuralTrainer(
            architecture_builder=self._architecture_builder,
            config=self.config,
        )
        metadata = trainer.fit_prepared(
            train_X=fold.train_X,
            train_y=train_y,
            val_X=fold.val_X,
            val_y=val_y,
            scaler=scaler,
            seed=int(seed),
            forecast_origin=int(window.forecast_origin),
            training_start=fold.metadata.training_start,
            training_end=fold.metadata.training_end,
        )
        self._trainer = trainer
        self._scaler = scaler
        self.metadata_ = metadata
        self._last_lookback_raw = (
            train_series.to_numpy(dtype=float)[-self.config.lookback:].copy()
        )
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        """One-shot H-step sequence forecast; inverse-transform once."""
        if self._trainer is None or self._trainer.model_ is None:
            raise RuntimeError("EncoderDecoderLSTM.predict called before fit")
        if self._scaler is None or self._last_lookback_raw is None:
            raise RuntimeError("EncoderDecoderLSTM missing scaler or lookback state")

        horizon = len(window.target_dates)
        if horizon < 1:
            raise ValueError("window.target_dates must be non-empty")
        if horizon != int(self.config.horizon):
            raise ValueError(
                f"window horizon {horizon} != model horizon {self.config.horizon}"
            )

        scaled_lookback = self._scaler.transform(self._last_lookback_raw)
        x = np.asarray(scaled_lookback, dtype=float).reshape(1, self.config.lookback, 1)
        try:
            scaled_preds = self._trainer.model_.predict(x, verbose=0)
        except TypeError:
            scaled_preds = self._trainer.model_.predict(x)
        scaled_preds = np.asarray(scaled_preds, dtype=float).reshape(-1)
        if scaled_preds.shape[0] != horizon:
            raise RuntimeError(
                f"encoder-decoder predict returned {scaled_preds.shape[0]} values; "
                f"expected {horizon}"
            )

        raw_preds = self._scaler.inverse_transform_y(scaled_preds).reshape(-1)

        meta: dict[str, Any] = {}
        if self.metadata_ is not None:
            meta = {
                "parameter_count": self.metadata_.parameter_count,
                "epochs_ran": self.metadata_.epochs_ran,
                "best_epoch": self.metadata_.best_epoch,
                "best_val_loss": self.metadata_.best_val_loss,
                "random_seed": self.metadata_.random_seed,
            }

        return ForecastResult(
            model_name=self.name,
            predictions=tuple(float(v) for v in raw_preds.tolist()),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(h) for h in window.horizons),
            metadata=meta,
        )
