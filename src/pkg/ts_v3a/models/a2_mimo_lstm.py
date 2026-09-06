"""V3A A2: mimo_lstm.

Single LSTM + Dense(horizon), trained DIRECT/MIMO via :class:`NeuralTrainer`.
Forecast is one-shot (no recursive feedback).
"""
from __future__ import annotations

from typing import Any, Optional

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


def build_mimo_lstm(config: NeuralExperimentConfig) -> Any:
    """Fresh Keras graph: Input(lookback,1) → LSTM(hidden) → Dense(horizon)."""
    from keras import Sequential
    from keras.layers import Dense, Input, LSTM

    lookback = int(config.lookback)
    units = int(config.hidden_units)
    horizon = int(config.horizon)
    return Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(units, activation="tanh"),
            Dense(horizon),
        ],
        name="mimo_lstm",
    )


def default_a2_config(**overrides: Any) -> NeuralExperimentConfig:
    """Config defaults for A2 — same training hyperparams as A1, MIMO architecture."""
    base = {
        "architecture_name": ArchitectureName.A2_MIMO_LSTM.value,
        "lookback": 12,
        "hidden_units": 32,
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


class MimoLSTM(BaseNeuralForecastModel):
    """A2 MIMO LSTM wrapper around the shared :class:`NeuralTrainer`."""

    name: str = ArchitectureName.A2_MIMO_LSTM.value

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        cfg = config or default_a2_config()
        if cfg.architecture_name != ArchitectureName.A2_MIMO_LSTM.value:
            cfg = cfg.with_architecture(ArchitectureName.A2_MIMO_LSTM)
        super().__init__(cfg)
        self._trainer: Optional[NeuralTrainer] = None
        self._scaler: Optional[FoldScaler] = None
        self._last_lookback_raw: Optional[np.ndarray] = None
        self._architecture_builder = build_mimo_lstm

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "MimoLSTM":
        """Fit DIRECT/MIMO windows with :class:`NeuralTrainer`."""
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

        trainer = NeuralTrainer(
            architecture_builder=self._architecture_builder,
            config=self.config,
        )
        metadata = trainer.fit_prepared(
            train_X=fold.train_X,
            train_y=fold.train_y,
            val_X=fold.val_X,
            val_y=fold.val_y,
            scaler=scaler,
            seed=int(seed),
            forecast_origin=int(window.forecast_origin),
            training_start=fold.metadata.training_start,
            training_end=fold.metadata.training_end,
            validation_start=fold.metadata.validation_start,
            validation_end=fold.metadata.validation_end,
        )
        self._trainer = trainer
        self._scaler = scaler
        self.metadata_ = metadata
        self._last_lookback_raw = (
            train_series.to_numpy(dtype=float)[-self.config.lookback:].copy()
        )
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        """One-shot H-step forecast; inverse-transform once; no recursion/postprocess."""
        if self._trainer is None or self._trainer.model_ is None:
            raise RuntimeError("MimoLSTM.predict called before fit")
        if self._scaler is None or self._last_lookback_raw is None:
            raise RuntimeError("MimoLSTM missing scaler or lookback state")

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
                f"MIMO predict returned {scaled_preds.shape[0]} values; "
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
