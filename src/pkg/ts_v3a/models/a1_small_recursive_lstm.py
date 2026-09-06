"""V3A A1: small_recursive_lstm.

Single LSTM + Dense(1), trained one-step (RECURSIVE) via :class:`NeuralTrainer`.
Multi-step forecasts use pure recursive rollout in scaled space.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.models.base import BaseNeuralForecastModel
from pkg.ts_v3a.models.recursive_rollout import rollout_recursive_forecast
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import NeuralTrainer
from pkg.ts_v3a.types import TargetMode, TrainMetadata


def build_small_recursive_lstm(config: NeuralExperimentConfig) -> Any:
    """Fresh Keras graph: Input(lookback,1) → LSTM(hidden) → Dense(1)."""
    from keras import Sequential
    from keras.layers import Dense, Input, LSTM

    lookback = int(config.lookback)
    units = int(config.hidden_units)
    return Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(units, activation="tanh"),
            Dense(1),
        ],
        name="small_recursive_lstm",
    )


def default_a1_config(**overrides: Any) -> NeuralExperimentConfig:
    """Config defaults for A1 (lookback=12, hidden_units=32)."""
    base = {
        "architecture_name": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
        "lookback": 12,
        "hidden_units": 32,
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


class SmallRecursiveLSTM(BaseNeuralForecastModel):
    """A1 recursive LSTM wrapper around the shared :class:`NeuralTrainer`."""

    name: str = ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        cfg = config or default_a1_config()
        if cfg.architecture_name != ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value:
            cfg = cfg.with_architecture(ArchitectureName.A1_SMALL_RECURSIVE_LSTM)
        super().__init__(cfg)
        self._trainer: Optional[NeuralTrainer] = None
        self._scaler: Optional[FoldScaler] = None
        self._last_lookback_raw: Optional[np.ndarray] = None
        self._architecture_builder = build_small_recursive_lstm

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "SmallRecursiveLSTM":
        """Fit one-step RECURSIVE windows with :class:`NeuralTrainer`."""
        if len(train_series) < self.config.lookback:
            raise ValueError(
                f"Need at least lookback={self.config.lookback} months; "
                f"got {len(train_series)}"
            )
        fold, scaler = prepare_neural_fold(
            train_series,
            forecast_origin=int(window.forecast_origin),
            config=self.config,
            seed=int(seed),
            mode=TargetMode.RECURSIVE,
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
        """Recursive H-step forecast; inverse-transform once; no postprocess."""
        if self._trainer is None or self._trainer.model_ is None:
            raise RuntimeError("SmallRecursiveLSTM.predict called before fit")
        if self._scaler is None or self._last_lookback_raw is None:
            raise RuntimeError("SmallRecursiveLSTM missing scaler or lookback state")

        horizon = len(window.target_dates)
        if horizon < 1:
            raise ValueError("window.target_dates must be non-empty")

        scaled_lookback = self._scaler.transform(self._last_lookback_raw)
        scaled_preds = rollout_recursive_forecast(
            self._trainer.model_,
            scaled_lookback,
            horizon=horizon,
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
            predictions=tuple(float(x) for x in raw_preds.tolist()),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(h) for h in window.horizons),
            metadata=meta,
        )
