"""V3A A5: bidirectional_mimo_lstm.

Bidirectional(LSTM) + Dense(horizon), trained DIRECT/MIMO via :class:`NeuralTrainer`.
One-shot forecast (no recursive feedback).

Bidirectional processing operates **only** over the historical input window
``(batch, lookback, 1)``. It never includes the forecast origin, future target
months, or outer CV test observations — those months are excluded before
windowing by ``date < forecast_origin`` and are not present in the lookback
tensor passed to the network.
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


def build_bidirectional_mimo_lstm(config: NeuralExperimentConfig) -> Any:
    """Input → Bidirectional(LSTM) → Dense(horizon).

    The Bidirectional wrapper scans only the historical lookback axis. It does
    not see forecast_origin, future targets, or outer-test months.
    """
    from keras import Sequential
    from keras.layers import Bidirectional, Dense, Input, LSTM

    lookback = int(config.lookback)
    units = int(config.hidden_units)
    horizon = int(config.horizon)
    return Sequential(
        [
            Input(shape=(lookback, 1)),
            Bidirectional(LSTM(units, activation="tanh")),
            Dense(horizon),
        ],
        name="bidirectional_mimo_lstm",
    )


def default_a5_config(**overrides: Any) -> NeuralExperimentConfig:
    """Config defaults for A5 — same training hyperparams as A2."""
    base = {
        "architecture_name": ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM.value,
        "lookback": 12,
        "hidden_units": 32,
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


class BidirectionalMimoLSTM(BaseNeuralForecastModel):
    """A5 bidirectional MIMO LSTM using the shared :class:`NeuralTrainer`."""

    name: str = ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM.value

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        cfg = config or default_a5_config()
        if cfg.architecture_name != ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM.value:
            cfg = cfg.with_architecture(ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM)
        super().__init__(cfg)
        self._trainer: Optional[NeuralTrainer] = None
        self._scaler: Optional[FoldScaler] = None
        self._last_lookback_raw: Optional[np.ndarray] = None
        self._last_lookback_dates: Optional[tuple[int, ...]] = None
        self._architecture_builder = build_bidirectional_mimo_lstm

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "BidirectionalMimoLSTM":
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
        )
        self._trainer = trainer
        self._scaler = scaler
        self.metadata_ = metadata
        lookback = int(self.config.lookback)
        self._last_lookback_raw = (
            train_series.to_numpy(dtype=float)[-lookback:].copy()
        )
        # Dates for leakage assertions: last lookback months are all < origin.
        try:
            dates = [int(d) for d in train_series.index[-lookback:]]
            self._last_lookback_dates = tuple(dates)
        except (TypeError, ValueError):
            self._last_lookback_dates = None
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        """One-shot H-step forecast; inverse-transform once; no postprocess."""
        if self._trainer is None or self._trainer.model_ is None:
            raise RuntimeError("BidirectionalMimoLSTM.predict called before fit")
        if self._scaler is None or self._last_lookback_raw is None:
            raise RuntimeError("BidirectionalMimoLSTM missing scaler or lookback state")

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
                f"bidirectional MIMO predict returned {scaled_preds.shape[0]} "
                f"values; expected {horizon}"
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
