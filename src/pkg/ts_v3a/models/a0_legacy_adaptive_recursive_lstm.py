"""V3A A0: legacy_adaptive_recursive_lstm.

Preserves the V0/V1 history-dependent two-layer LSTM sizes while training under
the V3A contract (fold-local scaling, early stopping, recursive 15-step forecast).

Architecture is resolved **only** from the history length available inside the
current fold (``len(train_series)`` with ``date < forecast_origin``). Never use
the SKU's eventual/full future history.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.models.base import BaseNeuralForecastModel
from pkg.ts_v3a.models.recursive_rollout import rollout_recursive_forecast
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.trainer import NeuralTrainer
from pkg.ts_v3a.types import TargetMode


@dataclass(frozen=True)
class LegacyArchitectureSpec:
    """V1-style size tier resolved from fold history length only."""

    history_length: int
    l1: int
    l2: int
    lookback: int
    max_epochs: int

    def as_dict(self) -> dict[str, int]:
        return {
            "history_length": int(self.history_length),
            "l1": int(self.l1),
            "l2": int(self.l2),
            "lookback": int(self.lookback),
            "max_epochs": int(self.max_epochs),
        }


def resolve_legacy_architecture(history_length: int) -> LegacyArchitectureSpec:
    """Map fold history length to V1 LSTM sizes (exact thresholds from forecast.py)."""
    n = int(history_length)
    if n < 0:
        raise ValueError(f"history_length must be >= 0, got {n}")
    if n > 36:
        l1, l2, lookback, epochs = 512, 256, 12, 500
    elif n > 24:
        l1, l2, lookback, epochs = 128, 64, 6, 250
    elif n > 12:
        l1, l2, lookback, epochs = 128, 64, 3, 100
    elif n > 6:
        l1, l2, lookback, epochs = 16, 4, 3, 100
    else:
        l1, l2, lookback, epochs = 16, 4, 1, 100
    return LegacyArchitectureSpec(
        history_length=n,
        l1=l1,
        l2=l2,
        lookback=lookback,
        max_epochs=epochs,
    )


def config_from_legacy_spec(
    spec: LegacyArchitectureSpec,
    **overrides: Any,
) -> NeuralExperimentConfig:
    """Build a V3A config from a fold-local legacy architecture resolution."""
    base = {
        "architecture_name": ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
        "lookback": int(spec.lookback),
        "hidden_units": int(spec.l1),
        "second_hidden_units": int(spec.l2),
        "number_layers": 2,
        "max_epochs": int(spec.max_epochs),
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


def default_a0_config(**overrides: Any) -> NeuralExperimentConfig:
    """Placeholder A0 config; real sizes are resolved per fold in ``fit``."""
    # Defaults match the >36 tier until fit resolves fold-local sizes.
    base = {
        "architecture_name": ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
        "lookback": 12,
        "hidden_units": 512,
        "second_hidden_units": 256,
        "number_layers": 2,
        "max_epochs": 500,
        "horizon": 15,
    }
    base.update(overrides)
    return NeuralExperimentConfig(**base)


def build_legacy_adaptive_recursive_lstm(config: NeuralExperimentConfig) -> Any:
    """Fresh Keras graph: LSTM(l1, return_sequences=True) → LSTM(l2) → Dense(1)."""
    from keras import Sequential
    from keras.layers import Dense, Input, LSTM

    lookback = int(config.lookback)
    l1 = int(config.hidden_units)
    l2 = int(config.second_hidden_units)
    return Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(l1, activation="tanh", return_sequences=True),
            LSTM(l2, activation="tanh"),
            Dense(1),
        ],
        name="legacy_adaptive_recursive_lstm",
    )


class LegacyAdaptiveRecursiveLSTM(BaseNeuralForecastModel):
    """A0 legacy adaptive recursive LSTM under the V3A training contract."""

    name: str = ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        # Base config holds shared V3A knobs; lookback/l1/l2/epochs overwritten in fit.
        cfg = config or default_a0_config()
        if cfg.architecture_name != ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value:
            cfg = cfg.with_architecture(
                ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM
            )
        super().__init__(cfg)
        self._trainer: Optional[NeuralTrainer] = None
        self._scaler: Optional[FoldScaler] = None
        self._last_lookback_raw: Optional[np.ndarray] = None
        self._resolved_spec: Optional[LegacyArchitectureSpec] = None
        self._architecture_builder = build_legacy_adaptive_recursive_lstm
        self._fit_overrides: dict[str, Any] = {}

    def with_fit_overrides(self, **overrides: Any) -> "LegacyAdaptiveRecursiveLSTM":
        """Store overrides applied after fold resolution (e.g. short max_epochs in tests)."""
        self._fit_overrides = dict(overrides)
        return self

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "LegacyAdaptiveRecursiveLSTM":
        """Resolve architecture from fold history length, then train recursively."""
        history_length = len(train_series)
        spec = resolve_legacy_architecture(history_length)
        self._resolved_spec = spec

        # Preserve shared V3A knobs; allow post-resolve max_epochs (tests) only.
        # Fold-local lookback/l1/l2 always come from ``spec``.
        shared = self.config.as_parameters_dict()
        shared.update(self._fit_overrides)
        for key in (
            "architecture_name",
            "lookback",
            "hidden_units",
            "second_hidden_units",
            "number_layers",
            "horizon",
        ):
            shared.pop(key, None)
        fold_cfg = config_from_legacy_spec(spec, **shared)
        # Re-assert size fields in case overrides tried to change them.
        fold_params = fold_cfg.as_parameters_dict()
        fold_params["architecture_name"] = (
            ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value
        )
        fold_params["lookback"] = int(spec.lookback)
        fold_params["hidden_units"] = int(spec.l1)
        fold_params["second_hidden_units"] = int(spec.l2)
        fold_params["number_layers"] = 2
        fold_params["horizon"] = 15
        fold_cfg = NeuralExperimentConfig(**fold_params)
        self.config = fold_cfg

        if history_length < fold_cfg.lookback:
            raise ValueError(
                f"Need at least lookback={fold_cfg.lookback} months for resolved "
                f"tier (history_length={history_length})"
            )

        fold, scaler = prepare_neural_fold(
            train_series,
            forecast_origin=int(window.forecast_origin),
            config=fold_cfg,
            seed=int(seed),
            mode=TargetMode.RECURSIVE,
            require_eligible=True,
        )
        assert fold.val_X is not None and fold.val_y is not None

        trainer = NeuralTrainer(
            architecture_builder=self._architecture_builder,
            config=fold_cfg,
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
        params = dict(metadata.parameters)
        params.update(spec.as_dict())  # resolved history_length/l1/l2/lookback/max_epochs
        params["l1"] = int(spec.l1)
        params["l2"] = int(spec.l2)
        if int(fold_cfg.max_epochs) != int(spec.max_epochs):
            params["max_epochs_effective"] = int(fold_cfg.max_epochs)
        self.metadata_ = replace(metadata, parameters=params)
        self._trainer = trainer
        self._scaler = scaler
        self._last_lookback_raw = (
            train_series.to_numpy(dtype=float)[-fold_cfg.lookback:].copy()
        )
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        """Recursive H-step forecast; inverse-transform once; no postprocess."""
        if self._trainer is None or self._trainer.model_ is None:
            raise RuntimeError("LegacyAdaptiveRecursiveLSTM.predict called before fit")
        if self._scaler is None or self._last_lookback_raw is None:
            raise RuntimeError(
                "LegacyAdaptiveRecursiveLSTM missing scaler or lookback state"
            )

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
        if self._resolved_spec is not None:
            meta.update(self._resolved_spec.as_dict())
        if self.metadata_ is not None:
            meta.update(
                {
                    "parameter_count": self.metadata_.parameter_count,
                    "epochs_ran": self.metadata_.epochs_ran,
                    "best_epoch": self.metadata_.best_epoch,
                    "best_val_loss": self.metadata_.best_val_loss,
                    "random_seed": self.metadata_.random_seed,
                }
            )

        return ForecastResult(
            model_name=self.name,
            predictions=tuple(float(x) for x in raw_preds.tolist()),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(h) for h in window.horizons),
            metadata=meta,
        )
