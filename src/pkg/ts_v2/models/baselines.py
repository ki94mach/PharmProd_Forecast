"""Simple statistical baselines: naive, seasonal naive, and drift.

These emit raw-scale point forecasts. No clipping, rounding, or smoothing.
Insufficient history is :class:`ModelUnavailableError` (no silent fallback).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG
from pkg.ts_v2.models.base import BaseForecastModel
from pkg.ts_v2.models.errors import ModelContractError, ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def _finite_values(train_series: pd.Series, *, model_name: str) -> np.ndarray:
    if train_series is None or len(train_series) == 0:
        raise ModelUnavailableError(
            "empty training series",
            model_name=model_name,
            details={"n": 0},
        )
    values = pd.to_numeric(train_series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ModelUnavailableError(
            "training series contains missing values",
            model_name=model_name,
            details={"n": int(len(values)), "n_nan": int(np.isnan(values).sum())},
        )
    return values


def _point_forecast(
    model_name: str,
    predictions: Sequence[float],
    target_dates: Sequence[int],
    metadata: Optional[dict] = None,
) -> ForecastResult:
    dates = tuple(int(d) for d in target_dates)
    preds = tuple(float(p) for p in predictions)
    if len(preds) != len(dates):
        raise ModelContractError(
            f"internal length mismatch preds={len(preds)} dates={len(dates)}",
            model_name=model_name,
        )
    return ForecastResult(
        model_name=model_name,
        predictions=preds,
        target_dates=dates,
        horizons=tuple(range(1, len(dates) + 1)),
        metadata=dict(metadata or {}),
    )


class NaiveModel(BaseForecastModel):
    """All future values equal the last observed training value."""

    name = "naive"

    def __init__(self) -> None:
        self._last: Optional[float] = None
        self._n: int = 0

    def fit(self, train_series: pd.Series) -> "NaiveModel":
        values = _finite_values(train_series, model_name=self.name)
        if len(values) < 1:
            raise ModelUnavailableError(
                "naive requires at least 1 training observation",
                model_name=self.name,
                details={"n": int(len(values))},
            )
        self._last = float(values[-1])
        self._n = int(len(values))
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._last is None:
            raise ModelUnavailableError("naive is not fitted", model_name=self.name)
        last = self._last
        preds = tuple(last for _ in range(horizon))
        return _point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={"last_value": last, "n_train": self._n},
        )


class SeasonalNaiveModel(BaseForecastModel):
    """Repeat the latest complete seasonal cycle (period 12 by default).

    Horizon ``h`` uses the value from the last cycle at position
    ``(h - 1) % period`` (1-based ``h``). Horizons past one cycle wrap; this is
    recursive repetition of the same cycle, not a different algorithm.

    Requires at least ``period`` finite observations. Does **not** fall back to
    non-seasonal naive.
    """

    name = "seasonal_naive"

    def __init__(self, period: Optional[int] = None) -> None:
        self.period = int(DEFAULT_CONFIG.seasonal_period if period is None else period)
        self._cycle: Optional[np.ndarray] = None
        self._n: int = 0

    def fit(self, train_series: pd.Series) -> "SeasonalNaiveModel":
        values = _finite_values(train_series, model_name=self.name)
        n = int(len(values))
        if n < self.period:
            raise ModelUnavailableError(
                f"seasonal_naive requires at least {self.period} observations, got {n}",
                model_name=self.name,
                details={"n": n, "period": self.period},
            )
        self._cycle = np.asarray(values[-self.period :], dtype=float)
        self._n = n
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._cycle is None:
            raise ModelUnavailableError(
                "seasonal_naive is not fitted", model_name=self.name
            )
        cycle = self._cycle
        period = self.period
        preds = tuple(float(cycle[(h - 1) % period]) for h in range(1, horizon + 1))
        return _point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={"period": period, "n_train": self._n, "cycle": tuple(float(x) for x in cycle)},
        )


class DriftModel(BaseForecastModel):
    """Straight-line drift from the first to the last training observation.

    ``yhat[h] = y_T + h * (y_T - y_1) / (T - 1)`` for 1-based horizon ``h``.
    Requires ``T >= 2``. A single observation is unavailable (not naive).
    """

    name = "drift"

    def __init__(self) -> None:
        self._last: Optional[float] = None
        self._slope: Optional[float] = None
        self._n: int = 0

    def fit(self, train_series: pd.Series) -> "DriftModel":
        values = _finite_values(train_series, model_name=self.name)
        n = int(len(values))
        if n < 2:
            raise ModelUnavailableError(
                f"drift requires at least 2 training observations, got {n}",
                model_name=self.name,
                details={"n": n},
            )
        first = float(values[0])
        last = float(values[-1])
        self._last = last
        self._slope = (last - first) / float(n - 1)
        self._n = n
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._last is None or self._slope is None:
            raise ModelUnavailableError("drift is not fitted", model_name=self.name)
        last = self._last
        slope = self._slope
        preds = tuple(last + h * slope for h in range(1, horizon + 1))
        return _point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={"last_value": last, "slope": slope, "n_train": self._n},
        )
