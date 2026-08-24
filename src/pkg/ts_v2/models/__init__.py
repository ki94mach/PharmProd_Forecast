"""Forecasting model registry for V2.

Models are not implemented in this scaffold step. Future entries will be
univariate SKU models (e.g. seasonal naive, ETS, ARIMA) selected by
``TSForecastConfig.selection_metric``.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.types import ForecastOrigin


class ForecastModel(Protocol):
    """Minimal interface every V2 model must satisfy.

    Implementations must not drop months themselves (no ``history[:-1]``).
    The caller supplies history already cut with ``date < forecast_origin``
    via :func:`~pkg.ts_v2.dates.make_forecast_window` /
    :func:`~pkg.ts_v2.data.filter_training_history`.
    """

    name: str

    def fit(self, history: pd.Series, config: TSForecastConfig) -> "ForecastModel":
        """Fit on history with ``date < origin`` only (caller-enforced)."""
        ...

    def predict(
        self,
        origin: ForecastOrigin,
        horizons: Sequence[int],
        config: TSForecastConfig,
    ) -> Sequence[float]:
        """Return point forecasts for the requested 1-based horizons."""
        ...


def available_models() -> tuple[str, ...]:
    """Registered model names. Empty until models are implemented."""
    return ()
