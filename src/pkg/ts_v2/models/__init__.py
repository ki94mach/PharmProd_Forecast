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
    """Minimal interface every V2 model must satisfy."""

    name: str

    def fit(self, history: pd.Series, config: TSForecastConfig) -> "ForecastModel":
        """Fit on history ending strictly before the evaluation origin."""
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
