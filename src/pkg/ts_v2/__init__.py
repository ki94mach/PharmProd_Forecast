"""V2 time-series forecasting baseline (scaffold).

This package is intentionally separate from V1
(``pkg.sales_forecasting`` / ``pkg.forecast``). Do not change V1 behavior from
here; historical benchmark CSVs depend on the existing engine.
"""
from __future__ import annotations

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.models import run_model
from pkg.ts_v2.types import ForecastOrigin, ForecastResult, ForecastWindow, PreparedSeries

__all__ = [
    "DEFAULT_CONFIG",
    "TSForecastConfig",
    "ForecastOrigin",
    "ForecastResult",
    "ForecastWindow",
    "PreparedSeries",
    "make_forecast_window",
    "prepare_monthly_series",
    "run_model",
]