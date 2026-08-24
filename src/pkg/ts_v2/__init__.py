"""V2 time-series forecasting baseline (scaffold).

This package is intentionally separate from V1
(``pkg.sales_forecasting`` / ``pkg.forecast``). Do not change V1 behavior from
here; historical benchmark CSVs depend on the existing engine.
"""
from __future__ import annotations

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig

__all__ = ["DEFAULT_CONFIG", "TSForecastConfig"]
