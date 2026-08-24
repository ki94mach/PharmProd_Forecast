"""Shared types for the V2 time-series forecasting baseline.

These are structural placeholders. Model implementations come in later steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class ForecastOrigin:
    """Explicit forecast origin (first month of the forecast window).

    Shamsi ``YYYYMM`` integer, e.g. ``140501``. Training uses months strictly
    before this origin; there is no implicit last-month drop.
    """

    shamsi_yyyymm: int


@dataclass(frozen=True)
class ProductSeries:
    """One SKU's monthly history available at a given origin."""

    product: str
    origin: ForecastOrigin
    history: pd.Series  # index: Shamsi YYYYMM (int), values: sales
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HorizonForecast:
    """Point forecast for a single (product, origin, horizon) cell."""

    product: str
    origin: ForecastOrigin
    horizon: int
    target_shamsi_yyyymm: int
    yhat: float
    model_name: str


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of model selection for one series at one origin."""

    product: str
    origin: ForecastOrigin
    best_model_name: str
    scores: Mapping[str, float]
    metric: str


@dataclass(frozen=True)
class BacktestFold:
    """One multi-horizon evaluation origin in a rolling backtest."""

    origin: ForecastOrigin
    train_end_exclusive: int  # Shamsi YYYYMM; train is date < this (== origin)
    horizons: Sequence[int]


@dataclass(frozen=True)
class EngineResult:
    """Container for a V2 run (backtest and/or final refit forecasts)."""

    config_name: str = "default"
    selection: Optional[SelectionResult] = None
    forecasts: Sequence[HorizonForecast] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
