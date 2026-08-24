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
class ForecastWindow:
    """Explicit V2 origin / training / target-date contract for one run.

    Attributes:
        forecast_origin: First forecast month (Shamsi YYYYMM), e.g. ``140501``.
        training_end: Last inclusive training month (``forecast_origin - 1``).
        target_dates: Exactly ``H`` Shamsi months; index ``i`` is horizon ``i+1``.
        horizons: ``(1, 2, ..., H)`` aligned with ``target_dates``.

    Training rule: ``date < forecast_origin`` (never include the origin month).
    """

    forecast_origin: int
    training_end: int
    target_dates: tuple[int, ...]
    horizons: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.target_dates) != len(self.horizons):
            raise ValueError(
                "target_dates and horizons must have the same length: "
                f"{len(self.target_dates)} != {len(self.horizons)}"
            )
        if self.horizons and self.horizons != tuple(range(1, len(self.horizons) + 1)):
            raise ValueError(f"horizons must be 1..H contiguous, got {self.horizons!r}")
        if self.target_dates and self.target_dates[0] != self.forecast_origin:
            raise ValueError(
                "horizon 1 target must equal forecast_origin: "
                f"{self.target_dates[0]} != {self.forecast_origin}"
            )


@dataclass(frozen=True)
class ProductSeries:
    """One SKU's monthly history available at a given origin.

    ``history`` must contain only months with ``date < origin`` (no origin row,
    no ``series[:-1]`` trimming by models).
    """

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
    window: Optional[ForecastWindow] = None


@dataclass(frozen=True)
class EngineResult:
    """Container for a V2 run (backtest and/or final refit forecasts)."""

    config_name: str = "default"
    selection: Optional[SelectionResult] = None
    forecasts: Sequence[HorizonForecast] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
