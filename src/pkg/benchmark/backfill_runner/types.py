"""Shared types for the historical forecast backfill runner.

The runner is orchestration-only. Forecasting logic lives in engine adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.vintages import VintageSpec


@dataclass(frozen=True)
class EngineJobRequest:
    """One product × vintage job handed to a forecasting engine.

    The runner always enforces ``sales.date < forecast_origin`` before calling
    the engine. Engines must not rely on seeing post-origin rows.
    """

    engine: str
    product: str
    quarter: str
    forecast_origin: int
    horizon: int
    target_dates: tuple[int, ...]
    training_sales: pd.DataFrame
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def training_cutoff_exclusive(self) -> int:
        return int(self.forecast_origin)


@dataclass(frozen=True)
class EngineJobResult:
    """Normalized engine output for one product × vintage."""

    success: bool
    product: str
    quarter: str
    forecast_origin: int
    selected_model: Optional[str] = None
    forecasts: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobKey:
    engine: str
    quarter: str
    product: str
    forecast_origin: int

    @property
    def slug(self) -> str:
        return f"{self.quarter}__{self.product}"


@dataclass
class JobLogRecord:
    engine: str
    quarter: str
    forecast_origin: int
    product: str
    start_time_utc: str
    end_time_utc: str
    duration_seconds: float
    success: bool
    selected_model: Optional[str] = None
    error_message: Optional[str] = None
    status: str = "complete"  # complete | failed | skipped


@runtime_checkable
class ForecastEngine(Protocol):
    """Versioned forecasting backend used by the backfill runner."""

    name: str

    def forecast_product(self, request: EngineJobRequest) -> EngineJobResult:
        """Fit/predict one SKU. Must not mutate shared runner state."""
        ...


def target_dates_for_origin(forecast_origin: int, horizon: int) -> tuple[int, ...]:
    origin = int(forecast_origin)
    h = int(horizon)
    return tuple(shamsi_add_months(origin, i) for i in range(h))


def request_from_vintage(
    *,
    engine: str,
    product: str,
    vintage: VintageSpec,
    training_sales: pd.DataFrame,
    meta: Optional[Mapping[str, Any]] = None,
) -> EngineJobRequest:
    targets = target_dates_for_origin(vintage.forecast_origin, vintage.horizon)
    if len(targets) != int(vintage.horizon):
        raise ValueError("target_dates length must equal horizon")
    return EngineJobRequest(
        engine=str(engine),
        product=str(product),
        quarter=str(vintage.quarter),
        forecast_origin=int(vintage.forecast_origin),
        horizon=int(vintage.horizon),
        target_dates=targets,
        training_sales=training_sales,
        meta=dict(meta or {}),
    )
