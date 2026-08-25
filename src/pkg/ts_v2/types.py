"""Shared types for the V2 time-series forecasting baseline.

These are structural placeholders. Model implementations come in later steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

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
    no ``series[:-1]`` trimming by models). Prefer :class:`PreparedSeries` for
    full diagnostics (gaps, activity start, observation counts).
    """

    product: str
    origin: ForecastOrigin
    history: pd.Series  # index: Shamsi YYYYMM (int), values: sales
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedSeries:
    """Monthly training series in **raw sales units** for one product / origin.

    No MinMax, Yeo–Johnson, ADF, or other global transforms. Models must use
    ``values`` as-is (aside from optional nonnegativity at forecast emit time).

    Attributes:
        product: English product key.
        values: Monthly sales (float); index is Shamsi YYYYMM.
        dates: Same Shamsi YYYYMM labels as ``values.index`` (explicit monthly grid).
        forecast_origin: Exclusive training cutoff (train is ``date < origin``).
        last_training_month: Inclusive last month on the prepared grid
            (``forecast_origin - 1`` when the series is non-empty).
        first_active_month: First month kept after the activity-start policy.
        n_observations: Length of the monthly grid (``len(values)``).
        is_missing_month: True where the calendar month had **no** warehouse row
            before gap filling (distinct from an explicit observed zero).
        n_observed_months: Count of months with at least one warehouse row.
        n_gap_months: Count of calendar gaps (``is_missing_month`` True).
        missing_month_policy: Policy used to fill ``values`` at gaps.
        activity_start_min_sales: Threshold used for activity trimming (or None).
        zero_month_proportion: Share of months with sales ``<= 0`` (NaN ignored).
        average_inter_demand_interval: Mean gap (in months) between consecutive
            positive-demand months; ``None`` if fewer than two demand events.
            Diagnostic only — does not auto-route to Croston/TSB.
        n_demand_months: Count of months with sales ``> 0``.
    """

    product: str
    values: pd.Series
    dates: tuple[int, ...]
    forecast_origin: int
    last_training_month: Optional[int]
    first_active_month: Optional[int]
    n_observations: int
    is_missing_month: pd.Series
    n_observed_months: int
    n_gap_months: int
    missing_month_policy: str
    activity_start_min_sales: Optional[float] = None
    zero_month_proportion: Optional[float] = None
    average_inter_demand_interval: Optional[float] = None
    n_demand_months: int = 0

    @property
    def history(self) -> pd.Series:
        """Alias for model-facing raw values (date < origin only)."""
        return self.values

@dataclass(frozen=True)
class ForecastResult:
    """Point forecast from one V2 model for one origin / window.

    Contract (enforced by :func:`pkg.ts_v2.models.run_model`):

    - ``len(predictions) == len(target_dates) == len(horizons) == horizon``
    - ``target_dates`` is exactly the caller-supplied window (models must not
      shift, drop, or skip the first month)
    - values are raw-scale floats: no rounding, quarterly smoothing, or
      arbitrary bias correction inside the model
    """

    model_name: str
    predictions: tuple[float, ...]
    target_dates: tuple[int, ...]
    horizons: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lower: Optional[tuple[float, ...]] = None
    upper: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class ModelFailure:
    """Typed failure for one model on one series; does not abort the SKU run."""

    model_name: str
    reason: str
    error_type: str
    details: Mapping[str, Any] = field(default_factory=dict)


ModelOutcome = Union[ForecastResult, ModelFailure]


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
