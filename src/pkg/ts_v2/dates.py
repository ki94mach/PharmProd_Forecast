"""Date helpers for V2 (explicit origins, Shamsi month arithmetic).

V2 never infers the forecast start as ``max(history) + 1``. Callers pass an
explicit Shamsi ``YYYYMM`` forecast start / origin.

Production time contract (default ``make_forecast_window``)
----------------------------------------------------------
- ``forecast_start``: first *delivered* business forecast month.
- ``current_partial_month = forecast_start - 1`` (bridge; predicted then discarded).
- ``last_complete_month = forecast_start - 2`` (inclusive training end).
- Generate ``H+1`` internal steps, discard the bridge, deliver ``H`` horizons
  starting at ``forecast_start``.

Screening contract (``make_screening_forecast_window``, A2–A5)
-------------------------------------------------------------
- ``training_end = origin - 1``; no bridge; deliver ``H`` months from origin.

All Shamsi month arithmetic uses :func:`pkg.benchmark.calendar.shamsi_add_months`
— never subtract Jalali ``YYYYMM`` integers directly. Shamsi ↔ pandas
``YYYYMM`` offset conversion lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import pandas as pd

from pkg.benchmark.calendar import (
    shamsi_add_months,
    shamsi_month_diff,
    shamsi_month_start_gregorian,
)
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow

# V1/warehouse convention: Shamsi YYYYMM + 62100 yields a pseudo-Gregorian
# YYYYMM that pandas can parse with ``format="%Y%m"``. Keep the offset here only.
SHAMSI_TO_PANDAS_YYYYMM_OFFSET = 62100

OriginLike = Union[int, ForecastOrigin]


@dataclass(frozen=True)
class ProductionTimeContract:
    """Shared production calendar for V2, A0, and A1.

    Attributes:
        forecast_start: First delivered business forecast month (Shamsi YYYYMM).
        current_partial_month: Bridge month (``forecast_start - 1``).
        last_complete_month: Inclusive training end (``forecast_start - 2``).
        delivered_horizon: Number of delivered horizons (default 15).
    """

    forecast_start: int
    current_partial_month: int
    last_complete_month: int
    delivered_horizon: int = 15

    @property
    def internal_steps(self) -> int:
        """Bridge month plus delivered horizons."""
        return int(self.delivered_horizon) + 1


def _as_shamsi_yyyymm(origin: OriginLike) -> int:
    if isinstance(origin, ForecastOrigin):
        return int(origin.shamsi_yyyymm)
    return int(origin)


def validate_shamsi_yyyymm(shamsi_yyyymm: int) -> int:
    """Return ``shamsi_yyyymm`` if it is a valid Shamsi ``YYYYMM``."""
    ym = int(shamsi_yyyymm)
    year, month = divmod(ym, 100)
    if year < 1300 or year > 1599 or month < 1 or month > 12:
        raise ValueError(f"Invalid Shamsi YYYYMM: {shamsi_yyyymm!r}")
    return ym


def shamsi_to_pandas_yyyymm(shamsi_yyyymm: int) -> int:
    """Shamsi ``YYYYMM`` → pandas-parseable pseudo-Gregorian ``YYYYMM`` (+62100)."""
    return validate_shamsi_yyyymm(shamsi_yyyymm) + SHAMSI_TO_PANDAS_YYYYMM_OFFSET


def pandas_yyyymm_to_shamsi(pandas_yyyymm: int) -> int:
    """Pseudo-Gregorian ``YYYYMM`` → Shamsi ``YYYYMM`` (-62100)."""
    return validate_shamsi_yyyymm(int(pandas_yyyymm) - SHAMSI_TO_PANDAS_YYYYMM_OFFSET)


def parse_origin(shamsi_yyyymm: int) -> ForecastOrigin:
    """Build a :class:`ForecastOrigin` from a Shamsi ``YYYYMM`` integer."""
    return ForecastOrigin(shamsi_yyyymm=validate_shamsi_yyyymm(shamsi_yyyymm))


def resolve_production_time_contract(
    forecast_start: OriginLike,
    *,
    delivered_horizon: int = 15,
) -> ProductionTimeContract:
    """Resolve production months via :func:`shamsi_add_months` only."""
    h = int(delivered_horizon)
    if h < 1:
        raise ValueError(f"delivered_horizon must be >= 1, got {h}")
    start = validate_shamsi_yyyymm(_as_shamsi_yyyymm(forecast_start))
    partial = shamsi_add_months(start, -1)
    last_complete = shamsi_add_months(start, -2)
    return ProductionTimeContract(
        forecast_start=start,
        current_partial_month=partial,
        last_complete_month=last_complete,
        delivered_horizon=h,
    )


def delivered_target_dates(contract: ProductionTimeContract) -> tuple[int, ...]:
    """``H`` delivered months starting at ``forecast_start``."""
    start = int(contract.forecast_start)
    h = int(contract.delivered_horizon)
    return tuple(shamsi_add_months(start, i) for i in range(h))


def internal_target_dates(contract: ProductionTimeContract) -> tuple[int, ...]:
    """``H+1`` internal months: bridge (partial) then delivered targets."""
    return (int(contract.current_partial_month),) + delivered_target_dates(contract)


def map_internal_predictions_to_delivered(
    predictions: Sequence[float],
    *,
    delivered_horizon: Optional[int] = None,
) -> tuple[float, ...]:
    """Drop the bridge (index 0) and keep the next ``delivered_horizon`` values."""
    preds = tuple(float(x) for x in predictions)
    if len(preds) < 2:
        raise ValueError(
            f"internal predictions must include bridge + at least one delivered "
            f"step, got length {len(preds)}"
        )
    delivered = preds[1:]
    if delivered_horizon is not None:
        h = int(delivered_horizon)
        if len(delivered) < h:
            raise ValueError(
                f"need {h} delivered predictions after bridge, got {len(delivered)}"
            )
        delivered = delivered[:h]
    return delivered


def target_month(origin: OriginLike, horizon: int) -> int:
    """Shamsi YYYYMM for delivered horizon ``h`` (1-based): ``h=1`` is start."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    origin_ym = validate_shamsi_yyyymm(_as_shamsi_yyyymm(origin))
    return shamsi_add_months(origin_ym, horizon - 1)


def months_between(start_yyyymm: int, end_yyyymm: int) -> int:
    """Inclusive month count from ``start`` to ``end`` (Shamsi YYYYMM)."""
    return shamsi_month_diff(
        validate_shamsi_yyyymm(end_yyyymm),
        validate_shamsi_yyyymm(start_yyyymm),
    ) + 1


def make_forecast_window(
    forecast_origin: OriginLike,
    *,
    config: Optional[TSForecastConfig] = None,
    horizon: Optional[int] = None,
) -> ForecastWindow:
    """Build the **production** date contract for one forecast start.

    Contract
    --------
    - ``forecast_origin`` / ``forecast_start``: first *delivered* target month.
    - ``current_partial_month``: bridge month (``start - 1``).
    - ``training_end``: ``last_complete_month`` (``start - 2``).
    - Training rule: ``date <= training_end`` (partial excluded).
    - Delivered ``target_dates``: ``H`` months starting at ``forecast_start``.
    - Internal steps: bridge + delivered (``H+1``); discard bridge on delivery.
    """
    cfg = config or DEFAULT_CONFIG
    h = int(cfg.forecast_horizon if horizon is None else horizon)
    contract = resolve_production_time_contract(forecast_origin, delivered_horizon=h)
    horizons = tuple(range(1, h + 1))
    return ForecastWindow(
        forecast_origin=contract.forecast_start,
        training_end=contract.last_complete_month,
        target_dates=delivered_target_dates(contract),
        horizons=horizons,
        current_partial_month=contract.current_partial_month,
    )


def make_screening_forecast_window(
    forecast_origin: OriginLike,
    *,
    config: Optional[TSForecastConfig] = None,
    horizon: Optional[int] = None,
) -> ForecastWindow:
    """Build the historical **screening** contract (A2–A5 bake-off).

    - ``training_end = origin - 1`` (partial month included in training).
    - No bridge month (``current_partial_month is None``).
    - Exactly ``H`` delivered targets starting at ``origin``.
    """
    cfg = config or DEFAULT_CONFIG
    h = int(cfg.forecast_horizon if horizon is None else horizon)
    if h < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {h}")

    origin_ym = validate_shamsi_yyyymm(_as_shamsi_yyyymm(forecast_origin))
    training_end = shamsi_add_months(origin_ym, -1)
    horizons = tuple(range(1, h + 1))
    target_dates = tuple(shamsi_add_months(origin_ym, i) for i in range(h))

    return ForecastWindow(
        forecast_origin=origin_ym,
        training_end=training_end,
        target_dates=target_dates,
        horizons=horizons,
        current_partial_month=None,
    )


def is_training_month(shamsi_yyyymm: int, window: ForecastWindow) -> bool:
    """True iff ``shamsi_yyyymm`` is on or before ``window.training_end``."""
    ym = validate_shamsi_yyyymm(shamsi_yyyymm)
    return ym <= int(window.training_end)


def exclusive_training_cutoff(window: ForecastWindow) -> int:
    """Exclusive upper bound for supervised windows / leakage asserts.

    Production: ``current_partial_month`` (history must be ``date < partial``).
    Screening: ``forecast_origin`` (history must be ``date < origin``).
    """
    if window.current_partial_month is not None:
        return int(window.current_partial_month)
    return int(window.forecast_origin)


def shamsi_to_month_start_timestamp(shamsi_yyyymm: int) -> pd.Timestamp:
    """Shamsi YYYYMM → Gregorian month-start Timestamp (Prophet ``freq='MS'``).

    Uses real Jalali→Gregorian conversion, not the V1 ``+62100`` fake year.
    """
    d = shamsi_month_start_gregorian(validate_shamsi_yyyymm(shamsi_yyyymm))
    return pd.Timestamp(year=d.year, month=d.month, day=1)


def shamsi_months_to_ms_index(shamsi_yyyymms: Sequence[int]) -> pd.DatetimeIndex:
    """Ordered DatetimeIndex of month-starts for Shamsi YYYYMM labels."""
    return pd.DatetimeIndex(
        [shamsi_to_month_start_timestamp(ym) for ym in shamsi_yyyymms],
        name="ds",
    )


__all__ = [
    "SHAMSI_TO_PANDAS_YYYYMM_OFFSET",
    "ProductionTimeContract",
    "validate_shamsi_yyyymm",
    "shamsi_to_pandas_yyyymm",
    "pandas_yyyymm_to_shamsi",
    "shamsi_to_month_start_timestamp",
    "shamsi_months_to_ms_index",
    "parse_origin",
    "resolve_production_time_contract",
    "delivered_target_dates",
    "internal_target_dates",
    "map_internal_predictions_to_delivered",
    "target_month",
    "months_between",
    "make_forecast_window",
    "make_screening_forecast_window",
    "is_training_month",
    "exclusive_training_cutoff",
    "shamsi_add_months",
    "shamsi_month_diff",
]
