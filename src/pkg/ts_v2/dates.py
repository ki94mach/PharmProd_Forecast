"""Date helpers for V2 (explicit origins, Shamsi month arithmetic).

V2 never infers the forecast start as ``max(history) + 1``. Callers pass an
explicit Shamsi ``YYYYMM`` forecast origin. Training is always
``date < forecast_origin``; target months are exactly horizons ``1..H``.

Models must not invent their own month-skipping (no ``series[:-1]``).
All Shamsi ↔ pandas ``YYYYMM`` offset conversion lives here.
"""
from __future__ import annotations

from typing import Optional, Union

from pkg.benchmark.calendar import shamsi_add_months, shamsi_month_diff
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow

# V1/warehouse convention: Shamsi YYYYMM + 62100 yields a pseudo-Gregorian
# YYYYMM that pandas can parse with ``format="%Y%m"``. Keep the offset here only.
SHAMSI_TO_PANDAS_YYYYMM_OFFSET = 62100

OriginLike = Union[int, ForecastOrigin]


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


def target_month(origin: OriginLike, horizon: int) -> int:
    """Shamsi YYYYMM for horizon ``h`` (1-based): ``h=1`` is the origin month."""
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
    """Build the explicit V2 date contract for one forecast origin.

    Contract
    --------
    - ``forecast_origin``: first target month (CLI/business Shamsi YYYYMM).
    - ``training_end``: last month allowed in training (= origin − 1 month).
    - Training rule used everywhere: ``date < forecast_origin``
      (equivalently ``date <= training_end``).
    - ``target_dates[h-1]`` is the Shamsi month for horizon ``h``;
      length equals ``forecast_horizon`` (default 15).
    - No implicit last-month drop; models must not skip months themselves.
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
    )


def is_training_month(shamsi_yyyymm: int, window: ForecastWindow) -> bool:
    """True iff ``shamsi_yyyymm`` is strictly before ``window.forecast_origin``."""
    ym = validate_shamsi_yyyymm(shamsi_yyyymm)
    return ym < window.forecast_origin


__all__ = [
    "SHAMSI_TO_PANDAS_YYYYMM_OFFSET",
    "validate_shamsi_yyyymm",
    "shamsi_to_pandas_yyyymm",
    "pandas_yyyymm_to_shamsi",
    "parse_origin",
    "target_month",
    "months_between",
    "make_forecast_window",
    "is_training_month",
    "shamsi_add_months",
    "shamsi_month_diff",
]
