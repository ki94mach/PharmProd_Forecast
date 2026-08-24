"""Date helpers for V2 (explicit origins, Shamsi month arithmetic).

V2 never infers the forecast start as ``max(history) + 1``. Callers pass an
explicit :class:`~pkg.ts_v2.types.ForecastOrigin`.
"""
from __future__ import annotations

from pkg.benchmark.calendar import shamsi_add_months, shamsi_month_diff
from pkg.ts_v2.types import ForecastOrigin


def parse_origin(shamsi_yyyymm: int) -> ForecastOrigin:
    """Build a :class:`ForecastOrigin` from a Shamsi ``YYYYMM`` integer."""
    ym = int(shamsi_yyyymm)
    year, month = divmod(ym, 100)
    if year < 1300 or year > 1599 or month < 1 or month > 12:
        raise ValueError(f"Invalid Shamsi YYYYMM origin: {shamsi_yyyymm!r}")
    return ForecastOrigin(shamsi_yyyymm=ym)


def target_month(origin: ForecastOrigin, horizon: int) -> int:
    """Shamsi YYYYMM for horizon ``h`` (1-based) after ``origin``."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return shamsi_add_months(origin.shamsi_yyyymm, horizon - 1)


def months_between(start_yyyymm: int, end_yyyymm: int) -> int:
    """Inclusive month count from ``start`` to ``end`` (Shamsi YYYYMM)."""
    return shamsi_month_diff(int(end_yyyymm), int(start_yyyymm)) + 1


__all__ = [
    "parse_origin",
    "target_month",
    "months_between",
    "shamsi_add_months",
    "shamsi_month_diff",
]
