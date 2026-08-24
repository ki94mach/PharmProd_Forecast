"""Data loading and series preparation for V2.

Principles:

- Train on months strictly before the explicit forecast origin.
- Do not silently drop the last warehouse month (no ``series[:-1]``).
- Fit any transforms only on training history (no preprocessing leakage).
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, validate_shamsi_yyyymm
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow, ProductSeries

DateLike = Union[int, ForecastOrigin, ForecastWindow]


def load_monthly_sales() -> pd.DataFrame:
    """Load warehouse monthly sales.

    Not implemented in this scaffold step. Production V1 loaders stay in
    ``pkg.db.query.sales``; V2 will wrap them without changing V1 behavior.
    """
    raise NotImplementedError("V2 sales loading is not implemented yet")


def _window_from(origin_or_window: DateLike, config: Optional[TSForecastConfig]) -> ForecastWindow:
    if isinstance(origin_or_window, ForecastWindow):
        return origin_or_window
    if isinstance(origin_or_window, ForecastOrigin):
        return make_forecast_window(origin_or_window, config=config)
    return make_forecast_window(int(origin_or_window), config=config)


def filter_training_frame(
    frame: pd.DataFrame,
    origin_or_window: DateLike,
    *,
    date_col: str = "date",
    config: Optional[TSForecastConfig] = None,
) -> pd.DataFrame:
    """Keep rows with Shamsi ``date < forecast_origin``.

    The forecast-origin month never enters training. This is the only allowed
    as-of cut; callers must not apply an extra last-month drop afterward.
    """
    window = _window_from(origin_or_window, config)
    if frame is None or frame.empty:
        return frame.iloc[0:0].copy() if frame is not None else pd.DataFrame()
    work = frame.copy()
    work[date_col] = work[date_col].map(lambda x: validate_shamsi_yyyymm(int(x)))
    return work.loc[work[date_col] < window.forecast_origin].copy()


def filter_training_history(
    history: pd.Series,
    origin_or_window: DateLike,
    *,
    config: Optional[TSForecastConfig] = None,
) -> pd.Series:
    """Keep series points with index ``date < forecast_origin`` (Shamsi YYYYMM)."""
    window = _window_from(origin_or_window, config)
    if history is None or len(history) == 0:
        return pd.Series(dtype=float)
    idx = pd.Index([validate_shamsi_yyyymm(int(x)) for x in history.index], name=history.index.name)
    values = pd.to_numeric(history.to_numpy(), errors="coerce")
    out = pd.Series(values, index=idx, name=history.name)
    out = out.loc[out.index < window.forecast_origin].sort_index()
    return out


def assert_training_before_origin(
    history: pd.Series,
    origin_or_window: DateLike,
    *,
    config: Optional[TSForecastConfig] = None,
) -> None:
    """Raise if any training index is at/after the forecast origin."""
    window = _window_from(origin_or_window, config)
    if history is None or len(history) == 0:
        return
    bad = [validate_shamsi_yyyymm(int(x)) for x in history.index if int(x) >= window.forecast_origin]
    if bad:
        raise ValueError(
            f"training history includes forecast_origin or later months "
            f"(origin={window.forecast_origin}, bad={sorted(set(bad))}); "
            "models must not see the origin month"
        )


def series_as_of(
    sales: pd.DataFrame,
    product: str,
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ProductSeries:
    """Build a product series available at ``origin`` (``date < origin``).

    Not fully implemented (sales load / product join) in this step; the
    training cut itself is available via :func:`filter_training_frame`.
    """
    raise NotImplementedError("V2 series_as_of is not implemented yet")


def assert_min_history(series: ProductSeries, config: Optional[TSForecastConfig] = None) -> None:
    """Raise if history length is below ``config.min_train_months``."""
    cfg = config or DEFAULT_CONFIG
    assert_training_before_origin(series.history, series.origin, config=cfg)
    n = int(len(series.history))
    if n < cfg.min_train_months:
        raise ValueError(
            f"{series.product!r} at origin {series.origin.shamsi_yyyymm}: "
            f"need >= {cfg.min_train_months} months, got {n}"
        )
