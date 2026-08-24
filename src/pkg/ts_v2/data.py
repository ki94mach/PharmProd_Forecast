"""Data loading and monthly series preparation for V2.

Principles:

- Models receive sales in **raw units** (no MinMax / Yeo–Johnson / ADF).
- Train on months strictly before the explicit forecast origin.
- Do not silently drop the last warehouse month (no ``series[:-1]``).
- Never fit transforms on post-origin rows (there are no such transforms).
- Aggregate duplicate product/month rows; enforce a monthly calendar grid.
- Calendar gaps vs explicit zeros are tracked via ``is_missing_month``.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, parse_origin, validate_shamsi_yyyymm
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow, PreparedSeries, ProductSeries

DateLike = Union[int, ForecastOrigin, ForecastWindow]


def load_monthly_sales() -> pd.DataFrame:
    """Load warehouse monthly sales.

    Not implemented in this step. Production V1 loaders stay in
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


def _empty_prepared(
    product: str,
    window: ForecastWindow,
    config: TSForecastConfig,
) -> PreparedSeries:
    empty_idx = pd.Index([], dtype="int64", name="date")
    return PreparedSeries(
        product=product,
        values=pd.Series(dtype=float, index=empty_idx, name="sales"),
        dates=(),
        forecast_origin=window.forecast_origin,
        last_training_month=None,
        first_active_month=None,
        n_observations=0,
        is_missing_month=pd.Series(dtype=bool, index=empty_idx, name="is_missing_month"),
        n_observed_months=0,
        n_gap_months=0,
        missing_month_policy=config.missing_month_policy,
        activity_start_min_sales=config.activity_start_min_sales,
    )


def _month_range_inclusive(start_ym: int, end_ym: int) -> list[int]:
    start = validate_shamsi_yyyymm(start_ym)
    end = validate_shamsi_yyyymm(end_ym)
    if start > end:
        return []
    out = [start]
    cur = start
    while cur < end:
        cur = shamsi_add_months(cur, 1)
        out.append(cur)
    return out


def _aggregate_product_months(
    sales: pd.DataFrame,
    product: str,
    *,
    product_col: str,
    date_col: str,
    sales_col: str,
) -> pd.Series:
    """Sum duplicate product/month rows; index is Shamsi YYYYMM."""
    if sales is None or sales.empty:
        return pd.Series(dtype=float, name=sales_col)
    work = sales.loc[sales[product_col].astype(str) == str(product)].copy()
    if work.empty:
        return pd.Series(dtype=float, name=sales_col)
    work[date_col] = work[date_col].map(lambda x: validate_shamsi_yyyymm(int(x)))
    work[sales_col] = pd.to_numeric(work[sales_col], errors="coerce")
    # Observed NaN sales become 0 only for aggregation of present rows; calendar
    # gaps are handled separately and remain marked missing.
    work[sales_col] = work[sales_col].fillna(0.0)
    grouped = work.groupby(date_col, sort=True)[sales_col].sum()
    grouped.index = grouped.index.astype("int64")
    grouped.index.name = "date"
    grouped.name = "sales"
    return grouped.astype(float)


def prepare_monthly_series(
    sales: pd.DataFrame,
    product: str,
    origin_or_window: DateLike,
    *,
    config: Optional[TSForecastConfig] = None,
    product_col: str = "product",
    date_col: str = "date",
    sales_col: str = "sales",
) -> PreparedSeries:
    """Build a raw-unit monthly training series for ``product`` as-of an origin.

    Steps (all exclusive of post-origin data):

    1. Keep rows with ``date < forecast_origin``.
    2. Aggregate duplicate product/month rows (sum).
    3. Optionally trim leading months before meaningful activity
       (``activity_start_min_sales``, V1-compatible default ``5``).
    4. Reindex to a contiguous Shamsi monthly grid through ``training_end``.
    5. Apply ``missing_month_policy`` to gap values; always set ``is_missing_month``.
    """
    cfg = config or DEFAULT_CONFIG
    window = _window_from(origin_or_window, cfg)

    train_frame = filter_training_frame(
        sales, window, date_col=date_col, config=cfg
    )
    observed = _aggregate_product_months(
        train_frame,
        product,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    if observed.empty:
        return _empty_prepared(product, window, cfg)

    threshold = cfg.activity_start_min_sales
    if threshold is None:
        first_active = int(observed.index.min())
    else:
        active = observed.loc[observed > float(threshold)]
        if active.empty:
            return _empty_prepared(product, window, cfg)
        first_active = int(active.index.min())

    last_training = int(window.training_end)
    if first_active > last_training:
        return _empty_prepared(product, window, cfg)

    # Drop leading pre-activity months from the observed map, then build grid.
    observed = observed.loc[observed.index >= first_active]
    grid = _month_range_inclusive(first_active, last_training)
    idx = pd.Index(grid, dtype="int64", name="date")
    is_missing = pd.Series(~idx.isin(observed.index), index=idx, name="is_missing_month")

    values = observed.reindex(idx)
    if cfg.missing_month_policy == "zero":
        values = values.fillna(0.0)
    elif cfg.missing_month_policy == "missing":
        # Leave NaN for gaps; observed zeros stay 0.0.
        pass
    else:
        raise ValueError(
            f"Unknown missing_month_policy={cfg.missing_month_policy!r}; "
            "expected 'zero' or 'missing'"
        )
    values = values.astype(float)
    values.name = "sales"

    assert_training_before_origin(values.dropna(), window, config=cfg)
    if (values.index >= window.forecast_origin).any():
        raise RuntimeError("internal error: prepared series leaked forecast_origin")

    n_gap = int(is_missing.sum())
    n_obs_months = int((~is_missing).sum())
    return PreparedSeries(
        product=str(product),
        values=values,
        dates=tuple(int(x) for x in idx.tolist()),
        forecast_origin=window.forecast_origin,
        last_training_month=last_training,
        first_active_month=first_active,
        n_observations=int(len(values)),
        is_missing_month=is_missing,
        n_observed_months=n_obs_months,
        n_gap_months=n_gap,
        missing_month_policy=cfg.missing_month_policy,
        activity_start_min_sales=cfg.activity_start_min_sales,
    )


def series_as_of(
    sales: pd.DataFrame,
    product: str,
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ProductSeries:
    """Build a :class:`ProductSeries` from :func:`prepare_monthly_series`."""
    prepared = prepare_monthly_series(sales, product, origin, config=config)
    return ProductSeries(
        product=prepared.product,
        origin=parse_origin(prepared.forecast_origin),
        history=prepared.values,
        meta={
            "prepared": prepared,
            "first_active_month": prepared.first_active_month,
            "last_training_month": prepared.last_training_month,
            "n_gap_months": prepared.n_gap_months,
            "is_missing_month": prepared.is_missing_month,
        },
    )


def assert_min_history(
    series: Union[ProductSeries, PreparedSeries],
    config: Optional[TSForecastConfig] = None,
) -> None:
    """Raise if history length is below ``config.min_train_months``."""
    cfg = config or DEFAULT_CONFIG
    if isinstance(series, PreparedSeries):
        history = series.values
        product = series.product
        origin_ym = series.forecast_origin
    else:
        history = series.history
        product = series.product
        origin_ym = series.origin.shamsi_yyyymm
    assert_training_before_origin(history.dropna(), origin_ym, config=cfg)
    n = int(len(history))
    if n < cfg.min_train_months:
        raise ValueError(
            f"{product!r} at origin {origin_ym}: "
            f"need >= {cfg.min_train_months} months, got {n}"
        )


def assert_no_post_origin_leakage(
    sales: pd.DataFrame,
    prepared: PreparedSeries,
    *,
    date_col: str = "date",
    sales_col: str = "sales",
    product_col: str = "product",
) -> None:
    """Diagnostic: prepared observed months match pre-origin aggregates only."""
    origin = prepared.forecast_origin
    if any(int(d) >= origin for d in prepared.dates):
        raise AssertionError("prepared dates include forecast_origin or later")
    # Re-aggregate only pre-origin rows and compare observed (non-gap) cells.
    pre = filter_training_frame(sales, origin, date_col=date_col)
    expected = _aggregate_product_months(
        pre,
        prepared.product,
        product_col=product_col,
        date_col=date_col,
        sales_col=sales_col,
    )
    for ym, is_gap in prepared.is_missing_month.items():
        if bool(is_gap):
            continue
        if int(ym) not in expected.index:
            raise AssertionError(f"observed month {ym} not in pre-origin aggregate")
        got = float(prepared.values.loc[ym])
        exp = float(expected.loc[int(ym)])
        if not np.isclose(got, exp, equal_nan=True):
            raise AssertionError(
                f"value mismatch at {ym}: prepared={got} expected_raw={exp}"
            )
