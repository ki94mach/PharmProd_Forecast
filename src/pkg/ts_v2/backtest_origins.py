"""Historical origin discovery for V2 expanding-window backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, parse_origin, validate_shamsi_yyyymm
from pkg.ts_v2.types import ForecastOrigin, ForecastWindow


@dataclass(frozen=True)
class OriginCoverage:
    """Evaluable horizons at one historical origin."""

    origin: ForecastOrigin
    window: ForecastWindow
    evaluable_target_dates: tuple[int, ...]
    evaluable_horizons: tuple[int, ...]
    max_evaluated_horizon: int
    full_horizon_coverage: bool


def discover_origins(
    monthly_sales: pd.Series,
    *,
    config: Optional[TSForecastConfig] = None,
    explicit_origins: Optional[Sequence[int]] = None,
) -> list[OriginCoverage]:
    """List backtest origins with explicit per-origin horizon coverage.

    Training at origin ``O`` uses months ``date < O``. Targets are
    ``O .. O+H-1``; only horizons whose ``target_date`` exists in
    ``monthly_sales`` are evaluated (no fabricated actuals).

    When any origin supports the full ``forecast_horizon``, origins are
    sorted so full-coverage origins come first (preference, not exclusion).
    """
    cfg = config or DEFAULT_CONFIG
    h_max = int(cfg.forecast_horizon)
    min_train = int(cfg.min_train_months)

    if monthly_sales is None or len(monthly_sales) == 0:
        return []

    sales = monthly_sales.copy()
    sales.index = pd.Index([validate_shamsi_yyyymm(int(x)) for x in sales.index])
    sales = pd.to_numeric(sales, errors="coerce")
    sales = sales.sort_index()
    observed_dates = set(int(x) for x in sales.index[sales.notna()])

    if explicit_origins is not None:
        origin_candidates = [validate_shamsi_yyyymm(int(o)) for o in explicit_origins]
    else:
        first = int(sales.index.min())
        last = int(sales.index.max())
        origin_candidates = []
        cur = shamsi_add_months(first, min_train)
        while cur <= last:
            origin_candidates.append(cur)
            cur = shamsi_add_months(cur, 1)

    covers: list[OriginCoverage] = []
    for origin_ym in origin_candidates:
        window = make_forecast_window(origin_ym, config=cfg)
        train_dates = [int(d) for d in sales.index if int(d) < window.forecast_origin]
        if len(train_dates) < min_train:
            continue

        eval_dates: list[int] = []
        eval_horizons: list[int] = []
        for horizon, target in zip(window.horizons, window.target_dates):
            t = int(target)
            if t in observed_dates:
                eval_dates.append(t)
                eval_horizons.append(int(horizon))

        if not eval_dates:
            continue

        max_h = int(max(eval_horizons))
        covers.append(
            OriginCoverage(
                origin=parse_origin(window.forecast_origin),
                window=window,
                evaluable_target_dates=tuple(eval_dates),
                evaluable_horizons=tuple(eval_horizons),
                max_evaluated_horizon=max_h,
                full_horizon_coverage=(max_h >= h_max and len(eval_horizons) >= h_max),
            )
        )

    # Prefer full 15-month windows, then later origins (more training data).
    covers.sort(
        key=lambda c: (
            not c.full_horizon_coverage,
            -c.origin.shamsi_yyyymm,
        )
    )
    return covers


def eval_window_for_origin(coverage: OriginCoverage) -> ForecastWindow:
    """Build the forecast window passed to models (evaluable targets only)."""
    return ForecastWindow(
        forecast_origin=coverage.window.forecast_origin,
        training_end=coverage.window.training_end,
        target_dates=coverage.evaluable_target_dates,
        horizons=coverage.evaluable_horizons,
    )
