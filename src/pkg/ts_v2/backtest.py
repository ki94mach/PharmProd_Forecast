"""Multi-origin / multi-horizon backtesting for V2.

Unlike V1's single 80/20 rolling 1-step RMSE on a scaled series, V2 evaluates
explicit origins and horizons against raw actuals (once implemented).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import BacktestFold, ForecastOrigin, HorizonForecast


def make_folds(
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[BacktestFold]:
    """Build backtest folds: train is ``date < origin``, horizons ``1..H``."""
    cfg = config or DEFAULT_CONFIG
    horizons = tuple(range(1, cfg.forecast_horizon + 1))
    return [
        BacktestFold(
            origin=origin,
            train_end_exclusive=origin.shamsi_yyyymm,
            horizons=horizons,
        )
        for origin in origins
    ]


def run_backtest(
    products: Iterable[str],
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[HorizonForecast]:
    """Evaluate candidates across origins and horizons.

    Not implemented in this scaffold step.
    """
    raise NotImplementedError("V2 backtest is not implemented yet")
