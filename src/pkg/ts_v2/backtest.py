"""Multi-origin / multi-horizon backtesting for V2.

Unlike V1's single 80/20 rolling 1-step RMSE on a scaled series, V2 evaluates
explicit origins and horizons against raw actuals (once implemented).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, parse_origin
from pkg.ts_v2.models.base import ForecastModel, run_model
from pkg.ts_v2.types import (
    BacktestFold,
    ForecastOrigin,
    ForecastWindow,
    HorizonForecast,
    ModelOutcome,
)


def make_folds(
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[BacktestFold]:
    """Build backtest folds from :func:`~pkg.ts_v2.dates.make_forecast_window`."""
    cfg = config or DEFAULT_CONFIG
    folds: list[BacktestFold] = []
    for origin in origins:
        window = make_forecast_window(origin, config=cfg)
        folds.append(
            BacktestFold(
                origin=parse_origin(window.forecast_origin),
                train_end_exclusive=window.forecast_origin,
                horizons=window.horizons,
                window=window,
            )
        )
    return folds


def windows_for_origins(
    origins: Sequence[ForecastOrigin],
    *,
    config: Optional[TSForecastConfig] = None,
) -> list[ForecastWindow]:
    """Explicit forecast windows for each evaluation origin."""
    cfg = config or DEFAULT_CONFIG
    return [make_forecast_window(origin, config=cfg) for origin in origins]


def forecast_fold(
    model: ForecastModel,
    train_series,
    fold: BacktestFold,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ModelOutcome:
    """Run one candidate on one fold via the shared :func:`run_model` interface."""
    cfg = config or DEFAULT_CONFIG
    window = fold.window or make_forecast_window(fold.origin, config=cfg)
    return run_model(model, train_series, window)


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
