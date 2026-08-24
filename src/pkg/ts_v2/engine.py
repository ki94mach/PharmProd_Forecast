"""V2 forecasting engine (orchestration).

Pipeline (intended):

1. Resolve explicit :class:`~pkg.ts_v2.types.ForecastOrigin`.
2. Build history with ``date < origin`` (no implicit last-month removal).
3. Multi-origin / multi-horizon backtest + selection.
4. Final full-history refit of the winning model.
5. Emit raw monthly forecasts (no quarterly smoothing).

Models are invoked only through :func:`pkg.ts_v2.models.run_model` (same as backtest).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.models.base import ForecastModel, run_model
from pkg.ts_v2.types import (
    EngineResult,
    ForecastOrigin,
    ForecastWindow,
    ModelOutcome,
    PreparedSeries,
)


def forecast_series(
    model: ForecastModel,
    train_series: pd.Series,
    window: ForecastWindow,
) -> ModelOutcome:
    """Fit/predict one model on a prepared training series (shared model API)."""
    return run_model(model, train_series, window)


def forecast_prepared(
    model: ForecastModel,
    prepared: PreparedSeries,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ModelOutcome:
    """Run ``model`` on a :class:`PreparedSeries` using its forecast origin window."""
    cfg = config or DEFAULT_CONFIG
    window = make_forecast_window(prepared.forecast_origin, config=cfg)
    return run_model(model, prepared.values, window)


def forecast_products(
    products: Iterable[str],
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> EngineResult:
    """Run selection (if needed) and full-history refit for one origin.

    Not implemented in this step (no production models yet).
    """
    raise NotImplementedError("V2 engine.forecast_products is not implemented yet")


def forecast_with_backtest(
    products: Iterable[str],
    origins: Sequence[ForecastOrigin],
    final_origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> EngineResult:
    """Backtest across ``origins``, then refit and forecast at ``final_origin``.

    Not implemented in this step.
    """
    raise NotImplementedError("V2 engine.forecast_with_backtest is not implemented yet")


def default_engine_config() -> TSForecastConfig:
    """Return the frozen default V2 config (copy-safe via frozen dataclass)."""
    return DEFAULT_CONFIG
