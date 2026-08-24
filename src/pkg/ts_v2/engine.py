"""V2 forecasting engine (orchestration only in this scaffold).

Pipeline (intended):

1. Resolve explicit :class:`~pkg.ts_v2.types.ForecastOrigin`.
2. Build history with ``date < origin`` (no implicit last-month removal).
3. Multi-origin / multi-horizon backtest + selection.
4. Final full-history refit of the winning model.
5. Emit raw monthly forecasts (no quarterly smoothing).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import EngineResult, ForecastOrigin


def forecast_products(
    products: Iterable[str],
    origin: ForecastOrigin,
    *,
    config: Optional[TSForecastConfig] = None,
) -> EngineResult:
    """Run selection (if needed) and full-history refit for one origin.

    Not implemented in this scaffold step.
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

    Not implemented in this scaffold step.
    """
    raise NotImplementedError("V2 engine.forecast_with_backtest is not implemented yet")


def default_engine_config() -> TSForecastConfig:
    """Return the frozen default V2 config (copy-safe via frozen dataclass)."""
    return DEFAULT_CONFIG
