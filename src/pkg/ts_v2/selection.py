"""Model selection for V2 (metric-driven, leakage-safe).

Selection uses :attr:`~pkg.ts_v2.config.TSForecastConfig.selection_metric`
on holdout forecasts from :mod:`pkg.ts_v2.backtest`. After a winner is chosen,
the production path refits on full history via :mod:`pkg.ts_v2.engine`.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import ForecastOrigin, SelectionResult


def select_best_model(
    scores: Mapping[str, float],
    *,
    product: str,
    origin: ForecastOrigin,
    config: Optional[TSForecastConfig] = None,
) -> SelectionResult:
    """Pick the model with the lowest selection metric score."""
    cfg = config or DEFAULT_CONFIG
    if not scores:
        raise ValueError("scores must be non-empty")
    best_name = min(scores, key=scores.get)
    return SelectionResult(
        product=product,
        origin=origin,
        best_model_name=best_name,
        scores=dict(scores),
        metric=cfg.selection_metric,
    )


def score_candidates(
    product: str,
    origin: ForecastOrigin,
    candidate_names: Sequence[str],
    *,
    config: Optional[TSForecastConfig] = None,
) -> SelectionResult:
    """Fit/score each candidate at ``origin``.

    Not implemented in this scaffold step (no models yet).
    """
    raise NotImplementedError("V2 model scoring is not implemented yet")
