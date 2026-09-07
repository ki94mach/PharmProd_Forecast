"""V3A outer-backtest metrics (raw units, equal-weight horizon MAE).

Wraps V2 metric helpers so the V3A report surface uses ``mean_horizon_MAE``
as the primary architecture score. Horizons receive equal weight; row counts
must not dominate shorter horizons.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.metrics import (
    aggregate_metrics,
    horizon_bias,
    horizon_mae,
    horizon_rmse,
    horizon_wmape,
    selection_mae_from_horizons,
)


def mean_horizon_mae(predictions: pd.DataFrame) -> float:
    """Equal-weight mean of available horizon-level MAEs (not row-weighted)."""
    return float(selection_mae_from_horizons(horizon_mae(predictions)))


def metrics_summary_row(
    product: str,
    architecture: str,
    predictions: pd.DataFrame,
    *,
    number_of_origins: int,
    number_of_predictions: int,
    evaluated_horizons: tuple[int, ...],
    max_evaluated_horizon: int,
    unavailable_fold_count: int,
    forecast_horizon: int = 15,
    v2_config: Optional[TSForecastConfig] = None,
) -> dict[str, Any]:
    """Flat metrics row for one (product, architecture) slice."""
    cfg = v2_config or V2_DEFAULT_CONFIG
    # aggregate_metrics expects product/model columns only for filtering by caller;
    # pass the already-sliced predictions frame.
    m = aggregate_metrics(predictions, config=cfg)
    h_mae: pd.Series = m["horizon_mae"]  # type: ignore[assignment]
    row: dict[str, Any] = {
        "product": product,
        "architecture": architecture,
        "mean_horizon_MAE": m["selection_mae"],
        "overall_rmse": m["overall_rmse"],
        "overall_bias": m["overall_bias"],
        "overall_wmape": m["overall_wmape"],
        "number_of_origins": int(number_of_origins),
        "number_of_predictions": int(number_of_predictions),
        "evaluated_horizons": evaluated_horizons,
        "max_evaluated_horizon": int(max_evaluated_horizon),
        "unavailable_fold_count": int(unavailable_fold_count),
    }
    for h in range(1, int(forecast_horizon) + 1):
        row[f"mae_h{h}"] = float(h_mae[h]) if h in h_mae.index else float("nan")
    return row


__all__ = [
    "horizon_bias",
    "horizon_mae",
    "horizon_rmse",
    "horizon_wmape",
    "mean_horizon_mae",
    "metrics_summary_row",
    "selection_mae_from_horizons",
]
