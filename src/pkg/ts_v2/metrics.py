"""V2 backtest metrics (raw units, horizon-equal selection score)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig


def _finite_actual_pred(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    work = frame.copy()
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    return work.loc[work["actual"].notna() & work["prediction"].notna()].copy()


def horizon_mae(predictions: pd.DataFrame) -> pd.Series:
    """Mean absolute error by horizon (raw units)."""
    work = _finite_actual_pred(predictions)
    if work.empty:
        return pd.Series(dtype=float)
    err = (work["actual"] - work["prediction"]).abs()
    return err.groupby(work["horizon"].astype(int)).mean().sort_index()


def horizon_rmse(predictions: pd.DataFrame) -> pd.Series:
    work = _finite_actual_pred(predictions)
    if work.empty:
        return pd.Series(dtype=float)
    err = work["actual"] - work["prediction"]
    return np.sqrt((err ** 2).groupby(work["horizon"].astype(int)).mean()).sort_index()


def horizon_bias(predictions: pd.DataFrame) -> pd.Series:
    """Mean signed error (prediction - actual) by horizon."""
    work = _finite_actual_pred(predictions)
    if work.empty:
        return pd.Series(dtype=float)
    err = work["prediction"] - work["actual"]
    return err.groupby(work["horizon"].astype(int)).mean().sort_index()


def horizon_wmape(predictions: pd.DataFrame) -> pd.Series:
    """Weighted MAPE by horizon when sum(|actual|) > 0."""
    work = _finite_actual_pred(predictions)
    if work.empty:
        return pd.Series(dtype=float)
    out: dict[int, float] = {}
    for h, g in work.groupby(work["horizon"].astype(int)):
        denom = float(g["actual"].abs().sum())
        if denom <= 0.0:
            out[int(h)] = float("nan")
        else:
            out[int(h)] = float((g["actual"] - g["prediction"]).abs().sum() / denom)
    return pd.Series(out).sort_index()


def selection_mae_from_horizons(horizon_maes: pd.Series) -> float:
    """Equal-weight mean of available horizon-level MAEs (not row-weighted)."""
    if horizon_maes is None or horizon_maes.empty:
        return float("nan")
    vals = horizon_maes.dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def aggregate_metrics(
    predictions: pd.DataFrame,
    *,
    config: Optional[TSForecastConfig] = None,
) -> dict[str, float | pd.Series]:
    """Diagnostic metrics for one model/product prediction slice."""
    cfg = config or DEFAULT_CONFIG
    h_mae = horizon_mae(predictions)
    h_rmse = horizon_rmse(predictions)
    h_bias = horizon_bias(predictions)
    h_wmape = horizon_wmape(predictions)
    sel = selection_mae_from_horizons(h_mae)
    work = _finite_actual_pred(predictions)
    overall_rmse = float("nan")
    overall_wmape = float("nan")
    if not work.empty:
        err = work["actual"] - work["prediction"]
        overall_rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = float(work["actual"].abs().sum())
        if denom > 0.0:
            overall_wmape = float(err.abs().sum() / denom)
    return {
        "selection_metric": cfg.selection_metric,
        "selection_mae": sel,
        "horizon_mae": h_mae,
        "horizon_rmse": h_rmse,
        "horizon_bias": h_bias,
        "horizon_wmape": h_wmape,
        "overall_rmse": overall_rmse,
        "overall_wmape": overall_wmape,
        "overall_bias": float(h_bias.mean()) if not h_bias.empty else float("nan"),
    }


def metrics_summary_row(
    product: str,
    model: str,
    predictions: pd.DataFrame,
    coverage: dict,
    *,
    config: Optional[TSForecastConfig] = None,
) -> dict:
    """Flat summary row for reporting."""
    m = aggregate_metrics(predictions, config=config)
    h_mae: pd.Series = m["horizon_mae"]  # type: ignore[assignment]
    row = {
        "product": product,
        "model": model,
        "selection_mae": m["selection_mae"],
        "overall_rmse": m["overall_rmse"],
        "overall_bias": m["overall_bias"],
        "overall_wmape": m["overall_wmape"],
        **coverage,
    }
    for h in range(1, int((config or DEFAULT_CONFIG).forecast_horizon) + 1):
        row[f"mae_h{h}"] = float(h_mae[h]) if h in h_mae.index else float("nan")
    return row
