"""Historical Human (Line Budget) reliability features — point-in-time safe.

Only outcomes with ``target_date < origin`` are used when scoring a row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import BIAS_AF_EPS

HUMAN_FEATURE_NAMES: tuple[str, ...] = (
    "human_bias_product",
    "human_mae_product",
    "human_bias_horizon",
    "human_mae_horizon",
    "human_bias_product_horizon",
    "historical_actual_budget_ratio",
    "mean_human_adjustment",
    "mean_abs_human_adjustment",
)


def _origin_col(df: pd.DataFrame, origin_col: Optional[str]) -> str:
    if origin_col is not None:
        return origin_col
    if "origin" in df.columns:
        return "origin"
    if "budget_origin" in df.columns:
        return "budget_origin"
    if "ts_origin" in df.columns:
        return "ts_origin"
    raise ValueError("panel needs origin / budget_origin / ts_origin")


def _horizon_series(df: pd.DataFrame) -> pd.Series:
    if "horizon" in df.columns:
        return df["horizon"].astype(int)
    if "budget_horizon" in df.columns:
        return df["budget_horizon"].astype(int)
    if "ts_horizon" in df.columns:
        return df["ts_horizon"].astype(int)
    raise ValueError("panel needs horizon column")


def _pit_maps(hist: pd.DataFrame) -> dict:
    """Build PIT aggregates from known Budget outcomes."""
    if hist.empty:
        return {
            "global_bias": 0.0,
            "global_mae": 0.0,
            "global_af": 1.0,
            "by_product_bias": {},
            "by_product_mae": {},
            "by_horizon_bias": {},
            "by_horizon_mae": {},
            "by_ph_bias": {},
        }

    resid = hist["sales"].astype(float) - hist["budget_forecast"].astype(float)
    global_bias = float(resid.mean())
    global_mae = float(resid.abs().mean())
    bud = hist["budget_forecast"].astype(float)
    safe = bud.abs() >= BIAS_AF_EPS
    if safe.any():
        af = (hist.loc[safe, "sales"].astype(float) / bud.loc[safe]).mean()
        global_af = float(af) if np.isfinite(af) else 1.0
    else:
        global_af = 1.0

    tmp = hist.copy()
    tmp["_resid"] = resid.to_numpy()
    tmp["_abs"] = resid.abs().to_numpy()
    tmp["_h"] = _horizon_series(tmp)

    by_product_bias = tmp.groupby("product")["_resid"].mean().to_dict()
    by_product_mae = tmp.groupby("product")["_abs"].mean().to_dict()
    by_horizon_bias = tmp.groupby("_h")["_resid"].mean().to_dict()
    by_horizon_mae = tmp.groupby("_h")["_abs"].mean().to_dict()
    by_ph_bias = tmp.groupby(["product", "_h"])["_resid"].mean().to_dict()

    return {
        "global_bias": global_bias,
        "global_mae": global_mae,
        "global_af": global_af,
        "by_product_bias": by_product_bias,
        "by_product_mae": by_product_mae,
        "by_horizon_bias": by_horizon_bias,
        "by_horizon_mae": by_horizon_mae,
        "by_ph_bias": by_ph_bias,
    }


def _matched_adj_maps(matched_hist: pd.DataFrame) -> tuple[float, float]:
    if matched_hist is None or matched_hist.empty:
        return 0.0, 0.0
    if "human_adjustment" in matched_hist.columns:
        adj = matched_hist["human_adjustment"].astype(float)
    elif {"budget_forecast", "ts_forecast"}.issubset(matched_hist.columns):
        adj = (
            matched_hist["budget_forecast"].astype(float)
            - matched_hist["ts_forecast"].astype(float)
        )
    else:
        return 0.0, 0.0
    adj = adj.replace([np.inf, -np.inf], np.nan).dropna()
    if adj.empty:
        return 0.0, 0.0
    return float(adj.mean()), float(adj.abs().mean())


def add_human_features(
    df: pd.DataFrame,
    budget_hist: pd.DataFrame,
    matched_hist: Optional[pd.DataFrame] = None,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach historical Human reliability features (PIT vs each row origin)."""
    out = df.copy()
    ocol = _origin_col(out, origin_col)
    horizons = _horizon_series(out)

    bud = budget_hist.copy()
    bud["target_date"] = bud["target_date"].astype(int)
    if "budget_forecast" not in bud.columns and "forecast" in bud.columns:
        bud = bud.rename(columns={"forecast": "budget_forecast"})

    matched = None
    if matched_hist is not None and len(matched_hist):
        matched = matched_hist.copy()
        matched["target_date"] = matched["target_date"].astype(int)

    origins = sorted(out[ocol].astype(int).unique())
    pit_by_origin: dict[int, dict] = {}
    adj_by_origin: dict[int, tuple[float, float]] = {}

    for O in origins:
        hist = bud.loc[bud["target_date"] < O]
        pit_by_origin[O] = _pit_maps(hist)
        if matched is not None:
            mhist = matched.loc[matched["target_date"] < O]
            adj_by_origin[O] = _matched_adj_maps(mhist)
        else:
            adj_by_origin[O] = (0.0, 0.0)

    bias_p, mae_p, bias_h, mae_h, bias_ph, af_list = [], [], [], [], [], []
    mean_adj, mean_abs_adj = [], []

    for product, O, h in zip(
        out["product"].astype(str), out[ocol].astype(int), horizons.astype(int)
    ):
        maps = pit_by_origin[int(O)]
        h = int(h)
        bias_p.append(maps["by_product_bias"].get(product, maps["global_bias"]))
        mae_p.append(maps["by_product_mae"].get(product, maps["global_mae"]))
        bias_h.append(maps["by_horizon_bias"].get(h, maps["global_bias"]))
        mae_h.append(maps["by_horizon_mae"].get(h, maps["global_mae"]))
        if (product, h) in maps["by_ph_bias"]:
            bias_ph.append(maps["by_ph_bias"][(product, h)])
        elif product in maps["by_product_bias"]:
            bias_ph.append(maps["by_product_bias"][product])
        else:
            bias_ph.append(maps["global_bias"])
        af_list.append(maps["global_af"])
        ma, maa = adj_by_origin[int(O)]
        mean_adj.append(ma)
        mean_abs_adj.append(maa)

    out["human_bias_product"] = np.asarray(bias_p, dtype=float)
    out["human_mae_product"] = np.asarray(mae_p, dtype=float)
    out["human_bias_horizon"] = np.asarray(bias_h, dtype=float)
    out["human_mae_horizon"] = np.asarray(mae_h, dtype=float)
    out["human_bias_product_horizon"] = np.asarray(bias_ph, dtype=float)
    out["historical_actual_budget_ratio"] = np.asarray(af_list, dtype=float)
    out["mean_human_adjustment"] = np.asarray(mean_adj, dtype=float)
    out["mean_abs_human_adjustment"] = np.asarray(mean_abs_adj, dtype=float)

    for col in HUMAN_FEATURE_NAMES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
