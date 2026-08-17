"""F2 shrunk Human-reliability features (Budget-only, point-in-time).

Does not require a TS twin (full ``budget_universe`` training remains valid).
Does not expose origin-constant regime features from F1B
(``historical_actual_budget_ratio``, ``mean_human_adjustment``).

Shrinkage constant ``k`` is fixed at 5 and is not tuned on PRIMARY origins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

SHRINKAGE_K = 5.0

HUMAN_F2_FEATURE_NAMES: tuple[str, ...] = (
    "human_n_product",
    "human_n_product_horizon",
    "human_bias_product_shrunk",
    "human_bias_product_horizon_shrunk",
    "human_mae_product_shrunk",
)

DEFERRED_REGIME_FEATURES: tuple[str, ...] = (
    "historical_actual_budget_ratio",
    "mean_human_adjustment",
    "mean_abs_human_adjustment",
)


def shrink(n: float, raw: float, parent: float, k: float = SHRINKAGE_K) -> float:
    """Empirical-Bayes shrink of ``raw`` toward ``parent`` with strength ``k``.

    n = 0 → parent; large n → raw.
    """
    n = float(n)
    k = float(k)
    return float((n * raw + k * parent) / (n + k))


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
    """PIT aggregates + support counts from Budget outcomes with target_date < O."""
    if hist.empty:
        return {
            "n_global": 0,
            "global_bias": 0.0,
            "global_mae": 0.0,
            "by_product_n": {},
            "by_product_bias": {},
            "by_product_mae": {},
            "by_horizon_n": {},
            "by_horizon_bias": {},
            "by_horizon_mae": {},
            "by_ph_n": {},
            "by_ph_bias": {},
            "by_ph_mae": {},
        }

    resid = hist["sales"].astype(float) - hist["budget_forecast"].astype(float)
    tmp = hist.copy()
    tmp["_resid"] = resid.to_numpy()
    tmp["_abs"] = resid.abs().to_numpy()
    tmp["_h"] = _horizon_series(tmp)
    tmp["_p"] = tmp["product"].astype(str)

    return {
        "n_global": int(len(hist)),
        "global_bias": float(resid.mean()),
        "global_mae": float(resid.abs().mean()),
        "by_product_n": tmp.groupby("_p").size().to_dict(),
        "by_product_bias": tmp.groupby("_p")["_resid"].mean().to_dict(),
        "by_product_mae": tmp.groupby("_p")["_abs"].mean().to_dict(),
        "by_horizon_n": tmp.groupby("_h").size().to_dict(),
        "by_horizon_bias": tmp.groupby("_h")["_resid"].mean().to_dict(),
        "by_horizon_mae": tmp.groupby("_h")["_abs"].mean().to_dict(),
        "by_ph_n": tmp.groupby(["_p", "_h"]).size().to_dict(),
        "by_ph_bias": tmp.groupby(["_p", "_h"])["_resid"].mean().to_dict(),
        "by_ph_mae": tmp.groupby(["_p", "_h"])["_abs"].mean().to_dict(),
    }


def _row_features(product: str, h: int, maps: dict, k: float) -> dict:
    n_p = int(maps["by_product_n"].get(product, 0))
    n_h = int(maps["by_horizon_n"].get(h, 0))
    n_ph = int(maps["by_ph_n"].get((product, h), 0))
    g_bias = float(maps["global_bias"])
    g_mae = float(maps["global_mae"])
    raw_p_bias = float(maps["by_product_bias"].get(product, g_bias))
    raw_p_mae = float(maps["by_product_mae"].get(product, g_mae))
    raw_ph_bias = float(maps["by_ph_bias"].get((product, h), raw_p_bias))

    shrunk_p_bias = shrink(n_p, raw_p_bias, g_bias, k)
    shrunk_p_mae = shrink(n_p, raw_p_mae, g_mae, k)
    shrunk_ph_bias = shrink(n_ph, raw_ph_bias, shrunk_p_bias, k)

    return {
        "human_n_global": float(maps["n_global"]),
        "human_n_product": float(n_p),
        "human_n_horizon": float(n_h),
        "human_n_product_horizon": float(n_ph),
        "raw_bias_product": raw_p_bias,
        "raw_bias_product_horizon": raw_ph_bias,
        "raw_mae_product": raw_p_mae,
        "global_bias": g_bias,
        "global_mae": g_mae,
        "human_bias_product_shrunk": shrunk_p_bias,
        "human_bias_product_horizon_shrunk": shrunk_ph_bias,
        "human_mae_product_shrunk": shrunk_p_mae,
        "fallback_ph": n_ph == 0,
        "fallback_product": n_p == 0,
    }


def add_human_f2_features(
    df: pd.DataFrame,
    budget_hist: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
    k: float = SHRINKAGE_K,
    extras: bool = False,
) -> pd.DataFrame:
    """Attach shrunk Human reliability features (PIT vs each row origin).

    ``extras=True`` also attaches diagnostic columns (raw bias, counts, fallbacks)
    that are **not** part of the scored F2B feature set.
    """
    out = df.copy()
    ocol = _origin_col(out, origin_col)
    horizons = _horizon_series(out)

    bud = budget_hist.copy()
    bud["target_date"] = bud["target_date"].astype(int)
    if "budget_forecast" not in bud.columns and "forecast" in bud.columns:
        bud = bud.rename(columns={"forecast": "budget_forecast"})

    origins = sorted(out[ocol].astype(int).unique())
    pit_by_origin: dict[int, dict] = {}
    for O in origins:
        hist = bud.loc[bud["target_date"] < O]
        pit_by_origin[int(O)] = _pit_maps(hist)

    extra_names = [
        "human_n_global",
        "human_n_horizon",
        "raw_bias_product",
        "raw_bias_product_horizon",
        "raw_mae_product",
        "global_bias",
        "global_mae",
        "fallback_ph",
        "fallback_product",
    ]
    collected = {c: [] for c in HUMAN_F2_FEATURE_NAMES}
    extra_collected = {c: [] for c in extra_names}

    for product, O, h in zip(
        out["product"].astype(str), out[ocol].astype(int), horizons.astype(int)
    ):
        row = _row_features(product, int(h), pit_by_origin[int(O)], k)
        for c in HUMAN_F2_FEATURE_NAMES:
            collected[c].append(row[c])
        if extras:
            for c in extra_names:
                extra_collected[c].append(row[c])

    for c, vals in collected.items():
        out[c] = np.asarray(vals, dtype=float)
    if extras:
        for c, vals in extra_collected.items():
            if c.startswith("fallback"):
                out[c] = np.asarray(vals, dtype=bool)
            else:
                out[c] = np.asarray(vals, dtype=float)

    for col in HUMAN_F2_FEATURE_NAMES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
