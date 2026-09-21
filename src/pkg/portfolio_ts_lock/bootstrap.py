"""Bootstrap and sensitivity helpers (vectorized for large panels)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _wmape_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    m = np.isfinite(a) & np.isfinite(f)
    a, f = a[m], f[m]
    denom = float(np.sum(np.abs(a)))
    if denom <= 0 or a.size == 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(f - a)) / denom)


def _wmape_lift_arrays(
    actual: np.ndarray, v2: np.ndarray, hy: np.ndarray
) -> float:
    return _wmape_pct(actual, v2) - _wmape_pct(actual, hy)


def _wmape_lift(g: pd.DataFrame) -> float:
    return _wmape_lift_arrays(
        g["actual"].to_numpy(dtype=float),
        g["f_v2_policy"].to_numpy(dtype=float),
        g["f_a0_hybrid"].to_numpy(dtype=float),
    )


def _prepare_sku_blocks(
    df: pd.DataFrame,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    skus: list[str] = []
    acts: list[np.ndarray] = []
    v2s: list[np.ndarray] = []
    hys: list[np.ndarray] = []
    for product, g in df.groupby(df["product"].astype(str), sort=False):
        skus.append(str(product))
        acts.append(g["actual"].to_numpy(dtype=float))
        v2s.append(g["f_v2_policy"].to_numpy(dtype=float))
        hys.append(g["f_a0_hybrid"].to_numpy(dtype=float))
    return skus, acts, v2s, hys


def sku_cluster_bootstrap(
    df: pd.DataFrame,
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    skus, acts, v2s, hys = _prepare_sku_blocks(df)
    n = len(skus)
    rng = np.random.default_rng(seed)
    lifts = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        draw = rng.integers(0, n, size=n)
        a = np.concatenate([acts[i] for i in draw])
        v = np.concatenate([v2s[i] for i in draw])
        h = np.concatenate([hys[i] for i in draw])
        lifts[b] = _wmape_lift_arrays(a, v, h)
    return {
        "B": int(B),
        "lift_mean": float(np.nanmean(lifts)),
        "lift_ci95_low": float(np.nanquantile(lifts, 0.025)),
        "lift_ci95_high": float(np.nanquantile(lifts, 0.975)),
        "frac_positive": float(np.mean(lifts > 0)),
    }


def origin_block_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    origins = sorted(df["forecast_origin"].astype(int).unique().tolist())
    for o in origins:
        g = df.loc[df["forecast_origin"].astype(int) != o]
        rows.append(
            {"drop_origin": o, "lift_wmape_pp": _wmape_lift(g), "n_rows": int(len(g))}
        )
    return pd.DataFrame(rows)


def hierarchical_origin_then_sku_bootstrap(
    df: pd.DataFrame,
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Resample origins, then SKUs within each drawn origin block."""
    rng = np.random.default_rng(seed)
    by_origin: dict[int, tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}
    for origin, g in df.groupby(df["forecast_origin"].astype(int), sort=False):
        by_origin[int(origin)] = _prepare_sku_blocks(g)
    origins = list(by_origin.keys())
    lifts = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        drawn_origins = rng.choice(origins, size=len(origins), replace=True)
        a_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        h_parts: list[np.ndarray] = []
        for o in drawn_origins:
            skus, acts, v2s, hys = by_origin[int(o)]
            if not skus:
                continue
            n = len(skus)
            draw = rng.integers(0, n, size=n)
            a_parts.extend(acts[i] for i in draw)
            v_parts.extend(v2s[i] for i in draw)
            h_parts.extend(hys[i] for i in draw)
        if not a_parts:
            lifts[b] = float("nan")
            continue
        lifts[b] = _wmape_lift_arrays(
            np.concatenate(a_parts),
            np.concatenate(v_parts),
            np.concatenate(h_parts),
        )
    return {
        "B": int(B),
        "lift_mean": float(np.nanmean(lifts)),
        "lift_ci95_low": float(np.nanquantile(lifts, 0.025)),
        "lift_ci95_high": float(np.nanquantile(lifts, 0.975)),
        "frac_positive": float(np.mean(lifts > 0)),
    }


def leave_one_origin(df: pd.DataFrame) -> pd.DataFrame:
    return origin_block_sensitivity(df)


def leave_one_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-SKU using total abs-error accounting (exact WMAPE recomputation)."""
    skus, acts, v2s, hys = _prepare_sku_blocks(df)
    # Precompute per-SKU abs-error sums and abs-actual sums for both policies
    ae_v2 = np.array([float(np.nansum(np.abs(v - a))) for a, v in zip(acts, v2s)])
    ae_hy = np.array([float(np.nansum(np.abs(h - a))) for a, h in zip(acts, hys)])
    abs_a = np.array([float(np.nansum(np.abs(a))) for a in acts])
    n_rows = np.array([int(a.size) for a in acts])
    tot_ae_v2 = float(ae_v2.sum())
    tot_ae_hy = float(ae_hy.sum())
    tot_abs_a = float(abs_a.sum())
    tot_n = int(n_rows.sum())
    rows = []
    for i, s in enumerate(skus):
        denom = tot_abs_a - abs_a[i]
        if denom <= 0:
            lift = float("nan")
        else:
            wm_v2 = 100.0 * (tot_ae_v2 - ae_v2[i]) / denom
            wm_hy = 100.0 * (tot_ae_hy - ae_hy[i]) / denom
            lift = float(wm_v2 - wm_hy)
        rows.append(
            {
                "drop_sku": s,
                "lift_wmape_pp": lift,
                "n_rows": int(tot_n - n_rows[i]),
            }
        )
    return pd.DataFrame(rows)


def drop_top_k_hybrid_gainers(df: pd.DataFrame, k: int = 3) -> dict[str, Any]:
    """Drop SKUs with largest hybrid AE improvement; report remaining lift."""
    gains = []
    for product, g in df.groupby("product", sort=False):
        m = g[["actual", "f_v2_policy", "f_a0_hybrid"]].dropna()
        if m.empty:
            continue
        ae_v2 = float(np.mean(np.abs(m["f_v2_policy"] - m["actual"])))
        ae_hy = float(np.mean(np.abs(m["f_a0_hybrid"] - m["actual"])))
        gains.append((str(product), ae_v2 - ae_hy))  # positive = hybrid better
    gains.sort(key=lambda x: x[1], reverse=True)
    drop = {p for p, _ in gains[: int(k)]}
    remain = df.loc[~df["product"].astype(str).isin(drop)]
    lift = _wmape_lift(remain) if not remain.empty else float("nan")
    return {
        "k": int(k),
        "dropped_skus": sorted(drop),
        "lift_wmape_pp": float(lift),
        "lift_positive": bool(lift > 0) if lift == lift else False,
    }
