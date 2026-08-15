"""Point-in-time demand / sales-dynamics features (keyed by forecast origin).

Uses frozen ``raw/sales.parquet`` only — never live SQL.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months

EPS = 1.0

DEMAND_FEATURE_NAMES: tuple[str, ...] = (
    "sales_roll6",
    "sales_roll12",
    "sales_std3",
    "sales_std6",
    "sales_std12",
    "trend_3m",
    "trend_6m",
    "sales_yoy_change",
    "sales_vs_roll12",
    "recent_growth",
    "recent_acceleration",
)


def _sales_at(sales_pivot: pd.Series, product: str, ym: int) -> float:
    try:
        return float(sales_pivot.loc[(product, int(ym))])
    except KeyError:
        return np.nan


def _window_values(
    sales_pivot: pd.Series, product: str, origin: int, n: int
) -> list[float]:
    """Sales at origin-1 .. origin-n (inclusive), oldest-first."""
    vals = []
    for k in range(n, 0, -1):
        vals.append(_sales_at(sales_pivot, product, shamsi_add_months(origin, -k)))
    return vals


def _nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def _nanstd(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) < 2:
        return 0.0
    return float(np.std(finite, ddof=0))


def _rel_change(a: float, b: float) -> float:
    """(a - b) / max(|b|, EPS); NaNs -> 0."""
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float((a - b) / max(abs(b), EPS))


def load_frozen_sales(root) -> pd.DataFrame:
    """Load ``{root}/raw/sales.parquet``."""
    path = root / "raw" / "sales.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Frozen sales history missing: {path}")
    sales = pd.read_parquet(path)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = sales["date"].astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")
    return sales


def add_demand_features(
    df: pd.DataFrame,
    sales_hist: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach demand features for each row using sales known before ``origin``.

    ``origin_col`` defaults to ``origin``, else ``ts_origin``, else ``budget_origin``.
    """
    out = df.copy()
    if origin_col is None:
        if "origin" in out.columns:
            origin_col = "origin"
        elif "ts_origin" in out.columns:
            origin_col = "ts_origin"
        elif "budget_origin" in out.columns:
            origin_col = "budget_origin"
        else:
            raise ValueError("panel needs origin / ts_origin / budget_origin")

    sales_pivot = (
        sales_hist.groupby(["product", "date"], as_index=False)["sales"]
        .sum()
        .set_index(["product", "date"])["sales"]
    )

    # Cache per (product, origin)
    cache: dict[tuple[str, int], dict[str, float]] = {}
    rows = []
    for product, origin in zip(out["product"].astype(str), out[origin_col].astype(int)):
        key = (product, int(origin))
        if key not in cache:
            o = int(origin)
            w12 = _window_values(sales_pivot, product, o, 12)
            w6 = w12[-6:]
            w3 = w12[-3:]
            lag1 = w12[-1] if w12 else np.nan
            lag3 = w12[-3] if len(w12) >= 3 else np.nan
            lag6 = w12[-6] if len(w12) >= 6 else np.nan
            lag12 = w12[0] if len(w12) >= 12 else np.nan
            roll12 = _nanmean(w12)
            growth_1_3 = _rel_change(lag1, lag3)
            growth_3_6 = _rel_change(lag3, lag6)
            cache[key] = {
                "sales_roll6": _nanmean(w6),
                "sales_roll12": roll12,
                "sales_std3": _nanstd(w3),
                "sales_std6": _nanstd(w6),
                "sales_std12": _nanstd(w12),
                "trend_3m": _rel_change(lag1, lag3),
                "trend_6m": _rel_change(lag1, lag6),
                "sales_yoy_change": _rel_change(lag1, lag12),
                "sales_vs_roll12": (
                    0.0
                    if not np.isfinite(lag1) or not np.isfinite(roll12)
                    else float(lag1 / max(roll12, EPS) - 1.0)
                ),
                "recent_growth": growth_1_3,
                "recent_acceleration": growth_1_3 - growth_3_6,
            }
        rows.append(cache[key])

    feat = pd.DataFrame(rows, index=out.index)
    for col in DEMAND_FEATURE_NAMES:
        out[col] = feat[col].fillna(0.0).astype(float)
    return out
