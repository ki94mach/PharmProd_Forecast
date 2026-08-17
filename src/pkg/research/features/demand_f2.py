"""F2 robust demand-state features (point-in-time, keyed by forecast origin).

Compact redesign after F1A: drop raw relative-change ratios, drop
``recent_growth`` / ``trend_3m`` duplicates, drop ``sales_roll12`` (highly
correlated with F0 lags/roll3). Uses signed-log differences instead of
``(a-b)/max(|b|, EPS)``.

Uses frozen ``raw/sales.parquet`` only — never live SQL.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.features.demand import (
    _nanstd,
    _window_values,
    load_frozen_sales,
)

DEMAND_F2_FEATURE_NAMES: tuple[str, ...] = (
    "sales_std6",
    "sales_std12",
    "trend_log_3m",
    "trend_log_6m",
    "yoy_log_change",
    "sales_history_months",
    "sales_history_coverage_3m",
    "sales_history_coverage_12m",
)

# ``sales_roll12`` omitted: F1 audit |r| > 0.97 vs F0 lags / sales_roll3.
# ``sales_history_coverage_6m`` omitted: intermediate of 3m and 12m coverage.


def signed_log(x: float) -> float:
    """sign(x) * log1p(|x|). Missing/non-finite → 0 (coverage flags disambiguate)."""
    if not np.isfinite(x):
        return 0.0
    return float(np.sign(x) * np.log1p(abs(x)))


def _coverage(window: list[float], n: int) -> float:
    if n <= 0:
        return 0.0
    vals = window[-n:] if len(window) >= n else window
    obs = sum(1 for v in vals if np.isfinite(v))
    return float(obs / n)


def add_demand_f2_features(
    df: pd.DataFrame,
    sales_hist: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach F2 demand-state features using sales known strictly before origin."""
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

    cache: dict[tuple[str, int], dict[str, float]] = {}
    rows = []
    for product, origin in zip(out["product"].astype(str), out[origin_col].astype(int)):
        key = (product, int(origin))
        if key not in cache:
            o = int(origin)
            w12 = _window_values(sales_pivot, product, o, 12)
            w6 = w12[-6:]
            lag1 = w12[-1] if w12 else np.nan
            lag3 = w12[-3] if len(w12) >= 3 else np.nan
            lag6 = w12[-6] if len(w12) >= 6 else np.nan
            lag12 = w12[0] if len(w12) >= 12 else np.nan
            sl1 = signed_log(lag1)
            cache[key] = {
                "sales_std6": _nanstd(w6),
                "sales_std12": _nanstd(w12),
                "trend_log_3m": sl1 - signed_log(lag3),
                "trend_log_6m": sl1 - signed_log(lag6),
                "yoy_log_change": sl1 - signed_log(lag12),
                "sales_history_months": float(sum(1 for v in w12 if np.isfinite(v))),
                "sales_history_coverage_3m": _coverage(w12, 3),
                "sales_history_coverage_12m": _coverage(w12, 12),
            }
        rows.append(cache[key])

    feat = pd.DataFrame(rows, index=out.index)
    for col in DEMAND_F2_FEATURE_NAMES:
        out[col] = feat[col].fillna(0.0).astype(float)
    return out
