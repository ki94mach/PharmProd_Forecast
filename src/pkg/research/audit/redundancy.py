"""Section 2: Demand feature redundancy audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import CLEAN_QUANT_FEATURES
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.research.audit.common import audit_output_dir, primary_test_predictions, save_csv
from pkg.research.features.demand import (
    DEMAND_FEATURE_NAMES,
    _nanmean,
    _nanstd,
    _rel_change,
    _window_values,
    add_demand_features,
    load_frozen_sales,
)

F0_SALES = [
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
]


def _recompute_demand_internals(
    sales_pivot: pd.Series,
    product: str,
    origin: int,
) -> dict[str, float]:
    """Mirror demand.py cache logic for cross-check."""
    w12 = _window_values(sales_pivot, product, origin, 12)
    w6 = w12[-6:]
    w3 = w12[-3:]
    lag1 = w12[-1] if w12 else np.nan
    lag3 = w12[-3] if len(w12) >= 3 else np.nan
    lag6 = w12[-6] if len(w12) >= 6 else np.nan
    lag12 = w12[0] if len(w12) >= 12 else np.nan
    roll12 = _nanmean(w12)
    growth_1_3 = _rel_change(lag1, lag3)
    growth_3_6 = _rel_change(lag3, lag6)
    return {
        "lag1": lag1,
        "lag3": lag3,
        "lag6": lag6,
        "lag12": lag12,
        "sales_roll6": _nanmean(w6),
        "sales_roll12": roll12,
        "sales_std3": _nanstd(w3),
        "sales_std6": _nanstd(w6),
        "sales_std12": _nanstd(w12),
        "trend_3m": growth_1_3,
        "trend_6m": _rel_change(lag1, lag6),
        "sales_yoy_change": _rel_change(lag1, lag12),
        "sales_vs_roll12": (
            0.0
            if not np.isfinite(lag1) or not np.isfinite(roll12)
            else float(lag1 / max(roll12, 1.0) - 1.0)
        ),
        "recent_growth": growth_1_3,
        "recent_acceleration": growth_1_3 - growth_3_6,
    }


def analyze_demand_redundancy(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Correlation, algebraic duplicates, F0 vs demand cross-check."""
    out_dir = out_dir or audit_output_dir()
    test = primary_test_predictions(ds)
    sales_hist = load_frozen_sales(ds.root)
    test = add_demand_features(test, sales_hist, origin_col="origin")

    sales_cols = F0_SALES + list(DEMAND_FEATURE_NAMES)
    avail = [c for c in sales_cols if c in test.columns]
    corr = test[avail].corr()

    # Algebraic duplicate check: trend_3m vs recent_growth
    if "trend_3m" in test.columns and "recent_growth" in test.columns:
        exact_dup = (test["trend_3m"] - test["recent_growth"]).abs()
        dup_max_diff = float(exact_dup.max())
        dup_n_exact = int((exact_dup == 0).sum())
    else:
        dup_max_diff = float("nan")
        dup_n_exact = 0

    # High correlation pairs
    redundancy_rows = []
    if dup_max_diff == 0.0 or dup_max_diff < 1e-12:
        redundancy_rows.append(
            {
                "feature_a": "trend_3m",
                "feature_b": "recent_growth",
                "relationship": "exact",
                "correlation": 1.0,
                "notes": "Both equal _rel_change(lag1, lag3) in demand.py",
            }
        )

    cols = avail
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if np.isnan(r):
                continue
            if abs(r) > 0.95 and not (
                a == "trend_3m" and b == "recent_growth"
            ):
                redundancy_rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "relationship": "high_corr",
                        "correlation": float(r),
                        "notes": f"|r|={abs(r):.4f}",
                    }
                )

    # F0 mapping notes
    f0_mappings = [
        ("sales_lag_1", "demand lag1 (origin-1)", "exact_definition"),
        ("sales_lag_3", "demand lag3 (origin-3)", "exact_definition"),
        ("sales_roll3", "mean(lag1,lag2,lag3)", "exact_definition"),
        ("sales_lag_2", "demand origin-2", "exact_definition"),
        ("sales_lag_12", "demand lag12 (origin-12)", "exact_definition"),
    ]
    for fa, fb, rel in f0_mappings:
        redundancy_rows.append(
            {
                "feature_a": fa,
                "feature_b": fb,
                "relationship": rel,
                "correlation": float("nan"),
                "notes": "Same sales window keyed by origin",
            }
        )

    redundancy_df = pd.DataFrame(redundancy_rows)

    # Cross-source equivalence: F0 lags vs recomputed from sales parquet
    sales_pivot = (
        sales_hist.groupby(["product", "date"], as_index=False)["sales"]
        .sum()
        .set_index(["product", "date"])["sales"]
    )
    # Spot-check lag1/lag3/roll3 alignment
    cross_check = []
    for _, row in test.head(500).iterrows():  # spot-check sample
        product = str(row["product"])
        origin = int(row["origin"])
        internal = _recompute_demand_internals(sales_pivot, product, origin)
        f0_lag1 = float(row.get("sales_lag_1", 0))
        rec_lag1 = internal["lag1"] if np.isfinite(internal["lag1"]) else 0.0
        f0_lag3 = float(row.get("sales_lag_3", 0))
        rec_lag3 = internal["lag3"] if np.isfinite(internal["lag3"]) else 0.0
        w3_vals = _window_values(sales_pivot, product, origin, 3)
        rec_roll3 = _nanmean(w3_vals)
        cross_check.append(
            {
                "product": product,
                "origin": origin,
                "f0_lag1": f0_lag1,
                "recomputed_lag1": rec_lag1,
                "lag1_match": abs(f0_lag1 - rec_lag1) < 1e-6,
                "f0_lag3": f0_lag3,
                "recomputed_lag3": rec_lag3,
                "lag3_match": abs(f0_lag3 - rec_lag3) < 1e-6,
                "f0_roll3": float(row.get("sales_roll3", 0)),
                "recomputed_roll3": rec_roll3,
                "roll3_match": abs(float(row.get("sales_roll3", 0)) - rec_roll3) < 1e-6,
            }
        )
    cross_df = pd.DataFrame(cross_check)

    save_csv(corr.reset_index().rename(columns={"index": "feature"}), out_dir, "demand_correlation.csv")
    save_csv(redundancy_df, out_dir, "demand_redundancy.csv")
    save_csv(cross_df, out_dir, "demand_f0_crosscheck.csv")

    return {
        "correlation": corr,
        "redundancy": redundancy_df,
        "cross_check": cross_df,
        "trend_recent_growth_max_diff": dup_max_diff,
        "trend_recent_growth_n_exact": dup_n_exact,
    }
