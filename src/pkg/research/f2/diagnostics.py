"""Pre-model diagnostics for F2 demand and Human features."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, prep_lags
from pkg.research.f2.config import f2_output_dir
from pkg.research.features.demand_f2 import DEMAND_F2_FEATURE_NAMES, add_demand_f2_features
from pkg.research.features.demand import load_frozen_sales
from pkg.research.features.human_f2 import (
    HUMAN_F2_FEATURE_NAMES,
    SHRINKAGE_K,
    add_human_f2_features,
)

F0_SALES = [
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
]


def _primary_test(ds: BenchmarkDataset) -> pd.DataFrame:
    panel = prep_lags(ds.matched_universe)
    origins = set(int(o) for o in PRIMARY_ORIGINS)
    return panel.loc[panel["origin"].astype(int).isin(origins)].copy()


def _dist_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "min": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "n": 0,
            "n_zero": 0,
            "pct_zero": np.nan,
        }
    return {
        "min": float(x.min()),
        "p1": float(x.quantile(0.01)),
        "p5": float(x.quantile(0.05)),
        "median": float(x.median()),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
        "n": int(len(x)),
        "n_zero": int((x == 0).sum()),
        "pct_zero": float((x == 0).mean() * 100),
    }


def run_demand_diagnostics(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f2_output_dir()
    test = _primary_test(ds)
    sales = load_frozen_sales(ds.root)
    test = add_demand_f2_features(test, sales, origin_col="origin")

    dist_rows = []
    for col in DEMAND_F2_FEATURE_NAMES:
        dist_rows.append({"feature": col, **_dist_stats(test[col])})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / "demand_f2_distribution.csv", index=False)

    corr_cols = [c for c in F0_SALES + list(DEMAND_F2_FEATURE_NAMES) if c in test.columns]
    corr = test[corr_cols].corr()
    corr.reset_index().rename(columns={"index": "feature"}).to_csv(
        out_dir / "demand_f2_correlation.csv", index=False
    )

    high = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.95:
                high.append({"feature_a": a, "feature_b": b, "correlation": float(r)})
    high_df = pd.DataFrame(high)
    high_df.to_csv(out_dir / "demand_f2_high_corr.csv", index=False)

    cov = (
        test.groupby("origin")[
            [
                "sales_history_months",
                "sales_history_coverage_3m",
                "sales_history_coverage_12m",
            ]
        ]
        .mean()
        .reset_index()
    )
    cov.to_csv(out_dir / "demand_f2_coverage_by_origin.csv", index=False)

    return {"distribution": dist, "correlation": corr, "high_corr": high_df, "coverage": cov}


def run_human_diagnostics(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    out_dir = out_dir or f2_output_dir()
    test = _primary_test(ds)
    test = add_human_f2_features(
        test, ds.budget_universe, origin_col="origin", k=SHRINKAGE_K, extras=True
    )

    dist_rows = []
    for col in HUMAN_F2_FEATURE_NAMES + (
        "raw_bias_product",
        "raw_bias_product_horizon",
        "human_n_product",
        "human_n_product_horizon",
    ):
        if col in test.columns:
            dist_rows.append({"feature": col, **_dist_stats(test[col])})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / "human_f2_distribution.csv", index=False)

    # Amount of shrinkage
    test["shrink_delta_ph"] = (
        test["raw_bias_product_horizon"] - test["human_bias_product_horizon_shrunk"]
    ).abs()
    test["n_ph_bucket"] = pd.cut(
        test["human_n_product_horizon"],
        bins=[-0.5, 0.5, 1.5, 3.5, 5.5, np.inf],
        labels=["0", "1", "2-3", "4-5", ">5"],
    )
    shrink_by_n = (
        test.groupby("n_ph_bucket", observed=False)
        .agg(
            n=("shrink_delta_ph", "count"),
            mean_abs_shrink=("shrink_delta_ph", "mean"),
            median_n_ph=("human_n_product_horizon", "median"),
        )
        .reset_index()
    )
    shrink_by_n.to_csv(out_dir / "human_f2_shrinkage_by_n.csv", index=False)

    n_hist = (
        test["n_ph_bucket"].value_counts().rename_axis("n_ph_bucket").reset_index(name="n")
    )
    n_hist.to_csv(out_dir / "human_f2_n_ph_histogram.csv", index=False)

    # Variability within origin (should NOT be constant like F1 regime features)
    var_rows = []
    for feat in HUMAN_F2_FEATURE_NAMES:
        for origin, g in test.groupby(test["origin"].astype(int)):
            var_rows.append(
                {
                    "feature": feat,
                    "origin": int(origin),
                    "nunique": int(g[feat].nunique()),
                    "constant_per_origin": int(g[feat].nunique() == 1),
                }
            )
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(out_dir / "human_f2_variability_by_origin.csv", index=False)

    fallback = pd.DataFrame(
        [
            {
                "pct_fallback_ph": float(test["fallback_ph"].mean() * 100),
                "pct_fallback_product": float(test["fallback_product"].mean() * 100),
                "k": SHRINKAGE_K,
            }
        ]
    )
    fallback.to_csv(out_dir / "human_f2_fallback.csv", index=False)

    examples = test.nlargest(15, "human_n_product_horizon")[
        [
            "product",
            "horizon",
            "origin",
            "human_n_product_horizon",
            "raw_bias_product_horizon",
            "human_bias_product_horizon_shrunk",
            "raw_bias_product",
            "global_bias",
        ]
    ].copy()
    examples["kind"] = "high_n"
    low = test.nsmallest(15, "human_n_product_horizon")[
        [
            "product",
            "horizon",
            "origin",
            "human_n_product_horizon",
            "raw_bias_product_horizon",
            "human_bias_product_horizon_shrunk",
            "raw_bias_product",
            "global_bias",
        ]
    ].copy()
    low["kind"] = "low_n"
    examples = pd.concat([examples, low], ignore_index=True)
    examples.to_csv(out_dir / "human_f2_shrinkage_examples.csv", index=False)

    return {
        "distribution": dist,
        "shrinkage_by_n": shrink_by_n,
        "variability": var_df,
        "fallback": fallback,
        "examples": examples,
    }
