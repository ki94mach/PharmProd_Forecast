"""Sections 3-4: Human feature granularity and sample-size audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import BIAS_AF_EPS, PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.research.audit.common import audit_output_dir, primary_test_predictions, save_csv
from pkg.research.features.human import HUMAN_FEATURE_NAMES, add_human_features


def _horizon_col(df: pd.DataFrame) -> pd.Series:
    if "horizon" in df.columns:
        return df["horizon"].astype(int)
    if "budget_horizon" in df.columns:
        return df["budget_horizon"].astype(int)
    return df["ts_horizon"].astype(int)


def compute_human_sample_counts(
    budget_hist: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostic human_n_* counts per test row (PIT)."""
    bud = budget_hist.copy()
    bud["target_date"] = bud["target_date"].astype(int)
    if "budget_forecast" not in bud.columns and "forecast" in bud.columns:
        bud = bud.rename(columns={"forecast": "budget_forecast"})

    horizons = _horizon_col(test)
    rows = []
    for product, origin, h in zip(
        test["product"].astype(str),
        test["origin"].astype(int),
        horizons.astype(int),
    ):
        hist = bud.loc[bud["target_date"] < origin]
        hp = hist.loc[hist["product"].astype(str) == product]
        hh = hist.loc[_horizon_col(hist) == h]
        hph = hist.loc[
            (hist["product"].astype(str) == product) & (_horizon_col(hist) == h)
        ]
        rows.append(
            {
                "product": product,
                "origin": int(origin),
                "horizon": int(h),
                "human_n_product": len(hp),
                "human_n_horizon": len(hh),
                "human_n_product_horizon": len(hph),
            }
        )
    return pd.DataFrame(rows)


def _bias_fallback_level(
    product: str,
    h: int,
    maps: dict,
) -> str:
    if (product, h) in maps["by_ph_bias"]:
        return "product_horizon"
    if product in maps["by_product_bias"]:
        return "product"
    return "global"


def analyze_human_granularity(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Granularity and per-origin nunique for Human features."""
    out_dir = out_dir or audit_output_dir()
    test = primary_test_predictions(ds)
    test = add_human_features(
        test,
        ds.budget_universe,
        matched_hist=ds.matched_universe,
        origin_col="origin",
    )

    feature_levels = {
        "human_bias_product": "product (fallback global)",
        "human_mae_product": "product (fallback global)",
        "human_bias_horizon": "horizon (fallback global)",
        "human_mae_horizon": "horizon (fallback global)",
        "human_bias_product_horizon": "product×horizon (fallback product→global)",
        "historical_actual_budget_ratio": "global per origin",
        "mean_human_adjustment": "matched-history global per origin",
        "mean_abs_human_adjustment": "matched-history global per origin",
    }

    gran_rows = []
    for feat in HUMAN_FEATURE_NAMES:
        for origin in PRIMARY_ORIGINS:
            sub = test.loc[test["origin"].astype(int) == int(origin)]
            if sub.empty:
                continue
            nunique = int(sub[feat].nunique())
            pct_const = 100.0 if nunique <= 1 else float(
                sub.groupby(feat).size().max() / len(sub) * 100.0
            )
            gran_rows.append(
                {
                    "feature": feat,
                    "origin": int(origin),
                    "aggregation_level": feature_levels.get(feat, "unknown"),
                    "nunique": nunique,
                    "pct_rows_constant_within_origin": pct_const,
                    "constant_per_origin": nunique == 1,
                    "regime_indicator": feat
                    in {
                        "historical_actual_budget_ratio",
                        "mean_human_adjustment",
                        "mean_abs_human_adjustment",
                    }
                    and nunique == 1,
                }
            )
    granularity_df = pd.DataFrame(gran_rows)
    save_csv(granularity_df, out_dir, "human_granularity.csv")

    return {"granularity": granularity_df, "feature_levels": feature_levels}


def analyze_human_sample_sizes(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """PIT sample-size distributions for Human reliability stats."""
    out_dir = out_dir or audit_output_dir()
    test = primary_test_predictions(ds)
    counts = compute_human_sample_counts(ds.budget_universe, test)
    test = test.reset_index(drop=True)
    counts = counts.reset_index(drop=True)
    full = pd.concat([test, counts[["human_n_product", "human_n_horizon", "human_n_product_horizon"]]], axis=1)

    # Fallback level for product_horizon bias
    from pkg.research.features.human import _pit_maps

    bud = ds.budget_universe.copy()
    bud["target_date"] = bud["target_date"].astype(int)
    fallback_rows = []
    for origin in PRIMARY_ORIGINS:
        hist = bud.loc[bud["target_date"] < int(origin)]
        maps = _pit_maps(hist)
        sub = full.loc[full["origin"].astype(int) == int(origin)]
        for _, row in sub.iterrows():
            fb = _bias_fallback_level(str(row["product"]), int(row["horizon"]), maps)
            fallback_rows.append({"origin": int(origin), "fallback_level": fb})
    fb_df = pd.DataFrame(fallback_rows)
    fallback_summary = (
        fb_df.groupby(["origin", "fallback_level"]).size().reset_index(name="n")
    )

    def _bucket(n: int) -> str:
        if n <= 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 5:
            return "4-5"
        return ">5"

    full["n_ph_bucket"] = full["human_n_product_horizon"].map(_bucket)
    bucket_by_origin = (
        full.groupby(["origin", "n_ph_bucket"]).size().reset_index(name="n")
    )
    bucket_overall = full["n_ph_bucket"].value_counts().reset_index()
    bucket_overall.columns = ["n_ph_bucket", "n"]

    save_csv(full, out_dir, "human_sample_counts.csv")
    save_csv(bucket_by_origin, out_dir, "human_n_ph_by_origin.csv")
    save_csv(bucket_overall, out_dir, "human_n_ph_overall.csv")
    save_csv(fallback_summary, out_dir, "human_bias_fallback.csv")

    shrinkage_doc = (
        "Proposed shrinkage (F2 design only, not implemented):\n"
        "  shrunk_bias_ph = (n_ph * bias_ph + k * bias_product) / (n_ph + k)\n"
        "  shrunk_bias_product = (n_p * bias_product + k * bias_global) / (n_p + k)\n"
        "  k candidates: 3, 5 (no test-set tuning in audit)"
    )

    return {
        "counts": full,
        "bucket_by_origin": bucket_by_origin,
        "bucket_overall": bucket_overall,
        "fallback_summary": fallback_summary,
        "shrinkage_proposal": shrinkage_doc,
    }
