"""Section 6: Ratio / growth instability audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.research.audit.common import (
    audit_output_dir,
    distribution_stats,
    enrich_test_panel,
    primary_test_predictions,
    save_csv,
)
from pkg.research.features.demand import (
    DEMAND_FEATURE_NAMES,
    EPS,
    _nanmean,
    _window_values,
    add_demand_features,
    load_frozen_sales,
)
from pkg.research.features.human import add_human_features

RATIO_FEATURES = [
    "trend_3m",
    "trend_6m",
    "sales_yoy_change",
    "sales_vs_roll12",
    "recent_growth",
    "recent_acceleration",
    "historical_actual_budget_ratio",
]


def profile_ratio_features(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Distribution stats and denominator diagnostics for ratio features."""
    out_dir = out_dir or audit_output_dir()

    # Test panel (PRIMARY)
    test = primary_test_predictions(ds)
    sales_hist = load_frozen_sales(ds.root)
    test = add_demand_features(test, sales_hist, origin_col="origin")
    test = add_human_features(
        test,
        ds.budget_universe,
        matched_hist=ds.matched_universe,
        origin_col="origin",
    )
    test["panel_split"] = "test"

    # Train panel (all matched rows before max PRIMARY origin, for context)
    train = ds.matched_universe.copy()
    max_o = max(PRIMARY_ORIGINS)
    train = train.loc[train["origin"].astype(int) < max_o]
    train = add_demand_features(train, sales_hist, origin_col="origin")
    train = add_human_features(
        train,
        ds.budget_universe,
        matched_hist=ds.matched_universe,
        origin_col="origin",
    )
    train["panel_split"] = "train"

    combined = pd.concat([train, test], ignore_index=True)

    stat_rows = []
    for feat in RATIO_FEATURES:
        if feat not in combined.columns:
            continue
        for split in ("test", "train", "all"):
            sub = combined if split == "all" else combined.loc[combined["panel_split"] == split]
            stats = distribution_stats(sub[feat])
            stat_rows.append({"feature": feat, "panel_split": split, **stats})
    stats_df = pd.DataFrame(stat_rows)

    # Denominator diagnostics for demand ratios
    sales_pivot = (
        sales_hist.groupby(["product", "date"], as_index=False)["sales"]
        .sum()
        .set_index(["product", "date"])["sales"]
    )
    denom_rows = []
    for _, row in test.iterrows():
        product = str(row["product"])
        origin = int(row["origin"])
        w12 = _window_values(sales_pivot, product, origin, 12)
        lag1 = w12[-1] if w12 else np.nan
        lag3 = w12[-3] if len(w12) >= 3 else np.nan
        lag6 = w12[-6] if len(w12) >= 6 else np.nan
        lag12 = w12[0] if len(w12) >= 12 else np.nan
        roll12 = _nanmean(w12)
        denom_rows.append(
            {
                "product": product,
                "origin": origin,
                "horizon": int(row["horizon"]),
                "lag3_abs": abs(lag3) if np.isfinite(lag3) else np.nan,
                "lag6_abs": abs(lag6) if np.isfinite(lag6) else np.nan,
                "lag12_abs": abs(lag12) if np.isfinite(lag12) else np.nan,
                "roll12": roll12,
                "roll12_le_zero": roll12 <= 0 if np.isfinite(roll12) else True,
                "roll12_abs_lt_eps": abs(roll12) < EPS if np.isfinite(roll12) else True,
                "lag3_abs_lt_eps": abs(lag3) < EPS if np.isfinite(lag3) else True,
                "sales_vs_roll12": float(row.get("sales_vs_roll12", 0)),
            }
        )
    denom_df = pd.DataFrame(denom_rows)

    # Top extremes on test
    extreme_rows = []
    for feat in RATIO_FEATURES:
        if feat not in test.columns:
            continue
        top = test.nlargest(10, feat)[
            ["product", "origin", "horizon", feat]
        ].copy()
        top["feature"] = feat
        top["direction"] = "max"
        extreme_rows.append(top)
        bottom = test.nsmallest(10, feat)[
            ["product", "origin", "horizon", feat]
        ].copy()
        bottom["feature"] = feat
        bottom["direction"] = "min"
        extreme_rows.append(bottom)
    extremes_df = pd.concat(extreme_rows, ignore_index=True) if extreme_rows else pd.DataFrame()

    save_csv(stats_df, out_dir, "ratio_distribution_stats.csv")
    save_csv(denom_df, out_dir, "ratio_denominator_diagnostics.csv")
    save_csv(extremes_df, out_dir, "ratio_extremes.csv")

    denom_summary = {
        "roll12_le_zero": int(denom_df["roll12_le_zero"].sum()),
        "roll12_abs_lt_eps": int(denom_df["roll12_abs_lt_eps"].sum()),
        "lag3_abs_lt_eps": int(denom_df["lag3_abs_lt_eps"].sum()),
    }

    return {
        "stats": stats_df,
        "denominator": denom_df,
        "denominator_summary": denom_summary,
        "extremes": extremes_df,
    }
