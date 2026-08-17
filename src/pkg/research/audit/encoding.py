"""Section 5: Missing-history encoding audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.dataset import BenchmarkDataset
from pkg.research.audit.common import audit_output_dir, primary_test_predictions, save_csv
from pkg.research.audit.human_audit import compute_human_sample_counts
from pkg.research.features.demand import DEMAND_FEATURE_NAMES, add_demand_features, load_frozen_sales
from pkg.research.features.human import HUMAN_FEATURE_NAMES, add_human_features


def _sales_history_diagnostics(
    sales_pivot: pd.Series,
    product: str,
    origin: int,
) -> dict:
    """Months with observed sales in origin-12..origin-1 window."""
    months = []
    for k in range(12, 0, -1):
        ym = shamsi_add_months(origin, -k)
        try:
            v = float(sales_pivot.loc[(product, int(ym))])
            months.append(np.isfinite(v))
        except KeyError:
            months.append(False)
    n_obs = sum(months)
    return {
        "sales_history_months": n_obs,
        "sales_history_coverage_3m": sum(months[-3:]) / 3.0,
        "sales_history_coverage_6m": sum(months[-6:]) / 6.0,
        "sales_history_coverage_12m": n_obs / 12.0,
    }


def analyze_missing_history_encoding(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Quantify zeros from missing history vs genuine values."""
    out_dir = out_dir or audit_output_dir()
    test = primary_test_predictions(ds)
    sales_hist = load_frozen_sales(ds.root)
    test = add_demand_features(test, sales_hist, origin_col="origin")
    test = add_human_features(
        test,
        ds.budget_universe,
        matched_hist=ds.matched_universe,
        origin_col="origin",
    )
    counts = compute_human_sample_counts(ds.budget_universe, test)
    test = test.reset_index(drop=True)
    counts = counts.reset_index(drop=True)
    test = pd.concat(
        [
            test,
            counts[
                ["human_n_product", "human_n_horizon", "human_n_product_horizon"]
            ],
        ],
        axis=1,
    )

    sales_pivot = (
        sales_hist.groupby(["product", "date"], as_index=False)["sales"]
        .sum()
        .set_index(["product", "date"])["sales"]
    )
    hist_diag = []
    for product, origin in zip(test["product"].astype(str), test["origin"].astype(int)):
        hist_diag.append(_sales_history_diagnostics(sales_pivot, product, origin))
    hist_df = pd.DataFrame(hist_diag)
    test = pd.concat([test, hist_df], axis=1)

    demand_features = list(DEMAND_FEATURE_NAMES)
    human_features = list(HUMAN_FEATURE_NAMES)
    all_feats = demand_features + human_features

    zero_rows = []
    for feat in all_feats:
        if feat not in test.columns:
            continue
        is_zero = test[feat].astype(float) == 0.0
        n_zero = int(is_zero.sum())
        if n_zero == 0:
            continue

        # Classify zero reason
        if feat in demand_features:
            missing_hist = test["sales_history_months"] == 0
            missing_window = test["sales_history_coverage_12m"] < 0.25
            reason_missing = is_zero & (missing_hist | missing_window)
            ratio_feats = {
                "trend_3m",
                "trend_6m",
                "recent_growth",
                "recent_acceleration",
                "sales_yoy_change",
            }
            if feat in ratio_feats:
                reason_genuine = is_zero & ~reason_missing & (
                    test["sales_history_coverage_3m"] >= 0.67
                )
            elif feat.startswith("sales_std"):
                reason_genuine = is_zero & ~reason_missing & (
                    test["sales_history_months"] >= 2
                )
            else:
                reason_genuine = is_zero & ~reason_missing
            reason_computed = is_zero & ~reason_missing & ~reason_genuine
        else:
            # Human features
            no_hist = test["human_n_product"] == 0
            reason_missing = is_zero & no_hist
            reason_genuine = is_zero & ~no_hist & (feat.endswith("_horizon") | feat.endswith("_product"))
            reason_computed = is_zero & ~reason_missing & ~reason_genuine

        for origin in sorted(test["origin"].astype(int).unique()):
            sub = test.loc[test["origin"].astype(int) == origin]
            z = is_zero.loc[sub.index]
            zero_rows.append(
                {
                    "feature": feat,
                    "origin": int(origin),
                    "n_rows": len(sub),
                    "n_zero": int(z.sum()),
                    "pct_zero": float(z.mean() * 100),
                    "n_zero_missing_history": int(reason_missing.loc[sub.index].sum()),
                    "pct_zero_missing_history": float(
                        reason_missing.loc[sub.index].mean() * 100
                    )
                    if z.any()
                    else 0.0,
                    "n_zero_genuine": int(reason_genuine.loc[sub.index].sum()),
                    "n_zero_computed": int(reason_computed.loc[sub.index].sum()),
                }
            )

    zero_df = pd.DataFrame(zero_rows)
    save_csv(test, out_dir, "encoding_diagnostics_panel.csv")
    save_csv(zero_df, out_dir, "encoding_zero_summary.csv")

    return {"panel": test, "zero_summary": zero_df}
