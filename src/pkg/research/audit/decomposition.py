"""Section 7: Error decomposition F1 vs F0."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import HORIZON_BUCKETS, PRIMARY_ORIGINS
from pkg.benchmark.dataset import BenchmarkDataset, horizon_bucket
from pkg.benchmark.evaluate import wmape
from pkg.research.audit.common import (
    ROW_KEYS,
    audit_output_dir,
    run_experiment_backtest,
    run_frozen_backtest,
    save_csv,
    weighted_portfolio_wmape,
)
from pkg.research.evaluate_features import _product_stats, _rel_wmape


def _merge_predictions(f0: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    b = f0[
        ["product", "qrt", "target_date", "test_origin", "horizon", "actual", "prediction"]
    ].rename(columns={"prediction": "pred_f0"})
    c = cand[
        ["product", "qrt", "target_date", "test_origin", "prediction"]
    ].rename(columns={"prediction": "pred_cand"})
    m = b.merge(c, on=list(ROW_KEYS), how="inner")
    m["ae_f0"] = (m["actual"] - m["pred_f0"]).abs()
    m["ae_cand"] = (m["actual"] - m["pred_cand"]).abs()
    m["delta_ae"] = m["ae_cand"] - m["ae_f0"]
    m["horizon_bucket"] = m["horizon"].map(horizon_bucket)
    return m


def decompose_error_delta(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
    experiments: tuple[str, ...] = ("F1A", "F1B", "F1C"),
) -> dict:
    """Aggregate absolute-error delta vs frozen F0 by product/origin/horizon."""
    out_dir = out_dir or audit_output_dir()
    all_summaries = []
    product_tables = []

    for anchor in ("ts", "human"):
        f0_res = run_frozen_backtest(ds, anchor)
        f0_preds = f0_res.predictions.copy()
        f0_preds["actual"] = f0_preds["sales"].astype(float)

        for exp_name in experiments:
            cand_res = run_experiment_backtest(ds, exp_name, anchor)
            cand_preds = cand_res.predictions.copy()
            m = _merge_predictions(f0_preds, cand_preds)

            net_delta = float(m["delta_ae"].sum())
            total_deterioration = float(m.loc[m["delta_ae"] > 0, "delta_ae"].sum())
            total_improvement = float(-m.loc[m["delta_ae"] < 0, "delta_ae"].sum())

            # By product
            by_product = (
                m.groupby("product")
                .agg(
                    delta_ae=("delta_ae", "sum"),
                    actual_volume=("actual", lambda s: float(np.abs(s).sum())),
                    n=("delta_ae", "count"),
                )
                .reset_index()
            )
            by_product["wmape_f0"] = by_product["product"].map(
                lambda p: wmape(
                    m.loc[m["product"] == p, "actual"],
                    m.loc[m["product"] == p, "pred_f0"],
                )
            )
            by_product["wmape_cand"] = by_product["product"].map(
                lambda p: wmape(
                    m.loc[m["product"] == p, "actual"],
                    m.loc[m["product"] == p, "pred_cand"],
                )
            )
            by_product = by_product.sort_values("delta_ae", ascending=False)

            top10_det = by_product.head(10)
            top10_imp = by_product.sort_values("delta_ae").head(10)

            det_sorted = by_product.loc[by_product["delta_ae"] > 0].sort_values(
                "delta_ae", ascending=False
            )
            top5_share = (
                float(det_sorted.head(5)["delta_ae"].sum() / total_deterioration * 100)
                if total_deterioration > 0
                else 0.0
            )
            top10_share = (
                float(det_sorted.head(10)["delta_ae"].sum() / total_deterioration * 100)
                if total_deterioration > 0
                else 0.0
            )

            pstats = _product_stats(f0_res, cand_res)
            w_wmape_f0 = weighted_portfolio_wmape(
                m["actual"].to_numpy(), m["pred_f0"].to_numpy()
            )
            w_wmape_cand = weighted_portfolio_wmape(
                m["actual"].to_numpy(), m["pred_cand"].to_numpy()
            )

            all_summaries.append(
                {
                    "experiment": exp_name,
                    "anchor": anchor,
                    "net_delta_ae": net_delta,
                    "total_deterioration": total_deterioration,
                    "total_improvement": total_improvement,
                    "top5_deterioration_share_pct": top5_share,
                    "top10_deterioration_share_pct": top10_share,
                    "product_win_rate": pstats["product_win_rate"],
                    "median_product_improvement_pct": pstats[
                        "median_product_improvement_pct"
                    ],
                    "n_products": pstats["n_products"],
                    "wmape_f0": float(f0_res.overall["wmape"].iloc[0]),
                    "wmape_cand": float(cand_res.overall["wmape"].iloc[0]),
                    "weighted_wmape_f0": w_wmape_f0,
                    "weighted_wmape_cand": w_wmape_cand,
                }
            )

            for _, row in top10_det.iterrows():
                product_tables.append(
                    {
                        "experiment": exp_name,
                        "anchor": anchor,
                        "direction": "deterioration",
                        "product": row["product"],
                        "delta_ae": row["delta_ae"],
                        "actual_volume": row["actual_volume"],
                        "wmape_f0": row["wmape_f0"],
                        "wmape_cand": row["wmape_cand"],
                        "n": row["n"],
                    }
                )
            for _, row in top10_imp.iterrows():
                product_tables.append(
                    {
                        "experiment": exp_name,
                        "anchor": anchor,
                        "direction": "improvement",
                        "product": row["product"],
                        "delta_ae": row["delta_ae"],
                        "actual_volume": row["actual_volume"],
                        "wmape_f0": row["wmape_f0"],
                        "wmape_cand": row["wmape_cand"],
                        "n": row["n"],
                    }
                )

            save_csv(by_product, out_dir, f"decomp_by_product_{exp_name}_{anchor}.csv")

            # By origin
            by_origin = m.groupby("test_origin")["delta_ae"].sum().reset_index()
            save_csv(by_origin, out_dir, f"decomp_by_origin_{exp_name}_{anchor}.csv")

            # By horizon bucket
            by_hz = m.groupby("horizon_bucket")["delta_ae"].sum().reset_index()
            save_csv(by_hz, out_dir, f"decomp_by_horizon_{exp_name}_{anchor}.csv")

            # Product x origin
            by_po = (
                m.groupby(["product", "test_origin"])["delta_ae"]
                .sum()
                .reset_index()
                .sort_values("delta_ae", ascending=False)
            )
            save_csv(by_po, out_dir, f"decomp_by_product_origin_{exp_name}_{anchor}.csv")

    summary_df = pd.DataFrame(all_summaries)
    product_df = pd.DataFrame(product_tables)
    save_csv(summary_df, out_dir, "decomposition_summary.csv")
    save_csv(product_df, out_dir, "decomposition_top_products.csv")

    return {"summary": summary_df, "top_products": product_df}
