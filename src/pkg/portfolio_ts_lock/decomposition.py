"""Portfolio-impact decomposition on residual_development only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.bootstrap import drop_top_k_hybrid_gainers
from pkg.portfolio_ts_lock.completeness import expected_targets
from pkg.portfolio_ts_lock.constants import (
    FALLBACK_NAIVE,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import last_train_value
from pkg.portfolio_ts_lock.freeze import score_frame


def residual_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        df["forecast_origin"].astype(int).isin(RESIDUAL_DEVELOPMENT_ORIGINS)
    ].copy()


def resolve_naive_deployable_partials(
    *,
    partial: pd.DataFrame,
    dense: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn partial POs into naive 15h deployable rows when train history exists.

    Returns (naive_rows_15h, no_forecast_available_po).
    Never intended for common-native.
    """
    if partial is None or partial.empty:
        empty = pd.DataFrame()
        return empty, empty

    naive_rows: list[dict[str, Any]] = []
    no_fc: list[dict[str, Any]] = []
    dense_map = {
        (str(r.product), int(r.date)): float(r.actual)
        for r in dense.itertuples(index=False)
    }

    for r in partial.itertuples(index=False):
        product = str(r.product)
        origin = int(r.forecast_origin)
        if origin not in RESIDUAL_DEVELOPMENT_ORIGINS:
            # still resolve, but decomposition filters residual later
            pass
        naive = last_train_value(dense, product, origin)
        cutoff = shamsi_add_months(origin, -2)
        has_hist = dense.loc[
            (dense["product"].astype(str) == product)
            & (pd.to_numeric(dense["date"], errors="coerce") <= cutoff)
        ]
        legitimate = bool(len(has_hist) > 0 and np.isfinite(naive))
        if not legitimate:
            no_fc.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "cohort": getattr(r, "cohort", ""),
                    "origin_role": getattr(r, "origin_role", ""),
                    "coverage_status": "NO_FORECAST_AVAILABLE",
                    "reason": "no_sales_through_origin_minus_2_for_naive_last_value",
                }
            )
            continue

        exp = expected_targets(origin)
        missing_td = [
            int(td)
            for td in exp.values()
            if (product, int(td)) not in dense_map
            or not np.isfinite(dense_map[(product, int(td))])
        ]
        if missing_td:
            no_fc.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "cohort": getattr(r, "cohort", ""),
                    "origin_role": getattr(r, "origin_role", ""),
                    "coverage_status": "NO_FORECAST_AVAILABLE",
                    "reason": "missing_dense_actuals:"
                    + ",".join(str(x) for x in missing_td),
                }
            )
            continue

        for h, td in exp.items():
            actual = float(dense_map[(product, int(td))])
            naive_rows.append(
                {
                    "product": product,
                    "forecast_origin": origin,
                    "horizon": int(h),
                    "target_date": int(td),
                    "actual": actual,
                    "cohort": getattr(r, "cohort", ""),
                    "origin_role": getattr(r, "origin_role", ""),
                    "v21_ok": False,
                    "a0_ok": False,
                    "f_v21": float("nan"),
                    "f_a0": float("nan"),
                    "f_naive": float(naive),
                    "f_v2_policy": float(naive),
                    "f_a0_only": float("nan"),
                    "f_a0_hybrid": float(naive),
                    "fallback_category": FALLBACK_NAIVE,
                    "joint_availability": False,
                    "resolved_from_partial": True,
                }
            )

    return pd.DataFrame(naive_rows), pd.DataFrame(no_fc)


def cohort_volume_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """By cohort: volume share, summed AE, AE reduction, contribution to hybrid gain."""
    if df.empty:
        return pd.DataFrame()
    total_abs_actual = float(np.nansum(np.abs(df["actual"].to_numpy(dtype=float))))
    rows = []
    # portfolio-level AE sums for contribution
    tot_ae_v2 = float(np.nansum(np.abs(df["f_v2_policy"] - df["actual"])))
    tot_ae_hy = float(np.nansum(np.abs(df["f_a0_hybrid"] - df["actual"])))
    tot_reduction = tot_ae_v2 - tot_ae_hy  # positive = hybrid better

    for cohort, g in df.groupby("cohort", sort=False):
        abs_a = float(np.nansum(np.abs(g["actual"].to_numpy(dtype=float))))
        ae_v2 = float(np.nansum(np.abs(g["f_v2_policy"] - g["actual"])))
        ae_hy = float(np.nansum(np.abs(g["f_a0_hybrid"] - g["actual"])))
        reduction = ae_v2 - ae_hy
        rows.append(
            {
                "cohort": cohort,
                "n_rows": int(len(g)),
                "n_product_origins": int(g.groupby(["product", "forecast_origin"]).ngroups),
                "actual_volume_abs": abs_a,
                "share_of_portfolio_actual_volume": (
                    abs_a / total_abs_actual if total_abs_actual > 0 else float("nan")
                ),
                "sum_ae_v2": ae_v2,
                "sum_ae_hybrid": ae_hy,
                "ae_reduction_v2_minus_hybrid": reduction,
                "contribution_to_total_hybrid_gain": (
                    reduction / tot_reduction if tot_reduction != 0 else float("nan")
                ),
                "wmape_v2": score_frame(g["actual"], g["f_v2_policy"])["portfolio_wmape_pct"],
                "wmape_hybrid": score_frame(g["actual"], g["f_a0_hybrid"])[
                    "portfolio_wmape_pct"
                ],
            }
        )
    return pd.DataFrame(rows)


def original55_deep_dive(df55: pd.DataFrame) -> dict[str, Any]:
    """By-origin lift, drop-top-3, top-10 regressors, concentration flag."""
    if df55.empty:
        return {"empty": True}

    by_origin = []
    for origin, g in df55.groupby(df55["forecast_origin"].astype(int), sort=True):
        s_v2 = score_frame(g["actual"], g["f_v2_policy"])
        s_hy = score_frame(g["actual"], g["f_a0_hybrid"])
        by_origin.append(
            {
                "forecast_origin": int(origin),
                "n_rows": int(len(g)),
                "wmape_v2": s_v2["portfolio_wmape_pct"],
                "wmape_hybrid": s_hy["portfolio_wmape_pct"],
                "lift_wmape_pp": float(
                    s_v2["portfolio_wmape_pct"] - s_hy["portfolio_wmape_pct"]
                ),
            }
        )
    by_origin_df = pd.DataFrame(by_origin)

    drop3 = drop_top_k_hybrid_gainers(df55, k=3)

    # Top 10 products by hybrid regression (most negative AE reduction)
    prod_rows = []
    for product, g in df55.groupby("product", sort=False):
        ae_v2 = float(np.nansum(np.abs(g["f_v2_policy"] - g["actual"])))
        ae_hy = float(np.nansum(np.abs(g["f_a0_hybrid"] - g["actual"])))
        reduction = ae_v2 - ae_hy
        prod_rows.append(
            {
                "product": str(product),
                "sum_ae_v2": ae_v2,
                "sum_ae_hybrid": ae_hy,
                "ae_reduction": reduction,
                "wmape_v2": score_frame(g["actual"], g["f_v2_policy"])["portfolio_wmape_pct"],
                "wmape_hybrid": score_frame(g["actual"], g["f_a0_hybrid"])[
                    "portfolio_wmape_pct"
                ],
            }
        )
    prod_df = pd.DataFrame(prod_rows).sort_values("ae_reduction")
    top10_regress = prod_df.head(10).copy()

    # Concentration: share of total AE loss (negative reduction) from top 3 regressors
    losses = prod_df.loc[prod_df["ae_reduction"] < 0, "ae_reduction"]
    total_loss = float(losses.sum()) if not losses.empty else 0.0
    top3_loss = float(prod_df.head(3)["ae_reduction"].sum())
    if total_loss < 0:
        top3_share_of_loss = float(top3_loss / total_loss)
    else:
        top3_share_of_loss = float("nan")
    n_regressor = int((prod_df["ae_reduction"] < 0).sum())
    n_products = int(len(prod_df))
    concentrated = bool(
        (n_regressor > 0 and top3_share_of_loss >= 0.5)
        or (n_regressor > 0 and n_regressor <= max(3, int(0.15 * n_products)))
    )

    return {
        "by_origin": by_origin_df,
        "drop_top3": drop3,
        "top10_regressing_products": top10_regress,
        "n_products": n_products,
        "n_regressing_products": n_regressor,
        "top3_share_of_negative_ae_reduction": top3_share_of_loss,
        "loss_pattern": "concentrated" if concentrated else "broad",
    }


def build_impact_report(
    *,
    deployable: pd.DataFrame,
    common: pd.DataFrame,
) -> dict[str, Any]:
    dep_r = residual_only(deployable)
    com_r = residual_only(common)

    pooled = {
        "deployable_residual": {
            "v2": score_frame(dep_r["actual"], dep_r["f_v2_policy"]),
            "hybrid": score_frame(dep_r["actual"], dep_r["f_a0_hybrid"]),
            "lift_wmape_pp": float(
                score_frame(dep_r["actual"], dep_r["f_v2_policy"])["portfolio_wmape_pct"]
                - score_frame(dep_r["actual"], dep_r["f_a0_hybrid"])["portfolio_wmape_pct"]
            ),
            "n_rows": int(len(dep_r)),
            "n_product_origins": int(
                dep_r.groupby(["product", "forecast_origin"]).ngroups
            ),
        },
        "common_native_residual": {
            "v2": score_frame(com_r["actual"], com_r["f_v2_policy"]),
            "hybrid": score_frame(com_r["actual"], com_r["f_a0_hybrid"]),
            "lift_wmape_pp": float(
                score_frame(com_r["actual"], com_r["f_v2_policy"])["portfolio_wmape_pct"]
                - score_frame(com_r["actual"], com_r["f_a0_hybrid"])["portfolio_wmape_pct"]
            ),
            "n_rows": int(len(com_r)),
            "n_product_origins": int(
                com_r.groupby(["product", "forecast_origin"]).ngroups
            ),
        },
    }

    cohort_dep = cohort_volume_decomposition(dep_r)
    cohort_com = cohort_volume_decomposition(com_r)
    df55 = dep_r.loc[dep_r["cohort"] == "original_55"].copy()
    dive = original55_deep_dive(df55)

    return {
        "pooled": pooled,
        "cohort_deployable": cohort_dep,
        "cohort_common_native": cohort_com,
        "original55_dive": dive,
    }
