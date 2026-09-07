"""Section D: diagnostics on V2 itself.

Every rule that flags a forecast is stated explicitly and reported with counts.
Nothing here alters a forecast; suspicious rows are only labelled.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pkg.research.v2_eval.config import DiagnosticThresholds, V2EvalConfig
from pkg.research.v2_eval.load import BackfillInputs
from pkg.research.v2_eval.metrics import (
    relative_improvement_pct,
    wmape_pct,
)


def horizon_degradation(matched: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon MAE, bias and relative gain, for trend inspection."""
    rows = []
    for horizon, group in matched.groupby("horizon"):
        mae_legacy = float(group["absolute_error_legacy"].mean())
        mae_v2 = float(group["absolute_error_v2"].mean())
        rows.append(
            {
                "horizon": int(horizon),
                "n": int(len(group)),
                "mae_legacy": mae_legacy,
                "mae_v2": mae_v2,
                "relative_improvement_pct": relative_improvement_pct(
                    mae_legacy, mae_v2
                ),
                "bias_legacy": float(group["signed_error_legacy"].mean()),
                "bias_v2": float(group["signed_error_v2"].mean()),
                "wmape_legacy": wmape_pct(group["actual"], group["legacy_prediction"]),
                "wmape_v2": wmape_pct(group["actual"], group["v2_prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def origin_consistency(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, group in matched.groupby("forecast_origin"):
        wmape_legacy = wmape_pct(group["actual"], group["legacy_prediction"])
        wmape_v2 = wmape_pct(group["actual"], group["v2_prediction"])
        rows.append(
            {
                "forecast_origin": int(origin),
                "v2_quarter": ",".join(sorted(set(group["quarter"]))),
                "n": int(len(group)),
                "max_horizon_scored": int(group["horizon"].max()),
                "wmape_legacy": wmape_legacy,
                "wmape_v2": wmape_v2,
                "relative_improvement_pct": relative_improvement_pct(
                    wmape_legacy, wmape_v2
                ),
                "v2_better": bool(wmape_v2 < wmape_legacy),
            }
        )
    return pd.DataFrame(rows).sort_values("forecast_origin").reset_index(drop=True)


def product_outcomes(
    product_table: pd.DataFrame,
    thresholds: DiagnosticThresholds,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify products as improved / tied / regressed with an explicit band."""
    frame = product_table.copy()
    eligible = frame["n"] >= thresholds.min_rows_per_product
    rel = frame["relative_improvement_pct"]

    outcome = np.where(
        ~eligible | ~np.isfinite(rel),
        "insufficient_rows",
        np.where(
            rel > thresholds.product_tie_band_pct,
            "improved",
            np.where(rel < -thresholds.product_tie_band_pct, "regressed", "tied"),
        ),
    )
    frame["outcome"] = outcome
    scored = frame.loc[frame["outcome"] != "insufficient_rows"]
    counts = scored["outcome"].value_counts().to_dict()
    total = int(len(scored))
    summary = {
        "rule": (
            f"relative WMAPE change beyond +/-{thresholds.product_tie_band_pct}% "
            f"counts as improved/regressed; products need at least "
            f"{thresholds.min_rows_per_product} evaluated rows"
        ),
        "n_products_scored": total,
        "n_improved": int(counts.get("improved", 0)),
        "n_tied": int(counts.get("tied", 0)),
        "n_regressed": int(counts.get("regressed", 0)),
        "pct_improved": (100.0 * counts.get("improved", 0) / total) if total else float("nan"),
        "pct_tied": (100.0 * counts.get("tied", 0) / total) if total else float("nan"),
        "pct_regressed": (100.0 * counts.get("regressed", 0) / total) if total else float("nan"),
        "median_relative_improvement_pct": float(
            scored["relative_improvement_pct"].median()
        )
        if total
        else float("nan"),
    }
    return frame, summary


def error_concentration(
    product_table: pd.DataFrame,
    thresholds: DiagnosticThresholds,
) -> dict[str, Any]:
    """How concentrated are the portfolio gains and losses across SKUs?"""
    frame = product_table.copy()
    gains = frame.loc[frame["total_absolute_error_reduction"] > 0].sort_values(
        "total_absolute_error_reduction", ascending=False
    )
    losses = frame.loc[frame["total_absolute_error_reduction"] < 0].sort_values(
        "total_absolute_error_reduction"
    )
    total_gain = float(gains["total_absolute_error_reduction"].sum())
    total_loss = float(-losses["total_absolute_error_reduction"].sum())
    net = float(frame["total_absolute_error_reduction"].sum())

    top1_share = (
        float(gains.iloc[0]["total_absolute_error_reduction"] / total_gain)
        if total_gain > 0 and len(gains)
        else 0.0
    )
    top5_share = (
        float(gains.head(5)["total_absolute_error_reduction"].sum() / total_gain)
        if total_gain > 0
        else 0.0
    )
    flags = []
    if top1_share > thresholds.concentration_top1_share:
        flags.append("one_product_over_25pct_of_gains")
    if top5_share > thresholds.concentration_top5_share:
        flags.append("top5_over_50pct_of_gains")

    volume = frame["actual_volume"].sum()
    top5_volume_share = (
        float(frame.nlargest(5, "actual_volume")["actual_volume"].sum() / volume)
        if volume > 0
        else float("nan")
    )
    return {
        "net_absolute_error_reduction": net,
        "total_gain": total_gain,
        "total_loss": total_loss,
        "top1_gain_share": top1_share,
        "top5_gain_share": top5_share,
        "top5_volume_share": top5_volume_share,
        "flags": flags,
        "top_gainers": gains.head(10),
        "top_regressors": losses.head(10),
    }


def forecast_shape_flags(
    v2_scored: pd.DataFrame,
    sales: pd.DataFrame,
    thresholds: DiagnosticThresholds,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag flat, extreme, clipped, negative or non-finite V2 forecasts."""
    frame = v2_scored.copy()

    history_max = {}
    for (product, origin), _group in frame.groupby(["product_id", "forecast_origin"]):
        history = sales.loc[
            (sales["product_id"] == product) & (sales["target_date"] < origin),
            "actual",
        ]
        history_max[(product, origin)] = (
            float(history.max()) if len(history) else float("nan")
        )

    per_job = []
    for (product, origin), group in frame.groupby(["product_id", "forecast_origin"]):
        values = group["forecast"].to_numpy(dtype=float)
        hmax = history_max[(product, origin)]
        distinct = int(pd.unique(np.round(values, 6)).size)
        max_value = float(np.max(values)) if values.size else float("nan")
        ratio = (
            float(max_value / hmax) if np.isfinite(hmax) and hmax > 0 else float("nan")
        )
        per_job.append(
            {
                "product_id": product,
                "forecast_origin": int(origin),
                "quarter": group["quarter"].iloc[0],
                "model": group["model"].iloc[0],
                "distinct_values": distinct,
                "is_flat": bool(distinct <= thresholds.flat_distinct_values),
                "max_forecast": max_value,
                "history_max_pre_origin": hmax,
                "max_over_history_max": ratio,
                "is_extreme": bool(
                    np.isfinite(ratio) and ratio > thresholds.extreme_ratio_vs_history_max
                ),
                "n_raw_negative_clipped": int((group["raw_forecast"] < 0).sum()),
                "n_nonfinite": int((~np.isfinite(values)).sum()),
                "n_negative_delivered": int((values < 0).sum()),
                "all_zero": bool(np.all(values == 0)),
            }
        )
    jobs = pd.DataFrame(per_job)

    summary = {
        "rules": {
            "flat": (
                f"all 15 horizons collapse to <= "
                f"{thresholds.flat_distinct_values} distinct value(s)"
            ),
            "extreme": (
                "max forecast exceeds "
                f"{thresholds.extreme_ratio_vs_history_max}x the SKU's maximum "
                "pre-origin monthly sales"
            ),
            "clipped": "raw_forecast < 0 replaced by 0 under max(raw, 0)",
        },
        "n_jobs": int(len(jobs)),
        "n_flat_jobs": int(jobs["is_flat"].sum()),
        "pct_flat_jobs": float(100.0 * jobs["is_flat"].mean()) if len(jobs) else float("nan"),
        "n_extreme_jobs": int(jobs["is_extreme"].sum()),
        "n_all_zero_jobs": int(jobs["all_zero"].sum()),
        "n_clipped_rows": int(jobs["n_raw_negative_clipped"].sum()),
        "n_nonfinite_rows": int(jobs["n_nonfinite"].sum()),
        "n_negative_delivered_rows": int(jobs["n_negative_delivered"].sum()),
        "flat_by_model": jobs.loc[jobs["is_flat"], "model"].value_counts().to_dict(),
    }
    return jobs, summary


def flat_accuracy_split(
    matched: pd.DataFrame,
    shape_jobs: pd.DataFrame,
) -> pd.DataFrame:
    """Is V2 better or worse on the SKU/origins where it emits a flat line?"""
    flags = shape_jobs[["product_id", "forecast_origin", "is_flat"]]
    joined = matched.merge(flags, on=["product_id", "forecast_origin"], how="left")
    rows = []
    for is_flat, group in joined.groupby(joined["is_flat"].fillna(False)):
        wmape_legacy = wmape_pct(group["actual"], group["legacy_prediction"])
        wmape_v2 = wmape_pct(group["actual"], group["v2_prediction"])
        rows.append(
            {
                "v2_forecast_is_flat": bool(is_flat),
                "n": int(len(group)),
                "wmape_legacy": wmape_legacy,
                "wmape_v2": wmape_v2,
                "relative_improvement_pct": relative_improvement_pct(
                    wmape_legacy, wmape_v2
                ),
            }
        )
    return pd.DataFrame(rows)


def coverage_bias_check(
    matched: pd.DataFrame,
    v2_scored: pd.DataFrame,
    product_table: pd.DataFrame,
) -> pd.DataFrame:
    """Does missing coverage hide hard products?

    Compares each product's evaluated row count against how many V2 rows it
    produced, so a product that is mostly unmatched cannot quietly look good.
    """
    produced = v2_scored.groupby("product_id").size().rename("v2_rows_produced")
    scored = matched.groupby("product_id").size().rename("rows_matched")
    frame = pd.concat([produced, scored], axis=1).fillna(0)
    frame["match_rate"] = frame["rows_matched"] / frame["v2_rows_produced"]
    frame = frame.reset_index().merge(
        product_table[["product_id", "wmape_legacy", "wmape_v2", "actual_volume", "n"]],
        on="product_id",
        how="left",
    )
    return frame.sort_values("match_rate").reset_index(drop=True)


def selection_and_runtime(data: BackfillInputs) -> dict[str, Any]:
    """Model-selection frequencies, runtimes and what the run did not record."""
    jobs = data.jobs
    success = jobs.loc[jobs["status"] == "SUCCESS"]
    model_counts = success["selected_model"].value_counts()

    by_origin = (
        success.groupby(["forecast_origin", "selected_model"])
        .size()
        .rename("n")
        .reset_index()
    )

    # How often does a product's selected model change between consecutive origins?
    switches = 0
    comparisons = 0
    for _product, group in success.sort_values("forecast_origin").groupby("product_id"):
        models = group["selected_model"].tolist()
        for previous, current in zip(models, models[1:]):
            comparisons += 1
            if previous != current:
                switches += 1

    runtimes = pd.to_numeric(success["runtime_seconds"], errors="coerce").dropna()
    strategies = (
        data.results["selected_strategy"].value_counts().to_dict()
        if "selected_strategy" in data.results.columns
        else {}
    )

    return {
        "model_counts": model_counts.to_dict(),
        "model_share_pct": (100.0 * model_counts / len(success)).round(2).to_dict(),
        "by_origin": by_origin,
        "selection_switch_rate_pct": (
            100.0 * switches / comparisons if comparisons else float("nan")
        ),
        "selection_switches": switches,
        "selection_comparisons": comparisons,
        "runtime_total_hours": float(runtimes.sum() / 3600.0),
        "runtime_mean_seconds": float(runtimes.mean()),
        "runtime_median_seconds": float(runtimes.median()),
        "runtime_max_seconds": float(runtimes.max()),
        "runtime_by_model": (
            success.assign(
                runtime_seconds=pd.to_numeric(
                    success["runtime_seconds"], errors="coerce"
                )
            )
            .groupby("selected_model")["runtime_seconds"]
            .agg(["count", "mean", "median", "max"])
            .round(1)
            .reset_index()
        ),
        "selected_strategies": strategies,
        "n_training_observations": (
            pd.to_numeric(
                data.results.get("n_training_observations"), errors="coerce"
            ).describe().to_dict()
            if "n_training_observations" in data.results.columns
            else {}
        ),
        "unavailable_evidence": [
            "per-model CV / candidate forecasts (backtests/ is empty)",
            "ensemble membership and weights (strategy was best_model only)",
            "fallback frequency and reasons (never recorded by the engine)",
            "per-fold training windows used at selection time",
        ],
        "backtest_files": data.backtest_files,
    }


def raw_vs_constrained(v2_scored: pd.DataFrame) -> dict[str, Any]:
    """V2-only diagnostic: effect of the non-negativity clip on scored rows."""
    scored = v2_scored.dropna(subset=["actual"])
    clipped = scored.loc[scored["raw_forecast"] < 0]
    if clipped.empty:
        return {
            "n_clipped_scored_rows": 0,
            "note": "no scored row had a negative raw forecast",
        }
    ae_raw = (clipped["raw_forecast"] - clipped["actual"]).abs().sum()
    ae_constrained = (clipped["forecast"] - clipped["actual"]).abs().sum()
    return {
        "n_clipped_scored_rows": int(len(clipped)),
        "sum_abs_error_raw": float(ae_raw),
        "sum_abs_error_constrained": float(ae_constrained),
        "clip_absolute_error_change": float(ae_constrained - ae_raw),
        "note": (
            "Legacy never saved a pre-postprocess series, so this raw-vs-"
            "constrained delta is a V2-only diagnostic and is not a paired "
            "raw-output comparison."
        ),
    }


def build_diagnostics(
    data: BackfillInputs,
    matched: pd.DataFrame,
    v2_scored: pd.DataFrame,
    product_table: pd.DataFrame,
    config: V2EvalConfig,
) -> dict[str, Any]:
    thresholds = config.thresholds
    shape_jobs, shape_summary = forecast_shape_flags(v2_scored, data.sales, thresholds)
    products, product_summary = product_outcomes(product_table, thresholds)
    return {
        "horizon_degradation": horizon_degradation(matched),
        "origin_consistency": origin_consistency(matched),
        "product_outcomes": products,
        "product_outcome_summary": product_summary,
        "concentration": error_concentration(product_table, thresholds),
        "shape_jobs": shape_jobs,
        "shape_summary": shape_summary,
        "flat_accuracy": flat_accuracy_split(matched, shape_jobs),
        "coverage_bias": coverage_bias_check(matched, v2_scored, product_table),
        "selection_runtime": selection_and_runtime(data),
        "raw_vs_constrained": raw_vs_constrained(v2_scored),
    }
