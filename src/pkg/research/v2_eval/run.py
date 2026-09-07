"""Orchestration for the V2-vs-legacy backfill evaluation.

Reads the completed experiment, builds the matched panel, computes metrics and
diagnostics, and writes every artifact under ``config.out_dir``.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.v2_eval import charts as charts_module
from pkg.research.v2_eval.audit import build_coverage_table, build_methodology_audit
from pkg.research.v2_eval.config import (
    HORIZON_GROUPS,
    MATCH_KEYS,
    SHIFTED_ORIGINS,
    V2EvalConfig,
    default_config,
)
from pkg.research.v2_eval.diagnostics import build_diagnostics
from pkg.research.v2_eval.load import load_all, sha256_frame
from pkg.research.v2_eval.match import (
    build_matched_panel,
    fully_actualised_pairs,
    validate_matched_panel,
)
from pkg.research.v2_eval.metrics import (
    category_metrics,
    horizon_metrics,
    origin_metrics,
    overall_metrics,
    product_metrics,
    relative_improvement_pct,
)
from pkg.research.v2_eval.report import write_docs_summary, write_report

REPRODUCE_COMMAND = "python -m pkg.research.evaluate_v2_backfill"

METRIC_DEFINITIONS = {
    "match_key": " + ".join(MATCH_KEYS),
    "signed_error": "prediction - actual (positive = overforecast)",
    "absolute_error_reduction": "absolute_error_legacy - absolute_error_v2",
    "mean_horizon_mae": (
        "equally weighted mean of horizon-level MAEs, as in "
        "pkg.ts_v2.metrics.selection_mae_from_horizons"
    ),
    "wmape": (
        "100 * sum|actual - prediction| / sum|actual| over the slice "
        "(pkg.benchmark.evaluate.wmape); not the mean of per-SKU WMAPEs; "
        "NaN when sum|actual| == 0"
    ),
    "wmape_scale_note": (
        "pkg.benchmark.evaluate.wmape returns percent while "
        "pkg.ts_v2.metrics.horizon_wmape returns the same quantity as a ratio; "
        "this analysis always reports percent"
    ),
    "rmse": "sqrt(mean((prediction - actual)^2))",
    "signed_bias": "mean(prediction - actual); compared by distance from zero",
    "relative_improvement_pct": (
        "100 * (legacy_error - v2_error) / legacy_error; NaN when the legacy "
        "value is zero or non-finite; never applied to signed bias"
    ),
    "horizon_groups": {name: [low, high] for name, low, high in HORIZON_GROUPS},
    "negative_actuals": (
        "kept as-is; they enter the WMAPE denominator through sum|actual|"
    ),
    "missing_values": (
        "rows without both forecasts and an actual are excluded and reported as "
        "coverage; never imputed with zero"
    ),
}


def _sensitivities(matched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    excluded = matched.loc[~matched["forecast_origin"].isin(SHIFTED_ORIGINS)]
    full15 = fully_actualised_pairs(matched)
    label_shift = (
        "exclude_shifted_origins_"
        + "_".join(str(o) for o in SHIFTED_ORIGINS)
    )
    return {
        label_shift: excluded,
        "sku_origin_pairs_with_all_15_actual_horizons": full15,
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def evaluate_v2_backfill(
    *,
    config: Optional[V2EvalConfig] = None,
    make_charts: bool = True,
) -> dict[str, Any]:
    cfg = config or default_config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(cfg)
    panel = build_matched_panel(data)
    problems = validate_matched_panel(panel)
    if problems:
        raise AssertionError("matched panel validation failed: " + "; ".join(problems))

    matched = panel.matched
    sensitivities = _sensitivities(matched)

    overall = overall_metrics(matched, sensitivities)
    horizons = horizon_metrics(matched)
    origins = origin_metrics(matched)
    products = product_metrics(matched)
    categories = category_metrics(matched)

    audit = build_methodology_audit(data)
    coverage = build_coverage_table(data, matched, panel.v2_scored)

    diagnostics = build_diagnostics(data, matched, panel.v2_scored, products, cfg)

    # Products table gains its outcome classification from the diagnostics pass.
    products = diagnostics["product_outcomes"]

    chart_paths: dict[str, Any] = {}
    if make_charts:
        chart_paths = charts_module.build_charts(
            cfg,
            matched,
            data.sales,
            diagnostics["horizon_degradation"],
            diagnostics["origin_consistency"],
            products,
        )

    matched.to_parquet(cfg.out_dir / "matched_predictions.parquet", index=False)
    _write_csv(overall, cfg.out_dir / "overall_metrics.csv")
    _write_csv(horizons, cfg.out_dir / "horizon_metrics.csv")
    _write_csv(products, cfg.out_dir / "product_metrics.csv")
    _write_csv(origins, cfg.out_dir / "origin_metrics.csv")
    _write_csv(coverage, cfg.out_dir / "coverage_and_failures.csv")
    _write_csv(audit, cfg.out_dir / "methodology_audit.csv")
    if not categories.empty:
        _write_csv(categories, cfg.out_dir / "category_metrics.csv")
    _write_csv(
        panel.unmatched_v2[
            [*MATCH_KEYS, "quarter", "forecast", "actual", "reason"]
        ],
        cfg.out_dir / "unmatched_v2_rows.csv",
    )
    _write_csv(
        panel.unmatched_legacy[
            [*MATCH_KEYS, "legacy_quarter", "legacy_prediction", "reason"]
        ],
        cfg.out_dir / "unmatched_legacy_rows.csv",
    )
    _write_csv(
        diagnostics["shape_jobs"], cfg.out_dir / "forecast_shape_flags.csv"
    )
    _write_csv(
        diagnostics["selection_runtime"]["by_origin"],
        cfg.out_dir / "model_selection_by_origin.csv",
    )
    _write_csv(diagnostics["coverage_bias"], cfg.out_dir / "product_coverage.csv")

    verdict = build_verdict(overall, diagnostics, panel.coverage_summary)

    report = {
        "config": cfg,
        "data": data,
        "panel": panel,
        "overall": overall,
        "horizons": horizons,
        "origins": origins,
        "products": products,
        "categories": categories,
        "audit": audit,
        "coverage": coverage,
        "diagnostics": diagnostics,
        "sensitivities": {k: len(v) for k, v in sensitivities.items()},
        "charts": chart_paths,
        "verdict": verdict,
        "matched_hash": sha256_frame(matched),
    }

    manifest = build_manifest(cfg, data, panel, report)
    with open(cfg.out_dir / "analysis_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
    report["manifest"] = manifest

    write_report(report)
    write_docs_summary(report)
    return report


def build_verdict(
    overall: pd.DataFrame,
    diagnostics: dict[str, Any],
    coverage_summary: dict[str, int],
) -> dict[str, Any]:
    """Classify the outcome using the primary slice plus stability evidence."""
    primary = overall.loc[overall["scope"] == "overall"].set_index("metric")
    wmape_rel = float(primary.loc["wmape", "relative_improvement_pct"])
    mhm_rel = float(primary.loc["mean_horizon_mae", "relative_improvement_pct"])
    rmse_rel = float(primary.loc["rmse", "relative_improvement_pct"])

    origin_table = diagnostics["origin_consistency"]
    origins_better = int(origin_table["v2_better"].sum())
    origins_total = int(len(origin_table))
    product_summary = diagnostics["product_outcome_summary"]

    sensitivity = overall.loc[
        (overall["scope"] == "sensitivity") & (overall["metric"] == "wmape")
    ]
    sensitivity_rel = sensitivity["relative_improvement_pct"].tolist()

    headline_positive = wmape_rel > 0 and mhm_rel > 0
    sensitivities_agree = all(v > 0 for v in sensitivity_rel if pd.notna(v))
    # "Consistently" is reserved for a result that holds nearly everywhere, not
    # merely on the aggregate: at most one losing origin in ten and at most one
    # regressed product in ten.
    stable_origins = bool(origins_total) and origins_better / origins_total >= 0.90
    stable_products = product_summary["pct_regressed"] <= 10.0
    majority_products = product_summary["pct_improved"] > product_summary["pct_regressed"]

    if (
        headline_positive
        and stable_origins
        and stable_products
        and sensitivities_agree
    ):
        label = "consistently_better"
    elif headline_positive and sensitivities_agree and majority_products:
        label = "better_overall_but_uneven_by_segment"
    elif headline_positive:
        label = "better_on_the_primary_slice_only"
    elif abs(wmape_rel) < 1.0:
        label = "broadly_similar"
    else:
        label = "worse"

    losing_origins = (
        origin_table.loc[~origin_table["v2_better"], "forecast_origin"]
        .astype(int)
        .tolist()
    )

    return {
        "label": label,
        "wmape_relative_improvement_pct": wmape_rel,
        "mean_horizon_mae_relative_improvement_pct": mhm_rel,
        "rmse_relative_improvement_pct": rmse_rel,
        "origins_better": origins_better,
        "origins_total": origins_total,
        "losing_origins": losing_origins,
        "pct_products_improved": product_summary["pct_improved"],
        "pct_products_regressed": product_summary["pct_regressed"],
        "sensitivity_wmape_relative_improvement_pct": sensitivity_rel,
        "matched_rows": coverage_summary["matched_rows"],
    }


def build_manifest(
    cfg: V2EvalConfig,
    data,
    panel,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": "ts_v2_backfill_eval",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reproduce_command": REPRODUCE_COMMAND,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "inputs": {
            "experiment_dir": str(cfg.experiment_dir),
            "legacy_panel": str(cfg.legacy_panel),
            "raw_sales": str(cfg.raw_sales),
            "product_attrs": str(cfg.product_attrs),
            "vintage_manifest": str(cfg.vintage_manifest),
            "universe_manifest": str(cfg.universe_manifest),
        },
        "input_sha256": data.input_hashes,
        "experiment": {
            "experiment_id": data.manifest.get("experiment_id"),
            "engine_version": data.manifest.get("engine_version"),
            "config_hash": data.manifest.get("config_hash"),
            "forecast_horizon": data.manifest.get("forecast_horizon"),
            "training_cutoff_rule": data.manifest.get("training_cutoff_rule"),
            "run_meta": data.run_meta,
            "run_meta_note": (
                "run_meta.json records only the final resume pass, not the whole "
                "experiment; SQLite holds the cumulative job outcome"
            ),
        },
        "legacy_source": {
            "description": (
                "Delivered quarterly TS CSV vintages produced by the original "
                "pkg.forecast.SalesForecast pipeline, frozen into benchmark v1"
            ),
            "not": "V3A A0 or any re-run of the legacy engine",
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "filters": {
            "match_keys": list(MATCH_KEYS),
            "origin_alignment": (
                "matched on forecast_origin so both engines share the same "
                "information cutoff; legacy quarter labels may differ where the "
                "legacy CSV origin drifted"
            ),
            "shifted_origins": list(SHIFTED_ORIGINS),
            "row_inclusion": (
                "both predictions present and an actual present; no zero-filling"
            ),
            "sensitivities": report["sensitivities"],
        },
        "coverage": panel.coverage_summary,
        "outputs": {
            "report": "report.md",
            "matched_predictions": "matched_predictions.parquet",
            "matched_predictions_sha256": report["matched_hash"],
            "tables": [
                "overall_metrics.csv",
                "horizon_metrics.csv",
                "product_metrics.csv",
                "origin_metrics.csv",
                "coverage_and_failures.csv",
                "methodology_audit.csv",
                "category_metrics.csv",
                "unmatched_v2_rows.csv",
                "unmatched_legacy_rows.csv",
                "forecast_shape_flags.csv",
                "model_selection_by_origin.csv",
                "product_coverage.csv",
            ],
            "charts_dir": "charts/",
        },
        "verdict": report["verdict"],
        "limitations": [
            "backtests/ is empty, so candidate forecasts, per-model CV scores, "
            "ensemble weights and fallback reasons are unavailable",
            "legacy pre-postprocess forecasts were never saved, so the raw-output "
            "comparison is V2-only",
            "legacy effectively trains to origin-2 because of its last-month drop",
            "actuals end at 140504, so recent origins are only partly matured",
            "origins and horizons overlap, so rows are not independent and no "
            "significance testing is performed",
        ],
    }
