"""M2 orchestrator — residual learner model-class benchmark."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost

from pkg.benchmark.dataset import load_benchmark
from pkg.research.harness.gates import assert_freeze_unchanged, freeze_checksums
from pkg.research.model_benchmark.config import (
    EXPECTED_MATCHED_N,
    EXPECTED_MATCHED_ORIGINS,
    LIGHTGBM_PARAMS,
    MODELS,
    PRIMARY_ORIGINS_LOCKED,
    PREDICTION_REPEAT_TOL,
    docs_path,
    output_dir,
)
from pkg.research.model_benchmark.diagnostics import (
    build_watchlist,
    compare_vs_xgb,
    compute_verdicts_for_suite,
    fit_feature_importance,
    fit_linear_diagnostics,
    overall_m2_conclusion,
    run_repeatability_gate,
    run_tree_repeatability,
    slice_primary_from_broad,
)
from pkg.research.model_benchmark.evaluate import run_benchmark_suite
from pkg.research.model_benchmark.report import write_m2_report


def _require_dependencies() -> tuple[Any, Any]:
    try:
        catboost = importlib.import_module("catboost")
        lightgbm = importlib.import_module("lightgbm")
    except ImportError as exc:
        raise RuntimeError(
            "M2 requires catboost and lightgbm. Install via: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return catboost, lightgbm


def collect_environment_metadata(catboost_mod: Any, lightgbm_mod: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "catboost_version": catboost_mod.__version__,
        "lightgbm_version": lightgbm_mod.__version__,
        "operating_system": platform.platform(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lightgbm_deterministic_settings": {
            "deterministic": LIGHTGBM_PARAMS.get("deterministic"),
            "force_col_wise": LIGHTGBM_PARAMS.get("force_col_wise"),
            "n_jobs": LIGHTGBM_PARAMS.get("n_jobs"),
        },
        "lightgbm_categorical_approach": "train-fitted pandas Categorical dtype; categorical_feature by name",
        "random_seed": 42,
        "models_evaluated": list(MODELS),
        "no_hyperparameter_search": True,
    }


def _suite_meta(suite, overall_df: pd.DataFrame) -> dict[str, Any]:
    xgb_row = overall_df.loc[overall_df["model"] == "xgboost"].iloc[0]
    origins = suite.origins_used
    return {
        "origins": origins,
        "first_origin": origins[0] if origins else None,
        "last_origin": origins[-1] if origins else None,
        "n": int(xgb_row["n"]),
        "n_products": int(xgb_row["n_products"]),
        "n_origins": int(xgb_row["n_origins"]),
    }


def run_m2() -> dict[str, Any]:
    catboost_mod, lightgbm_mod = _require_dependencies()
    out = output_dir()

    # 1) Verify benchmark freeze
    ds = load_benchmark(verify_checksums=True)
    freeze_before = freeze_checksums(ds)

    # 2) Environment metadata
    env = collect_environment_metadata(catboost_mod, lightgbm_mod)
    print(json.dumps(env, indent=2))
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # 3) F0 reproduction + repeatability gate
    rep_df, rep_summary = run_repeatability_gate(ds, anchor="ts", n_runs=5)
    if rep_summary["max_abs_prediction_diff"] > PREDICTION_REPEAT_TOL:
        raise AssertionError(
            f"F0 repeatability failed: max diff={rep_summary['max_abs_prediction_diff']}"
        )

    ts_matched_gate = run_benchmark_suite(ds, "ts", slice_kind="matched_primary")
    hu_matched_gate = run_benchmark_suite(ds, "human", slice_kind="matched_primary")
    f0_ts_wmape = float(ts_matched_gate.results["xgboost"].overall["wmape"].iloc[0])
    f0_human_wmape = float(hu_matched_gate.results["xgboost"].overall["wmape"].iloc[0])

    # 6–8) M2A TS broad history
    print("Running M2A TS broad-history benchmark...")
    ts_suite = run_benchmark_suite(ds, "ts", slice_kind="broad")
    ts_overall, ts_by_origin, ts_by_horizon, ts_by_product, ts_conc = compare_vs_xgb(
        ts_suite, slice_label="ts_broad"
    )
    ts_primary_suite = slice_primary_from_broad(ts_suite)
    ts_primary_overall, _, _, _, _ = compare_vs_xgb(ts_primary_suite, slice_label="ts_primary")

    # M2B Human
    print("Running M2B Human benchmark...")
    human_suite = run_benchmark_suite(ds, "human", slice_kind="broad")
    human_overall, hu_by_origin, hu_by_horizon, hu_by_product, hu_conc = compare_vs_xgb(
        human_suite, slice_label="human_broad"
    )

    # Matched PRIMARY (both anchors)
    print("Running matched PRIMARY comparison...")
    ts_matched = run_benchmark_suite(ds, "ts", slice_kind="matched_primary")
    hu_matched = run_benchmark_suite(ds, "human", slice_kind="matched_primary")
    ts_m_overall, _, _, _, _ = compare_vs_xgb(ts_matched, slice_label="matched_primary")
    hu_m_overall, _, _, _, _ = compare_vs_xgb(hu_matched, slice_label="matched_primary")
    matched_primary = pd.concat([ts_m_overall, hu_m_overall], ignore_index=True)

    if int(ts_matched.results["xgboost"].overall["n"].iloc[0]) != EXPECTED_MATCHED_N:
        raise AssertionError(
            f"Matched PRIMARY n={ts_matched.results['xgboost'].overall['n'].iloc[0]} "
            f"!= expected {EXPECTED_MATCHED_N}"
        )
    if len(ts_matched.origins_used) != EXPECTED_MATCHED_ORIGINS:
        raise AssertionError(
            f"Matched PRIMARY origins={ts_matched.origins_used} "
            f"!= expected {EXPECTED_MATCHED_ORIGINS}"
        )

    # Diagnostics
    tree_rep = run_tree_repeatability(ds)
    repeat_out = pd.concat(
        [
            rep_df.assign(check="f0_5x"),
            tree_rep.assign(check="tree_2x"),
        ],
        ignore_index=True,
    )
    repeat_out.to_csv(out / "repeatability.csv", index=False)

    linear_rows = fit_linear_diagnostics(ds, "ts") + fit_linear_diagnostics(ds, "human")
    pd.DataFrame(linear_rows).to_csv(out / "linear_diagnostics.csv", index=False)

    fi_rows = fit_feature_importance(ds, "ts") + fit_feature_importance(ds, "human")
    pd.DataFrame(fi_rows).to_csv(out / "feature_importance.csv", index=False)

    # Verdicts
    verdicts: dict[str, dict[str, str]] = {}
    verdicts["ts"] = compute_verdicts_for_suite(ts_suite, slice_label="ts_broad", error_conc=ts_conc)
    verdicts["human"] = compute_verdicts_for_suite(
        human_suite, slice_label="human_broad", error_conc=hu_conc
    )
    conclusion = overall_m2_conclusion(verdicts)

    # Combine tables
    by_origin = pd.concat([ts_by_origin, hu_by_origin], ignore_index=True)
    by_horizon = pd.concat([ts_by_horizon, hu_by_horizon], ignore_index=True)
    by_product = pd.concat([ts_by_product, hu_by_product], ignore_index=True)
    error_conc = pd.concat([ts_conc, hu_conc], ignore_index=True)
    watchlist = pd.concat(
        [
            build_watchlist(ts_by_product, anchor="ts", slice_label="ts_broad"),
            build_watchlist(hu_by_product, anchor="human", slice_label="human_broad"),
        ],
        ignore_index=True,
    )

    model_comparison = pd.concat(
        [
            ts_overall.assign(table="A_ts_broad"),
            ts_primary_overall.assign(table="B_ts_primary"),
            human_overall.assign(table="C_human_broad"),
            matched_primary.assign(table="D_matched_primary"),
        ],
        ignore_index=True,
    )

    # Write outputs
    ts_suite.pooled_predictions.to_parquet(out / "ts_predictions.parquet", index=False)
    human_suite.pooled_predictions.to_parquet(out / "human_predictions.parquet", index=False)
    ts_overall.to_csv(out / "ts_overall.csv", index=False)
    human_overall.to_csv(out / "human_overall.csv", index=False)
    ts_primary_overall.to_csv(out / "ts_primary.csv", index=False)
    matched_primary.to_csv(out / "matched_primary.csv", index=False)
    by_origin.to_csv(out / "by_origin.csv", index=False)
    by_horizon.to_csv(out / "by_horizon.csv", index=False)
    by_product.to_csv(out / "by_product.csv", index=False)
    error_conc.to_csv(out / "error_concentration.csv", index=False)
    watchlist.to_csv(out / "high_volume_watchlist.csv", index=False)
    model_comparison.to_csv(out / "model_comparison.csv", index=False)

    ts_meta = _suite_meta(ts_suite, ts_overall)
    human_meta = _suite_meta(human_suite, human_overall)

    report_path = write_m2_report(
        out_dir=out,
        env=env,
        repeat_summary=rep_summary,
        f0_ts_primary_wmape=f0_ts_wmape,
        f0_human_primary_wmape=f0_human_wmape,
        ts_suite_meta=ts_meta,
        human_suite_meta=human_meta,
        ts_overall=ts_overall,
        human_overall=human_overall,
        ts_primary=ts_primary_overall,
        matched_primary=matched_primary,
        model_comparison=model_comparison,
        verdicts=verdicts,
        overall_conclusion=conclusion,
        tree_repeat=tree_rep,
    )

    assert_freeze_unchanged(ds, freeze_before)

    summary = {
        "output_dir": str(out),
        "report": str(report_path),
        "f0_ts_primary_wmape": f0_ts_wmape,
        "f0_human_primary_wmape": f0_human_wmape,
        "ts_origins": ts_meta["origins"],
        "human_origins": human_meta["origins"],
        "conclusion": conclusion,
        "verdicts": verdicts,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    print("=== M2 residual learner benchmark ===")
    try:
        run_m2()
    except Exception as exc:  # noqa: BLE001
        print(f"M2 FAILED: {exc}", file=sys.stderr)
        raise
    print("=== M2 COMPLETE ===")


if __name__ == "__main__":
    main()
