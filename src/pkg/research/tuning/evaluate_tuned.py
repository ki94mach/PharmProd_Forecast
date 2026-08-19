"""PRIMARY evaluation: baseline F0 gate + tuned model comparison.

Step order:
  1. Reproduce canonical F0 on PRIMARY; STOP if drift > 0.10 abs WMAPE.
  2. Save baseline predictions.
  3. Build tuned model callable from frozen params.
  4. Run backtest for each anchor; assert identical keys.
  5. Score all metrics; write CSVs and parquets.
  6. Compute verdict (PROMOTE / WEAK / REJECT) per anchor.

PRIMARY data never enters inner tuning or early stopping.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark import backtest, load_benchmark
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.harness.gates import (
    assert_freeze_unchanged,
    confirm_canonical_f0,
    freeze_checksums,
)
from pkg.research.harness.metrics import (
    assert_same_eval_rows,
    error_concentration,
    horizon_bucket_table,
    merge_ae,
    origin_pair_table,
    origin_summary,
    product_pair_table,
    product_summary,
    rel_wmape,
)
from pkg.research.tuning.config import (
    BASELINE_HUMAN_WMAPE_REF,
    BASELINE_STOP_TOL,
    BASELINE_TS_WMAPE_REF,
    EXPECTED_N,
    EXPECTED_N_ORIGINS,
    EXPECTED_PRIMARY_ORIGINS,
    F0_FEATURES,
    TRAIN_UNIVERSE,
    m1_output_dir,
)
from pkg.research.tuning.fit import make_primary_model

log = logging.getLogger(__name__)


# ── Baseline gate ─────────────────────────────────────────────────────────────

_SANITY_REF = {"ts": BASELINE_TS_WMAPE_REF, "human": BASELINE_HUMAN_WMAPE_REF}


def run_baseline_gate(ds) -> dict[str, BacktestResult]:
    """Reproduce canonical F0 for both anchors; STOP on drift."""
    canon = confirm_canonical_f0(ds)
    f0_results: dict[str, BacktestResult] = canon["results"]

    for anchor, ref in _SANITY_REF.items():
        got = float(f0_results[anchor].overall["wmape"].iloc[0])
        gap = abs(got - ref)
        if gap > BASELINE_STOP_TOL:
            raise EnvironmentError(
                f"F0 {anchor} WMAPE = {got:.4f} differs from reference {ref} "
                f"by {gap:.4f} (> {BASELINE_STOP_TOL}). "
                "Possible environment / reproducibility problem. STOP."
            )
        print(
            f"  [gate] F0 {anchor} WMAPE = {got:.4f} "
            f"(ref {ref}, gap {gap:.4f}) — OK"
        )

    return f0_results


def save_baseline_predictions(
    f0_results: dict[str, BacktestResult], out_dir: Path
) -> pd.DataFrame:
    """Concatenate both anchors' predictions and write baseline_predictions.parquet."""
    parts = []
    for anchor, res in f0_results.items():
        p = res.predictions.copy()
        p["anchor"] = anchor
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_dir / "baseline_predictions.parquet", index=False)
    return df


# ── PRIMARY key assertion ─────────────────────────────────────────────────────

def assert_primary_keys(
    baseline: BacktestResult,
    tuned: BacktestResult,
    anchor: str,
) -> None:
    """Assert that tuned predictions have exactly the same PRIMARY identities as baseline."""
    assert_same_eval_rows(baseline, tuned)
    n_baseline = len(baseline.predictions)
    n_tuned = len(tuned.predictions)
    if n_baseline != n_tuned:
        raise AssertionError(
            f"[{anchor}] row count mismatch: baseline n={n_baseline}, tuned n={n_tuned}"
        )
    if n_baseline != EXPECTED_N:
        raise AssertionError(
            f"[{anchor}] expected n={EXPECTED_N}, got n={n_baseline}"
        )
    n_origins = len(baseline.origins)
    if n_origins != EXPECTED_N_ORIGINS:
        raise AssertionError(
            f"[{anchor}] expected {EXPECTED_N_ORIGINS} origins, got {n_origins}"
        )


# ── Verdict ───────────────────────────────────────────────────────────────────

def classify_m1_verdict(
    *,
    wmape_baseline: float,
    wmape_tuned: float,
    product_win_rate: float,
    median_product_improvement_pct: float,
    origins_improved: int,
    origins_total: int,
    bias_baseline: float,
    bias_tuned: float,
    concentration_flags: list[str],
) -> str:
    """PROMOTE / WEAK_NEEDS_CONFIRMATION / REJECT.

    Rules are pre-specified before observing PRIMARY; identical logic to F2.
    """
    better = wmape_tuned < wmape_baseline
    median_ok = median_product_improvement_pct > 0
    win_ok = product_win_rate > 0.50
    origins_ok = origins_total > 0 and origins_improved > origins_total / 2
    bias_ok = (
        abs(bias_tuned) <= abs(bias_baseline) * 1.25
        or abs(bias_tuned) <= abs(bias_baseline) + 200.0
    )
    concentrated = bool(concentration_flags)

    if better and median_ok and win_ok and origins_ok and bias_ok and not concentrated:
        return "PROMOTE"
    if better and (not median_ok or not win_ok or not origins_ok or concentrated):
        return "WEAK_NEEDS_CONFIRMATION"
    if (not better) and (median_ok or win_ok):
        return "WEAK_NEEDS_CONFIRMATION"
    return "REJECT"


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_tuned_models(
    optuna_results: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """PRIMARY evaluation for all anchors with completed Optuna studies.

    Parameters
    ----------
    optuna_results : dict returned by run_all_optuna_studies()
    out_dir        : output directory (defaults to m1_output_dir())

    Returns
    -------
    dict with overall/by_origin/by_horizon/by_product DataFrames, verdicts, etc.
    """
    out_dir = out_dir or m1_output_dir()

    print("\n=== M1 PRIMARY Evaluation ===")
    ds = load_benchmark()
    freeze_before = freeze_checksums(ds)

    # Gate
    print("Running F0 baseline reproduction gate …")
    f0_results = run_baseline_gate(ds)
    save_baseline_predictions(f0_results, out_dir)

    overall_rows = []
    origin_rows = []
    horizon_rows = []
    product_rows = []
    conc_rows = []
    watch_rows = []
    all_tuned_preds = []
    verdicts: dict[str, str] = {}

    for anchor in ("ts", "human"):
        res = optuna_results.get(anchor, {})
        if "error" in res:
            print(f"  [{anchor}] Skipping PRIMARY eval — Optuna failed: {res['error']}")
            continue

        frozen_params = res["best_params"].copy()
        frozen_n_estimators = int(frozen_params.pop("frozen_n_estimators"))
        # strip metadata keys that are not XGB params
        xgb_param_keys = {
            "max_depth", "min_child_weight", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
            "objective", "random_state", "n_jobs",
        }
        xgb_params = {k: v for k, v in frozen_params.items() if k in xgb_param_keys}

        print(
            f"  [{anchor}] Building tuned model callable "
            f"(frozen_n_estimators={frozen_n_estimators}) …"
        )
        model_fn = make_primary_model(
            anchor, F0_FEATURES[anchor], xgb_params, frozen_n_estimators
        )

        print(f"  [{anchor}] Running PRIMARY backtest …")
        tuned_result = backtest(
            model_fn,
            dataset=ds,
            universe="matched",
            eligibility="primary",
            train_universe=TRAIN_UNIVERSE[anchor],
        )

        baseline_result = f0_results[anchor]

        # Key assertion (STOP on mismatch)
        assert_primary_keys(baseline_result, tuned_result, anchor)

        # Save tuned predictions
        tp = tuned_result.predictions.copy()
        tp["anchor"] = anchor
        all_tuned_preds.append(tp)

        # Metrics
        o_base = baseline_result.overall.iloc[0]
        o_tuned = tuned_result.overall.iloc[0]
        wmape_base = float(o_base["wmape"])
        wmape_tuned = float(o_tuned["wmape"])
        bias_base = float(o_base["bias"])
        bias_tuned = float(o_tuned["bias"])

        odf = origin_pair_table(baseline_result, tuned_result)
        osu = origin_summary(odf)
        pdf = product_pair_table(baseline_result, tuned_result)
        psu = product_summary(pdf)
        m = merge_ae(baseline_result, tuned_result)
        conc = error_concentration(m, f"M1_{anchor.upper()}", anchor)
        hb = horizon_bucket_table(baseline_result, tuned_result)

        verdict = classify_m1_verdict(
            wmape_baseline=wmape_base,
            wmape_tuned=wmape_tuned,
            product_win_rate=psu.get("product_win_rate", 0.0),
            median_product_improvement_pct=psu.get("median_product_improvement_pct", 0.0),
            origins_improved=osu.get("origins_improved", 0),
            origins_total=osu.get("origins_total", 0),
            bias_baseline=bias_base,
            bias_tuned=bias_tuned,
            concentration_flags=conc.get("flags", []),
        )
        verdicts[anchor] = verdict

        rel_imp = rel_wmape(wmape_base, wmape_tuned)
        print(
            f"  [{anchor}] baseline WMAPE={wmape_base:.4f}  "
            f"tuned WMAPE={wmape_tuned:.4f}  "
            f"rel_imp={rel_imp:.2f}%  verdict={verdict}"
        )

        overall_rows.append(
            {
                "anchor": anchor,
                "model": f"M1{'A' if anchor == 'ts' else 'B'}",
                "wmape_baseline": wmape_base,
                "wmape_tuned": wmape_tuned,
                "rel_wmape_improvement_pct": rel_imp,
                "rmse_baseline": float(o_base["rmse"]),
                "rmse_tuned": float(o_tuned["rmse"]),
                "mae_baseline": float(o_base["mae"]),
                "mae_tuned": float(o_tuned["mae"]),
                "bias_baseline": bias_base,
                "bias_tuned": bias_tuned,
                "bias_delta": bias_tuned - bias_base,
                "n": int(o_base["n"]),
                "verdict": verdict,
                **osu,
                **{f"product_{k}": v for k, v in psu.items() if k != "table"},
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
            }
        )

        for _, r in odf.iterrows():
            origin_rows.append({"anchor": anchor, **r.to_dict()})

        for _, r in hb.iterrows():
            horizon_rows.append({"anchor": anchor, **r.to_dict()})

        for _, r in pdf.iterrows():
            product_rows.append({"anchor": anchor, **r.to_dict()})

        conc_rows.append(
            {
                "anchor": anchor,
                "net_delta_ae": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "top5_improvement_share": conc.get("top5_improvement_share", float("nan")),
                "flags": ";".join(conc["flags"]),
            }
        )

        for sku in HIGH_VOLUME_WATCHLIST:
            sub = m.loc[m["product"] == sku]
            if sub.empty:
                continue
            watch_rows.append(
                {
                    "anchor": anchor,
                    "product": sku,
                    "delta_ae": float(sub["delta_ae"].sum()),
                    "actual_volume": float(np.abs(sub["actual"]).sum()),
                    "n": len(sub),
                    "wmape_baseline": wmape(sub["actual"], sub["pred_f0"]),
                    "wmape_tuned": wmape(sub["actual"], sub["pred_cand"]),
                }
            )

    # Write artifacts
    overall_df = pd.DataFrame(overall_rows)
    by_origin_df = pd.DataFrame(origin_rows)
    by_horizon_df = pd.DataFrame(horizon_rows)
    by_product_df = pd.DataFrame(product_rows)
    conc_df = pd.DataFrame(conc_rows)
    watch_df = pd.DataFrame(watch_rows)

    if all_tuned_preds:
        pd.concat(all_tuned_preds, ignore_index=True).to_parquet(
            out_dir / "tuned_predictions.parquet", index=False
        )

    overall_df.to_csv(out_dir / "overall.csv", index=False)
    by_origin_df.to_csv(out_dir / "by_origin.csv", index=False)
    by_horizon_df.to_csv(out_dir / "by_horizon.csv", index=False)
    by_product_df.to_csv(out_dir / "by_product.csv", index=False)
    conc_df.to_csv(out_dir / "error_concentration.csv", index=False)
    watch_df.to_csv(out_dir / "high_volume_watchlist.csv", index=False)

    # Optuna parameter importance (descriptive only)
    _write_param_importance(optuna_results, out_dir)

    assert_freeze_unchanged(ds, freeze_before)
    print("\n=== Freeze checksum unchanged — OK ===")

    return {
        "overall": overall_df,
        "by_origin": by_origin_df,
        "by_horizon": by_horizon_df,
        "by_product": by_product_df,
        "error_concentration": conc_df,
        "high_volume_watchlist": watch_df,
        "verdicts": verdicts,
        "f0_results": f0_results,
        "out_dir": out_dir,
    }


def _write_param_importance(
    optuna_results: dict[str, Any], out_dir: Path
) -> None:
    """Compute and write Optuna parameter importance (descriptive, not used for retuning)."""
    rows = []
    for anchor, res in optuna_results.items():
        if "error" in res or "study" not in res:
            continue
        study: "optuna.Study" = res["study"]
        try:
            import optuna.importance as oi
            importance = oi.get_param_importances(study)
            for param, imp in importance.items():
                rows.append({"anchor": anchor, "parameter": param, "importance": float(imp)})
        except Exception as exc:
            log.warning("Optuna importance failed for %s: %s", anchor, exc)
    pd.DataFrame(rows).to_csv(out_dir / "optuna_parameter_importance.csv", index=False)
