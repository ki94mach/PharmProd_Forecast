"""M1A2 pipeline: fixed-200 structural Optuna diagnostic (no early stopping)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from pkg.benchmark.dataset import load_benchmark, prep_lags
from pkg.benchmark.evaluate import wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.harness.gates import assert_freeze_unchanged, freeze_checksums
from pkg.research.harness.metrics import (
    error_concentration,
    horizon_bucket_table,
    merge_ae,
    origin_pair_table,
    origin_summary,
    product_pair_table,
    product_summary,
    rel_wmape,
)
from pkg.research.m1a2.config import (
    ANCHOR,
    EXPECTED_INNER_ORIGINS,
    EXPECTED_N,
    EXPECTED_N_ORIGINS,
    FEATURES,
    FIXED_N_ESTIMATORS,
    FORECAST_COL,
    M1R_F0_WMAPE_REF,
    M1R_RESULTS_DIR,
    OPTUNA_N_JOBS,
    OPTUNA_STUDY_NAME,
    OPTUNA_TRIALS,
    PRE_PRIMARY_CUTOFF,
    PRIMARY_ORIGINS_LOCKED,
    PREDICTION_REPEAT_TOL,
    SEED,
    XGB_DETERMINISTIC_FIXED,
    output_dir,
    optuna_db_url,
)
from pkg.research.m1a2.report import write_m1a2_report
from pkg.research.m1r.run import (
    _assert_det_params,
    _f0_param_dict,
    _metrics_from_preds,
    _predict_with_model,
    collect_environment_metadata,
    run_repeatability,
)
from pkg.research.tuning.folds import InnerFold, build_inner_folds
from pkg.research.tuning.search_space import suggest_params


def _build_xgb_params(structural: dict[str, Any]) -> dict[str, Any]:
    params = {
        **XGB_DETERMINISTIC_FIXED,
        **structural,
        "n_estimators": FIXED_N_ESTIMATORS,
    }
    _assert_det_params(params)
    assert params["n_estimators"] == FIXED_N_ESTIMATORS
    return params


def _fit_predict_fold(
    train: pd.DataFrame,
    val: pd.DataFrame,
    structural: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, XGBRegressor]:
    """Fit 200-tree residual model on one fold. No eval_set / early stopping."""
    tr = train.copy()
    va = val.copy()
    tr["residual"] = tr["sales"].astype(float) - tr[FORECAST_COL].astype(float)
    params = _build_xgb_params(structural)
    model = XGBRegressor(**params)
    sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)
    model.fit(tr[list(FEATURES)], tr["residual"], sample_weight=sw, verbose=False)
    assert model.get_params()["n_estimators"] == FIXED_N_ESTIMATORS
    resid = model.predict(va[list(FEATURES)])
    pred = np.maximum(0.0, va[FORECAST_COL].astype(float).to_numpy() + resid)
    actual = va["sales"].astype(float).to_numpy()
    return actual, pred, model


def _evaluate_inner_folds(
    folds: list[InnerFold],
    structural: dict[str, Any],
) -> tuple[float, dict[int, float], pd.DataFrame]:
    """Return pooled WMAPE, per-origin WMAPE, and row-level predictions."""
    all_actual: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    by_origin: dict[int, float] = {}
    rows: list[pd.DataFrame] = []

    for f in folds:
        v = int(f.origin)
        assert v < PRE_PRIMARY_CUTOFF
        assert int(f.train["target_date"].max()) < v
        actual, pred, _ = _fit_predict_fold(f.train, f.val, structural)
        by_origin[v] = float(wmape(actual, pred))
        all_actual.append(actual)
        all_pred.append(pred)
        part = f.val.copy()
        part["actual"] = actual
        part["prediction"] = pred
        part["inner_origin"] = v
        rows.append(part)

    ya = np.concatenate(all_actual)
    yp = np.concatenate(all_pred)
    pooled = float(wmape(ya, yp))
    return pooled, by_origin, pd.concat(rows, ignore_index=True)


def _verify_folds(folds: list[InnerFold]) -> list[int]:
    origins = sorted(int(f.origin) for f in folds)
    expected = list(EXPECTED_INNER_ORIGINS)
    if len(folds) != len(expected):
        raise AssertionError(
            f"Expected {len(expected)} inner folds, got {len(folds)}. Origins={origins}"
        )
    if origins != expected:
        raise AssertionError(
            f"Inner origins mismatch.\nExpected: {expected}\nGot:      {origins}\n"
            "STOP — do not tune with a different origin set."
        )
    for f in folds:
        v = int(f.origin)
        assert v < PRE_PRIMARY_CUTOFF
        assert v not in PRIMARY_ORIGINS_LOCKED
        assert int(f.train["target_date"].max()) < v
    return origins


def _verify_m1r_baseline(f0_pred: pd.DataFrame, f0_wmape: float) -> None:
    """Ensure deterministic F0 matches M1R environment if artifacts exist."""
    m1r_parquet = M1R_RESULTS_DIR / "deterministic_f0_predictions.parquet"
    if m1r_parquet.exists():
        m1r = pd.read_parquet(m1r_parquet)
        key_cols = ["product", "test_origin", "target_date", "horizon"]
        a = f0_pred[key_cols + ["prediction"]].sort_values(key_cols).reset_index(drop=True)
        b = m1r[key_cols + ["prediction"]].sort_values(key_cols).reset_index(drop=True)
        if not a[key_cols].equals(b[key_cols]):
            raise AssertionError("M1A2 F0 PRIMARY keys differ from M1R baseline parquet")
        d = np.abs(
            a["prediction"].to_numpy(dtype=float) - b["prediction"].to_numpy(dtype=float)
        )
        if float(d.max()) > PREDICTION_REPEAT_TOL:
            raise AssertionError(
                f"M1A2 F0 predictions differ from M1R baseline: max_diff={d.max()}"
            )
    if abs(f0_wmape - M1R_F0_WMAPE_REF) > 0.05:
        raise AssertionError(
            f"Deterministic F0 WMAPE {f0_wmape:.4f} differs materially from "
            f"M1R reference {M1R_F0_WMAPE_REF:.4f}"
        )


def _optuna_objective(
    folds: list[InnerFold],
    f0_by_origin: dict[int, float],
    trial: optuna.Trial,
) -> float:
    tuned = suggest_params(trial)
    pooled, by_origin, pred_df = _evaluate_inner_folds(folds, tuned)

    origin_wmapes = list(by_origin.values())
    beating = sum(1 for o, w in by_origin.items() if w < f0_by_origin.get(o, float("inf")))

    trial.set_user_attr("wmape_by_origin", json.dumps(by_origin))
    trial.set_user_attr(
        "median_origin_wmape",
        float(np.median(origin_wmapes)) if origin_wmapes else float("nan"),
    )
    trial.set_user_attr(
        "worst_origin_wmape",
        float(np.max(origin_wmapes)) if origin_wmapes else float("nan"),
    )
    trial.set_user_attr(
        "best_origin_wmape",
        float(np.min(origin_wmapes)) if origin_wmapes else float("nan"),
    )
    trial.set_user_attr("origins_beating_f0_inner", int(beating))
    trial.set_user_attr("n_inner_origins", len(folds))
    trial.set_user_attr("n_estimators", FIXED_N_ESTIMATORS)

    ya = pred_df["actual"].to_numpy(dtype=float)
    yp = pred_df["prediction"].to_numpy(dtype=float)
    trial.set_user_attr("pooled_mae", float(np.mean(np.abs(ya - yp))))
    trial.set_user_attr("pooled_rmse", float(np.sqrt(np.mean((ya - yp) ** 2))))
    trial.set_user_attr("pooled_bias", float(np.mean(yp - ya)))
    trial.set_user_attr("n_validation_rows", int(len(ya)))

    return pooled


def classify_verdict(
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
        return "WEAK / NEEDS CONFIRMATION"
    if (not better) and (median_ok or win_ok):
        return "WEAK / NEEDS CONFIRMATION"
    return "REJECT"


def classify_diagnostic(
    *,
    f0_primary: float,
    m1a2_primary: float,
    m1r_primary: float,
) -> str:
    """Diagnostic label for early-stopping vs structural transfer."""
    m1a2_vs_m1r = m1a2_primary < m1r_primary - 0.5  # materially better than M1R catastrophe
    m1a2_vs_f0 = m1a2_primary <= f0_primary + 0.25  # near or improves F0

    if m1a2_vs_m1r and m1a2_vs_f0:
        return "EARLY_STOPPING_WAS_MAJOR_FAILURE_SOURCE"
    if m1a2_vs_m1r and not m1a2_vs_f0:
        return "EARLY_STOPPING_ONLY_PART_OF_FAILURE"
    return "STRUCTURAL_TUNING_ALSO_FAILS_TO_TRANSFER"


def run_m1a2() -> dict[str, Any]:
    out_dir = output_dir()
    ds = load_benchmark(verify_checksums=True)
    freeze_before = freeze_checksums(ds)

    # 1) Environment metadata
    env = collect_environment_metadata()
    print(json.dumps(env, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # 2) F0 repeatability gate
    _, rep = run_repeatability(ds, out_dir)

    # 3) Build / verify inner folds
    ts_univ = prep_lags(ds.ts_universe)
    folds = build_inner_folds(ts_univ, ANCHOR, prepped=True)
    inner_origins = _verify_folds(folds)

    # 4) Inner canonical F0 reference (200 trees)
    f0_structural = _f0_param_dict(1)
    inner_f0_pooled, inner_f0_by_origin, inner_f0_preds = _evaluate_inner_folds(
        folds, f0_structural
    )
    inner_f0_rows = []
    for o, w in inner_f0_by_origin.items():
        inner_f0_rows.append({"origin": o, "wmape": w, "model": "canonical_f0_inner"})
    inner_f0_rows.append(
        {
            "origin": "pooled",
            "wmape": inner_f0_pooled,
            "model": "canonical_f0_inner",
        }
    )
    pd.DataFrame(inner_f0_rows).to_csv(out_dir / "inner_f0_metrics.csv", index=False)

    # 5) Fresh Optuna study (fixed 200, no early stopping)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=optuna_db_url(),
        load_if_exists=False,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(
        lambda t: _optuna_objective(folds, inner_f0_by_origin, t),
        n_trials=OPTUNA_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=False,
    )

    trial_rows = []
    diag_rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial_number": t.number, "pooled_wmape": t.value}
        row.update(t.params)
        row.update({f"ua_{k}": v for k, v in t.user_attrs.items()})
        trial_rows.append(row)
        diag_rows.append(
            {
                "trial_number": t.number,
                "pooled_wmape": t.value,
                "origins_beating_f0_inner": t.user_attrs.get("origins_beating_f0_inner"),
                "median_origin_wmape": t.user_attrs.get("median_origin_wmape"),
                "worst_origin_wmape": t.user_attrs.get("worst_origin_wmape"),
                "n_estimators": FIXED_N_ESTIMATORS,
            }
        )
    pd.DataFrame(trial_rows).to_csv(out_dir / "ts_trials.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "inner_trial_diagnostics.csv", index=False)

    best = study.best_trial
    rel_inner = (
        (inner_f0_pooled - float(best.value)) / inner_f0_pooled * 100.0
        if inner_f0_pooled > 0
        else float("nan")
    )
    best_payload = {
        "study_name": OPTUNA_STUDY_NAME,
        "trial_number": int(best.number),
        "selected_hyperparameters": best.params,
        "n_estimators": FIXED_N_ESTIMATORS,
        "best_inner_pooled_wmape": float(best.value),
        "inner_f0_pooled_wmape": float(inner_f0_pooled),
        "relative_inner_improvement_pct": float(rel_inner),
        "inner_origins": inner_origins,
        "sampler_seed": SEED,
        "deterministic_execution": {
            "tree_method": "hist",
            "xgboost_n_jobs": 1,
            "optuna_n_jobs": 1,
            "random_state": SEED,
            "n_estimators": FIXED_N_ESTIMATORS,
            "early_stopping": False,
        },
        "environment_file": "environment.json",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "ts_best_params.json").write_text(
        json.dumps(best_payload, indent=2), encoding="utf-8"
    )

    # 6) PRIMARY evaluation (once)
    f0_pred = _predict_with_model(ds, f0_structural, FIXED_N_ESTIMATORS)
    f0_pred.to_parquet(out_dir / "baseline_predictions.parquet", index=False)
    f0_metrics = _metrics_from_preds(f0_pred)
    if f0_metrics["n"] != EXPECTED_N or f0_metrics["n_origins"] != EXPECTED_N_ORIGINS:
        raise AssertionError(f"Unexpected F0 PRIMARY keys: {f0_metrics}")
    _verify_m1r_baseline(f0_pred, f0_metrics["wmape"])

    tuned_structural = {**best.params, **XGB_DETERMINISTIC_FIXED}
    tuned_pred = _predict_with_model(ds, tuned_structural, FIXED_N_ESTIMATORS)
    tuned_pred.to_parquet(out_dir / "tuned_predictions.parquet", index=False)
    tuned_metrics = _metrics_from_preds(tuned_pred)

    key_cols = ["product", "test_origin", "target_date", "horizon"]
    if not f0_pred[key_cols].sort_values(key_cols).reset_index(drop=True).equals(
        tuned_pred[key_cols].sort_values(key_cols).reset_index(drop=True)
    ):
        raise AssertionError("Baseline and M1A2 PRIMARY keys differ.")

    class _R:
        pass

    base_res = _R()
    tuned_res = _R()
    base_res.predictions = f0_pred.copy()
    tuned_res.predictions = tuned_pred.copy()
    base_res.by_origin = pd.DataFrame(
        [
            {
                "origin": int(o),
                "n": int(len(g)),
                "wmape": float(wmape(g["actual"], g["prediction"])),
            }
            for o, g in f0_pred.groupby("test_origin")
        ]
    )
    tuned_res.by_origin = pd.DataFrame(
        [
            {
                "origin": int(o),
                "n": int(len(g)),
                "wmape": float(wmape(g["actual"], g["prediction"])),
            }
            for o, g in tuned_pred.groupby("test_origin")
        ]
    )

    odf = origin_pair_table(base_res, tuned_res)
    osu = origin_summary(odf)
    hdf = horizon_bucket_table(base_res, tuned_res)
    pdf = product_pair_table(base_res, tuned_res)
    psu = product_summary(pdf)
    m = merge_ae(base_res, tuned_res)
    conc = error_concentration(m, "M1A2_TS", ANCHOR)

    watch_rows = []
    for sku in HIGH_VOLUME_WATCHLIST:
        sub = m.loc[m["product"] == sku]
        if sub.empty:
            continue
        watch_rows.append(
            {
                "product": sku,
                "delta_ae": float(sub["delta_ae"].sum()),
                "actual_volume": float(np.abs(sub["actual"]).sum()),
                "n": int(len(sub)),
                "wmape_baseline": float(wmape(sub["actual"], sub["pred_f0"])),
                "wmape_tuned": float(wmape(sub["actual"], sub["pred_cand"])),
            }
        )
    watch_df = pd.DataFrame(watch_rows)

    rel_imp = float(rel_wmape(f0_metrics["wmape"], tuned_metrics["wmape"]))
    verdict = classify_verdict(
        wmape_baseline=f0_metrics["wmape"],
        wmape_tuned=tuned_metrics["wmape"],
        product_win_rate=float(psu.get("product_win_rate", 0.0)),
        median_product_improvement_pct=float(psu.get("median_product_improvement_pct", 0.0)),
        origins_improved=int(osu.get("origins_improved", 0)),
        origins_total=int(osu.get("origins_total", 0)),
        bias_baseline=f0_metrics["bias"],
        bias_tuned=tuned_metrics["bias"],
        concentration_flags=conc.get("flags", []),
    )

    # Load M1R historical reference
    m1r_primary = float("nan")
    m1r_inner = float("nan")
    m1r_n_est = 27
    if (M1R_RESULTS_DIR / "overall.csv").exists():
        m1r_overall = pd.read_csv(M1R_RESULTS_DIR / "overall.csv")
        if len(m1r_overall):
            m1r_primary = float(m1r_overall["tuned_wmape"].iloc[0])
    if (M1R_RESULTS_DIR / "ts_best_params.json").exists():
        m1r_params = json.loads(
            (M1R_RESULTS_DIR / "ts_best_params.json").read_text(encoding="utf-8")
        )
        m1r_inner = float(m1r_params.get("best_inner_pooled_wmape", float("nan")))
        m1r_n_est = int(m1r_params.get("frozen_n_estimators", 27))

    diagnostic = classify_diagnostic(
        f0_primary=f0_metrics["wmape"],
        m1a2_primary=tuned_metrics["wmape"],
        m1r_primary=m1r_primary,
    )

    overall = pd.DataFrame(
        [
            {
                "baseline_wmape": f0_metrics["wmape"],
                "tuned_wmape": tuned_metrics["wmape"],
                "rel_wmape_improvement_pct": rel_imp,
                "baseline_rmse": f0_metrics["rmse"],
                "tuned_rmse": tuned_metrics["rmse"],
                "baseline_mae": f0_metrics["mae"],
                "tuned_mae": tuned_metrics["mae"],
                "baseline_bias": f0_metrics["bias"],
                "tuned_bias": tuned_metrics["bias"],
                "bias_delta": tuned_metrics["bias"] - f0_metrics["bias"],
                "n": f0_metrics["n"],
                "n_origins": f0_metrics["n_origins"],
                "origins_improved": osu.get("origins_improved", 0),
                "origins_total": osu.get("origins_total", 0),
                "median_origin_improvement": osu.get("median_origin_improvement", np.nan),
                "product_win_rate": psu.get("product_win_rate", np.nan),
                "median_product_improvement_pct": psu.get(
                    "median_product_improvement_pct", np.nan
                ),
                "p25_product_improvement_pct": psu.get("p25_product_improvement_pct", np.nan),
                "p75_product_improvement_pct": psu.get("p75_product_improvement_pct", np.nan),
                "net_delta_absolute_error": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "top5_improvement_share": conc.get("top5_improvement_share", np.nan),
                "verdict": verdict,
                "diagnostic_classification": diagnostic,
            }
        ]
    )

    overall.to_csv(out_dir / "overall.csv", index=False)
    odf.to_csv(out_dir / "by_origin.csv", index=False)
    hdf.to_csv(out_dir / "by_horizon.csv", index=False)
    pdf.to_csv(out_dir / "by_product.csv", index=False)
    pd.DataFrame(
        [
            {
                "net_delta_absolute_error": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "top5_improvement_share": conc.get("top5_improvement_share", np.nan),
                "flags": ";".join(conc["flags"]),
            }
        ]
    ).to_csv(out_dir / "error_concentration.csv", index=False)
    watch_df.to_csv(out_dir / "high_volume_watchlist.csv", index=False)

    try:
        import optuna.importance as oi

        imp = oi.get_param_importances(study)
        pd.DataFrame(
            [{"parameter": k, "importance": float(v)} for k, v in imp.items()]
        ).to_csv(out_dir / "parameter_importance.csv", index=False)
    except Exception:
        pd.DataFrame(columns=["parameter", "importance"]).to_csv(
            out_dir / "parameter_importance.csv", index=False
        )

    # Three-model comparison (M1R historical)
    m1r_rel = (
        (f0_metrics["wmape"] - m1r_primary) / f0_metrics["wmape"] * 100.0
        if np.isfinite(m1r_primary)
        else float("nan")
    )
    m1r_origins = "1/5"
    m1r_win = 0.3774
    if (M1R_RESULTS_DIR / "overall.csv").exists():
        mo = pd.read_csv(M1R_RESULTS_DIR / "overall.csv")
        if len(mo):
            m1r_origins = f"{int(mo['origins_improved'].iloc[0])}/{int(mo['origins_total'].iloc[0])}"
            m1r_win = float(mo["product_win_rate"].iloc[0])

    pd.DataFrame(
        [
            {
                "model": "Deterministic_F0",
                "n_estimators": FIXED_N_ESTIMATORS,
                "inner_pooled_wmape": inner_f0_pooled,
                "primary_wmape": f0_metrics["wmape"],
                "rel_primary_improvement_vs_f0_pct": 0.0,
                "origins_improved": "",
                "product_win_rate": "",
            },
            {
                "model": "M1R_historical",
                "n_estimators": m1r_n_est,
                "inner_pooled_wmape": m1r_inner,
                "primary_wmape": m1r_primary,
                "rel_primary_improvement_vs_f0_pct": m1r_rel,
                "origins_improved": m1r_origins,
                "product_win_rate": m1r_win,
            },
            {
                "model": "M1A2",
                "n_estimators": FIXED_N_ESTIMATORS,
                "inner_pooled_wmape": float(best.value),
                "primary_wmape": tuned_metrics["wmape"],
                "rel_primary_improvement_vs_f0_pct": rel_imp,
                "origins_improved": f"{int(osu.get('origins_improved', 0))}/{int(osu.get('origins_total', 0))}",
                "product_win_rate": float(psu.get("product_win_rate", np.nan)),
            },
        ]
    ).to_csv(out_dir / "model_comparison.csv", index=False)

    assert_freeze_unchanged(ds, freeze_before)

    write_m1a2_report(
        out_dir=out_dir,
        env=env,
        repeatability=rep,
        inner_origins=inner_origins,
        inner_f0_pooled=inner_f0_pooled,
        best_payload=best_payload,
        f0_metrics=f0_metrics,
        tuned_metrics=tuned_metrics,
        overall=overall.iloc[0].to_dict(),
        diagnostic=diagnostic,
        m1r_primary=m1r_primary,
        m1r_n_est=m1r_n_est,
        m1r_inner=m1r_inner,
    )

    return {
        "out_dir": str(out_dir),
        "reproducibility_verdict": "PASS",
        "m1a2_verdict": verdict,
        "diagnostic_classification": diagnostic,
    }
