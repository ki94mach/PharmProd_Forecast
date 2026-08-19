from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
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
from pkg.research.tuning.folds import build_inner_folds
from pkg.research.tuning.search_space import suggest_params

from .config import (
    ANCHOR,
    EXPECTED_N,
    EXPECTED_N_ORIGINS,
    FEATURES,
    FORECAST_COL,
    INNER_EARLY_STOPPING_ROUNDS,
    INNER_EVAL_METRIC,
    INNER_N_ESTIMATORS,
    OPTUNA_N_JOBS,
    OPTUNA_STUDY_NAME,
    OPTUNA_TRIALS,
    PRE_PRIMARY_CUTOFF,
    PRIMARY_ORIGINS_LOCKED,
    SEED,
    TRAIN_UNIVERSE,
    XGB_DETERMINISTIC_FIXED,
    docs_path,
    optuna_db_url,
    output_dir,
)


def _assert_det_params(params: dict[str, Any]) -> None:
    assert params["n_jobs"] == 1
    assert params["tree_method"] == "hist"


def collect_environment_metadata() -> dict[str, Any]:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "optuna_version": optuna.__version__,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": int((__import__("os")).cpu_count() or 0),
        "tree_method": "hist",
        "xgboost_n_jobs": 1,
        "optuna_n_jobs": OPTUNA_N_JOBS,
        "random_seed": SEED,
    }
    return meta


def _make_residual_model(
    xgb_params: dict[str, Any], n_estimators: int, *, enforce_deterministic: bool = True
):
    params = {**XGB_DETERMINISTIC_FIXED, **xgb_params, "n_estimators": int(n_estimators)}
    if enforce_deterministic:
        _assert_det_params(params)

    def _predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        tr = train_df.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[FORECAST_COL].astype(float)
        sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)
        m = XGBRegressor(**params)
        m.fit(tr[list(FEATURES)], tr["residual"], sample_weight=sw, verbose=False)
        resid_hat = m.predict(test_df[list(FEATURES)])
        return np.maximum(0.0, test_df[FORECAST_COL].astype(float).to_numpy() + resid_hat)

    _predict.__name__ = "m1r_ts_model"
    return _predict


def _f0_param_dict(n_jobs: int) -> dict[str, Any]:
    params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": n_jobs,
        "tree_method": "hist",
    }
    return params


def _predict_with_model(
    ds,
    xgb_params: dict[str, Any],
    n_estimators: int,
    origins: tuple[int, ...] = PRIMARY_ORIGINS_LOCKED,
    *,
    enforce_deterministic: bool = True,
) -> pd.DataFrame:
    ts = prep_lags(ds.ts_universe)
    matched = prep_lags(ds.matched_universe)
    preds = []
    for o in origins:
        train = ts.loc[ts["target_date"].astype(int) < int(o)].copy()
        test = matched.loc[matched["origin"].astype(int) == int(o)].copy()
        assert int(train["target_date"].max()) < int(o)
        model = _make_residual_model(
            xgb_params, n_estimators, enforce_deterministic=enforce_deterministic
        )
        yhat = model(train, test)
        out = test.copy()
        out["prediction"] = yhat
        out["actual"] = out["sales"].astype(float)
        out["anchor"] = out[FORECAST_COL].astype(float)
        out["residual_prediction"] = out["prediction"] - out["anchor"]
        out["test_origin"] = int(o)
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def _metrics_from_preds(df: pd.DataFrame) -> dict[str, Any]:
    y = df["actual"].to_numpy(dtype=float)
    yhat = df["prediction"].to_numpy(dtype=float)
    return {
        "wmape": float(wmape(y, yhat)),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "mae": float(np.mean(np.abs(y - yhat))),
        "bias": float(np.mean(yhat - y)),
        "n": int(len(df)),
        "n_origins": int(df["test_origin"].nunique()),
    }


def run_repeatability(ds, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs = []
    vectors = []
    pair_rows = []
    for i in range(1, 6):
        p = _predict_with_model(ds, _f0_param_dict(1), 200)
        p["run_id"] = i
        m = _metrics_from_preds(p)
        m["run_id"] = i
        runs.append(m)
        vectors.append(p["prediction"].to_numpy(dtype=float))
        keep_cols = [
            "run_id",
            "product",
            "test_origin",
            "target_date",
            "horizon",
            "actual",
            "anchor",
            "residual_prediction",
            "prediction",
        ]
        pair_rows.append(p[keep_cols])

    for (i, a), (j, b) in combinations(enumerate(vectors, start=1), 2):
        d = np.abs(a - b)
        pair_rows.append(
            pd.DataFrame(
                {
                    "run_id": [f"pair_{i}_{j}"],
                    "product": [""],
                    "test_origin": [0],
                    "target_date": [0],
                    "horizon": [0],
                    "actual": [np.nan],
                    "anchor": [np.nan],
                    "residual_prediction": [np.nan],
                    "prediction": [np.nan],
                }
            )
        )

    all_rows = pd.concat(pair_rows, ignore_index=True)
    all_rows.to_csv(out_dir / "repeatability_runs.csv", index=False)

    max_abs = 0.0
    mean_abs = 0.0
    nz = 0
    pairs = 0
    for a, b in combinations(vectors, 2):
        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max()))
        mean_abs += float(d.mean())
        nz += int((d > 0).sum())
        pairs += 1
    mean_abs = mean_abs / pairs if pairs else 0.0

    summary = {
        "run_metrics": runs,
        "max_abs_prediction_diff_any_pair": max_abs,
        "mean_abs_prediction_diff_any_pair": mean_abs,
        "nonzero_prediction_differences_any_pair_total": nz,
    }
    if max_abs > 1e-9:
        raise AssertionError(
            f"Determinism failed: max_abs_prediction_diff={max_abs} > 1e-9"
        )
    return all_rows, summary


def run_threading_diagnostic(ds, out_dir: Path) -> pd.DataFrame:
    base = _predict_with_model(ds, _f0_param_dict(1), 200, enforce_deterministic=True)
    rows = []
    for nj in (1, 2, -1):
        p = _predict_with_model(ds, _f0_param_dict(nj), 200, enforce_deterministic=(nj == 1))
        if not base[["product", "test_origin", "target_date", "horizon"]].reset_index(drop=True).equals(
            p[["product", "test_origin", "target_date", "horizon"]].reset_index(drop=True)
        ):
            raise AssertionError(f"Threading diagnostic key mismatch for n_jobs={nj}")
        d = np.abs(
            p["prediction"].to_numpy(dtype=float)
            - base["prediction"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "n_jobs": nj,
                "wmape": float(wmape(p["actual"], p["prediction"])),
                "max_abs_prediction_diff_vs_n_jobs_1": float(d.max()),
                "mean_abs_prediction_diff_vs_n_jobs_1": float(d.mean()),
                "changed_rows_vs_n_jobs_1": int((d > 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "threading_diagnostic.csv", index=False)
    return out


def _optuna_objective(folds, trial):
    # Hard leakage assertions.
    assert all(int(f.origin) < PRE_PRIMARY_CUTOFF for f in folds)
    assert all(int(f.origin) not in set(PRIMARY_ORIGINS_LOCKED) for f in folds)

    tuned = suggest_params(trial)
    all_actual = []
    all_pred = []
    by_origin = {}
    best_iter_by_origin = {}
    for f in folds:
        v = int(f.origin)
        assert v < PRE_PRIMARY_CUTOFF
        assert int(f.train["target_date"].max()) < v
        tr = f.train.copy()
        va = f.val.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[FORECAST_COL].astype(float)
        va["residual"] = va["sales"].astype(float) - va[FORECAST_COL].astype(float)
        params = {
            **XGB_DETERMINISTIC_FIXED,
            **tuned,
            "n_estimators": INNER_N_ESTIMATORS,
            "early_stopping_rounds": INNER_EARLY_STOPPING_ROUNDS,
            "eval_metric": INNER_EVAL_METRIC,
        }
        _assert_det_params(params)
        model = XGBRegressor(**params)
        sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)
        model.fit(
            tr[list(FEATURES)],
            tr["residual"],
            sample_weight=sw,
            eval_set=[(va[list(FEATURES)], va["residual"])],
            verbose=False,
        )
        best_iter = int(model.best_iteration) + 1
        best_iter_by_origin[v] = best_iter
        resid = model.predict(va[list(FEATURES)])
        pred = np.maximum(0.0, va[FORECAST_COL].to_numpy(dtype=float) + resid)
        actual = va["sales"].to_numpy(dtype=float)
        by_origin[v] = float(wmape(actual, pred))
        all_actual.append(actual)
        all_pred.append(pred)

    ya = np.concatenate(all_actual)
    yp = np.concatenate(all_pred)
    pooled = float(wmape(ya, yp))
    trial.set_user_attr("wmape_by_origin", json.dumps(by_origin))
    trial.set_user_attr("median_origin_wmape", float(np.median(list(by_origin.values()))))
    trial.set_user_attr("worst_origin_wmape", float(np.max(list(by_origin.values()))))
    trial.set_user_attr("pooled_mae", float(np.mean(np.abs(ya - yp))))
    trial.set_user_attr("pooled_rmse", float(np.sqrt(np.mean((ya - yp) ** 2))))
    trial.set_user_attr("pooled_bias", float(np.mean(yp - ya)))
    trial.set_user_attr("best_iteration_by_origin", json.dumps(best_iter_by_origin))
    trial.set_user_attr("median_best_iteration", float(np.median(list(best_iter_by_origin.values()))))
    trial.set_user_attr("n_folds", len(folds))
    trial.set_user_attr("n_validation_rows", int(len(ya)))
    return pooled


def run_m1r() -> dict[str, Any]:
    out_dir = output_dir()
    ds = load_benchmark(verify_checksums=True)
    freeze_before = freeze_checksums(ds)

    env = collect_environment_metadata()
    print(json.dumps(env, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # 1) Repeatability.
    _, rep = run_repeatability(ds, out_dir)

    # 2) Deterministic F0 baseline.
    f0_pred = _predict_with_model(ds, _f0_param_dict(1), 200)
    f0_pred.to_parquet(out_dir / "deterministic_f0_predictions.parquet", index=False)
    f0_metrics = _metrics_from_preds(f0_pred)
    if f0_metrics["n"] != EXPECTED_N or f0_metrics["n_origins"] != EXPECTED_N_ORIGINS:
        raise AssertionError(f"Unexpected deterministic F0 key count: {f0_metrics}")

    # 3) Threading diagnostic.
    threading_df = run_threading_diagnostic(ds, out_dir)

    # 4) Fresh deterministic TS-only Optuna study.
    ts_univ = prep_lags(ds.ts_universe)
    folds = build_inner_folds(ts_univ, "ts", prepped=True)
    for f in folds:
        assert int(f.origin) < PRE_PRIMARY_CUTOFF
        assert int(f.origin) not in set(PRIMARY_ORIGINS_LOCKED)
        assert int(f.train["target_date"].max()) < int(f.origin)
    inner_origins = [int(f.origin) for f in folds]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=optuna_db_url(),
        load_if_exists=False,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(lambda t: _optuna_objective(folds, t), n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        r = {"trial_number": t.number, "value": t.value}
        r.update(t.params)
        r.update({f"ua_{k}": v for k, v in t.user_attrs.items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "ts_trials.csv", index=False)

    best = study.best_trial
    best_iters = json.loads(best.user_attrs["best_iteration_by_origin"])
    frozen_n_estimators = int(round(float(np.median(list(best_iters.values())))))
    best_payload = {
        "study_name": OPTUNA_STUDY_NAME,
        "trial_number": int(best.number),
        "best_inner_pooled_wmape": float(best.value),
        "inner_origins": inner_origins,
        "selected_hyperparameters": best.params,
        "frozen_n_estimators": frozen_n_estimators,
        "sampler_seed": SEED,
        "deterministic_execution": {
            "tree_method": "hist",
            "xgboost_n_jobs": 1,
            "optuna_n_jobs": 1,
            "random_state": SEED,
        },
        "environment_file": "environment.json",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "ts_best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    # 5) PRIMARY evaluate deterministic tuned once.
    tuned_params = {
        **best.params,
        **XGB_DETERMINISTIC_FIXED,
    }
    tuned_pred = _predict_with_model(ds, tuned_params, frozen_n_estimators)
    tuned_pred.to_parquet(out_dir / "tuned_predictions.parquet", index=False)
    tuned_metrics = _metrics_from_preds(tuned_pred)

    # 6) Key equality.
    key_cols = ["product", "test_origin", "target_date", "horizon"]
    a = f0_pred[key_cols].sort_values(key_cols).reset_index(drop=True)
    b = tuned_pred[key_cols].sort_values(key_cols).reset_index(drop=True)
    if not a.equals(b):
        raise AssertionError("Baseline and tuned deterministic PRIMARY keys differ.")

    # 7) Diagnostics tables.
    # Convert to BacktestResult-like holders for reuse of helpers.
    class _R:
        pass
    base_res = _R()
    tuned_res = _R()
    base_res.predictions = f0_pred.copy()
    tuned_res.predictions = tuned_pred.copy()
    base_res.by_origin = pd.DataFrame(
        [
            {"origin": int(o), "n": int(len(g)), "wmape": float(wmape(g["actual"], g["prediction"]))}
            for o, g in f0_pred.groupby("test_origin")
        ]
    )
    tuned_res.by_origin = pd.DataFrame(
        [
            {"origin": int(o), "n": int(len(g)), "wmape": float(wmape(g["actual"], g["prediction"]))}
            for o, g in tuned_pred.groupby("test_origin")
        ]
    )

    odf = origin_pair_table(base_res, tuned_res)
    osu = origin_summary(odf)
    hdf = horizon_bucket_table(base_res, tuned_res)
    pdf = product_pair_table(base_res, tuned_res)
    psu = product_summary(pdf)
    m = merge_ae(base_res, tuned_res)
    conc = error_concentration(m, "M1R_TS", "ts")

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
    verdict = "PROMOTE"
    if not (
        tuned_metrics["wmape"] < f0_metrics["wmape"]
        and psu.get("product_win_rate", 0.0) > 0.5
        and psu.get("median_product_improvement_pct", 0.0) > 0.0
        and osu.get("origins_improved", 0) > osu.get("origins_total", 0) / 2
        and not conc["flags"]
    ):
        verdict = "REJECT"

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
                "median_product_improvement_pct": psu.get("median_product_improvement_pct", np.nan),
                "p25_product_improvement_pct": psu.get("p25_product_improvement_pct", np.nan),
                "p75_product_improvement_pct": psu.get("p75_product_improvement_pct", np.nan),
                "net_delta_absolute_error": conc["net_delta_ae"],
                "total_deterioration": conc["total_deterioration"],
                "total_improvement": conc["total_improvement"],
                "top1_deterioration_share": conc["top1_deterioration_share"],
                "top5_deterioration_share": conc["top5_deterioration_share"],
                "top10_deterioration_share": conc["top10_deterioration_share"],
                "verdict": verdict,
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

    # Compare to old M1 for documentation only.
    old_dir = Path(__file__).resolve().parents[3] / "data" / "results" / "m1_optuna"
    old_ts = {}
    old_primary = np.nan
    if (old_dir / "ts_best_params.json").exists():
        old_ts = json.loads((old_dir / "ts_best_params.json").read_text(encoding="utf-8"))
    if (old_dir / "overall.csv").exists():
        od = pd.read_csv(old_dir / "overall.csv")
        if "wmape_tuned" in od.columns and len(od):
            old_primary = float(od["wmape_tuned"].iloc[0])

    assert_freeze_unchanged(ds, freeze_before)

    # Report
    _write_report(
        out_dir=out_dir,
        env=env,
        repeatability=rep,
        f0_metrics=f0_metrics,
        threading_df=threading_df,
        inner_origins=inner_origins,
        best_payload=best_payload,
        tuned_metrics=tuned_metrics,
        overall=overall.iloc[0].to_dict(),
        old_ts=old_ts,
        old_primary=old_primary,
        human_note=(
            "M1B remains unavailable: insufficient eligible pre-PRIMARY Budget origins "
            "under existing maturity rules; MIN_PRIOR_BUDGET_VINTAGES was not relaxed."
        ),
    )
    _append_research_contract_note()

    return {
        "out_dir": str(out_dir),
        "reproducibility_verdict": "PASS",
        "m1r_ts_verdict": verdict,
    }


def _write_report(
    *,
    out_dir: Path,
    env: dict[str, Any],
    repeatability: dict[str, Any],
    f0_metrics: dict[str, Any],
    threading_df: pd.DataFrame,
    inner_origins: list[int],
    best_payload: dict[str, Any],
    tuned_metrics: dict[str, Any],
    overall: dict[str, Any],
    old_ts: dict[str, Any],
    old_primary: float,
    human_note: str,
) -> None:
    rel = overall["rel_wmape_improvement_pct"]
    win_rate_pct = (
        float(overall["product_win_rate"]) * 100.0
        if np.isfinite(float(overall["product_win_rate"]))
        else float("nan")
    )
    text = f"""# M1R — Deterministic XGBoost Reproducibility and M1A Rerun

## Reproducibility
1. Environment: `{json.dumps(env)}`
2. Five n_jobs=1 runs prediction-identical: **Yes**
3. Max prediction diff across repeated n_jobs=1 runs: `{repeatability['max_abs_prediction_diff_any_pair']}`
4. Deterministic F0 TS WMAPE: `{f0_metrics['wmape']:.4f}`
5. n_jobs=2 / -1 diagnostic differences are in `threading_diagnostic.csv`.
6. Single-thread contract verdict: **PASS**

Historical context only: prior reported TS values included ~37.23, ~37.20, ~38.28.
This run establishes the deterministic contract baseline for this environment.

## Optuna Rerun (TS only)
7. Pre-PRIMARY origins used: `{inner_origins}`
8. Leakage assertions passed: `V < 140404` and `train.target_date.max() < V` for all folds.
9. Selected trial: `{best_payload['trial_number']}`
10. Selected parameters: `{best_payload['selected_hyperparameters']}`
11. frozen_n_estimators: `{best_payload['frozen_n_estimators']}`
12. Best inner pooled WMAPE: `{best_payload['best_inner_pooled_wmape']:.4f}`
13. Deterministic tuned vs deterministic F0 on PRIMARY: `tuned {tuned_metrics['wmape']:.4f}` vs `baseline {f0_metrics['wmape']:.4f}`
14. Relative WMAPE improvement: `{rel:.2f}%`
15. Origins improved: `{int(overall['origins_improved'])}/{int(overall['origins_total'])}`
16. Product win rate: `{win_rate_pct:.2f}%`
17. Bias change: baseline `{f0_metrics['bias']:.4f}` -> tuned `{tuned_metrics['bias']:.4f}`
18. Concentration: top1 `{overall['top1_deterioration_share']:.4f}`, top5 `{overall['top5_deterioration_share']:.4f}`, top10 `{overall['top10_deterioration_share']:.4f}`
19. Old non-deterministic M1 vs M1R:
    - old best params: `{old_ts.get('max_depth', 'n/a')}, lr={old_ts.get('learning_rate', 'n/a')}, gamma={old_ts.get('gamma', 'n/a')}, frozen_n_estimators={old_ts.get('frozen_n_estimators', 'n/a')}`
    - old best inner WMAPE: `{old_ts.get('best_inner_pooled_wmape', 'n/a')}`
    - old PRIMARY tuned WMAPE: `{old_primary}`
    - new best params: `{best_payload['selected_hyperparameters']}`
    - new frozen_n_estimators: `{best_payload['frozen_n_estimators']}`
    - new best inner WMAPE: `{best_payload['best_inner_pooled_wmape']:.4f}`
    - new PRIMARY tuned WMAPE: `{tuned_metrics['wmape']:.4f}`
20. Replacement decision: **{overall['verdict']}**

## Human
21. {human_note}

## Verdicts
- Reproducibility: **PASS**
- M1R TS Tuning: **{overall['verdict']}**

All outputs are under `src/data/results/m1r_reproducibility/`.
"""
    docs_path().write_text(text, encoding="utf-8")


def _append_research_contract_note() -> None:
    readme = Path(__file__).resolve().parents[4] / "README.md"
    existing = readme.read_text(encoding="utf-8")
    marker = "## Deterministic Research Contract"
    if marker in existing:
        return
    note = """

## Deterministic Research Contract

For controlled forecasting research runs:

- XGBoost `tree_method` must be explicitly `hist`
- XGBoost `n_jobs` must be `1`
- Optuna `n_jobs` must be `1`
- Model random seed must be explicit
- Python/XGBoost/NumPy/etc versions must be recorded
- Frozen benchmark checksums must be verified before/after runs

Multithreaded XGBoost can be used for separate performance experiments, but not
as canonical research benchmark runs unless reproducibility is re-established.
"""
    readme.write_text(existing.rstrip() + note + "\n", encoding="utf-8")

