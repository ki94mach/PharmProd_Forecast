"""M2 diagnostics: repeatability, verdicts, linear/tree diagnostics."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from pkg.benchmark.dataset import BenchmarkDataset, prep_lags
from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
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
from pkg.research.model_benchmark.config import (
    ANCHOR_FORECAST_COL,
    CATBOOST_PARAMS,
    CATEGORICAL_FEATURES,
    COMPETITIVE_WMAPE_MARGIN,
    ELASTICNET_PARAMS,
    FEATURES_BY_ANCHOR,
    LIGHTGBM_PARAMS,
    PRIMARY_ORIGINS_LOCKED,
    RIDGE_PARAMS,
    XGBOOST_F0_PARAMS,
)
from pkg.research.model_benchmark.evaluate import (
    SuiteResult,
    build_folds,
    filter_predictions_to_origins,
    rolling_residual_backtest,
)
from pkg.research.model_benchmark.models import (
    CatBoostLearner,
    ElasticNetLearner,
    LightGBMLearner,
    RidgeLearner,
    XGBoostF0Learner,
    make_learner,
)
from pkg.research.model_benchmark.preprocessing import (
    linear_numeric_features,
    make_linear_preprocessor,
    prep_catboost_frame,
    prep_lightgbm_frame,
    prep_xgb_frame,
)


def classify_m2_verdict(
    *,
    wmape_xgb: float,
    wmape_candidate: float,
    product_win_rate: float,
    median_product_improvement_pct: float,
    origins_improved: int,
    origins_total: int,
    bias_xgb: float,
    bias_candidate: float,
    concentration_flags: list[str],
    horizon_buckets_improved: int,
) -> str:
    """BEATS_XGBOOST / COMPETITIVE / WEAKER_THAN_XGBOOST."""
    better = wmape_candidate < wmape_xgb
    near = abs(wmape_candidate - wmape_xgb) <= COMPETITIVE_WMAPE_MARGIN
    median_ok = median_product_improvement_pct > 0
    win_ok = product_win_rate > 0.50
    origins_ok = origins_total > 0 and origins_improved > origins_total / 2
    bias_ok = (
        abs(bias_candidate) <= abs(bias_xgb) * 1.25
        or abs(bias_candidate) <= abs(bias_xgb) + 200.0
    )
    concentrated = bool(concentration_flags)
    multi_horizon = horizon_buckets_improved > 1

    if (
        better
        and median_ok
        and win_ok
        and origins_ok
        and bias_ok
        and not concentrated
        and multi_horizon
    ):
        return "BEATS_XGBOOST"
    if better and (not median_ok or not win_ok or not origins_ok or concentrated or not multi_horizon):
        return "COMPETITIVE"
    if near and (median_ok or win_ok or origins_ok):
        return "COMPETITIVE"
    if (not better) and (median_ok or win_ok):
        return "COMPETITIVE"
    return "WEAKER_THAN_XGBOOST"


def overall_m2_conclusion(verdicts: dict[str, dict[str, str]]) -> str:
    """CASE A / B / C across anchors."""
    beats = [
        (anchor, model)
        for anchor, models in verdicts.items()
        for model, v in models.items()
        if model != "xgboost" and v == "BEATS_XGBOOST"
    ]
    competitive = [
        (anchor, model)
        for anchor, models in verdicts.items()
        for model, v in models.items()
        if model != "xgboost" and v == "COMPETITIVE"
    ]
    if beats:
        return "CASE_A"
    if competitive:
        return "CASE_B"
    return "CASE_C"


def _horizon_buckets_improved(base: BacktestResult, cand: BacktestResult) -> int:
    hdf = horizon_bucket_table(base, cand)
    if hdf.empty:
        return 0
    return int((hdf["rel_wmape_vs_f0_pct"] > 0).sum())


def compare_vs_xgb(
    suite: SuiteResult,
    *,
    slice_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall, by_origin, by_horizon, by_product, error_concentration tables."""
    xgb = suite.results["xgboost"]
    overall_rows = []
    origin_rows = []
    horizon_rows = []
    product_rows = []
    conc_rows = []

    for model_name, res in suite.results.items():
        o = res.overall.iloc[0]
        x = xgb.overall.iloc[0]
        rel = rel_wmape(float(x["wmape"]), float(o["wmape"])) if model_name != "xgboost" else 0.0
        oim, otot = (0, 0)
        pwin = float("nan")
        if model_name != "xgboost":
            odf = origin_pair_table(xgb, res)
            osum = origin_summary(odf)
            oim = osum["origins_improved"]
            otot = osum["origins_total"]
            pdf = product_pair_table(xgb, res)
            ps = product_summary(pdf)
            pwin = ps["product_win_rate"]
            origin_rows.append(
                odf.assign(
                    candidate=model_name,
                    anchor=suite.anchor,
                    slice=slice_label,
                )
            )
            horizon_rows.append(
                horizon_bucket_table(xgb, res).assign(
                    candidate=model_name,
                    anchor=suite.anchor,
                    slice=slice_label,
                )
            )
            product_rows.append(
                pdf.assign(
                    candidate=model_name,
                    anchor=suite.anchor,
                    slice=slice_label,
                )
            )
            mae = merge_ae(xgb, res)
            conc = error_concentration(mae, model_name, suite.anchor)
            conc_rows.append(
                {
                    "anchor": suite.anchor,
                    "slice": slice_label,
                    "candidate": model_name,
                    "net_delta_ae": conc["net_delta_ae"],
                    "total_deterioration": conc["total_deterioration"],
                    "total_improvement": conc["total_improvement"],
                    "top1_deterioration_share": conc["top1_deterioration_share"],
                    "top5_deterioration_share": conc["top5_deterioration_share"],
                    "top10_deterioration_share": conc["top10_deterioration_share"],
                    "top5_improvement_share": conc["top5_improvement_share"],
                    "concentration_flags": "|".join(conc["flags"]),
                }
            )

        overall_rows.append(
            {
                "anchor": suite.anchor,
                "slice": slice_label,
                "model": model_name,
                "wmape": float(o["wmape"]),
                "rmse": float(o["rmse"]),
                "mae": float(o["mae"]),
                "bias": float(o["bias"]),
                "n": int(o["n"]),
                "n_origins": len(res.origins),
                "n_products": int(res.predictions["product"].nunique()),
                "relative_wmape_vs_xgb_pct": rel,
                "origins_improved_vs_xgb": oim,
                "origins_total": otot,
                "product_win_rate": pwin,
            }
        )

    by_origin = pd.concat(origin_rows, ignore_index=True) if origin_rows else pd.DataFrame()
    by_horizon = pd.concat(horizon_rows, ignore_index=True) if horizon_rows else pd.DataFrame()
    by_product = pd.concat(product_rows, ignore_index=True) if product_rows else pd.DataFrame()
    error_conc = pd.DataFrame(conc_rows)
    return pd.DataFrame(overall_rows), by_origin, by_horizon, by_product, error_conc


def build_watchlist(
    by_product: pd.DataFrame,
    *,
    anchor: str,
    slice_label: str,
) -> pd.DataFrame:
    if by_product.empty:
        return pd.DataFrame()
    rows = []
    for product in HIGH_VOLUME_WATCHLIST:
        sub = by_product.loc[by_product["product"] == product]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            rows.append(
                {
                    "anchor": anchor,
                    "slice": slice_label,
                    "product": product,
                    "candidate": r["candidate"],
                    "actual_volume": r["actual_volume"],
                    "wmape_control": r["wmape_control"],
                    "wmape_candidate": r["wmape_candidate"],
                    "relative_improvement_pct": r["relative_improvement_pct"],
                    "delta_absolute_error": r["delta_absolute_error"],
                }
            )
    return pd.DataFrame(rows)


def run_repeatability_gate(
    ds: BenchmarkDataset,
    anchor: str = "ts",
    n_runs: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """5x deterministic XGBoost F0 on matched PRIMARY; max diff <= tol."""
    learner = XGBoostF0Learner()
    origins = list(PRIMARY_ORIGINS_LOCKED)
    vectors = []
    run_rows = []
    for i in range(1, n_runs + 1):
        res = rolling_residual_backtest(
            ds, anchor, [learner], origins, slice_kind="matched_primary"
        )
        preds = res.results["xgboost"].predictions
        vectors.append(preds["prediction"].to_numpy(dtype=float))
        run_rows.append(
            {
                "run_id": i,
                "model": "xgboost",
                "anchor": anchor,
                "wmape": float(wmape(preds["actual"], preds["prediction"])),
                "n": len(preds),
            }
        )

    pair_rows = []
    max_abs = 0.0
    for (i, a), (j, b) in combinations(enumerate(vectors, start=1), 2):
        d = np.abs(a - b)
        mx = float(d.max())
        max_abs = max(max_abs, mx)
        pair_rows.append({"run_a": i, "run_b": j, "max_abs_prediction_diff": mx})

    out = pd.DataFrame(run_rows + [{"run_id": "pairs", "model": "", "anchor": "", "wmape": max_abs, "n": len(pair_rows)}])
    summary = {
        "max_abs_prediction_diff": max_abs,
        "pair_comparisons": pair_rows,
        "run_metrics": run_rows,
    }
    return out, summary


def run_tree_repeatability(
    ds: BenchmarkDataset,
    models: Sequence[str] = ("xgboost", "catboost", "lightgbm"),
) -> pd.DataFrame:
    rows = []
    for anchor in ("ts", "human"):
        origins = list(PRIMARY_ORIGINS_LOCKED)
        for model_name in models:
            learner = make_learner(model_name)
            v1 = rolling_residual_backtest(
                ds, anchor, [learner], origins, slice_kind="matched_primary"
            ).results[model_name].predictions["prediction"].to_numpy(dtype=float)
            v2 = rolling_residual_backtest(
                ds, anchor, [learner], origins, slice_kind="matched_primary"
            ).results[model_name].predictions["prediction"].to_numpy(dtype=float)
            d = np.abs(v1 - v2)
            rows.append(
                {
                    "model": model_name,
                    "anchor": anchor,
                    "max_abs_repeated_prediction_diff": float(d.max()),
                    "mean_abs_repeated_prediction_diff": float(d.mean()),
                    "nonzero_diff_rows": int((d > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def fit_linear_diagnostics(
    ds: BenchmarkDataset,
    anchor: str,
) -> list[dict[str, Any]]:
    """Ridge/ElasticNet coef stats from last PRIMARY origin fit."""
    anchor_col = ANCHOR_FORECAST_COL[anchor]
    features = FEATURES_BY_ANCHOR[anchor]
    folds = build_folds(ds, anchor, PRIMARY_ORIGINS_LOCKED, slice_kind="matched_primary")
    fold = folds[-1]
    rows = []
    for learner_cls, params, model_name in (
        (Ridge, RIDGE_PARAMS, "ridge"),
        (ElasticNet, ELASTICNET_PARAMS, "elasticnet"),
    ):
        tr = fold.train.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
        sw = (1.0 / tr["horizon"].clip(lower=1).astype(float)).to_numpy()
        num_feats = linear_numeric_features(anchor_col)
        pre = make_linear_preprocessor(num_feats, CATEGORICAL_FEATURES)
        pipe = Pipeline([("pre", pre), ("model", learner_cls(**params))])
        pipe.fit(tr, tr["residual"], model__sample_weight=sw)
        Xt = pipe.named_steps["pre"].transform(tr)
        coef = pipe.named_steps["model"].coef_
        nz = int(np.sum(np.abs(coef) > 1e-12))
        mag = np.abs(coef)
        rows.append(
            {
                "anchor": anchor,
                "model": model_name,
                "origin": int(fold.origin),
                "n_transformed_features": int(Xt.shape[1]),
                "n_nonzero_coefficients": nz,
                "coef_abs_p25": float(np.quantile(mag, 0.25)),
                "coef_abs_median": float(np.median(mag)),
                "coef_abs_p75": float(np.quantile(mag, 0.75)),
                "coef_abs_max": float(mag.max()),
            }
        )
    return rows


def fit_feature_importance(
    ds: BenchmarkDataset,
    anchor: str,
) -> list[dict[str, Any]]:
    anchor_col = ANCHOR_FORECAST_COL[anchor]
    features = FEATURES_BY_ANCHOR[anchor]
    folds = build_folds(ds, anchor, PRIMARY_ORIGINS_LOCKED, slice_kind="matched_primary")
    fold = folds[-1]
    tr = fold.train.copy()
    tr["residual"] = tr["sales"].astype(float) - tr[anchor_col].astype(float)
    sw = (1.0 / tr["horizon"].clip(lower=1).astype(float)).to_numpy()
    rows: list[dict[str, Any]] = []

    # XGBoost
    tr_x = prep_xgb_frame(tr, features)
    xgb = XGBRegressor(**XGBOOST_F0_PARAMS)
    xgb.fit(tr_x[list(features)], tr["residual"], sample_weight=sw, verbose=False)
    for feat, imp in zip(features, xgb.feature_importances_):
        rows.append(
            {
                "anchor": anchor,
                "model": "xgboost",
                "origin": int(fold.origin),
                "feature": feat,
                "importance": float(imp),
            }
        )

    # CatBoost
    tr_cb = prep_catboost_frame(tr, features)
    cat_idx = [list(features).index(c) for c in CATEGORICAL_FEATURES if c in features]
    cb = CatBoostRegressor(**CATBOOST_PARAMS)
    cb.fit(Pool(tr_cb[list(features)], tr["residual"], cat_features=cat_idx, weight=sw))
    for feat, imp in zip(features, cb.get_feature_importance()):
        rows.append(
            {
                "anchor": anchor,
                "model": "catboost",
                "origin": int(fold.origin),
                "feature": feat,
                "importance": float(imp),
            }
        )

    # LightGBM
    tr_lgb, _ = prep_lightgbm_frame(tr, tr, features)
    cat_names = [c for c in CATEGORICAL_FEATURES if c in features]
    lgb = LGBMRegressor(**LIGHTGBM_PARAMS)
    lgb.fit(
        tr_lgb[list(features)],
        tr["residual"],
        sample_weight=sw,
        categorical_feature=cat_names,
    )
    for feat, imp in zip(features, lgb.feature_importances_):
        rows.append(
            {
                "anchor": anchor,
                "model": "lightgbm",
                "origin": int(fold.origin),
                "feature": feat,
                "importance": float(imp),
            }
        )
    return rows


def compute_verdicts_for_suite(
    suite: SuiteResult,
    *,
    slice_label: str,
    error_conc: pd.DataFrame,
) -> dict[str, str]:
    xgb = suite.results["xgboost"]
    x_w = float(xgb.overall["wmape"].iloc[0])
    x_b = float(xgb.overall["bias"].iloc[0])
    verdicts: dict[str, str] = {"xgboost": "REFERENCE"}
    for model_name, res in suite.results.items():
        if model_name == "xgboost":
            continue
        o_w = float(res.overall["wmape"].iloc[0])
        o_b = float(res.overall["bias"].iloc[0])
        pdf = product_pair_table(xgb, res)
        ps = product_summary(pdf)
        odf = origin_pair_table(xgb, res)
        osum = origin_summary(odf)
        flags_row = error_conc.loc[
            (error_conc["candidate"] == model_name)
            & (error_conc["anchor"] == suite.anchor)
            & (error_conc["slice"] == slice_label)
        ]
        flags = (
            flags_row["concentration_flags"].iloc[0].split("|")
            if len(flags_row) and flags_row["concentration_flags"].iloc[0]
            else []
        )
        flags = [f for f in flags if f]
        hb = _horizon_buckets_improved(xgb, res)
        verdicts[model_name] = classify_m2_verdict(
            wmape_xgb=x_w,
            wmape_candidate=o_w,
            product_win_rate=float(ps["product_win_rate"]),
            median_product_improvement_pct=float(ps["median_product_improvement_pct"]),
            origins_improved=int(osum["origins_improved"]),
            origins_total=int(osum["origins_total"]),
            bias_xgb=x_b,
            bias_candidate=o_b,
            concentration_flags=flags,
            horizon_buckets_improved=hb,
        )
    return verdicts


def slice_primary_from_broad(suite: SuiteResult) -> SuiteResult:
    """Filter broad-history suite to PRIMARY origins."""
    filtered_results: dict[str, BacktestResult] = {}
    for name, res in suite.results.items():
        preds = filter_predictions_to_origins(res.predictions, PRIMARY_ORIGINS_LOCKED)
        from pkg.research.model_benchmark.evaluate import _predictions_to_result

        filtered_results[name] = _predictions_to_result(preds, name)
    primary_origins = sorted(
        set(suite.origins_used) & set(PRIMARY_ORIGINS_LOCKED)
    )
    pooled = filter_predictions_to_origins(suite.pooled_predictions, PRIMARY_ORIGINS_LOCKED)
    return SuiteResult(
        anchor=suite.anchor,
        slice_kind="broad",
        results=filtered_results,
        pooled_predictions=pooled,
        origins_used=primary_origins,
        fold_diagnostics=suite.fold_diagnostics.loc[
            suite.fold_diagnostics["origin"].isin(primary_origins)
        ].copy(),
    )
