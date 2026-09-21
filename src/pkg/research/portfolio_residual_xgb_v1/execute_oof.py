"""Expanding-window OOF residual execution (VPS). Hard-fails on HB/post-HB.

Cursor must not run training; this module is for VPS --execute only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import (
    HB_ORIGINS,
    LOCKED_SCORE_END,
    POST_HB_ORIGINS,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import build_dense_actuals
from pkg.portfolio_ts_lock.decomposition import resolve_naive_deployable_partials
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb
from pkg.benchmark.universes import load_universe_product_names
from pkg.portfolio_ts_lock.completeness import audit_product_origin_completeness
from pkg.research.portfolio_residual_xgb_v1.constants import DETERMINISTIC_SEED
from pkg.research.portfolio_residual_xgb_v1.features_build import (
    add_calendar_and_sales_features,
    encode_categoricals_train_only,
    ensure_feature_columns,
)
from pkg.research.portfolio_residual_xgb_v1.methods import (
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
    fit_predict_residual_xgb,
    predict_frozen_ts,
    predict_median_correction,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    TS_RESID_FEATURE_LIST,
    filter_train_rows_pit,
    last_complete_month,
)
from pkg.research.portfolio_residual_xgb_v1.prepare_v1_1 import (  # noqa: E402
    hard_fail_hb,
    hard_fail_hb_post_hb,
)


def _load_residual_panel(
    *,
    sales_path: Path,
    v21_export: Path,
    a0_export: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_parquet(sales_path)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    panel_v21 = filter_non_hb(pd.read_parquet(v21_export / "forecasts_panel.parquet"))
    panel_a0 = filter_non_hb(pd.read_parquet(a0_export / "forecasts_panel.parquet"))
    hard_fail_hb(panel_v21, context="v21")
    hard_fail_hb(panel_a0, context="a0")
    status, _ = audit_product_origin_completeness(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        original55=set(load_universe_product_names("mvp_products")),
    )
    _c, deployable, _cats, partial = build_policy_panels(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        completeness_status=status,
    )
    naive_extra, _ = resolve_naive_deployable_partials(partial=partial, dense=dense)
    if not naive_extra.empty:
        deployable = pd.concat([deployable, naive_extra], ignore_index=True)
    residual = deployable.loc[
        deployable["forecast_origin"].astype(int).isin(RESIDUAL_DEVELOPMENT_ORIGINS)
    ].copy()
    hard_fail_hb_post_hb(residual, context="residual")
    return residual, sales


def run_oof_origin(
    *,
    residual_panel: pd.DataFrame,
    sales: pd.DataFrame,
    evaluation_origin: int,
    out_dir: Path,
    resume: bool = True,
    xgb_n_jobs: int | None = None,
) -> Path:
    o = int(evaluation_origin)
    if o in HB_ORIGINS or o in POST_HB_ORIGINS:
        raise RuntimeError(f"Refusing HB/post-HB evaluation origin {o}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"oof_predictions_origin_{o}.parquet"
    meta_path = out_dir / f"oof_meta_origin_{o}.json"
    if resume and pred_path.is_file() and meta_path.is_file():
        return pred_path

    o2 = shamsi_add_months(o, -2)
    train = filter_train_rows_pit(
        residual_panel, evaluation_origin=o, origin_minus_2=o2
    )
    test = residual_panel.loc[residual_panel["forecast_origin"].astype(int) == o].copy()
    hard_fail_hb_post_hb(train, context=f"train O={o}")
    hard_fail_hb_post_hb(test, context=f"test O={o}")
    if train.empty or test.empty:
        raise RuntimeError(f"Empty train/test for origin {o}")

    train_f = add_calendar_and_sales_features(train, sales)
    test_f = add_calendar_and_sales_features(test, sales)
    train_f, test_f, enc = encode_categoricals_train_only(train_f, test_f)
    train_f = ensure_feature_columns(train_f)
    test_f = ensure_feature_columns(test_f)

    pred_ts = predict_frozen_ts(test_f)
    pred_med, med = predict_median_correction(train_f, test_f)
    pred_xgb = fit_predict_residual_xgb(
        train_f,
        test_f,
        feature_cols=list(TS_RESID_FEATURE_LIST),
        n_jobs=xgb_n_jobs,
    )

    out = test_f[
        ["product", "forecast_origin", "horizon", "target_date", "actual", HYBRID_ANCHOR]
    ].copy()
    out[METHOD_FROZEN_TS] = pred_ts
    out[METHOD_MEDIAN] = pred_med
    out[METHOD_XGB] = pred_xgb
    # Magnification: XGB abs error worse than TS
    ae_ts = (out[METHOD_FROZEN_TS] - out["actual"]).abs()
    ae_xgb = (out[METHOD_XGB] - out["actual"]).abs()
    out["xgb_magnifies_ts_error"] = ae_xgb > ae_ts
    out["ae_delta_xgb_minus_ts"] = ae_xgb - ae_ts

    tmp = pred_path.with_suffix(pred_path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(pred_path)

    meta: dict[str, Any] = {
        "evaluation_origin": o,
        "origin_minus_2": o2,
        "n_train_rows": int(len(train_f)),
        "n_test_rows": int(len(test_f)),
        "max_sales_feature_date_train": int(
            train_f["forecast_origin"].astype(int).map(last_complete_month).max()
        ),
        "max_sales_feature_date_test": int(
            test_f["forecast_origin"].astype(int).map(last_complete_month).max()
        ),
        "median_residual": med,
        "seed": DETERMINISTIC_SEED,
        "xgb_n_jobs": xgb_n_jobs,
        "encoder": enc.to_state(),
        "hb_outcomes_inspected": False,
        "methods": [METHOD_FROZEN_TS, METHOD_MEDIAN, METHOD_XGB],
        "resume_safe": True,
    }
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    meta_tmp.replace(meta_path)
    return pred_path
