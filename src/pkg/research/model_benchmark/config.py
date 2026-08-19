"""M2 constants — predeclared learner configs, no tuning."""
from __future__ import annotations

from pathlib import Path

from pkg.benchmark.config import (
    BUDGET_RESID_FEATURES,
    PRIMARY_ORIGINS,
    TS_RESID_FEATURES,
)

SEED = 42
PREDICTION_REPEAT_TOL = 1e-9
M1R_F0_WMAPE_REF = 38.64273633337519

PRIMARY_ORIGINS_LOCKED = tuple(int(x) for x in PRIMARY_ORIGINS)
EXPECTED_MATCHED_N = 1877
EXPECTED_MATCHED_ORIGINS = 5

MODELS = ("xgboost", "ridge", "elasticnet", "catboost", "lightgbm")

ANCHOR_FORECAST_COL = {
    "ts": "ts_forecast",
    "human": "budget_forecast",
}

FEATURES_BY_ANCHOR = {
    "ts": tuple(TS_RESID_FEATURES),
    "human": tuple(BUDGET_RESID_FEATURES),
}

TRAIN_UNIVERSE_BY_ANCHOR = {
    "ts": "ts",
    "human": "budget",
}

ORIGIN_COL = {
    "ts": "ts_origin",
    "human": "budget_origin",
    "matched": "origin",
}

NUMERIC_BASE = [
    "horizon",
    "month",
    "quarter",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_12",
    "sales_roll3",
]

CATEGORICAL_FEATURES = [
    "model_enc",
    "field_enc",
    "form_enc",
    "provider_enc",
]

# Per-anchor numeric features include the anchor forecast column.
NUMERIC_FEATURES_BY_ANCHOR = {
    anchor: [ANCHOR_FORECAST_COL[anchor], *NUMERIC_BASE]
    for anchor in ANCHOR_FORECAST_COL
}

XGBOOST_F0_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": SEED,
}

RIDGE_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
}

ELASTICNET_PARAMS = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "fit_intercept": True,
    "max_iter": 10000,
    "random_state": SEED,
}

CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": 1,
    "deterministic": True,
    "verbosity": -1,
    "force_col_wise": True,
}

COMPETITIVE_WMAPE_MARGIN = 0.5


def output_dir() -> Path:
    p = Path(__file__).resolve().parents[3] / "data" / "results" / "m2_model_benchmark"
    p.mkdir(parents=True, exist_ok=True)
    return p


def docs_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "m2_residual_learner_benchmark.md"
