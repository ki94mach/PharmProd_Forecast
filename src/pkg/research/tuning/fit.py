"""Research-only XGB fit helpers for M1.

Deliberately separate from benchmark.models.fit_xgb so the inner-fold path
can use early stopping while the PRIMARY path cannot, without touching the
frozen benchmark adapter.

Fixed params (objective, random_state, n_jobs) are always merged in here.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pkg.research.tuning.config import (
    FORECAST_COL,
    INNER_EARLY_STOPPING_ROUNDS,
    INNER_EVAL_METRIC,
    INNER_N_ESTIMATORS,
    XGB_FIXED_PARAMS,
)


def _build_regressor(tuned_params: dict[str, Any], n_estimators: int) -> XGBRegressor:
    """Combine fixed + tuned params into an XGBRegressor."""
    params = {**XGB_FIXED_PARAMS, **tuned_params, "n_estimators": n_estimators}
    return XGBRegressor(**params)


def fit_inner_fold(
    anchor: str,
    features: Sequence[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    tuned_params: dict[str, Any],
) -> tuple[XGBRegressor, int]:
    """Fit residual XGB on one inner fold with early stopping.

    Returns (fitted_model, best_iteration + 1).
    Early stopping uses val residuals as eval_set. Never PRIMARY data.

    Note: early_stopping_rounds must be passed to XGBRegressor constructor
    in XGBoost >= 2.0 for best_iteration to be populated.
    """
    cols = list(features)
    forecast_col = FORECAST_COL[anchor]

    tr = train.copy()
    tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)

    vl = val.copy()
    vl["residual"] = vl["sales"].astype(float) - vl[forecast_col].astype(float)

    sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)

    params = {
        **XGB_FIXED_PARAMS,
        **tuned_params,
        "n_estimators": INNER_N_ESTIMATORS,
        "early_stopping_rounds": INNER_EARLY_STOPPING_ROUNDS,
    }
    model = XGBRegressor(**params)
    model.fit(
        tr[cols],
        tr["residual"],
        sample_weight=sw,
        eval_set=[(vl[cols], vl["residual"])],
        verbose=False,
    )
    best_iter = int(model.best_iteration)
    return model, best_iter + 1


def predict_inner_fold(
    anchor: str,
    features: Sequence[str],
    model: XGBRegressor,
    val: pd.DataFrame,
) -> np.ndarray:
    """Final forecast from inner fold: anchor + clipped residual_hat."""
    cols = list(features)
    forecast_col = FORECAST_COL[anchor]
    resid_hat = model.predict(val[cols])
    anchor_vals = val[forecast_col].astype(float).to_numpy()
    return np.maximum(0.0, anchor_vals + resid_hat)


def make_primary_model(
    anchor: str,
    features: Sequence[str],
    frozen_params: dict[str, Any],
    frozen_n_estimators: int,
):
    """Return a backtest-compatible callable for PRIMARY evaluation.

    No eval_set. No early stopping. Same frozen_params for every origin.
    Signature: (train_df, test_df) -> np.ndarray of forecasts.
    """
    cols = list(features)
    forecast_col = FORECAST_COL[anchor]
    params = {**frozen_params, "n_estimators": frozen_n_estimators}
    model_name = f"m1_{'ts' if anchor == 'ts' else 'human'}_tuned"

    def _predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        tr = train_df.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
        sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)
        model = XGBRegressor(**params)
        model.fit(tr[cols], tr["residual"], sample_weight=sw, verbose=False)
        resid_hat = model.predict(test_df[cols])
        anchor_vals = test_df[forecast_col].astype(float).to_numpy()
        return np.maximum(0.0, anchor_vals + resid_hat)

    _predict.__name__ = model_name
    return _predict
