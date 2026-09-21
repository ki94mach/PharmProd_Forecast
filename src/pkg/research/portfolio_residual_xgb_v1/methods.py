"""Fixed evaluation methods for residual-XGBoost v1.1 (no HUMAN_XGB)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pkg.benchmark.config import XGB_PARAMS
from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    TS_RESID_FEATURE_LIST,
)

METHOD_FROZEN_TS = "FROZEN_TS"
METHOD_MEDIAN = "TS_PLUS_TRAIN_MEDIAN_RESIDUAL_CORRECTION"
METHOD_XGB = "TS_PLUS_RESIDUAL_XGB"

EXECUTABLE_METHODS: tuple[str, ...] = (METHOD_FROZEN_TS, METHOD_MEDIAN, METHOD_XGB)


def method_registry() -> dict[str, Any]:
    return {
        "executable": list(EXECUTABLE_METHODS),
        "blocked": {"HUMAN_XGB": HUMAN_XGB_STATUS},
        "seed": 42,
        "xgb_params": dict(XGB_PARAMS),
        "anchor": HYBRID_ANCHOR,
        "features": list(TS_RESID_FEATURE_LIST),
        "clip_nonnegative": True,
    }


def predict_frozen_ts(frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, frame[HYBRID_ANCHOR].astype(float).to_numpy())


def predict_median_correction(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, float]:
    resid = train["actual"].astype(float) - train[HYBRID_ANCHOR].astype(float)
    med = float(np.nanmedian(resid.to_numpy(dtype=float)))
    pred = test[HYBRID_ANCHOR].astype(float).to_numpy() + med
    return np.maximum(0.0, pred), med


def fit_predict_residual_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Fit residual XGB on train; return clipped corrected forecasts on test.

    ``n_jobs`` overrides frozen ``XGB_PARAMS['n_jobs']`` for this study only
    (does not mutate the frozen benchmark dict).
    """
    from xgboost import XGBRegressor

    feats = list(feature_cols) if feature_cols is not None else list(TS_RESID_FEATURE_LIST)
    tr = train.dropna(subset=["actual", HYBRID_ANCHOR] + feats).copy()
    te = test.copy()
    y = tr["actual"].astype(float) - tr[HYBRID_ANCHOR].astype(float)
    params = dict(XGB_PARAMS)
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    model = XGBRegressor(**params)
    model.fit(tr[feats].astype(float), y)
    resid_hat = model.predict(te[feats].astype(float))
    pred = te[HYBRID_ANCHOR].astype(float).to_numpy() + np.asarray(resid_hat, dtype=float)
    return np.maximum(0.0, pred)
