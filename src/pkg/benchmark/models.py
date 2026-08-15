"""Frozen benchmark model adapters (Analysis A/B architectures)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pkg.benchmark.config import (
    BIAS_AF_EPS,
    BUDGET_RESID_FEATURES,
    INTEGRATED_FEATURES,
    RIDGE_ALPHA,
    TS_RESID_FEATURES,
    XGB_PARAMS,
)

ModelSpec = Union[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]

FROZEN_NAMES = frozenset(
    {
        "ts",
        "human",
        "ts_xgb",
        "human_xgb",
        "integrated",
        "bias_global",
        "bias_product",
        "bias_product_horizon",
        "af_ratio",
        "ridge",
    }
)

# Default train universe for each frozen name (matched TEST)
TRAIN_UNIVERSE = {
    "ts": None,
    "human": None,
    "ts_xgb": "ts",
    "human_xgb": "budget",
    "integrated": "matched",
    "bias_global": "budget",
    "bias_product": "budget",
    "bias_product_horizon": "budget",
    "af_ratio": "budget",
    "ridge": "budget",
}


def fit_xgb(features: Sequence[str], train: pd.DataFrame) -> XGBRegressor:
    """Fit XGB on train residuals with horizon sample weights (1/horizon)."""
    model = XGBRegressor(**XGB_PARAMS)
    sample_weight = 1.0 / train["horizon"].clip(lower=1).astype(float)
    model.fit(train[list(features)], train["residual"], sample_weight=sample_weight)
    return model


def _xtx_xty(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Form X'WX and X'Wy without BLAS gemm (broken OpenBLAS on some Windows envs)."""
    n, p = X.shape
    xtx = np.zeros((p, p), dtype=np.float64)
    xty = np.zeros(p, dtype=np.float64)
    for i in range(n):
        wi = float(w[i])
        if wi == 0.0:
            continue
        xi = X[i]
        yi = float(y[i])
        xty += wi * yi * xi
        # outer product xi xi' scaled by wi
        for a in range(p):
            xia = xi[a]
            if xia == 0.0:
                continue
            row = xtx[a]
            scale = wi * xia
            for b in range(p):
                row[b] += scale * xi[b]
    return xtx, xty


def _gauss_jordan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b for small dense A (p~14) without BLAS."""
    n = a.shape[0]
    m = np.zeros((n, n + 1), dtype=np.float64)
    m[:, :n] = a
    m[:, n] = b
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(m[col:, col])))
        if abs(m[pivot, col]) < 1e-18:
            raise np.linalg.LinAlgError("singular ridge system")
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
        m[col] /= m[col, col]
        for row in range(n):
            if row == col:
                continue
            m[row] -= m[row, col] * m[col]
    return m[:, n].copy()


def fit_ridge_weighted(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float = RIDGE_ALPHA,
):
    """Weighted Ridge with intercept; matches sklearn Ridge(alpha) defaults.

    Uses explicit loops instead of ``np.dot`` / sklearn because OpenBLAS
    gemm can segfault in some Windows conda envs while XGBoost still works.
    Returns (intercept, coef).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    # Center with weighted mean (sklearn fit_intercept=True)
    wsum = float(w.sum())
    if wsum <= 0:
        raise ValueError("sample_weight sum must be positive")
    x_mean = (w[:, None] * X).sum(axis=0) / wsum
    y_mean = float((w * y).sum() / wsum)
    Xc = X - x_mean
    yc = y - y_mean
    xtx, xty = _xtx_xty(Xc, yc, w)
    xtx = xtx + alpha * np.eye(xtx.shape[0], dtype=np.float64)
    coef = _gauss_jordan(xtx, xty)
    intercept = y_mean - float((x_mean * coef).sum())
    return intercept, coef


def ridge_predict(intercept: float, coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    # row-wise multiply-add without gemm
    out = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        out[i] = intercept + float((X[i] * coef).sum())
    return out


def _pit_bias_maps(train: pd.DataFrame):
    resid = train["sales"].astype(float) - train["budget_forecast"].astype(float)
    global_bias = float(resid.mean()) if len(train) else 0.0
    tmp = train.copy()
    tmp["_resid"] = resid.to_numpy()
    by_product = tmp.groupby("product")["_resid"].mean().to_dict()
    tmp["_h"] = tmp["horizon"].astype(int)
    by_ph = tmp.groupby(["product", "_h"])["_resid"].mean().to_dict()
    bud = train["budget_forecast"].astype(float)
    safe = bud.abs() >= BIAS_AF_EPS
    if safe.any():
        af = (train.loc[safe, "sales"].astype(float) / bud.loc[safe]).mean()
        global_af = float(af) if np.isfinite(af) else 1.0
    else:
        global_af = 1.0
    return global_bias, by_product, by_ph, global_af


def _apply_product_bias(products, global_bias, by_product):
    return np.array([by_product.get(p, global_bias) for p in products], dtype=float)


def _apply_product_horizon_bias(products, horizons, global_bias, by_product, by_ph):
    out = []
    for p, h in zip(products, horizons):
        h = int(h)
        if (p, h) in by_ph:
            out.append(by_ph[(p, h)])
        elif p in by_product:
            out.append(by_product[p])
        else:
            out.append(global_bias)
    return np.array(out, dtype=float)


def predict_frozen(
    name: str,
    train: Optional[pd.DataFrame],
    test: pd.DataFrame,
) -> np.ndarray:
    """Return point forecasts for a frozen architecture on ``test`` rows."""
    if name == "ts":
        return test["ts_forecast"].astype(float).to_numpy()
    if name == "human":
        return test["budget_forecast"].astype(float).to_numpy()

    if train is None or train.empty:
        raise ValueError(f"Model {name!r} requires a non-empty train fold")

    if name == "ts_xgb":
        tr = train.copy()
        tr["residual"] = tr["sales"] - tr["ts_forecast"]
        if "horizon" not in tr.columns:
            tr["horizon"] = tr["ts_horizon"]
        m = fit_xgb(TS_RESID_FEATURES, tr)
        resid = m.predict(test[TS_RESID_FEATURES])
        return np.maximum(0.0, test["ts_forecast"].astype(float).to_numpy() + resid)

    if name == "human_xgb":
        tr = train.copy()
        tr["residual"] = tr["sales"] - tr["budget_forecast"]
        if "horizon" not in tr.columns:
            tr["horizon"] = tr["budget_horizon"] if "budget_horizon" in tr.columns else tr["horizon"]
        m = fit_xgb(BUDGET_RESID_FEATURES, tr)
        resid = m.predict(test[BUDGET_RESID_FEATURES])
        return np.maximum(0.0, test["budget_forecast"].astype(float).to_numpy() + resid)

    if name == "integrated":
        tr = train.copy()
        tr["residual"] = tr["sales"] - tr["budget_forecast"]
        if "human_adjustment" not in tr.columns:
            tr["human_adjustment"] = tr["budget_forecast"] - tr["ts_forecast"]
        m = fit_xgb(INTEGRATED_FEATURES, tr)
        resid = m.predict(test[INTEGRATED_FEATURES])
        return np.maximum(0.0, test["budget_forecast"].astype(float).to_numpy() + resid)

    # Analysis A negative controls (Budget train)
    bud = test["budget_forecast"].astype(float).to_numpy()
    products = test["product"].tolist()
    horizons = test["horizon"].astype(int).tolist()
    global_bias, by_product, by_ph, global_af = _pit_bias_maps(train)

    if name == "bias_global":
        return np.maximum(0.0, bud + global_bias)
    if name == "bias_product":
        return np.maximum(0.0, bud + _apply_product_bias(products, global_bias, by_product))
    if name == "bias_product_horizon":
        return np.maximum(
            0.0,
            bud
            + _apply_product_horizon_bias(
                products, horizons, global_bias, by_product, by_ph
            ),
        )
    if name == "af_ratio":
        return np.maximum(0.0, bud * global_af)
    if name == "ridge":
        tr = train.copy()
        tr["residual"] = tr["sales"] - tr["budget_forecast"]
        sw = 1.0 / tr["horizon"].clip(lower=1).astype(float)
        intercept, coef = fit_ridge_weighted(
            tr[BUDGET_RESID_FEATURES].to_numpy(dtype=np.float64),
            tr["residual"].to_numpy(dtype=np.float64),
            sw.to_numpy(dtype=np.float64),
            alpha=RIDGE_ALPHA,
        )
        resid_hat = ridge_predict(
            intercept, coef, test[BUDGET_RESID_FEATURES].to_numpy(dtype=np.float64)
        )
        return np.maximum(0.0, bud + resid_hat)

    raise ValueError(f"Unknown frozen model: {name!r}")
