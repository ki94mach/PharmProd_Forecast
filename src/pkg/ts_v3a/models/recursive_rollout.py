"""Multi-step recursive forecast rollout (scaled space).

Recursion happens only here at forecast time. Callers must pass an already
scaled lookback window and inverse-transform the returned vector once.
No teacher forcing, clipping, rounding, smoothing, or scaler refitting.
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def _as_batch_window(initial_window: ArrayLike) -> np.ndarray:
    """Normalize to shape ``(1, lookback, 1)``."""
    arr = np.asarray(initial_window, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        # (lookback, 1) or (1, lookback)
        if arr.shape[0] == 1:
            return arr.reshape(1, arr.shape[1], 1)
        if arr.shape[1] == 1:
            return arr.reshape(1, arr.shape[0], 1)
        raise ValueError(f"Ambiguous 2D initial_window shape {arr.shape}")
    if arr.ndim == 3:
        if arr.shape[0] != 1 or arr.shape[2] != 1:
            raise ValueError(f"Expected (1, lookback, 1), got {arr.shape}")
        return arr.copy()
    raise ValueError(f"initial_window must be 1D/2D/3D, got ndim={arr.ndim}")


def _scalar_prediction(yhat: Any) -> float:
    arr = np.asarray(yhat, dtype=float).reshape(-1)
    if arr.size < 1:
        raise ValueError("model.predict returned an empty array")
    return float(arr[0])


def rollout_recursive_forecast(
    model: Any,
    initial_window: ArrayLike,
    *,
    horizon: int = 15,
) -> np.ndarray:
    """Autoregressive one-step recursion for ``horizon`` months.

    Parameters
    ----------
    model:
        Object with ``predict(X, verbose=0)`` returning a one-step forecast
        in the same scale as ``initial_window``.
    initial_window:
        Scaled lookback window: ``(lookback,)``, ``(lookback, 1)``, or
        ``(1, lookback, 1)``.
    horizon:
        Number of recursive steps (V2 contract default: 15).

    Returns
    -------
    np.ndarray
        Shape ``(horizon,)`` predictions in the model/input scale (not
        inverse-transformed).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    current = _as_batch_window(initial_window)
    lookback = int(current.shape[1])
    preds: list[float] = []

    for _ in range(int(horizon)):
        try:
            yhat = model.predict(current, verbose=0)
        except TypeError:
            yhat = model.predict(current)
        value = _scalar_prediction(yhat)
        preds.append(value)
        # Shift window and append prediction (no teacher forcing).
        next_step = np.array([[[value]]], dtype=float)
        current = np.concatenate([current[:, 1:, :], next_step], axis=1)
        assert current.shape == (1, lookback, 1)

    return np.asarray(preds, dtype=float)
