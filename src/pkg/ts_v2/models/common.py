"""Shared helpers for V2 forecasting models."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.models.errors import ModelContractError, ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def finite_values(train_series: pd.Series, *, model_name: str) -> np.ndarray:
    """Return a float array; raise unavailable on empty or NaN training."""
    if train_series is None or len(train_series) == 0:
        raise ModelUnavailableError(
            "empty training series",
            model_name=model_name,
            details={"n": 0},
        )
    values = pd.to_numeric(train_series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ModelUnavailableError(
            "training series contains missing values",
            model_name=model_name,
            details={"n": int(len(values)), "n_nan": int(np.isnan(values).sum())},
        )
    return values


def point_forecast(
    model_name: str,
    predictions: Sequence[float],
    target_dates: Sequence[int],
    metadata: Optional[dict] = None,
) -> ForecastResult:
    """Build a :class:`ForecastResult` with copied target dates (no rounding)."""
    dates = tuple(int(d) for d in target_dates)
    preds = tuple(float(p) for p in predictions)
    if len(preds) != len(dates):
        raise ModelContractError(
            f"internal length mismatch preds={len(preds)} dates={len(dates)}",
            model_name=model_name,
        )
    return ForecastResult(
        model_name=model_name,
        predictions=preds,
        target_dates=dates,
        horizons=tuple(range(1, len(dates) + 1)),
        metadata=dict(metadata or {}),
    )
