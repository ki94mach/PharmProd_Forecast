"""Croston SBA and TSB intermittent-demand baselines for V2.

Local implementations of standard formulations (no extra dependency).
Smoothing parameters come from :class:`~pkg.ts_v2.config.TSForecastConfig`
and are never tuned on holdout / future data.

Both methods produce a constant per-period mean demand rate for every
requested target date (classic Croston/TSB point forecast).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.models.base import BaseForecastModel
from pkg.ts_v2.models.common import finite_values, point_forecast
from pkg.ts_v2.models.errors import ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def _validate_alpha(name: str, value: float, *, model_name: str) -> float:
    a = float(value)
    if not (0.0 < a <= 1.0):
        raise ModelUnavailableError(
            f"{name} must be in (0, 1], got {a}",
            model_name=model_name,
            details={name: a},
        )
    return a


def fit_croston_sba(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """Fit Croston with Syntetos–Boylan approximation.

    Updates demand size ``z`` and inter-demand interval ``p`` only on
    positive-demand periods. Point forecast rate::

        rate = (1 - alpha / 2) * z / p

    Returns ``(rate, z, p)``.
    """
    demand_idx = np.flatnonzero(values > 0.0)
    if len(demand_idx) == 0:
        return 0.0, 0.0, 1.0

    z = float(values[demand_idx[0]])
    # Interval from series start to first demand (at least 1).
    p = float(demand_idx[0] + 1)
    for k in range(1, len(demand_idx)):
        q = float(demand_idx[k] - demand_idx[k - 1])
        y = float(values[demand_idx[k]])
        z = z + alpha * (y - z)
        p = p + beta * (q - p)
    p = max(p, 1e-12)
    rate = (1.0 - alpha / 2.0) * z / p
    return float(rate), float(z), float(p)


def fit_tsb(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """Fit Teunter–Syntetos–Babai (TSB) method.

    Updates demand probability every period; updates size only on demand.
    Point forecast rate::

        rate = p * z

    Returns ``(rate, z, p)``.
    """
    demand_idx = np.flatnonzero(values > 0.0)
    if len(demand_idx) == 0:
        return 0.0, 0.0, 0.0

    z = float(values[demand_idx[0]])
    # Initialize probability at the first demand occurrence.
    p = 1.0
    start = int(demand_idx[0])
    for t in range(start + 1, len(values)):
        y = float(values[t])
        if y > 0.0:
            z = z + alpha * (y - z)
            p = p + beta * (1.0 - p)
        else:
            p = p + beta * (0.0 - p)
    rate = p * z
    return float(rate), float(z), float(p)


class CrostonSBAModel(BaseForecastModel):
    """Croston method with Syntetos–Boylan bias correction."""

    name = "croston_sba"

    def __init__(self, config: Optional[TSForecastConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._rate: Optional[float] = None
        self._z: Optional[float] = None
        self._p: Optional[float] = None
        self._n: int = 0
        self._alpha: float = 0.1
        self._beta: float = 0.1

    def fit(self, train_series: pd.Series) -> "CrostonSBAModel":
        values = finite_values(train_series, model_name=self.name)
        if len(values) < 1:
            raise ModelUnavailableError(
                "croston_sba requires at least 1 observation",
                model_name=self.name,
            )
        alpha = _validate_alpha("croston_alpha", self.config.croston_alpha, model_name=self.name)
        beta_raw = self.config.croston_beta
        beta = alpha if beta_raw is None else _validate_alpha(
            "croston_beta", beta_raw, model_name=self.name
        )
        rate, z, p = fit_croston_sba(values, alpha=alpha, beta=beta)
        self._rate = rate
        self._z = z
        self._p = p
        self._n = int(len(values))
        self._alpha = alpha
        self._beta = beta
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._rate is None:
            raise ModelUnavailableError("croston_sba is not fitted", model_name=self.name)
        rate = self._rate
        preds = tuple(rate for _ in range(horizon))
        return point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={
                "rate": rate,
                "z": self._z,
                "p": self._p,
                "alpha": self._alpha,
                "beta": self._beta,
                "n_train": self._n,
            },
        )


class TSBModel(BaseForecastModel):
    """Teunter–Syntetos–Babai intermittent-demand method."""

    name = "tsb"

    def __init__(self, config: Optional[TSForecastConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._rate: Optional[float] = None
        self._z: Optional[float] = None
        self._p: Optional[float] = None
        self._n: int = 0
        self._alpha: float = 0.1
        self._beta: float = 0.1

    def fit(self, train_series: pd.Series) -> "TSBModel":
        values = finite_values(train_series, model_name=self.name)
        if len(values) < 1:
            raise ModelUnavailableError(
                "tsb requires at least 1 observation",
                model_name=self.name,
            )
        alpha = _validate_alpha("tsb_alpha", self.config.tsb_alpha, model_name=self.name)
        beta = _validate_alpha("tsb_beta", self.config.tsb_beta, model_name=self.name)
        rate, z, p = fit_tsb(values, alpha=alpha, beta=beta)
        self._rate = rate
        self._z = z
        self._p = p
        self._n = int(len(values))
        self._alpha = alpha
        self._beta = beta
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._rate is None:
            raise ModelUnavailableError("tsb is not fitted", model_name=self.name)
        rate = self._rate
        preds = tuple(rate for _ in range(horizon))
        return point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={
                "rate": rate,
                "z": self._z,
                "p": self._p,
                "alpha": self._alpha,
                "beta": self._beta,
                "n_train": self._n,
            },
        )
