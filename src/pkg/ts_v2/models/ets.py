"""ETS adapter (statsmodels) for V2.

Uses additive seasonal structure when history is long enough; otherwise
additive error+trend only. The same kwargs builder is used for every fit
(backtest and final refit).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig, use_seasonal
from pkg.ts_v2.models.base import BaseForecastModel
from pkg.ts_v2.models.common import finite_values, point_forecast
from pkg.ts_v2.models.errors import ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def ets_kwargs(n: int, config: Optional[TSForecastConfig] = None) -> dict:
    """Shared ETS specification for CV and final fits."""
    cfg = config or DEFAULT_CONFIG
    if use_seasonal(n, cfg):
        return {
            "error": "add",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": int(cfg.seasonal_period),
        }
    return {"error": "add", "trend": "add"}


class ETSModelAdapter(BaseForecastModel):
    """statsmodels ETSModel in raw sales units."""

    name = "ets"

    def __init__(self, config: Optional[TSForecastConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._result: Any = None
        self._n: int = 0
        self._kwargs: dict = {}

    def fit(self, train_series: pd.Series) -> "ETSModelAdapter":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        values = finite_values(train_series, model_name=self.name)
        n = int(len(values))
        kwargs = ets_kwargs(n, self.config)
        if kwargs.get("seasonal") == "add" and n < int(self.config.seasonal_period):
            raise ModelUnavailableError(
                "ets seasonal fit needs at least seasonal_period observations",
                model_name=self.name,
                details={"n": n, "seasonal_period": self.config.seasonal_period},
            )
        if n < 3:
            raise ModelUnavailableError(
                f"ets requires at least 3 observations, got {n}",
                model_name=self.name,
                details={"n": n},
            )
        series = pd.Series(values, dtype=float)
        self._kwargs = dict(kwargs)
        self._n = n
        self._result = ETSModel(series, **kwargs).fit(disp=False)
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._result is None:
            raise ModelUnavailableError("ets is not fitted", model_name=self.name)
        raw = self._result.forecast(steps=int(horizon))
        preds = tuple(float(x) for x in raw)
        return point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={"n_train": self._n, "ets_kwargs": dict(self._kwargs)},
        )
