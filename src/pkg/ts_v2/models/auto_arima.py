"""AutoARIMA adapter (pmdarima) for V2.

Seasonal SARIMA only when history exceeds ``seasonal_enable_after_months``.
Predicts exactly ``horizon`` steps — never 16-then-drop-1.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig, use_seasonal
from pkg.ts_v2.models.base import BaseForecastModel
from pkg.ts_v2.models.common import finite_values, point_forecast
from pkg.ts_v2.models.errors import ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def _fit_auto_arima(values, *, seasonal: bool, m: int):
    import pmdarima as pm

    common = dict(
        max_order=None,
        max_p=6,
        max_q=6,
        max_d=2,
        max_P=4,
        max_Q=4,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    if seasonal:
        try:
            return pm.auto_arima(
                values,
                seasonal=True,
                m=m,
                max_D=1,
                **common,
            )
        except ValueError:
            return pm.auto_arima(
                values,
                seasonal=True,
                m=m,
                max_D=1,
                seasonal_test="ch",
                **common,
            )
    return pm.auto_arima(values, seasonal=False, max_D=2, **common)


class AutoARIMAModel(BaseForecastModel):
    """pmdarima AutoARIMA in raw sales units."""

    name = "auto_arima"

    def __init__(self, config: Optional[TSForecastConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._model: Any = None
        self._n: int = 0
        self._seasonal: bool = False

    def fit(self, train_series: pd.Series) -> "AutoARIMAModel":
        values = finite_values(train_series, model_name=self.name)
        n = int(len(values))
        if n < 3:
            raise ModelUnavailableError(
                f"auto_arima requires at least 3 observations, got {n}",
                model_name=self.name,
                details={"n": n},
            )
        seasonal = use_seasonal(n, self.config)
        self._seasonal = seasonal
        self._n = n
        self._model = _fit_auto_arima(
            values,
            seasonal=seasonal,
            m=int(self.config.seasonal_period),
        )
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._model is None:
            raise ModelUnavailableError("auto_arima is not fitted", model_name=self.name)
        # Exactly ``horizon`` periods — never request horizon+1 and drop one.
        raw = self._model.predict(n_periods=int(horizon))
        preds = tuple(float(x) for x in raw)
        return point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={
                "n_train": self._n,
                "seasonal": self._seasonal,
                "m": int(self.config.seasonal_period) if self._seasonal else None,
            },
        )
