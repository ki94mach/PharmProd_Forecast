"""Prophet adapter for V2.

Linear growth, month-start timestamps, exact target window. No ×0.8 haircut,
no ``freq='M'``, no ``periods=len(history)+16`` padding, no cap/floor.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig, use_seasonal
from pkg.ts_v2.dates import shamsi_months_to_ms_index, shamsi_to_month_start_timestamp
from pkg.ts_v2.models.base import BaseForecastModel
from pkg.ts_v2.models.common import finite_values, point_forecast
from pkg.ts_v2.models.errors import ModelContractError, ModelUnavailableError
from pkg.ts_v2.types import ForecastResult


def build_prophet_future(
    train_ds: Sequence[pd.Timestamp],
    target_dates: Sequence[int],
) -> pd.DataFrame:
    """History ∪ requested target month-starts; never extends past last target."""
    hist = pd.DatetimeIndex(pd.to_datetime(list(train_ds))).normalize()
    targets = shamsi_months_to_ms_index(target_dates)
    last_target = targets.max()
    # History ∪ targets; never extend past the last requested target month.
    combined = hist.union(targets).sort_values()
    combined = combined[combined <= last_target]
    return pd.DataFrame({"ds": combined})


class ProphetModel(BaseForecastModel):
    """Prophet in raw sales units with a stable CV/final specification."""

    name = "prophet"

    def __init__(self, config: Optional[TSForecastConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._model: Any = None
        self._train_ds: Optional[pd.DatetimeIndex] = None
        self._n: int = 0
        self._yearly: bool = False

    def fit(self, train_series: pd.Series) -> "ProphetModel":
        from prophet import Prophet

        values = finite_values(train_series, model_name=self.name)
        n = int(len(values))
        if n < 2:
            raise ModelUnavailableError(
                f"prophet requires at least 2 observations, got {n}",
                model_name=self.name,
                details={"n": n},
            )
        index = train_series.index
        if len(index) != n:
            raise ModelUnavailableError(
                "prophet train index length mismatch",
                model_name=self.name,
            )
        try:
            shamsi_idx = [int(x) for x in index]
            ds = shamsi_months_to_ms_index(shamsi_idx)
        except (TypeError, ValueError):
            # Fallback if callers passed a DatetimeIndex already.
            ds = pd.DatetimeIndex(pd.to_datetime(index)).normalize()

        yearly = use_seasonal(n, self.config)
        growth = self.config.prophet_growth
        if growth != "linear":
            raise ModelUnavailableError(
                f"unsupported prophet_growth={growth!r}; V2 baseline is linear only",
                model_name=self.name,
            )
        model = Prophet(
            growth="linear",
            yearly_seasonality=yearly,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=float(self.config.prophet_changepoint_prior_scale),
        )
        frame = pd.DataFrame({"ds": ds, "y": values})
        model.fit(frame)
        self._model = model
        self._train_ds = ds
        self._n = n
        self._yearly = yearly
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        if self._model is None or self._train_ds is None:
            raise ModelUnavailableError("prophet is not fitted", model_name=self.name)
        if len(target_dates) != int(horizon):
            raise ModelContractError(
                f"horizon {horizon} != len(target_dates) {len(target_dates)}",
                model_name=self.name,
            )
        future = build_prophet_future(self._train_ds, target_dates)
        last_target = shamsi_to_month_start_timestamp(int(target_dates[-1]))
        if future["ds"].max() > last_target:
            raise ModelContractError(
                "prophet future extends beyond requested target window",
                model_name=self.name,
                details={
                    "future_max": str(future["ds"].max()),
                    "last_target": str(last_target),
                },
            )
        forecast = self._model.predict(future)
        target_ts = shamsi_months_to_ms_index(target_dates)
        by_ds = forecast.set_index("ds")["yhat"]
        try:
            preds = tuple(float(by_ds.loc[ts]) for ts in target_ts)
        except KeyError as exc:
            raise ModelContractError(
                "prophet forecast missing one or more target dates",
                model_name=self.name,
                details={"missing": str(exc)},
            ) from exc
        return point_forecast(
            self.name,
            preds,
            target_dates,
            metadata={
                "n_train": self._n,
                "yearly_seasonality": self._yearly,
                "changepoint_prior_scale": float(
                    self.config.prophet_changepoint_prior_scale
                ),
                "growth": "linear",
                "future_max_ds": str(future["ds"].max()),
            },
        )
