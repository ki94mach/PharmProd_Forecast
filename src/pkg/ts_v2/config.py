"""Explicit configuration for the V2 time-series forecasting baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SelectionMetric = Literal["mae", "rmse", "mape", "wmape"]


@dataclass(frozen=True)
class TSForecastConfig:
    """Defaults for V2 forecasting. Override per run; do not mutate V1 settings.

    Attributes:
        forecast_horizon: Number of months to forecast from each origin.
        selection_metric: Holdout metric used to pick the winning model.
        seasonal_period: Seasonal cycle length in months (e.g. 12 for yearly).
        min_train_months: Minimum history required before fitting a model.
        nonnegative_forecasts: If True, clip point forecasts at zero at the end.
    """

    forecast_horizon: int = 15
    selection_metric: SelectionMetric = "mae"
    seasonal_period: int = 12
    min_train_months: int = 12
    nonnegative_forecasts: bool = True


DEFAULT_CONFIG = TSForecastConfig()
