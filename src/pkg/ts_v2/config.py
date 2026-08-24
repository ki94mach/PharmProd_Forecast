"""Explicit configuration for the V2 time-series forecasting baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SelectionMetric = Literal["mae", "rmse", "mape", "wmape"]
GapPolicy = Literal["zero", "missing"]


@dataclass(frozen=True)
class TSForecastConfig:
    """Defaults for V2 forecasting. Override per run; do not mutate V1 settings.

    Attributes:
        forecast_horizon: Number of months to forecast from each origin.
        selection_metric: Holdout metric used to pick the winning model.
        seasonal_period: Seasonal cycle length in months (e.g. 12 for yearly).
        min_train_months: Minimum history required before fitting a model.
        nonnegative_forecasts: If True, clip point forecasts at zero at the end.
        activity_start_min_sales: Start the series at the first month whose
            aggregated sales are **strictly greater** than this value.
            Matches V1 ``sales > 5`` (see note below). ``None`` disables
            activity trimming and keeps history from the first observed month.
        missing_month_policy: How to fill calendar months with no warehouse row
            between ``first_active_month`` and ``last_training_month``.
            ``"zero"`` matches V1 ``asfreq(...).fillna(0)`` for model values;
            ``"missing"`` leaves NaN. In both cases ``is_missing_month`` marks
            calendar gaps so zero demand and absent rows stay distinguishable.

    Activity threshold (V1 compatibility)
    ------------------------------------
    V1 ``SalesForecast.preprocess_data`` drops months before the first sale
    ``> 5``. The method docstring says "first non-zero sale", but the live
    threshold is **5**, not 0 — almost certainly to ignore tiny residual /
    sampling / pre-launch shipments that are not meaningful commercial
    activity. V2 keeps that default for production-comparable history length,
    but only behind this named option so experiments can set ``None`` or
    another cutoff without hunting magic numbers in model code.
    """

    forecast_horizon: int = 15
    selection_metric: SelectionMetric = "mae"
    seasonal_period: int = 12
    min_train_months: int = 12
    nonnegative_forecasts: bool = True
    activity_start_min_sales: Optional[float] = 5.0
    missing_month_policy: GapPolicy = "zero"


DEFAULT_CONFIG = TSForecastConfig()
