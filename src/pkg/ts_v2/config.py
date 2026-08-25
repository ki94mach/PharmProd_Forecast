"""Explicit configuration for the V2 time-series forecasting baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SelectionMetric = Literal["mae", "rmse", "mape", "wmape"]
SelectionStrategy = Literal["best_model", "top3_mean", "top3_median", "top3_inverse_mae"]
GapPolicy = Literal["zero", "missing"]
ProphetGrowth = Literal["linear"]


@dataclass(frozen=True)
class TSForecastConfig:
    """Defaults for V2 forecasting. Override per run; do not mutate V1 settings.

    Attributes:
        forecast_horizon: Number of months to forecast from each origin.
        selection_metric: Holdout metric used to pick the winning model.
        seasonal_period: Seasonal cycle length in months (e.g. 12 for yearly).
        seasonal_enable_after_months: Enable annual seasonality when
            ``n > seasonal_enable_after_months`` (V1 used ``len(train) > 24``).
        min_train_months: Minimum history required before fitting a model.
        nonnegative_forecasts: If True, apply ``max(forecast, 0)`` centrally in
            :mod:`pkg.ts_v2.postprocess` after model/ensemble combination (never
            inside individual models).
        activity_start_min_sales: Start the series at the first month whose
            aggregated sales are **strictly greater** than this value.
            Matches V1 ``sales > 5`` (see note below). ``None`` disables
            activity trimming and keeps history from the first observed month.
        missing_month_policy: How to fill calendar months with no warehouse row
            between ``first_active_month`` and ``last_training_month``.
            ``"zero"`` matches V1 ``asfreq(...).fillna(0)`` for model values;
            ``"missing"`` leaves NaN. In both cases ``is_missing_month`` marks
            calendar gaps so zero demand and absent rows stay distinguishable.
        candidate_models: Registry names the backtest/engine will instantiate.
        prophet_changepoint_prior_scale: Fixed Prophet CPS (same for CV and refit).
        prophet_growth: Prophet growth mode (V2 baseline: linear only).
        croston_alpha: Smoothing for Croston SBA demand size (and interval if
            ``croston_beta`` is None). Fixed a priori — not tuned on holdout.
        croston_beta: Smoothing for Croston SBA inter-demand interval; defaults
            to ``croston_alpha`` when None.
        tsb_alpha: TSB demand-size smoothing parameter.
        tsb_beta: TSB demand-probability smoothing parameter.
        selection_tie_tolerance: When candidate ``selection_mae`` scores are within
            this absolute tolerance of the best score, prefer the simpler model
            (see ``selection_simplicity_order``).
        min_selection_origins: Minimum backtest origins required for a model to
            enter selection for a SKU.
        min_selection_predictions: Minimum out-of-fold prediction rows required
            for a model to enter selection for a SKU.
        selection_simplicity_order: Deterministic tie-break preference (lower index
            wins when scores are tied within ``selection_tie_tolerance``).
        selection_strategy: Forecast strategy for production (analysis compares
            all options; default remains ``best_model`` until empirically validated).
        ensemble_top_k: Number of models combined in top-k ensemble strategies.

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
    seasonal_enable_after_months: int = 24
    min_train_months: int = 12
    nonnegative_forecasts: bool = True
    activity_start_min_sales: Optional[float] = 5.0
    missing_month_policy: GapPolicy = "zero"
    candidate_models: tuple[str, ...] = (
        "naive",
        "seasonal_naive",
        "drift",
        "auto_arima",
        "ets",
        "prophet",
        "croston_sba",
        "tsb",
    )
    prophet_changepoint_prior_scale: float = 0.05
    prophet_growth: ProphetGrowth = "linear"
    croston_alpha: float = 0.1
    croston_beta: Optional[float] = None
    tsb_alpha: float = 0.1
    tsb_beta: float = 0.1
    selection_tie_tolerance: float = 1e-6
    min_selection_origins: int = 1
    min_selection_predictions: int = 1
    selection_simplicity_order: tuple[str, ...] = (
        "seasonal_naive",
        "naive",
        "drift",
        "ets",
        "auto_arima",
        "croston_sba",
        "tsb",
        "prophet",
    )
    selection_strategy: SelectionStrategy = "best_model"
    ensemble_top_k: int = 3


DEFAULT_CONFIG = TSForecastConfig()


def use_seasonal(n: int, config: Optional[TSForecastConfig] = None) -> bool:
    """True when history is long enough for annual seasonality (V1: ``n > 24``)."""
    cfg = config or DEFAULT_CONFIG
    return int(n) > int(cfg.seasonal_enable_after_months)
