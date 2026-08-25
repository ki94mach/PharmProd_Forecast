"""Forecasting model registry and shared interface for V2.

Baselines (naive, seasonal naive, drift) and library adapters
(auto_arima, ets, prophet) are registered on import.
"""
from __future__ import annotations

from pkg.ts_v2.models.auto_arima import AutoARIMAModel
from pkg.ts_v2.models.base import (
    BaseForecastModel,
    ForecastModel,
    is_failure,
    is_success,
    run_model,
    validate_forecast_result,
)
from pkg.ts_v2.models.baselines import DriftModel, NaiveModel, SeasonalNaiveModel
from pkg.ts_v2.models.errors import ModelContractError, ModelFailureError, ModelUnavailableError
from pkg.ts_v2.models.ets import ETSModelAdapter, ets_kwargs
from pkg.ts_v2.models.prophet import ProphetModel, build_prophet_future
from pkg.ts_v2.models.registry import (
    REGISTRY,
    available_models,
    get_model,
    models_from_config,
    register_model,
)
from pkg.ts_v2.types import ForecastResult, ModelFailure, ModelOutcome

register_model("naive", NaiveModel, replace=True)
register_model("seasonal_naive", SeasonalNaiveModel, replace=True)
register_model("drift", DriftModel, replace=True)
register_model("auto_arima", AutoARIMAModel, replace=True)
register_model("ets", ETSModelAdapter, replace=True)
register_model("prophet", ProphetModel, replace=True)

__all__ = [
    "AutoARIMAModel",
    "BaseForecastModel",
    "DriftModel",
    "ETSModelAdapter",
    "ForecastModel",
    "ForecastResult",
    "ModelContractError",
    "ModelFailure",
    "ModelFailureError",
    "ModelUnavailableError",
    "ModelOutcome",
    "NaiveModel",
    "ProphetModel",
    "REGISTRY",
    "SeasonalNaiveModel",
    "available_models",
    "build_prophet_future",
    "ets_kwargs",
    "get_model",
    "is_failure",
    "is_success",
    "models_from_config",
    "register_model",
    "run_model",
    "validate_forecast_result",
]
