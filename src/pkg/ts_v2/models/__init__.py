"""Forecasting model registry and shared interface for V2.

Simple baselines (naive, seasonal naive, drift) are registered on import.
ARIMA / ETS / Prophet are not implemented yet.
"""
from __future__ import annotations

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

__all__ = [
    "BaseForecastModel",
    "DriftModel",
    "ForecastModel",
    "ForecastResult",
    "ModelContractError",
    "ModelFailure",
    "ModelFailureError",
    "ModelUnavailableError",
    "ModelOutcome",
    "NaiveModel",
    "REGISTRY",
    "SeasonalNaiveModel",
    "available_models",
    "get_model",
    "is_failure",
    "is_success",
    "models_from_config",
    "register_model",
    "run_model",
    "validate_forecast_result",
]
