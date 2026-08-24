"""Forecasting model registry and shared interface for V2.

Production models (ARIMA / ETS / Prophet) are not implemented yet.
Register candidates on :data:`REGISTRY` and list them in
``TSForecastConfig.candidate_models``.
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
from pkg.ts_v2.models.errors import ModelContractError, ModelFailureError
from pkg.ts_v2.models.registry import (
    REGISTRY,
    available_models,
    get_model,
    models_from_config,
    register_model,
)
from pkg.ts_v2.types import ForecastResult, ModelFailure, ModelOutcome

__all__ = [
    "BaseForecastModel",
    "ForecastModel",
    "ForecastResult",
    "ModelContractError",
    "ModelFailure",
    "ModelFailureError",
    "ModelOutcome",
    "REGISTRY",
    "available_models",
    "get_model",
    "is_failure",
    "is_success",
    "models_from_config",
    "register_model",
    "run_model",
    "validate_forecast_result",
]
