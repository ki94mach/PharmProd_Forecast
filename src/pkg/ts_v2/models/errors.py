"""Typed model failures (one candidate can fail without aborting the SKU)."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from pkg.ts_v2.types import ModelFailure


class ModelFailureError(Exception):
    """Raised inside a model; :func:`run_model` converts this to ``ModelFailure``."""

    def __init__(
        self,
        reason: str,
        *,
        model_name: str = "",
        error_type: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.model_name = model_name
        self.error_type = error_type or type(self).__name__
        self.details = dict(details or {})

    def to_failure(self, default_name: str = "") -> ModelFailure:
        return ModelFailure(
            model_name=self.model_name or default_name,
            reason=self.reason,
            error_type=self.error_type,
            details=self.details,
        )


class ModelContractError(ModelFailureError):
    """The model returned a forecast that violates the V2 date/length contract."""


class ModelUnavailableError(ModelFailureError):
    """The candidate cannot be used on this series (e.g. insufficient history).

    Do not silently substitute another algorithm.
    """

    def __init__(
        self,
        reason: str,
        *,
        model_name: str = "",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            reason,
            model_name=model_name,
            error_type="ModelUnavailable",
            details=details,
        )
