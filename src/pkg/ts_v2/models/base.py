"""Common V2 forecasting model interface.

Every candidate is called the same way by backtest and engine::

    outcome = run_model(model, train_series, window)

``window.target_dates`` is authoritative. Models must not skip the first
forecast month, alter dates, round, smooth by quarter, or apply ad-hoc bias.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Protocol, Sequence, Union

import pandas as pd

from pkg.ts_v2.models.errors import ModelContractError, ModelFailureError
from pkg.ts_v2.types import ForecastResult, ForecastWindow, ModelFailure, ModelOutcome


class ForecastModel(Protocol):
    """Structural interface: ``fit`` then ``predict(horizon, target_dates)``."""

    name: str

    def fit(self, train_series: pd.Series) -> "ForecastModel":
        ...

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        ...


class BaseForecastModel(ABC):
    """Base class for V2 univariate models.

    Subclasses implement :meth:`fit` and :meth:`predict` only. Callers should
    use :func:`run_model` so contract checks and failure wrapping are shared.
    """

    name: str = "unnamed"

    @abstractmethod
    def fit(self, train_series: pd.Series) -> "BaseForecastModel":
        """Fit on raw training history (``date < origin`` already applied)."""

    @abstractmethod
    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        """Return one value per ``target_dates`` entry (length ``horizon``).

        Must copy ``target_dates`` unchanged into :class:`ForecastResult`.
        Must not skip ``target_dates[0]``. Must not round or smooth.
        """

    def forecast(
        self,
        train_series: pd.Series,
        horizon: int,
        target_dates: Sequence[int],
    ) -> ForecastResult:
        """Convenience: ``fit`` then ``predict`` (no failure wrapping)."""
        self.fit(train_series)
        return self.predict(horizon, target_dates)


def _as_float_tuple(values: Sequence[Any], *, field: str, model_name: str) -> tuple[float, ...]:
    out: list[float] = []
    for i, v in enumerate(values):
        try:
            x = float(v)
        except (TypeError, ValueError) as exc:
            raise ModelContractError(
                f"{field}[{i}] is not numeric: {v!r}",
                model_name=model_name,
                details={"field": field, "index": i},
            ) from exc
        if x != x:  # NaN
            raise ModelContractError(
                f"{field}[{i}] is NaN",
                model_name=model_name,
                details={"field": field, "index": i},
            )
        out.append(x)
    return tuple(out)


def validate_forecast_result(
    result: ForecastResult,
    *,
    target_dates: Sequence[int],
    horizon: int,
) -> ForecastResult:
    """Enforce length and date identity; return a frozen-normalized result."""
    name = result.model_name
    requested = tuple(int(d) for d in target_dates)
    if horizon != len(requested):
        raise ModelContractError(
            f"requested horizon {horizon} != len(target_dates) {len(requested)}",
            model_name=name,
        )
    got_dates = tuple(int(d) for d in result.target_dates)
    if got_dates != requested:
        raise ModelContractError(
            "models must not alter target_dates "
            f"(got {got_dates!r}, expected {requested!r})",
            model_name=name,
            details={"got": got_dates, "expected": requested},
        )
    preds = _as_float_tuple(result.predictions, field="predictions", model_name=name)
    if len(preds) != horizon:
        raise ModelContractError(
            f"predictions length {len(preds)} != requested horizon {horizon}",
            model_name=name,
        )
    horizons = tuple(int(h) for h in result.horizons)
    if not horizons:
        horizons = tuple(range(1, horizon + 1))
    if len(horizons) != horizon:
        raise ModelContractError(
            f"horizons length {len(horizons)} != requested horizon {horizon}",
            model_name=name,
        )

    lower = None
    if result.lower is not None:
        lower = _as_float_tuple(result.lower, field="lower", model_name=name)
        if len(lower) != horizon:
            raise ModelContractError(
                f"lower interval length {len(lower)} != horizon {horizon}",
                model_name=name,
            )
    upper = None
    if result.upper is not None:
        upper = _as_float_tuple(result.upper, field="upper", model_name=name)
        if len(upper) != horizon:
            raise ModelContractError(
                f"upper interval length {len(upper)} != horizon {horizon}",
                model_name=name,
            )

    meta: Mapping[str, Any] = dict(result.metadata) if result.metadata else {}
    return ForecastResult(
        model_name=name,
        predictions=preds,
        target_dates=requested,
        horizons=horizons,
        metadata=meta,
        lower=lower,
        upper=upper,
    )


def run_model(
    model: ForecastModel,
    train_series: pd.Series,
    window: ForecastWindow,
) -> ModelOutcome:
    """Identical call path for backtest and engine.

    On contract violations or unexpected exceptions, returns :class:`ModelFailure`
    instead of aborting the rest of the SKU's candidate set.
    """
    name = getattr(model, "name", type(model).__name__)
    horizon = len(window.horizons)
    try:
        model.fit(train_series)
        raw = model.predict(horizon, window.target_dates)
        if not isinstance(raw, ForecastResult):
            raise ModelContractError(
                f"predict() must return ForecastResult, got {type(raw).__name__}",
                model_name=name,
            )
        return validate_forecast_result(
            raw, target_dates=window.target_dates, horizon=horizon
        )
    except ModelFailureError as exc:
        return exc.to_failure(default_name=name)
    except Exception as exc:
        return ModelFailure(
            model_name=name,
            reason=str(exc) or repr(exc),
            error_type=type(exc).__name__,
            details={},
        )


def is_failure(outcome: ModelOutcome) -> bool:
    return isinstance(outcome, ModelFailure)


def is_success(outcome: Union[ForecastResult, ModelFailure]) -> bool:
    return isinstance(outcome, ForecastResult)
