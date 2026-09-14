"""Architecture adapters for V2.1 / A0 / A1 forecast_jobs."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult
from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import DEFAULT_CONFIG_V21
from pkg.ts_v2.dates import parse_origin
from pkg.ts_v2.engine import forecast_products
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import (
    STATUS_ERROR,
    STATUS_UNAVAILABLE,
    backtest_product_architectures,
)
from pkg.ts_v3a.config import DEFAULT_CONFIG as NEURAL_DEFAULT
from pkg.ts_v3a.screen import parse_architecture_aliases


_ARCH_ALIAS = {
    "a0": ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,
    "a1": ArchitectureName.A1_SMALL_RECURSIVE_LSTM,
}

# Shared join contract with V2.1 / backfill ``forecast.csv`` rows.
V21_COMPAT_FORECAST_COLUMNS = (
    "product",
    "quarter",
    "forecast_origin",
    "target_date",
    "horizon",
    "forecast",
    "raw_forecast",
    "model",
    "engine",
)

# Optional neural diagnostics kept after the V2.1 join keys (do not rename joins).
_NEURAL_EXTRA_COLUMNS = (
    "architecture",
    "seed",
    "actual",
    "prediction_kind",
    "product_id",
)

FALLBACK_MODEL = "naive_last_value"
FALLBACK_POLICY_NONE = "none"
FALLBACK_POLICY_NAIVE = "naive_last_value"

# Ineligibility reason codes (stable for triage / eval versioning).
REASON_V21_NO_ELIGIBLE_MODELS = "v21_no_eligible_models"
REASON_NEURAL_SHORT_HISTORY = "neural_short_history_ineligible"
REASON_NEURAL_EMPTY_PREDICTIONS = "neural_empty_predictions_ineligible"


def target_dates_for_origin(forecast_origin: int, horizon: int) -> tuple[int, ...]:
    origin = int(forecast_origin)
    return tuple(shamsi_add_months(origin, i) for i in range(int(horizon)))


def neural_predictions_to_v21_frame(
    preds: pd.DataFrame,
    *,
    request: EngineJobRequest,
    architecture: str,
    seed: int,
    model_name: str,
) -> pd.DataFrame:
    """Map screening-style OOF columns onto the V2.1 join contract.

    Renames ``origin`` → ``forecast_origin`` and ``prediction`` → ``forecast``,
    sets ``raw_forecast`` equal to the delivered forecast when no separate raw
    series exists, and drops the old alias columns so joins stay unambiguous.
    """
    if preds is None or preds.empty:
        return pd.DataFrame(columns=list(V21_COMPAT_FORECAST_COLUMNS))

    work = preds.copy()
    if "forecast_origin" not in work.columns:
        if "origin" not in work.columns:
            raise ValueError("neural predictions missing origin/forecast_origin")
        work["forecast_origin"] = pd.to_numeric(work["origin"], errors="coerce").astype(
            int
        )
    else:
        work["forecast_origin"] = pd.to_numeric(
            work["forecast_origin"], errors="coerce"
        ).astype(int)

    if "forecast" not in work.columns:
        if "prediction" not in work.columns:
            raise ValueError("neural predictions missing prediction/forecast")
        work["forecast"] = pd.to_numeric(work["prediction"], errors="coerce")
    else:
        work["forecast"] = pd.to_numeric(work["forecast"], errors="coerce")

    if "raw_forecast" not in work.columns:
        work["raw_forecast"] = work["forecast"]
    else:
        work["raw_forecast"] = pd.to_numeric(work["raw_forecast"], errors="coerce")

    if "product" in work.columns:
        work["product"] = work["product"].astype(str)
    else:
        work["product"] = str(request.product)
    work["quarter"] = str(request.quarter)
    work["engine"] = str(architecture).lower()
    work["model"] = str(model_name)
    work["target_date"] = pd.to_numeric(work["target_date"], errors="coerce").astype(int)
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype(int)
    if "seed" not in work.columns:
        work["seed"] = int(seed)

    # Drop screening aliases so consumers cannot join on the wrong name.
    drop_cols = [c for c in ("origin", "prediction") if c in work.columns]
    if drop_cols:
        work = work.drop(columns=drop_cols)

    extras = [c for c in _NEURAL_EXTRA_COLUMNS if c in work.columns]
    ordered: Sequence[str] = list(V21_COMPAT_FORECAST_COLUMNS) + extras
    return work.loc[:, list(ordered)].copy()


def _fallback_policy(meta: Mapping[str, Any]) -> str:
    raw = str(meta.get("fallback_policy", FALLBACK_POLICY_NAIVE)).strip().lower()
    return raw or FALLBACK_POLICY_NAIVE


def _fallback_version(meta: Mapping[str, Any]) -> str:
    return str(meta.get("fallback_version", "v1")).strip() or "v1"


def last_training_value(training_sales: pd.DataFrame, product: str) -> Optional[float]:
    """Return the last finite training sales value, or None if unavailable.

    Never invents a value: missing / non-finite history yields None (caller must
    fail rather than substitute zero for a missing forecast).
    """
    if training_sales is None or training_sales.empty:
        return None
    work = training_sales.copy()
    if "product" in work.columns:
        work = work.loc[work["product"].astype(str) == str(product)]
    if work.empty or "sales" not in work.columns or "date" not in work.columns:
        return None
    work = work.sort_values("date")
    values = pd.to_numeric(work["sales"], errors="coerce").to_numpy(dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return None
    return float(values[int(finite_idx[-1])])


def build_naive_last_value_forecasts(
    request: EngineJobRequest,
    *,
    engine: str,
    last_value: float,
    seed: Optional[int] = None,
    architecture: Optional[str] = None,
) -> pd.DataFrame:
    """Repeat the last observed training value across all target horizons."""
    rows = []
    for i, target in enumerate(request.target_dates, start=1):
        row: dict[str, Any] = {
            "product": str(request.product),
            "quarter": str(request.quarter),
            "forecast_origin": int(request.forecast_origin),
            "target_date": int(target),
            "horizon": int(i),
            "forecast": float(last_value),
            "raw_forecast": float(last_value),
            "model": FALLBACK_MODEL,
            "engine": str(engine).lower(),
        }
        if architecture is not None:
            row["architecture"] = str(architecture)
        if seed is not None:
            row["seed"] = int(seed)
        rows.append(row)
    cols = list(V21_COMPAT_FORECAST_COLUMNS)
    if architecture is not None:
        cols = cols + ["architecture"]
    if seed is not None:
        cols = cols + (["seed"] if "seed" not in cols else [])
    return pd.DataFrame(rows, columns=cols)


def _ineligible_result(
    request: EngineJobRequest,
    *,
    engine: str,
    reason_code: str,
    reason_detail: str,
    error_type: str,
    seed: Optional[int] = None,
    architecture: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> EngineJobResult:
    """Build INELIGIBLE + explicit naive-last-value fallback, or FAILED if none."""
    policy = _fallback_policy(request.meta)
    version = _fallback_version(request.meta)
    base_extras: dict[str, Any] = {
        "ineligible": True,
        "ineligibility_reason": reason_code,
        "ineligibility_detail": reason_detail,
        "fallback_policy": policy,
        "fallback_version": version,
        "fallback_reason": f"{reason_code}|policy={policy}|version={version}",
    }
    if architecture is not None:
        base_extras["architecture"] = architecture
    if seed is not None:
        base_extras["seed"] = int(seed)
    if extra:
        base_extras.update(dict(extra))

    if policy == FALLBACK_POLICY_NONE:
        return EngineJobResult(
            success=False,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            error_message=reason_detail,
            error_type=error_type,
            extras=base_extras,
            ineligible=False,
        )

    if policy != FALLBACK_POLICY_NAIVE:
        return EngineJobResult(
            success=False,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            error_message=f"unsupported fallback_policy {policy!r}",
            error_type="UnsupportedFallbackPolicy",
            extras=base_extras,
        )

    last = last_training_value(request.training_sales, request.product)
    if last is None:
        return EngineJobResult(
            success=False,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            error_message=(
                f"{reason_detail}; cannot apply {FALLBACK_MODEL} "
                "(no finite training sales — refusing zero fill)"
            ),
            error_type="FallbackUnavailable",
            extras=base_extras,
        )

    forecasts = build_naive_last_value_forecasts(
        request,
        engine=engine,
        last_value=last,
        seed=seed,
        architecture=architecture,
    )
    base_extras["fallback_last_value"] = float(last)
    base_extras["fallback_model"] = FALLBACK_MODEL
    return EngineJobResult(
        success=True,
        product=request.product,
        quarter=request.quarter,
        forecast_origin=request.forecast_origin,
        selected_model=FALLBACK_MODEL,
        forecasts=forecasts,
        error_message=reason_detail,
        error_type=error_type,
        extras=base_extras,
        ineligible=True,
    )


def _is_v21_no_eligible(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "no eligible models for selection" in msg or "insufficient origins" in msg


def _neural_unavailable_reason(fold_metadata: Optional[pd.DataFrame]) -> Optional[str]:
    if fold_metadata is None or fold_metadata.empty:
        return None
    if "status" not in fold_metadata.columns:
        return None
    unavailable = fold_metadata.loc[
        fold_metadata["status"].astype(str) == STATUS_UNAVAILABLE
    ]
    if unavailable.empty:
        return None
    if "unavailable_reason" in unavailable.columns:
        reasons = [
            str(r)
            for r in unavailable["unavailable_reason"].dropna().unique().tolist()
            if str(r).strip()
        ]
        if reasons:
            return "; ".join(reasons)
    return "insufficient_history"


def _neural_error_fold(
    fold_metadata: Optional[pd.DataFrame],
) -> Optional[dict[str, str]]:
    """Return error_type/message when any fold has status=error (not ineligible)."""
    if fold_metadata is None or fold_metadata.empty:
        return None
    if "status" not in fold_metadata.columns:
        return None
    errored = fold_metadata.loc[
        fold_metadata["status"].astype(str) == STATUS_ERROR
    ]
    if errored.empty:
        return None
    row = errored.iloc[0]
    err_type = row.get("error_type")
    err_msg = row.get("error_message")
    return {
        "error_type": str(err_type) if err_type is not None and str(err_type).strip() else "NeuralFoldError",
        "error_message": (
            str(err_msg)
            if err_msg is not None and str(err_msg).strip()
            else "neural fold status=error"
        ),
    }


def run_v21_job(request: EngineJobRequest) -> EngineJobResult:
    """Run one V2.1 product × origin job via ``DEFAULT_CONFIG_V21``."""
    try:
        origin = parse_origin(request.forecast_origin)
        engine_result = forecast_products(
            request.training_sales,
            [request.product],
            origin,
            config=DEFAULT_CONFIG_V21,
        )
        final = engine_result.final_forecasts.get(request.product)
        if final is None:
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                error_message="V2.1 returned no ProductFinalForecast",
                error_type="MissingFinalForecast",
            )
        rows = []
        for hf in final.horizon_forecasts:
            rows.append(
                {
                    "product": request.product,
                    "quarter": request.quarter,
                    "forecast_origin": int(request.forecast_origin),
                    "target_date": int(hf.target_shamsi_yyyymm),
                    "horizon": int(hf.horizon),
                    "forecast": float(hf.constrained_forecast),
                    "raw_forecast": float(hf.raw_forecast),
                    "model": str(final.selected_model),
                    "engine": "v2.1",
                }
            )
        got = tuple(int(r["target_date"]) for r in rows)
        if got != tuple(request.target_dates):
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                selected_model=str(final.selected_model),
                error_message=(
                    f"target_dates mismatch: got={got} expected={request.target_dates}"
                ),
                error_type="TargetDateMismatch",
            )
        selection = engine_result.selections.get(request.product)
        extras: dict[str, Any] = {
            "selected_strategy": final.selected_strategy,
            "n_training_observations": final.n_training_observations,
        }
        if selection is not None:
            extras["fallback_reason"] = selection.fallback_reason
        return EngineJobResult(
            success=True,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            selected_model=str(final.selected_model),
            forecasts=pd.DataFrame(rows, columns=list(V21_COMPAT_FORECAST_COLUMNS)),
            extras=extras,
        )
    except Exception as exc:  # noqa: BLE001 — per-job isolation
        if _is_v21_no_eligible(exc):
            return _ineligible_result(
                request,
                engine="v2.1",
                reason_code=REASON_V21_NO_ELIGIBLE_MODELS,
                reason_detail=str(exc),
                error_type="ExpectedIneligible",
            )
        return EngineJobResult(
            success=False,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )


def run_neural_job(
    request: EngineJobRequest,
    *,
    architecture: str,
    seed: int,
    full_sales: pd.DataFrame,
) -> EngineJobResult:
    """Run one A0/A1 product × origin × seed outer-backtest job.

    ``full_sales`` should be the panel available for origin discovery / actuals
    (typically the frozen sales frame). Training still ends at
    ``last_complete`` via the production ``ForecastWindow`` inside prepare.
    Persisted ``forecast.csv`` uses the V2.1 join column names.
    """
    try:
        arch = _ARCH_ALIAS.get(str(architecture).lower())
        if arch is None:
            arch = parse_architecture_aliases([architecture])[0]
        neural_cfg = NEURAL_DEFAULT
        # Deterministic seed is passed explicitly into backtest.
        result = backtest_product_architectures(
            full_sales,
            request.product,
            architectures=(arch,),
            seeds=(int(seed),),
            explicit_origins=(int(request.forecast_origin),),
            config=neural_cfg,
            min_successful_seeds=1,
        )
        preds = result.predictions
        if preds is None or preds.empty:
            fold_err = _neural_error_fold(result.fold_metadata)
            if fold_err is not None:
                # Runtime/training defects must be FAILED, not silent INELIGIBLE.
                return EngineJobResult(
                    success=False,
                    product=request.product,
                    quarter=request.quarter,
                    forecast_origin=request.forecast_origin,
                    error_message=fold_err["error_message"],
                    error_type=fold_err["error_type"],
                    extras={
                        "architecture": arch.value,
                        "seed": int(seed),
                        "empty_predictions": True,
                        "fold_status": STATUS_ERROR,
                    },
                )
            unavailable = _neural_unavailable_reason(result.fold_metadata)
            reason_code = (
                REASON_NEURAL_SHORT_HISTORY
                if unavailable
                else REASON_NEURAL_EMPTY_PREDICTIONS
            )
            detail = (
                f"neural backtest returned no predictions"
                + (f" ({unavailable})" if unavailable else "")
            )
            return _ineligible_result(
                request,
                engine=str(architecture).lower(),
                reason_code=reason_code,
                reason_detail=detail,
                error_type="ExpectedIneligible",
                seed=int(seed),
                architecture=arch.value,
                extra={
                    "empty_predictions": True,
                    "unavailable_reason": unavailable,
                },
            )
        model_name = f"{arch.value}:seed{int(seed)}"
        frame = neural_predictions_to_v21_frame(
            preds,
            request=request,
            architecture=str(architecture),
            seed=int(seed),
            model_name=model_name,
        )
        return EngineJobResult(
            success=True,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            selected_model=model_name,
            forecasts=frame,
            extras={
                "architecture": arch.value,
                "seed": int(seed),
                "n_prediction_rows": int(len(frame)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        # Eligibility exceptions from prepare/lookback → expected ineligible.
        name = type(exc).__name__
        msg = str(exc).lower()
        if name in {
            "IneligibleForTraining",
            "InsufficientHistoryError",
            "ModelUnavailableError",
        } or "insufficient" in msg:
            return _ineligible_result(
                request,
                engine=str(architecture).lower(),
                reason_code=REASON_NEURAL_SHORT_HISTORY,
                reason_detail=str(exc),
                error_type="ExpectedIneligible",
                seed=int(seed),
                architecture=str(architecture).lower(),
            )
        return EngineJobResult(
            success=False,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            error_message=str(exc),
            error_type=type(exc).__name__,
            extras={"architecture": architecture, "seed": int(seed)},
        )


def execute_architecture_job(
    *,
    architecture: str,
    request: EngineJobRequest,
    seed: Optional[int],
    sales_for_neural: pd.DataFrame,
) -> EngineJobResult:
    """Dispatch one planned job to the V2.1 or neural adapter."""
    key = str(architecture).lower()
    if key == "v2.1":
        return run_v21_job(request)
    if key in {"a0", "a1"}:
        if seed is None:
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                error_message="seed is required for a0/a1 jobs",
                error_type="MissingSeed",
            )
        return run_neural_job(
            request,
            architecture=key,
            seed=int(seed),
            full_sales=sales_for_neural,
        )
    return EngineJobResult(
        success=False,
        product=request.product,
        quarter=request.quarter,
        forecast_origin=request.forecast_origin,
        error_message=f"unsupported architecture {architecture!r}",
        error_type="UnsupportedArchitecture",
    )
