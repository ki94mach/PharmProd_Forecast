"""Architecture adapters for V2.1 / A0 / A1 forecast_jobs."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult
from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import DEFAULT_CONFIG_V21
from pkg.ts_v2.dates import parse_origin
from pkg.ts_v2.engine import forecast_products
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import backtest_product_architectures
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

    Uses ``full_sales`` (still truncated by the runner cutoff) so neural
    prepare can see the production/screening contract history it needs.
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
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                error_message="neural backtest returned no predictions",
                error_type="EmptyPredictions",
                extras={"architecture": arch.value, "seed": int(seed)},
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
