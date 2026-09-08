"""V2.1 forecasting engine adapter for the historical backfill runner."""
from __future__ import annotations

import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult
from pkg.ts_v2.config import DEFAULT_CONFIG_V21
from pkg.ts_v2.dates import parse_origin
from pkg.ts_v2.engine import forecast_products


class V21ForecastEngine:
    """Wraps ``pkg.ts_v2.engine.forecast_products`` with ``DEFAULT_CONFIG_V21``.

    Writes under backfill ``…/v2.1/`` via ``name = "v2.1"``. Same contract as
    :class:`~pkg.benchmark.backfill_runner.engines.v2_engine.V2ForecastEngine`,
    with intermittent-model eligibility gating.
    """

    name = "v2.1"

    def forecast_product(self, request: EngineJobRequest) -> EngineJobResult:
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
                    error_message="V2.1 engine returned no ProductFinalForecast",
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
                        "engine": self.name,
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
                        f"V2.1 target_dates mismatch: got={got} "
                        f"expected={request.target_dates}"
                    ),
                    error_type="TargetDateMismatch",
                )
            selection = engine_result.selections.get(request.product)
            extras = {
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
                forecasts=pd.DataFrame(rows),
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
