"""Dummy forecasting engine for orchestration tests (no real model fitting)."""
from __future__ import annotations

import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult


class DummyForecastEngine:
    """Deterministic constant-level forecasts for integration tests."""

    name = "dummy"

    def __init__(self, *, fail_products: frozenset[str] | None = None, level: float = 10.0):
        self.fail_products = fail_products or frozenset()
        self.level = float(level)

    def forecast_product(self, request: EngineJobRequest) -> EngineJobResult:
        if request.product in self.fail_products:
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                error_message=f"dummy forced failure for {request.product}",
                error_type="DummyFailure",
            )

        # Guard: refuse post-origin rows if caller forgot the cutoff.
        if not request.training_sales.empty:
            bad = request.training_sales.loc[
                request.training_sales["date"].astype(int) >= int(request.forecast_origin)
            ]
            if not bad.empty:
                return EngineJobResult(
                    success=False,
                    product=request.product,
                    quarter=request.quarter,
                    forecast_origin=request.forecast_origin,
                    error_message="training_sales contains date >= forecast_origin",
                    error_type="CutoffLeakage",
                )

        rows = []
        for h, target in enumerate(request.target_dates, start=1):
            rows.append(
                {
                    "product": request.product,
                    "quarter": request.quarter,
                    "forecast_origin": int(request.forecast_origin),
                    "target_date": int(target),
                    "horizon": int(h),
                    "forecast": self.level,
                    "model": "dummy_constant",
                    "engine": self.name,
                }
            )
        return EngineJobResult(
            success=True,
            product=request.product,
            quarter=request.quarter,
            forecast_origin=request.forecast_origin,
            selected_model="dummy_constant",
            forecasts=pd.DataFrame(rows),
            extras={"n_train_rows": int(len(request.training_sales))},
        )
