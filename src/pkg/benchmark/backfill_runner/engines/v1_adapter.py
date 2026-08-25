"""V1 forecasting adapter for historical backfill (does not modify V1 logic).

The runner truncates sales with ``date < forecast_origin`` (Shamsi) **before**
calling this adapter. V1 ``SalesForecast`` still applies its legacy internal
date handling (+62100 offset, ``max(history)+1`` forecast start, ``[:-1]``
transforms, smoothing). Those behaviors are preserved intentionally.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobRequest, EngineJobResult
from pkg.ts_v2.dates import SHAMSI_TO_PANDAS_YYYYMM_OFFSET, pandas_yyyymm_to_shamsi


class V1ForecastEngine:
    """Adapter around legacy ``pkg.forecast.SalesForecast``."""

    name = "v1"

    def forecast_product(self, request: EngineJobRequest) -> EngineJobResult:
        try:
            from pkg.forecast import SalesForecast
        except Exception as exc:  # noqa: BLE001
            return EngineJobResult(
                success=False,
                product=request.product,
                quarter=request.quarter,
                forecast_origin=request.forecast_origin,
                error_message=f"V1 import failed: {exc}",
                error_type=type(exc).__name__,
            )

        try:
            sale_df = _prepare_v1_sale_frame(request)
            if sale_df.empty:
                return EngineJobResult(
                    success=False,
                    product=request.product,
                    quarter=request.quarter,
                    forecast_origin=request.forecast_origin,
                    error_message="no pre-origin sales rows for V1 adapter",
                    error_type="EmptyHistory",
                )

            with tempfile.TemporaryDirectory(prefix="v1_backfill_") as tmp:
                out_csv = str(Path(tmp) / "forecast.csv")
                # Write header-only file so save_csv append path works if called.
                pd.DataFrame(
                    columns=[
                        "product",
                        "product_fa",
                        "date",
                        "provider",
                        "dep",
                        "status",
                        "forecast",
                        "model",
                    ]
                ).to_csv(out_csv, index=False, encoding="utf-8-sig")

                sf = SalesForecast(request.product, sale_df, out_csv)
                sf.preprocess_data()
                if len(sf.sale_series) < 4:
                    return EngineJobResult(
                        success=False,
                        product=request.product,
                        quarter=request.quarter,
                        forecast_origin=request.forecast_origin,
                        error_message="V1 history too short after preprocess (<4)",
                        error_type="InsufficientHistory",
                    )
                sf.model_selection()
                sf.predict()
                try:
                    sf.redistribute_smoothing()
                except Exception:
                    # Preserve V1 production path: smoothing errors are rare;
                    # keep raw predict output if smoothing blows up.
                    pass

                forecasts = _extract_v1_forecasts(sf, request)
                return EngineJobResult(
                    success=True,
                    product=request.product,
                    quarter=request.quarter,
                    forecast_origin=request.forecast_origin,
                    selected_model=str(getattr(sf, "best_model_type", None)),
                    forecasts=forecasts,
                    extras={
                        "v1_legacy_date_offset": SHAMSI_TO_PANDAS_YYYYMM_OFFSET,
                        "note": (
                            "V1 may start forecast at max(history)+1 internally; "
                            "runner already enforced date < forecast_origin."
                        ),
                    },
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


def _prepare_v1_sale_frame(request: EngineJobRequest) -> pd.DataFrame:
    """Convert runner Shamsi frame to V1's +62100 date convention."""
    src = request.training_sales.copy()
    if src.empty:
        return src
    # Hard guard (runner should already have filtered).
    src = src.loc[src["date"].astype(int) < int(request.forecast_origin)].copy()
    if src.empty:
        return src

    out = pd.DataFrame(
        {
            "product": request.product,
            "date": src["date"].astype(int) + SHAMSI_TO_PANDAS_YYYYMM_OFFSET,
            "sales": pd.to_numeric(src["sales"], errors="coerce").fillna(0.0),
        }
    )
    # V1 SalesForecast requires these columns for metadata.
    for col, default in (
        ("product_fa", request.meta.get("product_fa", request.product)),
        ("provider", request.meta.get("provider", "")),
        ("dep", request.meta.get("dep", request.meta.get("field", ""))),
        ("boxq", request.meta.get("boxq", 1)),
    ):
        out[col] = default if col not in src.columns else src[col].iloc[0]
    return out


def _extract_v1_forecasts(sf, request: EngineJobRequest) -> pd.DataFrame:
    """Map V1 forecast array onto the shared 15 target dates when possible."""
    values = np.asarray(sf.forecast, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("V1 produced an empty forecast")

    # Prefer aligning to request target_dates (canonical experiment contract).
    # If V1 emitted a different length, truncate/pad with NaN and record in frame.
    rows = []
    for h, target in enumerate(request.target_dates, start=1):
        pred = float(values[h - 1]) if h - 1 < len(values) else float("nan")
        rows.append(
            {
                "product": request.product,
                "quarter": request.quarter,
                "forecast_origin": int(request.forecast_origin),
                "target_date": int(target),
                "horizon": int(h),
                "forecast": pred,
                "model": str(getattr(sf, "best_model_type", "")),
                "engine": "v1",
            }
        )
    # Also expose V1's internal forecast index months when available.
    if hasattr(sf, "forecast_index") and sf.forecast_index is not None:
        v1_months = []
        for ts in list(sf.forecast_index)[:15]:
            try:
                yyyymm = int(pd.Timestamp(ts).strftime("%Y%m"))
                v1_months.append(pandas_yyyymm_to_shamsi(yyyymm))
            except Exception:
                v1_months.append(None)
        for i, row in enumerate(rows):
            if i < len(v1_months):
                row["v1_internal_target_date"] = v1_months[i]
    return pd.DataFrame(rows)
