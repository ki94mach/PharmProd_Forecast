"""Centralized V2 forecast post-processing (production path only).

V2 applies a single business rule after all model/ensemble combination:

    constrained = max(raw, 0)   when ``nonnegative_forecasts`` is enabled

Forbidden in V2 (V1-only; kept for historical reproducibility):

- ``redistribute_smoothing`` (quarterly / 3-month smoothing)
- Prophet ×0.8 or other model-specific bias haircuts
- ``replace_negative_sales`` and other model-specific negative replacement
- Internal rounding (rounding belongs at legacy export only)
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import ConstrainedForecastResult, ForecastResult

# V1 callables that must never run on the V2 path.
V2_FORBIDDEN_POSTPROCESS_NAMES: frozenset[str] = frozenset(
    {
        "redistribute_smoothing",
        "replace_negative_sales",
    }
)


def assert_v2_postprocess_allowed(name: str) -> None:
    """Raise if ``name`` is a forbidden V1 post-processing step."""
    key = str(name)
    if key in V2_FORBIDDEN_POSTPROCESS_NAMES:
        raise RuntimeError(
            f"V2 forbids V1 post-processing {key!r}; "
            "use apply_final_constraints() instead"
        )


def apply_nonnegativity(values: Sequence[float]) -> tuple[float, ...]:
    """Common business rule: ``max(value, 0)`` without rounding."""
    return tuple(max(0.0, float(v)) for v in values)


def apply_final_constraints(
    result: ForecastResult,
    *,
    config: Optional[TSForecastConfig] = None,
) -> ConstrainedForecastResult:
    """Apply centralized final constraints to raw model/ensemble output."""
    cfg = config or DEFAULT_CONFIG
    raw = tuple(float(p) for p in result.predictions)
    if cfg.nonnegative_forecasts:
        constrained = apply_nonnegativity(raw)
    else:
        constrained = raw
    meta = dict(result.metadata) if result.metadata else {}
    meta["postprocess"] = {
        "nonnegative_forecasts": bool(cfg.nonnegative_forecasts),
        "policy": "max(raw, 0)" if cfg.nonnegative_forecasts else "none",
    }
    meta["nonneg_adjustment"] = nonneg_adjustment_summary(raw, constrained)
    return ConstrainedForecastResult(
        model_name=str(result.model_name),
        raw_predictions=raw,
        constrained_predictions=constrained,
        target_dates=tuple(int(d) for d in result.target_dates),
        horizons=tuple(int(h) for h in result.horizons),
        metadata=meta,
        lower=result.lower,
        upper=result.upper,
    )


def nonneg_adjustment_summary(
    raw: Sequence[float],
    constrained: Sequence[float],
) -> Mapping[str, float | int]:
    """Summarize how many horizons the nonneg constraint changed."""
    changed = 0
    total_delta = 0.0
    for a, b in zip(raw, constrained):
        af = float(a)
        bf = float(b)
        if af != bf:
            changed += 1
            total_delta += bf - af
    return {
        "n_adjusted": int(changed),
        "total_delta": float(total_delta),
    }


def export_quantities(
    processed: ConstrainedForecastResult,
    *,
    round_for_legacy: bool = False,
) -> tuple[float, ...]:
    """Legacy/export boundary: optionally round constrained forecasts to integers."""
    vals = processed.constrained_predictions
    if round_for_legacy:
        return tuple(float(round(v)) for v in vals)
    return tuple(float(v) for v in vals)


def export_forecast_frame_rows(
    processed: ConstrainedForecastResult,
    *,
    product: str,
    round_for_legacy: bool = False,
) -> list[dict]:
    """Row dicts for CSV/Excel export with raw and constrained columns."""
    export_vals = export_quantities(processed, round_for_legacy=round_for_legacy)
    rows: list[dict] = []
    for h, target, raw, constrained, exported in zip(
        processed.horizons,
        processed.target_dates,
        processed.raw_predictions,
        processed.constrained_predictions,
        export_vals,
    ):
        rows.append(
            {
                "product": str(product),
                "model": processed.model_name,
                "horizon": int(h),
                "target_date": int(target),
                "raw_forecast": float(raw),
                "constrained_forecast": float(constrained),
                "export_quantity": float(exported),
            }
        )
    return rows
