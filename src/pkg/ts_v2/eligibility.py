"""Centralized intermittency eligibility for V2.1 intermittent models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import pandas as pd

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.intermittency import intermittency_stats
from pkg.ts_v2.types import PreparedSeries


@dataclass(frozen=True)
class CandidateEligibility:
    """Per-candidate intermittency gate outcome."""

    eligible: bool
    reason: str
    zero_month_proportion: Optional[float]
    adi: Optional[float]


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{float(value):.2f}"


def evaluate_intermittent_demand(
    *,
    zero_month_proportion: Optional[float],
    adi: Optional[float],
    min_zero_fraction: float,
    min_adi: float,
) -> tuple[bool, str]:
    """Return (eligible, reason) for intermittent-model routing.

    Eligible iff::

        (zero_month_proportion >= min_zero_fraction)
          OR (ADI is not None AND ADI > min_adi)
    """
    z = zero_month_proportion
    zero_ok = z is not None and float(z) >= float(min_zero_fraction)
    adi_ok = adi is not None and float(adi) > float(min_adi)
    eligible = bool(zero_ok or adi_ok)

    z_s = _format_float(z)
    adi_s = _format_float(adi)
    min_z_s = f"{float(min_zero_fraction):.2f}"
    min_adi_s = f"{float(min_adi):.2f}"

    if eligible:
        parts: list[str] = []
        if zero_ok:
            parts.append(f"zero_fraction={z_s} >= {min_z_s}")
        if adi_ok:
            parts.append(f"adi={adi_s} > {min_adi_s}")
        return True, "intermittent: " + "; ".join(parts)

    z_clause = (
        f"zero_fraction={z_s} < {min_z_s}"
        if z is not None
        else f"zero_fraction=None < {min_z_s}"
    )
    if adi is None:
        adi_clause = f"adi=None (need >=2 demand months); threshold > {min_adi_s}"
    else:
        adi_clause = f"adi={adi_s} <= {min_adi_s}"
    return False, f"not_intermittent: {z_clause}; {adi_clause}"


def stats_from_prepared(
    series: Union[PreparedSeries, pd.Series],
) -> tuple[Optional[float], Optional[float]]:
    """Extract (zero_month_proportion, ADI) from prepared history or raw values."""
    if isinstance(series, PreparedSeries):
        return series.zero_month_proportion, series.average_inter_demand_interval
    stats = intermittency_stats(series)
    return stats.zero_month_proportion, stats.average_inter_demand_interval


def eligibility_for_model(
    model_name: str,
    series: Union[PreparedSeries, pd.Series],
    config: TSForecastConfig,
) -> CandidateEligibility:
    """Gate one model: non-listed names are always eligible."""
    z, adi = stats_from_prepared(series)
    name = str(model_name)
    if name not in tuple(config.intermittent_model_names):
        return CandidateEligibility(
            eligible=True,
            reason="",
            zero_month_proportion=z,
            adi=adi,
        )
    eligible, reason = evaluate_intermittent_demand(
        zero_month_proportion=z,
        adi=adi,
        min_zero_fraction=float(config.intermittent_min_zero_fraction),
        min_adi=float(config.intermittent_min_adi),
    )
    return CandidateEligibility(
        eligible=eligible,
        reason=reason,
        zero_month_proportion=z,
        adi=adi,
    )


def build_candidate_eligibility(
    series: Union[PreparedSeries, pd.Series],
    config: TSForecastConfig,
    *,
    candidate_models: Optional[Sequence[str]] = None,
) -> dict[str, CandidateEligibility]:
    """Eligibility map for every candidate model name."""
    names = (
        tuple(candidate_models)
        if candidate_models is not None
        else tuple(config.candidate_models)
    )
    return {str(name): eligibility_for_model(str(name), series, config) for name in names}


def has_intermittent_gate(config: TSForecastConfig) -> bool:
    """True when config enables intermittency routing."""
    return bool(config.intermittent_model_names)
