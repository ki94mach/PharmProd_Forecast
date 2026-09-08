"""Training cutoff policies for forecast_jobs."""
from __future__ import annotations

import pandas as pd

from pkg.ts_v2.dates import resolve_production_time_contract


def apply_cutoff_policy(
    sales: pd.DataFrame,
    forecast_origin: int,
    *,
    policy_name: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Truncate sales according to the configured cutoff policy.

    ``production``
        Keep ``date <= last_complete_month`` (``forecast_start - 2``).
    ``origin_exclusive``
        Keep ``date < forecast_origin`` (legacy screening-style exclusive cut).
    """
    if sales is None or sales.empty:
        return sales.iloc[0:0].copy() if sales is not None else pd.DataFrame()
    work = sales.copy()
    work[date_col] = pd.to_numeric(work[date_col], errors="coerce").astype(int)
    origin = int(forecast_origin)
    name = str(policy_name).strip().lower()
    if name == "production":
        contract = resolve_production_time_contract(origin)
        cutoff = int(contract.last_complete_month)
        out = work.loc[work[date_col] <= cutoff].copy()
        if (out[date_col] > cutoff).any():
            raise RuntimeError("internal error: production cutoff leaked later rows")
        return out
    if name == "origin_exclusive":
        out = work.loc[work[date_col] < origin].copy()
        if (out[date_col] >= origin).any():
            raise RuntimeError("internal error: exclusive cutoff leaked origin rows")
        return out
    raise ValueError(f"unknown cutoff policy {policy_name!r}")
