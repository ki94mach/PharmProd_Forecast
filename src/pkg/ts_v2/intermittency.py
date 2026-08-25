"""Intermittent-demand diagnostics for prepared monthly series."""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd


class IntermittencyStats(NamedTuple):
    """ADI / zero-share diagnostics (no model routing)."""

    zero_month_proportion: Optional[float]
    average_inter_demand_interval: Optional[float]
    n_demand_months: int


def intermittency_stats(values: pd.Series) -> IntermittencyStats:
    """Compute zero-month share and average inter-demand interval (ADI).

    Demand months are positions where ``sales > 0``. ADI is the mean difference
    between consecutive demand indices (1-based month steps along the series).
    """
    if values is None or len(values) == 0:
        return IntermittencyStats(None, None, 0)

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return IntermittencyStats(None, None, 0)

    observed = arr[finite]
    zero_prop = float(np.mean(observed <= 0.0))
    # Demand indices on the full series positions (0-based).
    demand_pos = np.flatnonzero(np.isfinite(arr) & (arr > 0.0))
    n_demand = int(len(demand_pos))
    if n_demand < 2:
        return IntermittencyStats(zero_prop, None, n_demand)
    intervals = np.diff(demand_pos.astype(float))
    adi = float(np.mean(intervals))
    return IntermittencyStats(zero_prop, adi, n_demand)
