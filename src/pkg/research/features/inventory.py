"""Point-in-time inventory features keyed by forecast origin.

Uses frozen F3C Step 1 parquet only — never live SQL or FactInventory.

Temporal rule: for origin ``O``, inventory is measured at the exact last day
of the prior Shamsi month:

    origin_start = shamsi_month_start_gregorian(O)
    inventory_month_end = origin_start - 1 day

Equality join on ``product + snapshot_date == inventory_month_end``.
No merge_asof, no latest-prior, no freshness windows, no fill.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_month_start_gregorian
from pkg.research.harness.dataset import resolve_origin_col

SCORED_FEATURES: tuple[str, ...] = (
    "log_distributor_inventory_qty",
    "log_factory_inventory_qty",
)
FEATURE_NAMES: tuple[str, ...] = SCORED_FEATURES

RAW_QTY_NAMES: tuple[str, ...] = (
    "distributor_inventory_qty",
    "factory_inventory_qty",
)

DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "distributor_inventory_qty",
    "factory_inventory_qty",
    "distributor_missing_reason",
    "factory_missing_reason",
)


def load_frozen_distributor_inventory(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        from pkg.research.f3c.config import f3c_source_dir
        path = f3c_source_dir() / "distributor_inventory_daily.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen F3C distributor inventory missing: {path}. "
            "Run: python -m pkg.research.prepare_f3c"
        )
    df = pd.read_parquet(path)
    df["product"] = df["product"].astype(str)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def load_frozen_factory_inventory(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        from pkg.research.f3c.config import f3c_source_dir
        path = f3c_source_dir() / "factory_inventory_daily.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen F3C factory inventory missing: {path}. "
            "Run: python -m pkg.research.prepare_f3c"
        )
    df = pd.read_parquet(path)
    df["product"] = df["product"].astype(str)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def _origin_to_inventory_date(origin: int) -> pd.Timestamp:
    """Exact last day of the prior Shamsi month."""
    start = shamsi_month_start_gregorian(int(origin))
    return pd.Timestamp(start - timedelta(days=1))


def _safe_log1p(q: float) -> float:
    if not np.isfinite(q):
        return float("nan")
    if q < 0:
        return float("nan")
    return float(np.log1p(q))


def add_inventory_features(
    panel: pd.DataFrame,
    dist_hist: pd.DataFrame,
    fact_hist: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach exact month-end inventory features to a panel."""
    origin_col = resolve_origin_col(panel, origin_col)
    out = panel.copy()
    out["product"] = out["product"].astype(str)
    origins = out[origin_col].astype(int).unique()

    inv_date_map = {int(o): _origin_to_inventory_date(int(o)) for o in origins}

    # Build lookup dicts: (product, inv_date) → qty
    dist_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    if not dist_hist.empty:
        for inv_date in set(inv_date_map.values()):
            snap = dist_hist.loc[dist_hist["snapshot_date"] == inv_date]
            for _, row in snap.iterrows():
                dist_lookup[(str(row["product"]), inv_date)] = float(row["distributor_inventory_qty"])

    fact_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    if not fact_hist.empty:
        for inv_date in set(inv_date_map.values()):
            snap = fact_hist.loc[fact_hist["snapshot_date"] == inv_date]
            for _, row in snap.iterrows():
                fact_lookup[(str(row["product"]), inv_date)] = float(row["factory_inventory_qty"])

    dist_qty = []
    fact_qty = []
    dist_log = []
    fact_log = []
    dist_reason = []
    fact_reason = []

    for _, row in out.iterrows():
        product = str(row["product"])
        origin = int(row[origin_col])
        inv_date = inv_date_map[origin]
        origin_start = pd.Timestamp(shamsi_month_start_gregorian(origin))
        assert inv_date < origin_start, f"PIT violation: {inv_date} >= {origin_start}"

        # Distributor
        dkey = (product, inv_date)
        if dkey in dist_lookup:
            dq = dist_lookup[dkey]
            dist_qty.append(dq)
            if dq < 0:
                dist_log.append(float("nan"))
                dist_reason.append("NEGATIVE_QTY")
            else:
                dist_log.append(_safe_log1p(dq))
                dist_reason.append("AVAILABLE")
        else:
            dist_qty.append(float("nan"))
            dist_log.append(float("nan"))
            dist_reason.append("NO_EXACT_MONTH_END_PRODUCT_RECORD")

        # Factory
        fkey = (product, inv_date)
        if fkey in fact_lookup:
            fq = fact_lookup[fkey]
            fact_qty.append(fq)
            if fq < 0:
                fact_log.append(float("nan"))
                fact_reason.append("NEGATIVE_QTY")
            else:
                fact_log.append(_safe_log1p(fq))
                fact_reason.append("AVAILABLE")
        else:
            fact_qty.append(float("nan"))
            fact_log.append(float("nan"))
            fact_reason.append("NO_EXACT_MONTH_END_PRODUCT_RECORD")

    out["distributor_inventory_qty"] = dist_qty
    out["log_distributor_inventory_qty"] = dist_log
    out["factory_inventory_qty"] = fact_qty
    out["log_factory_inventory_qty"] = fact_log
    out["distributor_missing_reason"] = dist_reason
    out["factory_missing_reason"] = fact_reason

    return out


def assert_inventory_point_in_time(panel: pd.DataFrame, origin_col: str = "origin") -> None:
    """Verify no future-looking inventory in the panel."""
    for _, row in panel.iterrows():
        origin = int(row[origin_col])
        origin_start = pd.Timestamp(shamsi_month_start_gregorian(origin))
        inv_date = _origin_to_inventory_date(origin)
        assert inv_date < origin_start, f"PIT violation at origin {origin}"
