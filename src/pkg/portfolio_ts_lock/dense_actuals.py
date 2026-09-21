"""Dense monthly actuals for portfolio scoring."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import (
    ACTIVITY_THRESHOLD,
    HB_ORIGINS,
    LOCKED_SCORE_END,
)


def month_range(start: int, end: int) -> list[int]:
    out: list[int] = []
    cur = int(start)
    end_i = int(end)
    while cur <= end_i:
        out.append(cur)
        cur = shamsi_add_months(cur, 1)
    return out


def build_dense_actuals(
    sparse: pd.DataFrame,
    *,
    through_date: int = LOCKED_SCORE_END,
    activity_threshold: float = ACTIVITY_THRESHOLD,
) -> pd.DataFrame:
    """Contiguous monthly grid per product from first activity through ``through_date``.

    Absent transaction months inside the window are filled with 0.
    Months before first activity and after ``through_date`` are omitted.
    """
    rows: list[dict[str, Any]] = []
    sp = sparse.copy()
    sp["product"] = sp["product"].astype(str)
    sp["date"] = pd.to_numeric(sp["date"], errors="coerce").astype(int)
    sp["sales"] = pd.to_numeric(sp["sales"], errors="coerce")

    for product, g in sp.groupby("product", sort=False):
        observed = (
            g.groupby("date", sort=True)["sales"]
            .sum()
            .astype(float)
        )
        active = observed.loc[observed > float(activity_threshold)]
        if active.empty:
            pos = observed.loc[observed.fillna(0.0).ne(0.0)]
            if pos.empty:
                continue
            first_active = int(pos.index.min())
        else:
            first_active = int(active.index.min())
        if first_active > int(through_date):
            continue
        for ym in month_range(first_active, int(through_date)):
            if ym in observed.index and pd.notna(observed[ym]):
                val = float(observed[ym])
                is_miss = False
            else:
                val = 0.0
                is_miss = True
            rows.append(
                {
                    "product": str(product),
                    "date": int(ym),
                    "actual": val,
                    "is_missing_month": bool(is_miss),
                }
            )
    return pd.DataFrame(rows)


def assert_exported_actual_agreement(
    panel: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    atol: float = 1e-6,
    exclude_hb: bool = True,
) -> dict[str, Any]:
    """Where panel.actual is non-null, require agreement with dense actuals."""
    if "actual" not in panel.columns:
        return {"n_compared": 0, "n_mismatch": 0, "max_abs_diff": None}
    p = panel.copy()
    p["product"] = p["product"].astype(str)
    p["target_date"] = pd.to_numeric(p["target_date"], errors="coerce").astype(int)
    if exclude_hb and "forecast_origin" in p.columns:
        p = p.loc[~p["forecast_origin"].astype(int).isin(HB_ORIGINS)].copy()
    p = p.loc[p["actual"].notna()].copy()
    if p.empty:
        return {"n_compared": 0, "n_mismatch": 0, "max_abs_diff": None}
    d = dense.rename(columns={"date": "target_date", "actual": "dense_actual"})
    m = p.merge(d[["product", "target_date", "dense_actual"]], on=["product", "target_date"], how="left")
    diff = (m["actual"].astype(float) - m["dense_actual"].astype(float)).abs()
    mismatch = diff > float(atol)
    # Missing dense for a scored cell also counts as mismatch
    mismatch = mismatch | m["dense_actual"].isna()
    return {
        "n_compared": int(len(m)),
        "n_mismatch": int(mismatch.sum()),
        "max_abs_diff": float(diff.max()) if len(diff) else None,
        "ok": bool(mismatch.sum() == 0),
    }


def attach_dense_actuals(
    frame: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    forbid_hb: bool = True,
) -> pd.DataFrame:
    """Left-join dense actuals; refuse Human Budget origins when forbid_hb."""
    out = frame.copy()
    out["product"] = out["product"].astype(str)
    out["forecast_origin"] = pd.to_numeric(out["forecast_origin"], errors="coerce").astype(int)
    out["target_date"] = pd.to_numeric(out["target_date"], errors="coerce").astype(int)
    if forbid_hb:
        bad = out["forecast_origin"].isin(HB_ORIGINS)
        if bool(bad.any()):
            raise ValueError(
                f"Human Budget origins must not reach attach_dense_actuals; "
                f"found {sorted(out.loc[bad, 'forecast_origin'].unique().tolist())}"
            )
    d = dense.rename(columns={"date": "target_date"})
    out = out.drop(columns=[c for c in ("actual",) if c in out.columns], errors="ignore")
    out = out.merge(d[["product", "target_date", "actual"]], on=["product", "target_date"], how="left")
    return out


def last_train_value(
    dense: pd.DataFrame, product: str, origin: int
) -> float:
    """Production naive_last_value: last dense actual at or before origin-2."""
    cutoff = shamsi_add_months(int(origin), -2)
    g = dense.loc[
        (dense["product"].astype(str) == str(product))
        & (pd.to_numeric(dense["date"], errors="coerce") <= cutoff)
    ].sort_values("date")
    if g.empty:
        return float("nan")
    return float(g["actual"].iloc[-1])


def history_length_through_origin_minus_2(
    dense: pd.DataFrame, product: str, origin: int
) -> int:
    cutoff = shamsi_add_months(int(origin), -2)
    g = dense.loc[
        (dense["product"].astype(str) == str(product))
        & (pd.to_numeric(dense["date"], errors="coerce") <= cutoff)
    ]
    return int(len(g))
