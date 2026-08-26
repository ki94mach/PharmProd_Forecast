"""Drop inactive products from Excel packaging (CSV stays complete).

A product is inactive when it has no sales in the last complete 6 Shamsi months
and no positive distributor inventory on the latest snapshot.
"""
from __future__ import annotations

import pandas as pd

from pkg.benchmark.calendar import last_complete_6m


def products_with_recent_sales(sale_df: pd.DataFrame, origin_ym: int) -> set[str]:
    """Products with summed sales > 0 in last_complete_6m(origin)."""
    if sale_df is None or sale_df.empty:
        return set()
    start, end = last_complete_6m(int(origin_ym))
    work = sale_df.copy()
    work["product"] = work["product"].astype(str)
    work["date"] = work["date"].astype(int)
    work["sales"] = pd.to_numeric(work["sales"], errors="coerce").fillna(0.0)
    window = work.loc[(work["date"] >= start) & (work["date"] <= end)]
    if window.empty:
        return set()
    totals = window.groupby("product", sort=False)["sales"].sum()
    return set(totals[totals > 0].index.astype(str))


def products_with_distributor_inventory(dist_inv_df: pd.DataFrame) -> set[str]:
    """Products with distributor_inventory_qty > 0 on the latest snapshot_date."""
    if dist_inv_df is None or dist_inv_df.empty:
        return set()
    work = dist_inv_df.copy()
    if "product" not in work.columns or "snapshot_date" not in work.columns:
        return set()
    if "distributor_inventory_qty" not in work.columns:
        return set()
    work["product"] = work["product"].astype(str)
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"])
    work["distributor_inventory_qty"] = pd.to_numeric(
        work["distributor_inventory_qty"], errors="coerce"
    ).fillna(0.0)
    latest = work["snapshot_date"].max()
    snap = work.loc[work["snapshot_date"] == latest]
    if snap.empty:
        return set()
    totals = snap.groupby("product", sort=False)["distributor_inventory_qty"].sum()
    return set(totals[totals > 0].index.astype(str))


def products_inactive_for_excel(
    sale_df: pd.DataFrame,
    dist_inv_df: pd.DataFrame,
    origin_ym: int,
    products: set[str],
) -> set[str]:
    """Products with no sales in last_complete_6m(origin) AND no distributor stock."""
    active_sales = products_with_recent_sales(sale_df, origin_ym)
    has_inv = products_with_distributor_inventory(dist_inv_df)
    return {
        p
        for p in products
        if p not in active_sales and p not in has_inv
    }


def filter_forecast_for_excel(
    forecast_df: pd.DataFrame,
    sale_df: pd.DataFrame,
    dist_inv_df: pd.DataFrame,
    origin_ym: int,
) -> tuple[pd.DataFrame, set[str]]:
    """Return forecast rows excluding inactive products, plus the dropped set."""
    if forecast_df is None or forecast_df.empty:
        return forecast_df, set()
    products = set(forecast_df["product"].astype(str).unique())
    inactive = products_inactive_for_excel(
        sale_df, dist_inv_df, origin_ym, products
    )
    if not inactive:
        return forecast_df, set()
    mask = ~forecast_df["product"].astype(str).isin(inactive)
    return forecast_df.loc[mask].copy(), inactive
