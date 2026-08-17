"""Point-in-time product commercial-tenure features (keyed by forecast origin).

Observed age is tenure inside available frozen sales history, never true
commercial launch age. Uses frozen ``raw/sales.parquet`` only — never live SQL.

First commercial observation is the earliest month with strictly positive net
sales. Zeros are not a launch. Negatives may be returns/adjustments and are
not treated as first sale.

The Drug Launch event extract is a deferred / exploratory commercial-event
source and is not used here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_month_diff

SCORED_FEATURE = "months_since_first_observed_positive_sale"
FEATURE_NAMES: tuple[str, ...] = (SCORED_FEATURE,)

DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "first_positive_sale_month",
    "has_prior_positive_sale",
    "first_sale_left_censored",
    "earliest_available_sales_month",
    "first_nonzero_sale_month",
)

LAUNCH_EVENT_NAMES: tuple[str, ...] = (
    "launch_date",
    "launch_month",
    "months_since_generic_launch",
    "has_launch_event",
    "selected_date_share",
    "n_event_rows",
)


def _resolve_origin_col(df: pd.DataFrame, origin_col: Optional[str] = None) -> str:
    if origin_col is not None:
        return origin_col
    if "origin" in df.columns:
        return "origin"
    if "ts_origin" in df.columns:
        return "ts_origin"
    if "budget_origin" in df.columns:
        return "budget_origin"
    raise ValueError("panel needs origin / ts_origin / budget_origin")


def _aggregate_sales(sales_hist: pd.DataFrame) -> pd.DataFrame:
    if sales_hist is None or sales_hist.empty:
        return pd.DataFrame(columns=["product", "date", "sales"])
    out = sales_hist.copy()
    out["product"] = out["product"].astype(str)
    out["date"] = out["date"].astype(int)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")
    return out.groupby(["product", "date"], as_index=False)["sales"].sum()


def _months_by_product(sales_agg: pd.DataFrame, mask: pd.Series) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if sales_agg.empty:
        return out
    sub = sales_agg.loc[mask, ["product", "date"]]
    for product, g in sub.groupby("product"):
        dates = np.sort(g["date"].astype(int).to_numpy())
        out[str(product)] = dates
    return out


def _first_before(dates: Optional[np.ndarray], origin: int) -> float:
    """Earliest month strictly before origin; NaN if none."""
    if dates is None or len(dates) == 0:
        return np.nan
    # dates is sorted; first element is the earliest sale of this kind.
    first = int(dates[0])
    if first < int(origin):
        return float(first)
    return np.nan


def product_lifecycle_catalog(sales_hist: pd.DataFrame) -> pd.DataFrame:
    """Product-level catalog diagnostics (not origin-keyed; not scored).

    ``first_positive_sale_month`` here is the earliest positive month in the
    provided sales table. Left-censoring uses the global earliest sales month.
    """
    sales_agg = _aggregate_sales(sales_hist)
    if sales_agg.empty:
        return pd.DataFrame(
            columns=[
                "product",
                "first_positive_sale_month",
                "first_nonzero_sale_month",
                "earliest_available_sales_month",
                "first_sale_left_censored",
            ]
        )
    global_min = int(sales_agg["date"].min())
    pos = _months_by_product(sales_agg, sales_agg["sales"] > 0)
    nz = _months_by_product(sales_agg, sales_agg["sales"].fillna(0) != 0)
    at_start = sales_agg.loc[sales_agg["date"] == global_min]
    left = set(at_start.loc[at_start["sales"] > 0, "product"].astype(str))
    products = sorted(sales_agg["product"].astype(str).unique())
    rows = []
    for p in products:
        first_pos = pos[p][0] if p in pos and len(pos[p]) else np.nan
        first_nz = nz[p][0] if p in nz and len(nz[p]) else np.nan
        rows.append(
            {
                "product": p,
                "first_positive_sale_month": first_pos,
                "first_nonzero_sale_month": first_nz,
                "earliest_available_sales_month": global_min,
                "first_sale_left_censored": 1 if p in left else 0,
            }
        )
    return pd.DataFrame(rows)


def build_lifecycle_features(
    panel: pd.DataFrame,
    sales_history: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return lifecycle columns aligned to ``panel`` index.

    Point-in-time: only ``sales_history.date < origin`` may influence a row.
    First positive sale is never taken from the full series and then exposed
    to origins before that sale occurred.
    """
    origin_col = _resolve_origin_col(panel, origin_col)
    sales_agg = _aggregate_sales(sales_history)
    if sales_agg.empty:
        global_min = np.nan
        pos_by_p: dict[str, np.ndarray] = {}
        nz_by_p: dict[str, np.ndarray] = {}
        left: set[str] = set()
    else:
        global_min = int(sales_agg["date"].min())
        pos_by_p = _months_by_product(sales_agg, sales_agg["sales"] > 0)
        nz_by_p = _months_by_product(sales_agg, sales_agg["sales"].fillna(0) != 0)
        at_start = sales_agg.loc[sales_agg["date"] == global_min]
        left = set(at_start.loc[at_start["sales"] > 0, "product"].astype(str))

    products = panel["product"].astype(str)
    origins = panel[origin_col].astype(int)
    n = len(panel)
    first_pos = np.full(n, np.nan)
    first_nz = np.full(n, np.nan)
    has_prior = np.zeros(n, dtype=float)
    age = np.full(n, np.nan)
    censored = np.zeros(n, dtype=float)
    earliest = np.full(n, np.nan if not np.isfinite(global_min) else float(global_min))

    for i, (product, origin) in enumerate(zip(products, origins)):
        o = int(origin)
        fp = _first_before(pos_by_p.get(product), o)
        fn = _first_before(nz_by_p.get(product), o)
        first_pos[i] = fp
        first_nz[i] = fn
        if np.isfinite(fp):
            if not (fp < o):
                raise AssertionError(
                    f"PIT leak: first_positive_sale_month={fp} is not < origin={o} "
                    f"for product={product!r}"
                )
            has_prior[i] = 1.0
            age[i] = float(shamsi_month_diff(o, int(fp)))
        censored[i] = 1.0 if product in left else 0.0

    feat = pd.DataFrame(
        {
            SCORED_FEATURE: age,
            "first_positive_sale_month": first_pos,
            "has_prior_positive_sale": has_prior,
            "first_sale_left_censored": censored,
            "earliest_available_sales_month": earliest,
            "first_nonzero_sale_month": first_nz,
        },
        index=panel.index,
    )
    assert_point_in_time(feat, origins.to_numpy())
    return feat


def assert_point_in_time(feat: pd.DataFrame, origins: np.ndarray) -> None:
    """Every observed first-sale month must be strictly before its origin."""
    fp = feat["first_positive_sale_month"].to_numpy(dtype=float)
    mask = np.isfinite(fp)
    if not mask.any():
        return
    o = np.asarray(origins, dtype=int)
    bad = fp[mask] >= o[mask]
    if np.any(bad):
        raise AssertionError(
            "lifecycle PIT assertion failed: first_positive_sale_month >= origin "
            f"on {int(bad.sum())} rows"
        )


def add_lifecycle_features(
    df: pd.DataFrame,
    sales_hist: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Copy ``df`` and attach lifecycle columns. Modeling uses ``FEATURE_NAMES`` only."""
    out = df.copy()
    feat = build_lifecycle_features(out, sales_hist, origin_col=origin_col)
    for col in feat.columns:
        out[col] = feat[col]
    return out
