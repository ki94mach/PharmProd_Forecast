"""Point-in-time official consumer-price features (keyed by forecast origin).

Uses frozen F3B Step 1 ``price_history.parquet`` only — never live SQL or Excel.

Temporal rule (same as F3A/F2): a price observation is visible at origin ``O``
iff ``effective_month < O``. Origins are Shamsi YYYYMM ints treated as the
start of that month. A change in the origin month itself is not visible.

These features describe official price state known by forecast origin. They
do not describe future planned price changes and cannot use a post-origin
effective price even when the target month is after that change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_month_diff
from pkg.research.harness.dataset import resolve_origin_col

SCORED_FEATURES: tuple[str, ...] = (
    "log_consumer_price_asof_origin",
    "last_consumer_price_change_pct",
    "months_since_last_consumer_price_change",
)
FEATURE_NAMES: tuple[str, ...] = SCORED_FEATURES

DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "consumer_price_asof_origin",
    "last_price_effective_month",
    "previous_consumer_price",
    "last_change_month",
    "n_price_states_before_origin",
)


def load_frozen_price_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Step 1 frozen price history. Does not query SQL or open Excel."""
    if path is None:
        from pkg.research.f3b.config import f3b_source_dir

        path = f3b_source_dir() / "price_history.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen F3B price history missing: {path}. "
            "Run: python -m pkg.research.prepare_f3b"
        )
    hist = pd.read_parquet(path)
    hist["product"] = hist["product"].astype(str)
    hist["effective_month"] = hist["effective_month"].astype(int)
    if "effective_date" in hist.columns:
        hist["effective_date"] = pd.to_numeric(hist["effective_date"], errors="coerce")
    hist["consumer_price"] = pd.to_numeric(hist["consumer_price"], errors="coerce")
    return hist


def _positive_obs(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(
            columns=["product", "effective_month", "effective_date", "consumer_price"]
        )
    out = history.copy()
    out["product"] = out["product"].astype(str)
    out["effective_month"] = out["effective_month"].astype(int)
    if "effective_date" not in out.columns:
        out["effective_date"] = out["effective_month"] * 100 + 1
    out["effective_date"] = pd.to_numeric(out["effective_date"], errors="coerce")
    out["effective_date"] = out["effective_date"].fillna(out["effective_month"] * 100 + 1)
    out["consumer_price"] = pd.to_numeric(out["consumer_price"], errors="coerce")
    out = out.loc[np.isfinite(out["consumer_price"]) & (out["consumer_price"] > 0)]
    return out.sort_values(["product", "effective_month", "effective_date"])


def _collapse_states(months: np.ndarray, prices: np.ndarray) -> list[tuple[float, int, int]]:
    """Collapse consecutive identical prices into states.

    Returns list of (price, first_month, last_month) in time order.
    first_month is when this price became effective (a genuine change).
    last_month is the most recent observation of this same price.
    """
    states: list[tuple[float, int, int]] = []
    for month, price in zip(months.tolist(), prices.tolist()):
        month = int(month)
        price = float(price)
        if states and np.isclose(states[-1][0], price, rtol=0.0, atol=1e-6):
            p0, first, _ = states[-1]
            states[-1] = (p0, first, month)
        else:
            states.append((price, month, month))
    return states


def _states_by_product(obs: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if obs.empty:
        return out
    for product, g in obs.groupby("product", sort=False):
        out[str(product)] = (
            g["effective_month"].to_numpy(dtype=int),
            g["consumer_price"].to_numpy(dtype=float),
        )
    return out


def _features_for_states(
    months: np.ndarray,
    prices: np.ndarray,
    origin: int,
) -> dict[str, float]:
    mask = months < int(origin)
    vis_m = months[mask]
    vis_p = prices[mask]
    empty = {
        "consumer_price_asof_origin": np.nan,
        "log_consumer_price_asof_origin": np.nan,
        "last_consumer_price_change_pct": np.nan,
        "months_since_last_consumer_price_change": np.nan,
        "last_price_effective_month": np.nan,
        "previous_consumer_price": np.nan,
        "last_change_month": np.nan,
        "n_price_states_before_origin": 0.0,
    }
    if len(vis_m) == 0:
        return empty
    states = _collapse_states(vis_m, vis_p)
    current_price, first_month, last_month = states[-1]
    row = dict(empty)
    row["n_price_states_before_origin"] = float(len(states))
    row["consumer_price_asof_origin"] = float(current_price)
    row["log_consumer_price_asof_origin"] = float(np.log1p(current_price))
    row["last_price_effective_month"] = float(last_month)
    if len(states) >= 2:
        prev_price, _, _ = states[-2]
        row["previous_consumer_price"] = float(prev_price)
        row["last_change_month"] = float(first_month)
        if prev_price > 0:
            row["last_consumer_price_change_pct"] = float(
                (current_price - prev_price) / prev_price
            )
        row["months_since_last_consumer_price_change"] = float(
            shamsi_month_diff(int(origin), int(first_month))
        )
    return row


def build_price_features(
    panel: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return price columns aligned to ``panel`` index. Source-independent.

    Only ``price_history.effective_month < origin`` may influence a row.
    """
    ocol = resolve_origin_col(panel, origin_col)
    obs = _positive_obs(price_history)
    by_p = _states_by_product(obs)
    products = panel["product"].astype(str)
    origins = panel[ocol].astype(int)
    records = []
    for product, origin in zip(products, origins.to_numpy()):
        months, prices = by_p.get(str(product), (np.array([], dtype=int), np.array([], dtype=float)))
        records.append(_features_for_states(months, prices, int(origin)))
    feat = pd.DataFrame.from_records(records, index=panel.index)
    assert_price_point_in_time(feat, origins.to_numpy())
    return feat


def assert_price_point_in_time(feat: pd.DataFrame, origins: np.ndarray) -> None:
    """Visible price months must be strictly before origin (YYYYMM)."""
    o = np.asarray(origins, dtype=int)
    for col in ("last_price_effective_month", "last_change_month"):
        vals = feat[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        bad = vals[mask] >= o[mask]
        if np.any(bad):
            raise AssertionError(
                f"price PIT assertion failed: {col} >= origin on {int(bad.sum())} rows"
            )


def add_price_features(
    df: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    origin_col: Optional[str] = None,
) -> pd.DataFrame:
    """Copy ``df`` and attach price columns. Modeling uses ``FEATURE_NAMES`` only."""
    out = df.copy()
    feat = build_price_features(out, price_history, origin_col=origin_col)
    for col in feat.columns:
        out[col] = feat[col]
    return out
