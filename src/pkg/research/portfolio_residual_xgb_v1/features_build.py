"""Build residual features with production last_complete_month = origin−2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.research.portfolio_residual_xgb_v1.encoders import TrainOnlyCategoricalEncoder
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    TS_RESID_FEATURE_LIST,
    assert_no_sales_uses_origin_minus_1,
    last_complete_month,
)


def _sales_at(sales_pivot: pd.Series, product: str, ym: int) -> float:
    try:
        return float(sales_pivot.loc[(str(product), int(ym))])
    except KeyError:
        return np.nan


def add_calendar_and_sales_features(
    panel: pd.DataFrame,
    sales: pd.DataFrame,
    *,
    fillna: float = 0.0,
) -> pd.DataFrame:
    """Add horizon/month/quarter + production PIT sales lags (Fo-2 based)."""
    out = panel.copy()
    if "target_date" not in out.columns and "date" in out.columns:
        out = out.rename(columns={"date": "target_date"})
    out["horizon"] = out["horizon"].astype(int)
    out["month"] = out["target_date"].astype(int) % 100
    out["quarter"] = ((out["month"] - 1) // 3) + 1

    sales_agg = (
        sales.rename(columns={"date": "date", "sales": "sales"})
        .assign(product=lambda d: d["product"].astype(str), date=lambda d: d["date"].astype(int))
        .groupby(["product", "date"], as_index=False)["sales"]
        .sum()
    )
    pivot = sales_agg.set_index(["product", "date"])["sales"]

    lag1, lag2, lag3, lag12 = [], [], [], []
    for fo, product in zip(out["forecast_origin"].astype(int), out["product"].astype(str)):
        assert_no_sales_uses_origin_minus_1(fo)
        lc = last_complete_month(fo)
        lag1.append(_sales_at(pivot, product, lc))
        lag2.append(_sales_at(pivot, product, shamsi_add_months(lc, -1)))
        lag3.append(_sales_at(pivot, product, shamsi_add_months(lc, -2)))
        lag12.append(_sales_at(pivot, product, shamsi_add_months(lc, -11)))
    out["sales_lag_1"] = pd.Series(lag1, index=out.index).fillna(fillna)
    out["sales_lag_2"] = pd.Series(lag2, index=out.index).fillna(fillna)
    out["sales_lag_3"] = pd.Series(lag3, index=out.index).fillna(fillna)
    out["sales_lag_12"] = pd.Series(lag12, index=out.index).fillna(fillna)
    out["sales_roll3"] = out[["sales_lag_1", "sales_lag_2", "sales_lag_3"]].mean(axis=1)
    return out


def encode_categoricals_train_only(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, TrainOnlyCategoricalEncoder]:
    enc = TrainOnlyCategoricalEncoder().fit(train)
    return enc.transform(train), enc.transform(test), enc


def ensure_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in TS_RESID_FEATURE_LIST:
        if c == HYBRID_ANCHOR:
            continue
        if c not in out.columns:
            out[c] = 0.0
    return out
