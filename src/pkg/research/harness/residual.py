"""Shared residual XGB (frozen fit_xgb + clip). Family fillna policy is injected."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.models import fit_xgb


def make_residual_model(
    anchor: str,
    feature_cols: Sequence[str],
    *,
    fillna_extra: Iterable[str] = (),
    never_fillna: Iterable[str] = (),
    name: str | None = None,
):
    """Residual XGB using frozen XGB_PARAMS / fit_xgb.

    Fills ``sales_*``, ``human_*``, and ``fillna_extra`` with 0.
    Columns in ``never_fillna`` stay NaN (XGBoost native missing).
    """
    cols = list(feature_cols)
    extra = frozenset(fillna_extra)
    skip = frozenset(never_fillna)
    if anchor == "ts":
        forecast_col = "ts_forecast"
        default_name = "ts_xgb_research"
    elif anchor == "human":
        forecast_col = "budget_forecast"
        default_name = "human_xgb_research"
    else:
        raise ValueError(f"anchor must be 'ts' or 'human', got {anchor!r}")
    model_name = name or default_name

    def _predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        tr = train_df.copy()
        te = test_df.copy()
        tr["residual"] = tr["sales"].astype(float) - tr[forecast_col].astype(float)
        missing = [c for c in cols if c not in tr.columns or c not in te.columns]
        if missing:
            raise KeyError(f"missing feature columns for {model_name}: {missing}")
        for c in cols:
            if c in skip:
                continue
            if c.startswith("sales_") or c.startswith("human_") or c in extra:
                tr[c] = tr[c].fillna(0)
                te[c] = te[c].fillna(0)
        if "horizon" not in tr.columns:
            raise KeyError("train_df needs horizon for sample weights")
        model = fit_xgb(cols, tr)
        resid = model.predict(te[cols])
        return np.maximum(0.0, te[forecast_col].astype(float).to_numpy() + resid)

    _predict.__name__ = model_name
    return _predict
