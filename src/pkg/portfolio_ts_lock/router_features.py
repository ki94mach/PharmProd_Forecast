"""Router feature builders with leakage guards."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import HB_ORIGINS, POST_HB_ORIGINS


def assert_no_hb_origins(df: pd.DataFrame, *, where: str) -> None:
    if df.empty or "forecast_origin" not in df.columns:
        return
    bad = pd.to_numeric(df["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))
    if bool(bad.any()):
        raise ValueError(
            f"HB origins must not appear in {where}: "
            f"{sorted(df.loc[bad, 'forecast_origin'].astype(int).unique().tolist())}"
        )


def historical_error_mask(df: pd.DataFrame, origin: int) -> pd.Series:
    """Rows usable as history for origin O: target_date <= O-2."""
    cutoff = shamsi_add_months(int(origin), -2)
    return pd.to_numeric(df["target_date"], errors="coerce") <= cutoff


def demand_features_through_origin_minus_2(
    dense: pd.DataFrame, product: str, origin: int
) -> dict[str, float]:
    cutoff = shamsi_add_months(int(origin), -2)
    g = dense.loc[
        (dense["product"].astype(str) == str(product))
        & (pd.to_numeric(dense["date"], errors="coerce") <= cutoff)
    ]
    if g.empty:
        return {
            "train_n_months": 0,
            "train_mean_sales": float("nan"),
            "train_zero_share": float("nan"),
            "train_sum_sales": float("nan"),
        }
    sales = g["actual"].astype(float)
    return {
        "train_n_months": int(len(g)),
        "train_mean_sales": float(sales.mean()),
        "train_zero_share": float((sales == 0).mean()),
        "train_sum_sales": float(sales.sum()),
    }


def build_router_features_sku_origin(
    *,
    policy_rows: pd.DataFrame,
    dense: pd.DataFrame,
    seed_dispersion: pd.DataFrame | None = None,
    allow_post_hb_tag: bool = True,
) -> pd.DataFrame:
    """Per SKU×origin features without evaluation-period leakage.

    Historical errors use only target_date <= O-2.
    Demand uses sales through O-2.
    Current-origin disagreement / seed dispersion allowed (no outcomes).
    HB origins are never emitted.
    """
    assert_no_hb_origins(policy_rows, where="router_features_input")
    base = policy_rows.loc[
        ~pd.to_numeric(policy_rows["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))
    ].copy()
    if not allow_post_hb_tag:
        base = base.loc[
            ~pd.to_numeric(base["forecast_origin"], errors="coerce").isin(list(POST_HB_ORIGINS))
        ]

    rows: list[dict[str, Any]] = []
    for (product, origin), g in base.groupby(["product", "forecast_origin"], sort=False):
        origin_i = int(origin)
        product_s = str(product)
        hist = g.loc[historical_error_mask(g, origin_i)]
        # Historical AE under policies (need actuals on hist rows only)
        feat: dict[str, Any] = {
            "product": product_s,
            "forecast_origin": origin_i,
            "is_post_hb": origin_i in POST_HB_ORIGINS,
        }
        feat.update(demand_features_through_origin_minus_2(dense, product_s, origin_i))

        if not hist.empty and "actual" in hist.columns:
            for col, name in (
                ("f_v2_policy", "hist_mae_v2"),
                ("f_a0_hybrid", "hist_mae_hybrid"),
                ("f_a0_only", "hist_mae_a0_only"),
            ):
                if col not in hist.columns:
                    feat[name] = float("nan")
                    continue
                m = hist[["actual", col]].dropna()
                if m.empty:
                    feat[name] = float("nan")
                else:
                    feat[name] = float(np.mean(np.abs(m[col] - m["actual"])))
        else:
            feat["hist_mae_v2"] = float("nan")
            feat["hist_mae_hybrid"] = float("nan")
            feat["hist_mae_a0_only"] = float("nan")

        # Current-origin disagreement (no outcomes required)
        cur = g.copy()
        if "f_v21" in cur.columns and "f_a0" in cur.columns:
            both = cur.loc[cur["f_v21"].notna() & cur["f_a0"].notna()]
            if not both.empty:
                feat["curr_mean_abs_v2_a0_disagreement"] = float(
                    np.mean(np.abs(both["f_v21"] - both["f_a0"]))
                )
            else:
                feat["curr_mean_abs_v2_a0_disagreement"] = float("nan")
        else:
            feat["curr_mean_abs_v2_a0_disagreement"] = float("nan")

        if seed_dispersion is not None and not seed_dispersion.empty:
            sd = seed_dispersion.loc[
                (seed_dispersion["product"].astype(str) == product_s)
                & (seed_dispersion["forecast_origin"].astype(int) == origin_i)
            ]
            if not sd.empty:
                feat["a0_seed_std"] = float(sd["seed_std"].iloc[0])
                feat["a0_seed_range"] = float(sd["seed_range"].iloc[0])
            else:
                feat["a0_seed_std"] = float("nan")
                feat["a0_seed_range"] = float("nan")
        else:
            feat["a0_seed_std"] = float("nan")
            feat["a0_seed_range"] = float("nan")

        rows.append(feat)

    out = pd.DataFrame(rows)
    assert_no_hb_origins(out, where="router_features_output")
    return out
