"""Targeted tests for residual XGB evaluation_v1 analysis helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import (
    EvaluationOverwriteError,
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
    REC_INSUFFICIENT,
    REC_KEEP,
    REC_PROMOTE,
    ROUTER_WHITELIST,
    build_bootstrap_copy,
    check_complete_horizons,
    decide_recommendation,
    ensure_out_dir,
    join_router_features,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import HYBRID_ANCHOR


def _tiny_panel(*, xgb_better: bool) -> pd.DataFrame:
    """Two SKUs × one origin × horizons 1..15."""
    rows = []
    for product in ("SKU_A", "SKU_B"):
        for h in range(1, 16):
            actual = 10.0
            ts = 12.0
            # Better XGB => closer to actual
            xgb = 10.5 if xgb_better else 14.0
            med = 11.5
            anchor = 11.0
            rows.append(
                {
                    "product": product,
                    "forecast_origin": 140107,
                    "horizon": h,
                    "target_date": 140107 + h,
                    "actual": actual,
                    HYBRID_ANCHOR: anchor,
                    METHOD_FROZEN_TS: ts,
                    METHOD_MEDIAN: med,
                    METHOD_XGB: xgb,
                }
            )
    return pd.DataFrame(rows)


def test_complete_horizons_fail_on_missing():
    df = _tiny_panel(xgb_better=True)
    df = df.loc[df["horizon"] != 7].copy()
    r = check_complete_horizons(df)
    assert r["ok"] is False
    assert r["n_incomplete_groups"] == 2


def test_complete_horizons_pass():
    df = _tiny_panel(xgb_better=True)
    r = check_complete_horizons(df)
    assert r["ok"] is True
    assert r["n_incomplete_groups"] == 0


def test_recommendation_branches():
    assert (
        decide_recommendation(
            integrity_ok=False,
            non_computable_gates=[],
            promotion_gates_pass=True,
        )
        is None
    )
    assert (
        decide_recommendation(
            integrity_ok=True,
            non_computable_gates=["rmse_not_more_than_2pct_worse"],
            promotion_gates_pass=False,
        )
        == REC_INSUFFICIENT
    )
    assert (
        decide_recommendation(
            integrity_ok=True,
            non_computable_gates=[],
            promotion_gates_pass=True,
        )
        == REC_PROMOTE
    )
    assert (
        decide_recommendation(
            integrity_ok=True,
            non_computable_gates=[],
            promotion_gates_pass=False,
        )
        == REC_KEEP
    )


def test_router_column_leakage(tmp_path: Path):
    panel = _tiny_panel(xgb_better=True)
    router = pd.DataFrame(
        {
            "product": ["SKU_A", "SKU_B"],
            "forecast_origin": [140107, 140107],
            "train_n_months": [15, 20],
            "train_mean_sales": [10.0, 20.0],
            "train_zero_share": [0.1, 0.2],
            "hist_mae_v2": [1.0, 2.0],  # must not leak
            "secret_col": ["x", "y"],
        }
    )
    path = tmp_path / "router.csv"
    router.to_csv(path, index=False)
    joined = join_router_features(panel, path)
    for bad in ("hist_mae_v2", "secret_col"):
        assert bad not in joined.columns
    for c in ROUTER_WHITELIST:
        assert c in joined.columns


def test_lift_direction_positive_when_xgb_better():
    panel = _tiny_panel(xgb_better=True)
    boot = build_bootstrap_copy(panel)
    lift = float(boot.attrs["lift_wmape_pp_asserted"])
    assert lift > 0


def test_lift_direction_negative_when_xgb_worse():
    panel = _tiny_panel(xgb_better=False)
    boot = build_bootstrap_copy(panel)
    lift = float(boot.attrs["lift_wmape_pp_asserted"])
    assert lift < 0


def test_anchor_preservation():
    panel = _tiny_panel(xgb_better=True)
    # Ensure XGB differs from anchor
    assert not np.allclose(
        panel[HYBRID_ANCHOR].to_numpy(dtype=float),
        panel[METHOD_XGB].to_numpy(dtype=float),
    )
    before = panel[HYBRID_ANCHOR].astype(float).to_numpy().copy()
    boot = build_bootstrap_copy(panel)
    after = panel[HYBRID_ANCHOR].astype(float).to_numpy()
    assert np.allclose(before, after)
    assert not np.allclose(
        panel[HYBRID_ANCHOR].to_numpy(dtype=float),
        boot[HYBRID_ANCHOR].to_numpy(dtype=float),
    )


def test_overwrite_protection(tmp_path: Path):
    out = tmp_path / "evaluation_v1"
    out.mkdir()
    (out / "marker.txt").write_text("x", encoding="utf-8")
    with pytest.raises(EvaluationOverwriteError):
        ensure_out_dir(out, force=False)
    ensure_out_dir(out, force=True)  # must not raise
