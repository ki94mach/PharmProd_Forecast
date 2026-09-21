"""Tests for residual pipeline freeze decision (no production lock I/O)."""

from __future__ import annotations

import pytest

from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import (
    REC_INSUFFICIENT,
    REC_KEEP,
    REC_PROMOTE,
)
from pkg.research.portfolio_residual_xgb_v1.pipeline_lock import (
    PIPELINE_FROZEN_TS,
    PIPELINE_RESIDUAL_XGB,
    decide_pipeline_from_gates,
)


def _all_pass() -> dict[str, bool]:
    return {
        "pooled_oof_wmape_improvement_over_frozen_ts": True,
        "origins_improved_ge_4": True,
        "drop_top3_lift_positive": True,
        "rmse_not_more_than_2pct_worse": True,
        "p95_not_more_than_2pct_worse": True,
        "p99_not_more_than_2pct_worse": True,
        "no_horizon_group_regression_gt_2pp": True,
        "sku_cluster_bootstrap_frac_positive_ge_90": True,
    }


def test_decide_promote_when_all_gates_pass():
    d, pipe = decide_pipeline_from_gates(
        gates_result=_all_pass(), non_computable_gates=[]
    )
    assert d == REC_PROMOTE
    assert pipe == PIPELINE_RESIDUAL_XGB


def test_decide_keep_when_any_gate_fails():
    g = _all_pass()
    g["pooled_oof_wmape_improvement_over_frozen_ts"] = False
    d, pipe = decide_pipeline_from_gates(gates_result=g, non_computable_gates=[])
    assert d == REC_KEEP
    assert pipe == PIPELINE_FROZEN_TS


def test_decide_insufficient_non_computable():
    g = _all_pass()
    g["rmse_not_more_than_2pct_worse"] = None
    d, pipe = decide_pipeline_from_gates(
        gates_result=g, non_computable_gates=["rmse_not_more_than_2pct_worse"]
    )
    assert d == REC_INSUFFICIENT
    assert pipe is None
