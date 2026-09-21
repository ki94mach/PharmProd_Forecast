"""Immutable residual promotion-gate criteria (pre-registered before results)."""

from __future__ import annotations

from typing import Any

from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS
from pkg.research.portfolio_residual_xgb_v1.methods import EXECUTABLE_METHODS

PROMOTION_GATE_VERSION = "residual_promotion_gates_v1"

# Locked before seeing OOF results.
PROMOTION_GATES: dict[str, Any] = {
    "pooled_oof_wmape_improvement_over_frozen_ts": True,
    "origins_improved_ge": 4,
    "n_evaluation_origins": 7,
    "drop_top3_lift_positive": True,
    "rmse_not_more_than_2pct_worse_than_frozen_ts": True,
    "p95_not_more_than_2pct_worse_than_frozen_ts": True,
    "p99_not_more_than_2pct_worse_than_frozen_ts": True,
    "no_horizon_group_wmape_regression_gt_pp": 2.0,
    "sku_cluster_bootstrap_frac_positive_ge": 0.90,
    "bootstrap_B": 2000,
    "bootstrap_seed": 42,
}


def build_promotion_gate_artifact(
    *,
    contract_v1_1_sha256: str,
    eligible_evaluation_origins: list[int],
) -> dict[str, Any]:
    return {
        "gate_version": PROMOTION_GATE_VERSION,
        "locked_before_results": True,
        "hb_outcomes_inspected": False,
        "HUMAN_XGB": HUMAN_XGB_STATUS,
        "methods_evaluated": list(EXECUTABLE_METHODS),
        "promotion_candidate_method": "TS_PLUS_RESIDUAL_XGB",
        "baseline_method": "FROZEN_TS",
        "eligible_evaluation_origins": list(eligible_evaluation_origins),
        "residual_study_contract_v1_1_sha256": contract_v1_1_sha256,
        "gates": dict(PROMOTION_GATES),
        "gate_definitions": {
            "pooled_oof_wmape_improvement_over_frozen_ts": (
                "WMAPE(FROZEN_TS) - WMAPE(TS_PLUS_RESIDUAL_XGB) > 0 on pooled OOF"
            ),
            "origins_improved_ge": (
                "At least 4 of 7 evaluation origins have positive WMAPE lift vs FROZEN_TS"
            ),
            "drop_top3_lift_positive": (
                "After dropping top-3 SKUs by residual-XGB AE gain, remaining lift > 0"
            ),
            "rmse_p95_p99": (
                "RMSE, p95 AE, and p99 AE of residual-XGB each <= 1.02 × FROZEN_TS"
            ),
            "horizon_groups": (
                "No horizon-group WMAPE lift < -2.0 pp (lift = TS - XGB)"
            ),
            "bootstrap": (
                "SKU-cluster bootstrap P(lift>0) >= 0.90 with B=2000 seed=42"
            ),
        },
        "process_rule": (
            "Do not open HB outcomes until residual model + threshold are frozen "
            "against these gates."
        ),
    }
