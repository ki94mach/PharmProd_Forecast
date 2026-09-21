"""Deterministic post-HB deployment stability classifier (v2)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


STATUS_SUPPORTIVE = "SUPPORTIVE"
STATUS_CAUTION = "CAUTION"
STATUS_UNSUPPORTIVE = "UNSUPPORTIVE"


def deployment_stability_status_v2(
    *,
    deployable_wmape_lift: float,
    rmse_hybrid: float,
    rmse_v2: float,
    p95_hybrid: float,
    p95_v2: float,
    horizon_group_lifts: Mapping[str, float],
    leave_one_origin_lifts: Sequence[float],
    drop_top3_lift_wmape_pp: float,
) -> dict[str, Any]:
    """Fully deterministic SUPPORTIVE / CAUTION / UNSUPPORTIVE.

    SUPPORTIVE iff ALL gates pass:
      1. deployable WMAPE lift > 0
      2. RMSE_hy <= 1.02 * RMSE_v2
      3. p95_hy  <= 1.02 * p95_v2
      4. at least 3 of 4 horizon groups have lift >= 0
      5. both leave-one-origin lifts >= 0
      6. drop-top-3 lift > 0

    CAUTION: deployable lift > 0 and not SUPPORTIVE.
    UNSUPPORTIVE: deployable lift <= 0.
    """
    hz_vals = [float(v) for v in horizon_group_lifts.values()]
    n_hz_nonneg = sum(1 for v in hz_vals if v == v and v >= 0.0)
    loo = [float(v) for v in leave_one_origin_lifts]
    both_loo_nonneg = bool(loo) and all(v == v and v >= 0.0 for v in loo)

    rmse_ok = (
        rmse_v2 > 0
        and rmse_hybrid == rmse_hybrid
        and rmse_v2 == rmse_v2
        and rmse_hybrid <= rmse_v2 * 1.02
    )
    p95_ok = (
        p95_v2 > 0
        and p95_hybrid == p95_hybrid
        and p95_v2 == p95_v2
        and p95_hybrid <= p95_v2 * 1.02
    )

    gates = {
        "deployable_wmape_lift_gt_0": bool(deployable_wmape_lift > 0),
        "rmse_not_more_than_2pct_worse": bool(rmse_ok),
        "p95_not_more_than_2pct_worse": bool(p95_ok),
        "horizon_groups_nonneg_ge_3": bool(n_hz_nonneg >= 3),
        "both_leave_one_origin_lift_ge_0": bool(both_loo_nonneg),
        "drop_top3_lift_positive": bool(
            drop_top3_lift_wmape_pp == drop_top3_lift_wmape_pp
            and drop_top3_lift_wmape_pp > 0
        ),
    }
    all_pass = all(gates.values())

    if deployable_wmape_lift <= 0:
        status = STATUS_UNSUPPORTIVE
    elif all_pass:
        status = STATUS_SUPPORTIVE
    else:
        status = STATUS_CAUTION

    return {
        "deployment_stability_status": status,
        "classifier_gates": gates,
        "n_horizon_groups_lift_ge_0": int(n_hz_nonneg),
        "n_leave_one_origin": int(len(loo)),
    }
