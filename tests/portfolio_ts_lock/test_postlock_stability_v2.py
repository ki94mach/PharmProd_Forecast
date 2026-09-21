"""Unit tests for deterministic post-lock stability classifier v2."""

from __future__ import annotations

from pkg.portfolio_ts_lock.postlock_stability import (
    STATUS_CAUTION,
    STATUS_SUPPORTIVE,
    STATUS_UNSUPPORTIVE,
    deployment_stability_status_v2,
)


def _base_kwargs(**overrides):
    kw = dict(
        deployable_wmape_lift=0.65,
        rmse_hybrid=100.0,
        rmse_v2=100.0,
        p95_hybrid=200.0,
        p95_v2=200.0,
        horizon_group_lifts={"h1_3": 0.1, "h4_6": 0.1, "h7_12": 0.1, "h13_15": 0.1},
        leave_one_origin_lifts=[0.1, 0.2],
        drop_top3_lift_wmape_pp=0.5,
    )
    kw.update(overrides)
    return kw


def test_supportive_when_all_gates_pass():
    out = deployment_stability_status_v2(**_base_kwargs())
    assert out["deployment_stability_status"] == STATUS_SUPPORTIVE
    assert all(out["classifier_gates"].values())


def test_unsupportive_when_wmape_nonpositive():
    out = deployment_stability_status_v2(**_base_kwargs(deployable_wmape_lift=0.0))
    assert out["deployment_stability_status"] == STATUS_UNSUPPORTIVE
    out2 = deployment_stability_status_v2(**_base_kwargs(deployable_wmape_lift=-0.1))
    assert out2["deployment_stability_status"] == STATUS_UNSUPPORTIVE


def test_caution_rmse_worse_than_2pct():
    out = deployment_stability_status_v2(
        **_base_kwargs(rmse_hybrid=103.0, rmse_v2=100.0)
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["rmse_not_more_than_2pct_worse"] is False


def test_caution_p95_worse_than_2pct():
    out = deployment_stability_status_v2(
        **_base_kwargs(p95_hybrid=205.0, p95_v2=200.0)
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["p95_not_more_than_2pct_worse"] is False


def test_caution_fewer_than_3_horizon_groups_nonneg():
    out = deployment_stability_status_v2(
        **_base_kwargs(
            horizon_group_lifts={
                "h1_3": -0.6,
                "h4_6": -0.3,
                "h7_12": 3.0,
                "h13_15": -1.3,
            }
        )
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["horizon_groups_nonneg_ge_3"] is False
    assert out["n_horizon_groups_lift_ge_0"] == 1


def test_caution_leave_one_origin_negative():
    out = deployment_stability_status_v2(
        **_base_kwargs(leave_one_origin_lifts=[0.2, -0.1])
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["both_leave_one_origin_lift_ge_0"] is False


def test_caution_drop_top3_nonpositive():
    out = deployment_stability_status_v2(
        **_base_kwargs(drop_top3_lift_wmape_pp=0.0)
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["drop_top3_lift_positive"] is False


def test_fixture_matching_current_postlock_evidence_is_caution():
    """~0.65 pp WMAPE, ~6.4% RMSE worse, 1/4 horizon groups non-neg → CAUTION."""
    out = deployment_stability_status_v2(
        deployable_wmape_lift=0.637,
        rmse_hybrid=16324.5,
        rmse_v2=15341.6,
        p95_hybrid=15390.5,
        p95_v2=17015.4,
        horizon_group_lifts={
            "h1_3": -0.63,
            "h4_6": -0.28,
            "h7_12": 3.08,
            "h13_15": -1.34,
        },
        leave_one_origin_lifts=[0.5, 0.4],
        drop_top3_lift_wmape_pp=0.2,
    )
    assert out["deployment_stability_status"] == STATUS_CAUTION
    assert out["classifier_gates"]["deployable_wmape_lift_gt_0"] is True
    assert out["classifier_gates"]["rmse_not_more_than_2pct_worse"] is False
    assert out["classifier_gates"]["horizon_groups_nonneg_ge_3"] is False
