"""Tests for portfolio TS architecture lock guards and freeze rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkg.portfolio_ts_lock.constants import (
    HB_ORIGINS,
    HORIZON_GROUPS,
    LOCKED_SCORE_END,
)
from pkg.portfolio_ts_lock.dense_actuals import (
    attach_dense_actuals,
    build_dense_actuals,
)
from pkg.portfolio_ts_lock.freeze import (
    horizon_group_wmape_lifts,
    make_recommendation,
    sku_origin_wins,
)
from pkg.portfolio_ts_lock.lock_io import (
    LockExistsError,
    assert_lock_immutable_read,
    refuse_lock_mutation,
    write_lock_atomic,
)
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb
from pkg.portfolio_ts_lock.router_features import (
    build_router_features_sku_origin,
    historical_error_mask,
)


def test_hb_origins_never_reach_attach_or_scoring():
    dense = pd.DataFrame(
        {
            "product": ["P"] * 3,
            "date": [140101, 140102, 140103],
            "actual": [1.0, 0.0, 2.0],
            "is_missing_month": [False, True, False],
        }
    )
    frame = pd.DataFrame(
        {
            "product": ["P"],
            "forecast_origin": [140304],
            "horizon": [1],
            "target_date": [140304],
        }
    )
    with pytest.raises(ValueError, match="Human Budget"):
        attach_dense_actuals(frame, dense, forbid_hb=True)

    panel = pd.DataFrame(
        {
            "product": ["P", "P"],
            "forecast_origin": [140101, 140304],
            "horizon": [1, 1],
            "target_date": [140101, 140304],
            "forecast": [1.0, 9.0],
            "model": ["ets", "ets"],
            "job_status": ["SUCCESS", "SUCCESS"],
            "engine": ["v2.1", "v2.1"],
            "seed": [np.nan, np.nan],
        }
    )
    filtered = filter_non_hb(panel)
    assert set(filtered["forecast_origin"].astype(int)) == {140101}
    assert not set(filtered["forecast_origin"].astype(int)) & set(HB_ORIGINS)


def test_dense_fills_sparse_gaps_after_first_activity_only():
    sparse = pd.DataFrame(
        {
            "product": ["P", "P"],
            "date": [140103, 140105],
            "sales": [5.0, 7.0],
        }
    )
    dense = build_dense_actuals(sparse, through_date=140105)
    assert list(dense["date"]) == [140103, 140104, 140105]
    assert float(dense.loc[dense["date"] == 140104, "actual"].iloc[0]) == 0.0
    assert bool(dense.loc[dense["date"] == 140104, "is_missing_month"].iloc[0]) is True
    # no months before first activity
    assert dense["date"].min() == 140103
    # no months after through_date
    assert dense["date"].max() == 140105


def test_router_history_respects_origin_minus_2():
    df = pd.DataFrame(
        {
            "product": ["P"] * 4,
            "forecast_origin": [140201] * 4,
            "horizon": [1, 2, 3, 4],
            "target_date": [140111, 140112, 140201, 140202],
            "actual": [1, 1, 1, 1],
            "f_v2_policy": [1, 1, 1, 1],
            "f_a0_hybrid": [1, 1, 1, 1],
            "f_a0_only": [1, 1, 1, 1],
            "f_v21": [1, 1, 1, 1],
            "f_a0": [1.1, 1.1, 1.1, 1.1],
        }
    )
    # O=140201 → O-2 = 140111; only target_date <= 140111 allowed as history
    mask = historical_error_mask(df, 140201)
    assert set(df.loc[mask, "target_date"].tolist()) == {140111}

    dense = pd.DataFrame(
        {
            "product": ["P"] * 6,
            "date": [140106, 140107, 140108, 140109, 140110, 140111],
            "actual": [1, 0, 1, 0, 1, 2],
            "is_missing_month": [False] * 6,
        }
    )
    feats = build_router_features_sku_origin(policy_rows=df, dense=dense)
    assert not set(feats["forecast_origin"]) & set(HB_ORIGINS)
    # demand months only through O-2
    assert int(feats.iloc[0]["train_n_months"]) == 6  # 140106..140111


def test_freeze_requires_both_panel_lifts():
    wins = {"hybrid_wins": 10, "v2_wins": 3, "ties": 0}
    # deployable positive, common-native zero/neg → V2
    rec = make_recommendation(
        lift_deployable_wmape=1.0,
        lift_common_native_wmape=-0.1,
        wins=wins,
        bootstrap_frac_positive=0.95,
        drop_top3_lift_positive=True,
        rmse_hybrid=1.0,
        rmse_v2=1.0,
        p95_hybrid=1.0,
        p95_v2=1.0,
        horizon_group_lifts={n: 0.5 for n, _ in HORIZON_GROUPS},
        unexplained_a0_defect=False,
        pooled_deployable_lift_wmape=1.0,
    )
    assert rec["frozen_policy"] == "V2_POLICY"
    assert rec["gates"]["lift_common_native_positive"] is False

    # both positive + pooled not worse → hybrid
    rec2 = make_recommendation(
        lift_deployable_wmape=1.0,
        lift_common_native_wmape=0.5,
        wins=wins,
        bootstrap_frac_positive=0.95,
        drop_top3_lift_positive=True,
        rmse_hybrid=1.0,
        rmse_v2=1.0,
        p95_hybrid=1.0,
        p95_v2=1.0,
        horizon_group_lifts={n: 0.5 for n, _ in HORIZON_GROUPS},
        unexplained_a0_defect=False,
        pooled_deployable_lift_wmape=0.2,
    )
    assert rec2["frozen_policy"] == "A0_HYBRID_POLICY"

    # unseen pass but pooled worse with 55 volume dominance → IE
    rec3 = make_recommendation(
        lift_deployable_wmape=1.0,
        lift_common_native_wmape=0.5,
        wins=wins,
        bootstrap_frac_positive=0.95,
        drop_top3_lift_positive=True,
        rmse_hybrid=1.0,
        rmse_v2=1.0,
        p95_hybrid=1.0,
        p95_v2=1.0,
        horizon_group_lifts={n: 0.5 for n, _ in HORIZON_GROUPS},
        unexplained_a0_defect=False,
        pooled_deployable_lift_wmape=-1.0,
        original55_volume_share=0.7,
        original55_ae_reduction=-1e6,
        unseen_ae_reduction=1e5,
    )
    assert rec3["choice"] == "INSUFFICIENT_EVIDENCE"
    assert rec3["frozen_policy"] is None


def test_horizon_groups_are_fixed():
    names = [n for n, _ in HORIZON_GROUPS]
    assert names == ["h1_3", "h4_6", "h7_12", "h13_15"]
    assert HORIZON_GROUPS[0][1] == frozenset({1, 2, 3})
    assert HORIZON_GROUPS[1][1] == frozenset({4, 5, 6})
    assert HORIZON_GROUPS[2][1] == frozenset(range(7, 13))
    assert HORIZON_GROUPS[3][1] == frozenset({13, 14, 15})

    df = pd.DataFrame(
        {
            "horizon": list(range(1, 16)),
            "actual": [1.0] * 15,
            "f_v2_policy": [2.0] * 15,
            "f_a0_hybrid": [1.5] * 15,
        }
    )
    lifts = horizon_group_wmape_lifts(df, HORIZON_GROUPS)
    assert set(lifts) == set(names)


def test_lock_cannot_be_overwritten(tmp_path: Path):
    path = tmp_path / "ts_architecture_lock.json"
    write_lock_atomic(path, {"choice": "V2_POLICY", "v": 1})
    with pytest.raises(LockExistsError):
        write_lock_atomic(path, {"choice": "A0_HYBRID_POLICY", "v": 2})
    loaded = assert_lock_immutable_read(path)
    assert loaded["choice"] == "V2_POLICY"


def test_postlock_cannot_mutate_lock(tmp_path: Path):
    path = tmp_path / "ts_architecture_lock.json"
    write_lock_atomic(path, {"choice": "V2_POLICY"})
    with pytest.raises(PermissionError, match="must not mutate"):
        refuse_lock_mutation(path)
    # content unchanged
    assert assert_lock_immutable_read(path)["choice"] == "V2_POLICY"


def test_policy_fallback_categories_and_no_hb():
    from pkg.benchmark.calendar import shamsi_add_months
    from pkg.portfolio_ts_lock.completeness import audit_product_origin_completeness

    sparse = pd.DataFrame(
        {
            "product": ["P"] * 20,
            "date": [
                140001,
                140002,
                140003,
                140004,
                140005,
                140006,
                140007,
                140008,
                140009,
                140010,
                140011,
                140012,
                140101,
                140102,
                140103,
                140104,
                140105,
                140106,
                140107,
                140108,
            ],
            "sales": [1.0] * 20,
        }
    )
    # Extend sparse through full H=15 targets from 140101
    origin = 140101
    extra_dates = [shamsi_add_months(origin, h - 1) for h in range(1, 16)]
    # also need history before origin-2
    more = []
    for d in extra_dates:
        more.append({"product": "P", "date": d, "sales": 1.0})
    # pad earlier months for naive
    for d in [
        139901,
        139902,
        139903,
        139904,
        139905,
        139906,
        139907,
        139908,
        139909,
        139910,
        139911,
        139912,
    ]:
        more.append({"product": "P", "date": d, "sales": 1.0})
    sparse = pd.concat([sparse, pd.DataFrame(more)], ignore_index=True).drop_duplicates(
        ["product", "date"]
    )
    dense = build_dense_actuals(sparse, through_date=LOCKED_SCORE_END)

    rows_v = []
    rows_a = []
    for h in range(1, 16):
        td = shamsi_add_months(origin, h - 1)
        rows_v.append(
            {
                "product": "P",
                "forecast_origin": origin,
                "horizon": h,
                "target_date": td,
                "forecast": 2.0,
                "model": "ets",
                "job_status": "SUCCESS",
                "engine": "v2.1",
                "seed": np.nan,
            }
        )
        for seed in (41, 42, 43):
            rows_a.append(
                {
                    "product": "P",
                    "forecast_origin": origin,
                    "horizon": h,
                    "target_date": td,
                    "forecast": 4.0,
                    "model": "legacy_adaptive_recursive_lstm",
                    "job_status": "SUCCESS",
                    "engine": "a0",
                    "seed": seed,
                }
            )
    rows_v.append(
        {
            "product": "P",
            "forecast_origin": 140304,
            "horizon": 1,
            "target_date": 140304,
            "forecast": 99.0,
            "model": "ets",
            "job_status": "SUCCESS",
            "engine": "v2.1",
            "seed": np.nan,
        }
    )
    panel_v = pd.DataFrame(rows_v)
    panel_a = pd.DataFrame(rows_a)
    status, _inc = audit_product_origin_completeness(
        panel_v21=panel_v,
        panel_a0=panel_a,
        dense=dense,
        original55=set(),
    )
    common, deployable, cats, _partial = build_policy_panels(
        panel_v21=panel_v,
        panel_a0=panel_a,
        dense=dense,
        completeness_status=status,
    )
    assert not set(deployable["forecast_origin"].astype(int)) & set(HB_ORIGINS)
    assert int(len(deployable)) % 15 == 0
    assert "both_native" in set(cats["fallback_category"])
    assert abs(float(common["f_a0_hybrid"].iloc[0]) - 3.0) < 1e-9


def test_sku_origin_tie_eps():
    df = pd.DataFrame(
        {
            "product": ["A", "A"],
            "forecast_origin": [140101, 140101],
            "actual": [1.0, 1.0],
            "f_v2_policy": [1.0, 1.0],
            "f_a0_hybrid": [1.0, 1.0],
        }
    )
    w = sku_origin_wins(df)
    assert w["ties"] == 1
    assert w["hybrid_wins"] == 0
    assert w["v2_wins"] == 0
