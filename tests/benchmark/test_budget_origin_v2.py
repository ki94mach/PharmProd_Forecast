"""Unit tests for HB budget origin v2 (Version → quarter → origin_from_quarter)."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from pkg.benchmark.budget_origin_v2 import (
    BudgetOriginAmbiguityError,
    BudgetOriginDuplicateError,
    REASON_BEYOND_H15,
    REASON_PRE_ORIGIN,
    aggregate_forecast_mean,
    assert_no_duplicate_poh,
    assert_version_quarter_consistent,
    assign_budget_origin_from_version,
    build_budget_panel_v2,
    completeness_by_origin,
    filter_horizons_1_15,
    horizon_from_origin_target,
)
from pkg.benchmark.calendar import origin_from_quarter
from pkg.db.query.product_budget import version_to_qrt


def test_all_four_jalali_quarter_to_origin_mappings():
    assert origin_from_quarter("1403Q1") == 140301
    assert origin_from_quarter("1403Q2") == 140304
    assert origin_from_quarter("1403Q3") == 140307
    assert origin_from_quarter("1403Q4") == 140310
    assert version_to_qrt(14030101) == "1403Q1"
    assert version_to_qrt(14030201) == "1403Q2"
    assert version_to_qrt(14030301) == "1403Q3"
    assert version_to_qrt(14030401) == "1403Q4"


def test_year_rollover_140312_to_140401_only_for_1404q1():
    df = pd.DataFrame(
        {
            "product": ["P", "P", "P"],
            "generic": ["G", "G", "G"],
            "version": [14040101, 14040101, 14030401],
            "qrt": ["1404Q1", "1404Q1", "1403Q4"],
            "date": [140312, 140401, 140312],
            "forecast": [1.0, 2.0, 3.0],
        }
    )
    assigned = assign_budget_origin_from_version(df)
    # 1404Q1 + target 140312 → origin 140401, horizon 0
    q1 = assigned.loc[
        (assigned["source_budget_quarter"] == "1404Q1") & (assigned["date"] == 140312)
    ].iloc[0]
    assert int(q1["budget_origin"]) == 140401
    assert int(q1["horizon"]) == 0
    q4 = assigned.loc[assigned["source_budget_quarter"] == "1403Q4"].iloc[0]
    assert int(q4["budget_origin"]) == 140310
    assert int(q4["horizon"]) == 3


def test_horizon_0_and_16_exclusion():
    df = pd.DataFrame(
        {
            "product": ["P"] * 4,
            "generic": ["G"] * 4,
            "version": [14040101] * 4,
            "qrt": ["1404Q1"] * 4,
            "date": [140312, 140401, 140503, 140504],
            "forecast": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assigned = assign_budget_origin_from_version(df)
    kept, excl = filter_horizons_1_15(assigned)
    assert set(kept["target_date"].astype(int)) == {140401, 140503}
    assert set(kept["horizon"].astype(int)) == {1, 15}
    reasons = set(excl["exclusion_reason"])
    assert REASON_PRE_ORIGIN in reasons
    assert REASON_BEYOND_H15 in reasons
    pre = excl.loc[excl["exclusion_reason"] == REASON_PRE_ORIGIN]
    assert int(pre["target_date"].iloc[0]) == 140312
    beyond = excl.loc[excl["exclusion_reason"] == REASON_BEYOND_H15]
    assert int(beyond["target_date"].iloc[0]) == 140504


def test_correct_h1_and_h15_assignment():
    assert horizon_from_origin_target(140401, 140401) == 1
    assert horizon_from_origin_target(140401, 140503) == 15
    assert horizon_from_origin_target(140310, 140312) == 3


def test_incomplete_groups_remain_incomplete_no_rebasing():
    # After excluding h16, only horizons 1..14 present — incomplete, not rebased
    rows = []
    for h in range(1, 15):
        # origin 140401; target = origin + (h-1)
        # use assign path
        rows.append(
            {
                "product": "P",
                "generic": "G",
                "version": 14040101,
                "qrt": "1404Q1",
                "date": 140401 if h == 1 else None,  # fill below
                "forecast": float(h),
            }
        )
    # build dates via horizon formula inverse: target months from origin
    from pkg.benchmark.calendar import shamsi_add_months

    for i, r in enumerate(rows):
        r["date"] = shamsi_add_months(140401, i)
    df = pd.DataFrame(rows)
    # add a beyond-h15 row that gets excluded — must not shift others
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "product": "P",
                        "generic": "G",
                        "version": 14040101,
                        "qrt": "1404Q1",
                        "date": shamsi_add_months(140401, 15),
                        "forecast": 99.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    panel, excl, _stats = build_budget_panel_v2(df)
    assert set(panel["horizon"].astype(int)) == set(range(1, 15))
    assert 16 not in set(panel["horizon"].astype(int))
    # exclusion of h16 did not renumber h1..14
    assert int(panel.loc[panel["horizon"] == 1, "target_date"].iloc[0]) == 140401
    comp = completeness_by_origin(panel)
    assert comp["n_incomplete_groups"] == 1
    assert comp["incomplete_groups"][0]["missing"] == [15]


def test_no_horizon_rebasing_after_exclusions():
    df = pd.DataFrame(
        {
            "product": ["P", "P"],
            "generic": ["G", "G"],
            "version": [14040101, 14040101],
            "qrt": ["1404Q1", "1404Q1"],
            "date": [140312, 140402],  # h0 excluded; h2 kept as 2 not rebased to 1
            "forecast": [1.0, 2.0],
        }
    )
    panel, excl, _ = build_budget_panel_v2(df)
    assert len(excl) == 1
    assert int(panel["horizon"].iloc[0]) == 2
    assert int(panel["target_date"].iloc[0]) == 140402


def test_duplicate_detection_hard_fail():
    df = pd.DataFrame(
        {
            "product": ["P", "P"],
            "budget_origin": [140401, 140401],
            "horizon": [1, 1],
            "target_date": [140401, 140401],
        }
    )
    with pytest.raises(BudgetOriginDuplicateError):
        assert_no_duplicate_poh(df)


def test_qrt_mismatch_hard_fail():
    df = pd.DataFrame(
        {
            "product": ["P"],
            "version": [14040101],
            "qrt": ["1403Q4"],  # disagrees with Version
            "date": [140401],
            "forecast": [1.0],
        }
    )
    with pytest.raises(BudgetOriginAmbiguityError):
        assert_version_quarter_consistent(df)


def test_invalid_version_qq_hard_fail():
    df = pd.DataFrame(
        {
            "product": ["P"],
            "version": [14040501],  # QQ=5 invalid
            "qrt": ["1404Q5"],
            "date": [140401],
            "forecast": [1.0],
        }
    )
    with pytest.raises(BudgetOriginAmbiguityError):
        assert_version_quarter_consistent(df)


def test_deterministic_output_and_hashes():
    df = pd.DataFrame(
        {
            "product": ["B", "A", "A"],
            "generic": ["G", "G", "G"],
            "version": [14030201, 14030201, 14030201],
            "qrt": ["1403Q2", "1403Q2", "1403Q2"],
            "date": [140304, 140304, 140305],
            "forecast": [10.0, 10.0, 20.0],
        }
    )
    # duplicate (A,1403Q2,version,140304) twice with same forecast — aggregate mean
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "product": "A",
                        "generic": "G",
                        "version": 14030201,
                        "qrt": "1403Q2",
                        "date": 140304,
                        "forecast": 10.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    p1, e1, s1 = build_budget_panel_v2(df)
    p2, e2, s2 = build_budget_panel_v2(df)
    h1 = hashlib.sha256(
        p1.sort_values(["product", "horizon"]).to_csv(index=False).encode()
    ).hexdigest()
    h2 = hashlib.sha256(
        p2.sort_values(["product", "horizon"]).to_csv(index=False).encode()
    ).hexdigest()
    assert h1 == h2
    assert s1["n_rows_kept_1_15"] == s2["n_rows_kept_1_15"]
    # A@140304 appears once after aggregate
    assert len(p1.loc[(p1["product"] == "A") & (p1["horizon"] == 1)]) == 1


def test_aggregate_mean_is_documented_resolve():
    df = pd.DataFrame(
        {
            "product": ["P", "P"],
            "qrt": ["1404Q1", "1404Q1"],
            "version": [14040101, 14040101],
            "date": [140401, 140401],
            "forecast": [10.0, 20.0],
        }
    )
    agg = aggregate_forecast_mean(df)
    assert len(agg) == 1
    assert float(agg["forecast"].iloc[0]) == 15.0
