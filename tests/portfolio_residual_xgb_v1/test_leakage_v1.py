"""Leakage, PIT, origin-class, and HUMAN_XGB blocking tests for residual-xgb v1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import CLEAN_QUANT_FEATURES
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_residual_xgb_v1.constants import (
    HUMAN_XGB_STATUS,
    MIN_PRIOR_MATURED_ORIGINS,
    ORIGIN_CLASS_EVALUABLE,
    ORIGIN_CLASS_INSUFFICIENT,
    ORIGIN_CLASS_WARMUP,
)
from pkg.research.portfolio_residual_xgb_v1.contract import write_immutable_json
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    FEATURE_ASOF_RULES,
    TS_RESID_FEATURE_LIST,
    assert_all_features_have_pit_rules,
    filter_train_rows_pit,
)
from pkg.research.portfolio_residual_xgb_v1.prepare import (
    audit_hb_availability_budget_origins_only,
    build_job_plan,
)
from pkg.research.portfolio_residual_xgb_v1.splits import (
    classify_origin,
    sorted_residual_origins,
)


def test_every_clean_quant_feature_has_pit_rule():
    for name in CLEAN_QUANT_FEATURES:
        assert name in FEATURE_ASOF_RULES
    assert_all_features_have_pit_rules(list(TS_RESID_FEATURE_LIST))


def test_pit_filter_enforces_all_three_predicates():
    # O=140107 → O-2=140105; sales last_complete = Fo-2
    o = 140107
    o2 = shamsi_add_months(o, -2)
    frame = pd.DataFrame(
        [
            {
                "product": "P",
                "forecast_origin": 140104,
                "target_date": 140104,
                "horizon": 1,
            },
            {
                "product": "P",
                "forecast_origin": 140107,
                "target_date": 140105,
                "horizon": 1,
            },
            {
                "product": "P",
                "forecast_origin": 140104,
                "target_date": 140106,
                "horizon": 1,
            },
        ]
    )
    kept = filter_train_rows_pit(frame, evaluation_origin=o, origin_minus_2=o2)
    assert len(kept) == 1
    assert int(kept.iloc[0]["forecast_origin"]) == 140104
    assert int(kept.iloc[0]["target_date"]) <= o2
    from pkg.research.portfolio_residual_xgb_v1.pit_features import last_complete_month

    assert last_complete_month(int(kept.iloc[0]["forecast_origin"])) <= o2


def test_no_random_split_path_in_classify():
    origins = sorted_residual_origins()
    # first origins are warmup
    assert (
        classify_origin(
            evaluation_origin=origins[0],
            residual_origins_sorted=origins,
            train_rows=10_000,
            train_sku_origins=500,
        )
        == ORIGIN_CLASS_WARMUP
    )
    # past warmup but low rows → insufficient
    late = origins[MIN_PRIOR_MATURED_ORIGINS]
    assert (
        classify_origin(
            evaluation_origin=late,
            residual_origins_sorted=origins,
            train_rows=10,
            train_sku_origins=500,
        )
        == ORIGIN_CLASS_INSUFFICIENT
    )
    assert (
        classify_origin(
            evaluation_origin=late,
            residual_origins_sorted=origins,
            train_rows=5000,
            train_sku_origins=100,
        )
        == ORIGIN_CLASS_EVALUABLE
    )


def test_job_plan_human_xgb_blocked_zero_jobs():
    plan = build_job_plan(eligible_origins=[140201, 140204], n_products=10)
    assert plan["HUMAN_XGB"]["status"] == HUMAN_XGB_STATUS
    assert plan["HUMAN_XGB"]["n_jobs"] == 0
    assert plan["HUMAN_XGB"]["executable"] is False
    assert plan["total_executable_jobs"] == 2


def test_contract_create_only(tmp_path: Path):
    path = tmp_path / "residual_study_contract_v1.json"
    write_immutable_json(path, {"ok": True})
    with pytest.raises(LockExistsError):
        write_immutable_json(path, {"ok": False})


def test_hb_audit_reads_only_budget_origin(tmp_path: Path):
    # Minimal parquet with budget_origin only (+ decoy sales that must not be required)
    path = tmp_path / "budget_universe.parquet"
    pd.DataFrame(
        {
            "budget_origin": [140304, 140307, 140310, 140312],
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(path, index=False)
    audit = audit_hb_availability_budget_origins_only(path)
    assert audit["overlap_with_residual_development"] == []
    assert audit["human_xgb_status"] == HUMAN_XGB_STATUS
    assert audit["hb_outcomes_inspected"] is False
    assert audit["columns_read"] == ["budget_origin"]


def test_refuse_missing_pit_feature():
    with pytest.raises(ValueError, match="missing PIT"):
        assert_all_features_have_pit_rules(["not_a_real_feature"])
