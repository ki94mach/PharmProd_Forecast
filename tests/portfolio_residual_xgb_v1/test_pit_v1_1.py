"""Leakage / PIT / promotion-gate / job-inventory tests for residual-xgb v1.1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import CLEAN_QUANT_FEATURES
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_residual_xgb_v1.contract_v1_1 import write_immutable_json
from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS
from pkg.research.portfolio_residual_xgb_v1.encoders import TrainOnlyCategoricalEncoder
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    FEATURE_ASOF_RULES,
    SALES_FEATURE_NAMES,
    TS_RESID_FEATURE_LIST,
    assert_all_features_have_pit_rules,
    assert_no_sales_uses_origin_minus_1,
    filter_train_rows_pit,
    last_complete_month,
    sales_feature_month,
)
from pkg.research.portfolio_residual_xgb_v1.prepare_v1_1 import build_job_inventory_v1_1
from pkg.research.portfolio_residual_xgb_v1.promotion_gates import (
    PROMOTION_GATES,
    build_promotion_gate_artifact,
)


def test_sales_lags_use_last_complete_month_origin_minus_2():
    fo = 140107
    lc = last_complete_month(fo)
    assert lc == shamsi_add_months(fo, -2)
    assert sales_feature_month("sales_lag_1", fo) == lc
    assert sales_feature_month("sales_lag_2", fo) == shamsi_add_months(lc, -1)
    assert sales_feature_month("sales_lag_3", fo) == shamsi_add_months(lc, -2)
    assert sales_feature_month("sales_lag_12", fo) == shamsi_add_months(lc, -11)
    assert sales_feature_month("sales_roll3", fo) == lc
    assert_no_sales_uses_origin_minus_1(fo)
    # Explicitly: nothing equals Fo-1
    fo_m1 = shamsi_add_months(fo, -1)
    for name in SALES_FEATURE_NAMES:
        assert sales_feature_month(name, fo) < fo_m1 or sales_feature_month(name, fo) <= lc


def test_every_feature_has_pit_rule_including_clean_quant():
    for name in CLEAN_QUANT_FEATURES:
        assert name in FEATURE_ASOF_RULES
    assert_all_features_have_pit_rules(list(TS_RESID_FEATURE_LIST))


def test_pit_filter_three_predicates_with_sales_fo_minus_2():
    o = 140107
    o2 = shamsi_add_months(o, -2)
    frame = pd.DataFrame(
        [
            {"product": "P", "forecast_origin": 140104, "target_date": 140104, "horizon": 1},
            {"product": "P", "forecast_origin": 140107, "target_date": 140105, "horizon": 1},
            {"product": "P", "forecast_origin": 140104, "target_date": 140106, "horizon": 1},
        ]
    )
    kept = filter_train_rows_pit(frame, evaluation_origin=o, origin_minus_2=o2)
    assert len(kept) == 1
    assert int(kept.iloc[0]["forecast_origin"]) < o
    assert int(kept.iloc[0]["target_date"]) <= o2
    assert last_complete_month(int(kept.iloc[0]["forecast_origin"])) <= o2


def test_encoder_unknown_deterministic():
    train = pd.DataFrame(
        {
            "model": ["A", "B"],
            "Field": ["X", "Y"],
            "ProductForm": ["T", "T"],
            "Provider": ["P", "P"],
        }
    )
    test = pd.DataFrame(
        {
            "model": ["A", "UNSEEN"],
            "Field": ["X", "Z"],
            "ProductForm": ["T", "T"],
            "Provider": ["P", "P"],
        }
    )
    enc = TrainOnlyCategoricalEncoder().fit(train)
    out = enc.transform(test)
    assert "model_enc" in out.columns
    unk = enc.unknown_code_["model_enc"]
    assert int(out.loc[out["model"] == "UNSEEN", "model_enc"].iloc[0]) == unk


def test_job_inventory_smoke_and_full():
    origins = [140107, 140110, 140201, 140204, 140207, 140210, 140301]
    inv = build_job_inventory_v1_1(eligible_origins=origins)
    assert inv["smoke"]["origins"] == [140107]
    assert inv["smoke"]["n_xgb_fits"] == 1
    assert inv["full"]["n_xgb_fits"] == 7
    assert inv["outer_workers"] == 1
    assert inv["HUMAN_XGB"]["n_jobs"] == 0
    assert inv["HUMAN_XGB"]["status"] == HUMAN_XGB_STATUS


def test_promotion_gates_locked_before_results():
    art = build_promotion_gate_artifact(
        contract_v1_1_sha256="abc",
        eligible_evaluation_origins=[140107, 140110, 140201, 140204, 140207, 140210, 140301],
    )
    assert art["locked_before_results"] is True
    assert art["hb_outcomes_inspected"] is False
    assert art["gates"]["origins_improved_ge"] == 4
    assert art["gates"]["sku_cluster_bootstrap_frac_positive_ge"] == 0.90
    assert PROMOTION_GATES["p99_not_more_than_2pct_worse_than_frozen_ts"] is True


def test_immutable_write(tmp_path: Path):
    p = tmp_path / "residual_promotion_gates_v1.json"
    write_immutable_json(p, {"ok": True})
    with pytest.raises(LockExistsError):
        write_immutable_json(p, {"ok": False})
