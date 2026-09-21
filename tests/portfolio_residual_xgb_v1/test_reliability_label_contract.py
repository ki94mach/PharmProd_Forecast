"""Tests for reliability_label_contract_v1 decision rules and freeze I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_residual_xgb_v1.reliability_label_contract import (
    LABEL_HR,
    LABEL_IE,
    LABEL_MNIH,
    LABEL_MR,
    decide_sku_group_label,
    freeze_reliability_label_contract_v1,
    required_noninferior_origins,
    validity_range_prefix,
    wmape_denominator_valid,
)


def test_required_noninferior_origins_ceil():
    assert required_noninferior_origins(3) == 3
    assert required_noninferior_origins(4) == 3
    assert required_noninferior_origins(2) == 2


def test_wmape_denominator():
    assert wmape_denominator_valid(1.0) is True
    assert wmape_denominator_valid(0.0) is False
    assert wmape_denominator_valid(float("nan")) is False


def test_validity_range_mr_ie_is_h1_3_not_empty():
    r = validity_range_prefix([LABEL_MR, LABEL_IE, LABEL_HR, LABEL_MR])
    assert r["machine_reliable_groups"] == ["h1_3"]
    assert r["display_range"] == "h1–3"
    assert r["human_review_from_group"] == "h4_6"


def test_validity_range_empty_when_first_not_mr():
    r = validity_range_prefix([LABEL_IE, LABEL_MR, LABEL_MR, LABEL_MR])
    assert r["machine_reliable_groups"] == []
    assert r["human_review_from_group"] == "h1_3"


def _base(**kwargs):
    d = dict(
        n_complete_origins=3,
        group_horizons_exact=True,
        sum_abs_actual=100.0,
        machine_group_wmape=30.0,
        human_group_wmape=31.0,
        n_origins_noninferior=3,
        portfolio_activation_passed=True,
        deployable_branch="both_native",
        sku_group_ceiling_pct=40.0,
        portfolio_ceiling_pct=35.0,
        allow_machine_noninferior_without_sku_ceiling=False,
    )
    d.update(kwargs)
    return d


def test_label_insufficient_incomplete_group():
    assert decide_sku_group_label(**_base(group_horizons_exact=False)) == LABEL_IE


def test_label_insufficient_lt2_origins():
    assert decide_sku_group_label(**_base(n_complete_origins=1)) == LABEL_IE


def test_label_insufficient_bad_denominator():
    assert decide_sku_group_label(**_base(sum_abs_actual=0.0)) == LABEL_IE


def test_label_human_review_exactly_2_origins():
    assert decide_sku_group_label(**_base(n_complete_origins=2)) == LABEL_HR


def test_label_machine_reliable_pass():
    assert decide_sku_group_label(**_base()) == LABEL_MR


def test_label_human_review_when_relative_fails():
    assert (
        decide_sku_group_label(**_base(machine_group_wmape=34.0, human_group_wmape=31.0))
        == LABEL_HR
    )


def test_label_human_review_neither_naive_fallback():
    assert decide_sku_group_label(**_base(deployable_branch="neither_naive")) == LABEL_HR


def test_label_human_review_portfolio_gate_failed():
    assert (
        decide_sku_group_label(**_base(portfolio_activation_passed=False)) == LABEL_HR
    )


def test_label_absolute_ceiling_fail():
    assert (
        decide_sku_group_label(
            **_base(machine_group_wmape=45.0, sku_group_ceiling_pct=40.0)
        )
        == LABEL_HR
    )


def test_label_mnih_when_no_sku_ceiling_and_contingency():
    assert (
        decide_sku_group_label(
            **_base(
                sku_group_ceiling_pct=None,
                allow_machine_noninferior_without_sku_ceiling=True,
            )
        )
        == LABEL_MNIH
    )


def test_label_hr_when_no_sku_ceiling_without_contingency():
    assert decide_sku_group_label(**_base(sku_group_ceiling_pct=None)) == LABEL_HR


def test_fallback_incomplete_does_not_block_earlier_range():
    r = validity_range_prefix([LABEL_MR, LABEL_MR, LABEL_IE, LABEL_HR])
    assert r["machine_reliable_groups"] == ["h1_3", "h4_6"]
    assert r["human_review_from_group"] == "h7_12"


def test_origin_stability_insufficient_wins():
    # 3 origins require 3 noninferior; only 2 → HR
    assert decide_sku_group_label(**_base(n_origins_noninferior=2)) == LABEL_HR


def test_label_insufficient_nan_metrics():
    assert decide_sku_group_label(**_base(machine_group_wmape=float("nan"))) == LABEL_IE
    assert decide_sku_group_label(**_base(human_group_wmape=None)) == LABEL_IE


def test_horizon_group_widths_locked():
    from pkg.research.portfolio_residual_xgb_v1.reliability_label_contract import (
        GROUP_ORDER,
        GROUP_WIDTH,
    )
    from pkg.portfolio_ts_lock.constants import HORIZON_GROUPS

    assert GROUP_ORDER == ("h1_3", "h4_6", "h7_12", "h13_15")
    for name, hs in HORIZON_GROUPS:
        assert GROUP_WIDTH[name] == len(hs)


def test_overwrite_prevention(tmp_path: Path):
    family = tmp_path / "src" / "data" / "results" / "portfolio_residual_xgb_v1"
    family.mkdir(parents=True)
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps({"hb_outcomes_inspected": False}) + "\n", encoding="utf-8"
    )
    (family / "hb_reliability_coverage_audit_v1.json").write_text(
        json.dumps({"hb_outcomes_inspected": False}) + "\n", encoding="utf-8"
    )
    summary = freeze_reliability_label_contract_v1(repo_root=tmp_path)
    assert summary["ok"] is True
    assert summary["hb_outcomes_inspected"] is False
    path = family / "reliability_label_contract_v1.json"
    assert path.is_file()
    assert path.with_suffix(path.suffix + ".sha256").is_file()
    with pytest.raises(LockExistsError):
        freeze_reliability_label_contract_v1(repo_root=tmp_path)
