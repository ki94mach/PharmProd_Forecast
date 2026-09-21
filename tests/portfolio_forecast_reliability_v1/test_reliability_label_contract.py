"""Tests for portfolio_forecast_reliability_v1 label contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
    ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY,
    FAMILY,
    FAMILY_DIR_REL,
    WMAPE_DENOM_EPS,
    WMAPE_DENOM_RULE,
)
from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
    LABEL_HR,
    LABEL_IE,
    LABEL_MNIH,
    LABEL_MR,
    PlanningCeilingsRequired,
    decide_sku_group_label,
    freeze_reliability_label_contract_v1,
    required_noninferior_origins,
    validity_range_prefix,
    wmape_denominator_valid,
    write_corrected_review,
)


def test_family_name():
    assert FAMILY == "portfolio_forecast_reliability_v1"


def test_required_noninferior_origins_ceil():
    assert required_noninferior_origins(3) == 3
    assert required_noninferior_origins(4) == 3
    assert required_noninferior_origins(2) == 2


def test_wmape_denominator_gt_1e_9():
    assert WMAPE_DENOM_RULE == "sum(abs(actual)) > 1e-9"
    assert wmape_denominator_valid(1.0) is True
    assert wmape_denominator_valid(WMAPE_DENOM_EPS * 10) is True
    assert wmape_denominator_valid(WMAPE_DENOM_EPS) is False  # strict >
    assert wmape_denominator_valid(WMAPE_DENOM_EPS * 0.5) is False
    assert wmape_denominator_valid(0.0) is False
    assert wmape_denominator_valid(float("nan")) is False
    assert wmape_denominator_valid(float("inf")) is False


def test_review_structure_corrections():
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        apply_review_structure_corrections,
        canonical_per_group_label_rules,
    )

    review = {
        "two_scopes": {
            "portfolio_activation_gate": {
                "gates_apply_here_only": [
                    "coverage_distribution",
                    "p99_ae_ratio",
                    "sku_cluster_bootstrap",
                    "wmape_denominator_must_be_valid",
                ]
            },
            "sku_validity_group_labels": {
                "per_group_label_rule": {"ordered": []},
                "proposed_numeric_thresholds": {
                    "sku_group_machine_wmape_ceiling_pct": {"value": None}
                },
            },
        },
        "labels": {LABEL_IE: {"meaning": "old"}},
        "approval_checklist": [],
        "proposed_numeric_thresholds_flat_for_approval": {
            "sku_group.sku_group_machine_wmape_ceiling_pct": {"value": None}
        },
        "planning_ceilings_pending": {},
    }
    apply_review_structure_corrections(review)
    gates = review["two_scopes"]["portfolio_activation_gate"]["gates_apply_here_only"]
    assert "max_ae_ratio" in gates
    assert gates.index("max_ae_ratio") == gates.index("p99_ae_ratio") + 1
    ordered = review["two_scopes"]["sku_validity_group_labels"]["per_group_label_rule"][
        "ordered"
    ]
    labels = [b["label"] for b in ordered]
    assert LABEL_MNIH in labels
    assert ordered[-1]["label"] == LABEL_HR
    assert labels.index(LABEL_MR) < labels.index(LABEL_MNIH) < len(labels) - 1
    denom_rule = next(b for b in ordered if "wmape_denominator_invalid" in b["when"])
    assert "1e-9" in denom_rule["when"]
    assert "non-finite" in denom_rule["when"]
    seg = review["two_scopes"]["sku_validity_group_labels"]["proposed_numeric_thresholds"][
        "sku_group_machine_wmape_ceiling_pct"
    ]["segment_specific_ceilings"]
    assert seg["must_define_before_opening_outcomes"] is True
    assert "segment_assignment" in seg
    assert "ceiling_precedence" in seg
    assert "missing_segment_behavior" in seg
    assert "1e-9 or non-finite" in review["labels"][LABEL_IE]["meaning"]
    # canonical helper matches
    canon = canonical_per_group_label_rules()
    assert any(b["label"] == LABEL_MNIH for b in canon["ordered"])


def test_contingency_switch_defaults_false():
    assert ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT is False
    assert ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY == (
        "allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling"
    )


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
        allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling=False,
    )
    d.update(kwargs)
    return d


def test_label_insufficient_incomplete_group():
    assert decide_sku_group_label(**_base(group_horizons_exact=False)) == LABEL_IE


def test_label_insufficient_lt2_origins():
    assert decide_sku_group_label(**_base(n_complete_origins=1)) == LABEL_IE


def test_label_insufficient_bad_denominator():
    assert decide_sku_group_label(**_base(sum_abs_actual=0.0)) == LABEL_IE
    assert decide_sku_group_label(**_base(sum_abs_actual=1e-9)) == LABEL_IE


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


def test_label_mnih_when_no_sku_ceiling_and_contingency_true():
    assert (
        decide_sku_group_label(
            **_base(
                sku_group_ceiling_pct=None,
                allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling=True,
            )
        )
        == LABEL_MNIH
    )


def test_label_hr_when_no_sku_ceiling_contingency_default_false():
    assert decide_sku_group_label(**_base(sku_group_ceiling_pct=None)) == LABEL_HR


def test_fallback_incomplete_does_not_block_earlier_range():
    r = validity_range_prefix([LABEL_MR, LABEL_MR, LABEL_IE, LABEL_HR])
    assert r["machine_reliable_groups"] == ["h1_3", "h4_6"]
    assert r["human_review_from_group"] == "h7_12"


def test_origin_stability_insufficient_wins():
    assert decide_sku_group_label(**_base(n_origins_noninferior=2)) == LABEL_HR


def test_label_insufficient_nan_metrics():
    assert decide_sku_group_label(**_base(machine_group_wmape=float("nan"))) == LABEL_IE
    assert decide_sku_group_label(**_base(human_group_wmape=None)) == LABEL_IE


def test_horizon_group_widths_locked():
    from pkg.portfolio_ts_lock.constants import HORIZON_GROUPS
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        GROUP_ORDER,
        GROUP_WIDTH,
    )

    assert GROUP_ORDER == ("h1_3", "h4_6", "h7_12", "h13_15")
    for name, hs in HORIZON_GROUPS:
        assert GROUP_WIDTH[name] == len(hs)


def test_freeze_refuses_without_planning_ceilings(tmp_path: Path):
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps({"hb_outcomes_inspected": False, "family": FAMILY}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanningCeilingsRequired):
        freeze_reliability_label_contract_v1(repo_root=tmp_path)
    assert not (family / "reliability_label_contract_v1.json").exists()


def test_freeze_with_ceilings_and_overwrite_prevention(tmp_path: Path):
    # Minimal upstream residual lock for evidence
    up = (
        tmp_path
        / "src/data/results/portfolio_residual_xgb_v1"
    )
    up.mkdir(parents=True)
    (up / "residual_pipeline_lock_v1.json").write_text(
        json.dumps(
            {
                "decision": "KEEP_FROZEN_TS",
                "production_pipeline": "FROZEN_TS",
                "selected_pipeline": "FROZEN_TS",
                "promotion_gates_pass": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps({"hb_outcomes_inspected": False, "family": FAMILY}) + "\n",
        encoding="utf-8",
    )
    (family / "hb_reliability_coverage_audit_v1.json").write_text(
        json.dumps({"hb_outcomes_inspected": False}) + "\n", encoding="utf-8"
    )
    summary = freeze_reliability_label_contract_v1(
        repo_root=tmp_path,
        portfolio_machine_wmape_ceiling_pct=40.0,
        sku_group_machine_wmape_ceiling_pct=45.0,
    )
    assert summary["ok"] is True
    assert summary["family"] == FAMILY
    assert summary["hb_outcomes_inspected"] is False
    contract = json.loads(
        (family / "reliability_label_contract_v1.json").read_text(encoding="utf-8")
    )
    assert contract["family"] == FAMILY
    assert contract["locked_thresholds"][ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] is False
    assert contract["locked_thresholds"]["wmape_denominator"]["rule"] == WMAPE_DENOM_RULE
    assert (
        contract["upstream_evidence"]["residual_pipeline_lock_v1"][
            "lock_decision_confirms_FROZEN_TS"
        ]
        is True
    )
    with pytest.raises(LockExistsError):
        freeze_reliability_label_contract_v1(
            repo_root=tmp_path,
            portfolio_machine_wmape_ceiling_pct=40.0,
            sku_group_machine_wmape_ceiling_pct=45.0,
        )


def test_write_corrected_review_no_final_contract(tmp_path: Path, repo_fixture=None):
    # Use real repo residual review when available via monkeypath — build minimal source
    src_dir = tmp_path / "src/data/results/portfolio_residual_xgb_v1"
    src_dir.mkdir(parents=True)
    (src_dir / "residual_pipeline_lock_v1.json").write_text(
        json.dumps(
            {
                "decision": "KEEP_FROZEN_TS",
                "production_pipeline": "FROZEN_TS",
                "selected_pipeline": "FROZEN_TS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (src_dir / "reliability_label_contract_v1_review.json").write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "approval_checklist": [],
                "explicit_non_goals": [],
                "proposed_numeric_thresholds_flat_for_approval": {
                    "portfolio.wmape_denominator_must_be_valid": {
                        "status": "PROPOSED_FOR_APPROVAL",
                        "value": True,
                        "definition": "old Σ|actual|>0",
                    },
                    "portfolio.portfolio_machine_wmape_ceiling_pct": {
                        "status": "PROPOSED_FOR_APPROVAL_VALUE_REQUIRED_FROM_PLANNING",
                        "value": None,
                    },
                },
                "labels": {
                    LABEL_IE: {"meaning": "old"},
                    LABEL_MNIH: {"meaning": "contingency"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = write_corrected_review(repo_root=tmp_path)
    assert summary["ok"] is True
    assert summary["final_contract_written"] is False
    assert summary["family"] == FAMILY
    assert summary[ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] is False
    out = tmp_path / FAMILY_DIR_REL / "reliability_label_contract_v1_review.json"
    review = json.loads(out.read_text(encoding="utf-8"))
    assert review["family"] == FAMILY
    assert review["final_contract_written"] is False
    assert review["stop_before_final_contract"] is True
    assert review[ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] is False
    assert review["wmape_denominator"]["rule"] == WMAPE_DENOM_RULE
    assert not (tmp_path / FAMILY_DIR_REL / "reliability_label_contract_v1.json").exists()
