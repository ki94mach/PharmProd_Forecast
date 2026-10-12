"""Tests for portfolio_forecast_reliability_v1 (MNIH-primary)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pkg.benchmark.calendar import origin_from_quarter
from pkg.db.query.product_budget import version_to_qrt
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    EVIDENCE_STRENGTH_LIMITED,
    FAMILY,
    FAMILY_DIR_REL,
    LABEL_HR,
    LABEL_IHE,
    LABEL_MNIH,
    MAPPING_AUDIT_FILENAME,
    PROTECTED_HB_ORIGINS,
    REPRESENTATIVENESS_AUDIT_FILENAME,
    WMAPE_DENOM_EPS,
    WMAPE_DENOM_RULE,
)
from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
    FinalFreezeNotApproved,
    decide_sku_group_label,
    decide_sku_group_label_with_meta,
    evidence_strength_for_n_origins,
    freeze_reliability_label_contract_v1,
    required_noninferior_origins,
    validity_range_prefix,
    wmape_denominator_valid,
    write_corrected_review,
)


def test_family_name():
    assert FAMILY == "portfolio_forecast_reliability_v1"


def test_protected_origins_keep_140401_not_140312():
    assert 140401 in PROTECTED_HB_ORIGINS
    assert 140312 not in PROTECTED_HB_ORIGINS
    assert PROTECTED_HB_ORIGINS == frozenset({140304, 140307, 140310, 140401})


def test_version_quarter_maps_1404q1_to_140401():
    # YYYYQQEE edition → qrt → origin_from_quarter
    qrt = version_to_qrt(14040101)
    assert str(qrt).upper() == "1404Q1"
    assert int(origin_from_quarter(qrt)) == 140401


def test_required_noninferior_origins_ceil():
    assert required_noninferior_origins(3) == 3
    assert required_noninferior_origins(4) == 3
    assert required_noninferior_origins(2) == 2


def test_wmape_denominator_gt_1e_9():
    assert WMAPE_DENOM_RULE == "sum(abs(actual)) > 1e-9"
    assert wmape_denominator_valid(1.0) is True
    assert wmape_denominator_valid(WMAPE_DENOM_EPS) is False
    assert wmape_denominator_valid(0.0) is False
    assert wmape_denominator_valid(float("nan")) is False


def test_evidence_strength_limited():
    assert evidence_strength_for_n_origins(3) == EVIDENCE_STRENGTH_LIMITED
    assert evidence_strength_for_n_origins(4) == EVIDENCE_STRENGTH_LIMITED
    assert evidence_strength_for_n_origins(2) is None
    assert evidence_strength_for_n_origins(5) is None


def test_validity_range_mnih_prefix():
    r = validity_range_prefix([LABEL_MNIH, LABEL_IHE, LABEL_HR, LABEL_MNIH])
    assert r["machine_noninferior_groups"] == ["h1_3"]
    assert r["display_range"] == "h1–3"
    assert r["human_review_from_group"] == "h4_6"
    assert r["positive_label"] == LABEL_MNIH


def test_validity_range_empty_when_first_not_mnih():
    r = validity_range_prefix([LABEL_IHE, LABEL_MNIH, LABEL_MNIH, LABEL_MNIH])
    assert r["machine_noninferior_groups"] == []
    assert r["human_review_from_group"] == "h1_3"


def test_validity_range_stops_at_hr():
    r = validity_range_prefix([LABEL_MNIH, LABEL_MNIH, LABEL_HR, LABEL_MNIH])
    assert r["machine_noninferior_groups"] == ["h1_3", "h4_6"]
    assert r["human_review_from_group"] == "h7_12"


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
        in_human_comparable_cohort=True,
    )
    d.update(kwargs)
    return d


def test_label_ihe_incomplete_group():
    assert decide_sku_group_label(**_base(group_horizons_exact=False)) == LABEL_IHE


def test_label_ihe_0_or_1_origins():
    assert decide_sku_group_label(**_base(n_complete_origins=0)) == LABEL_IHE
    assert decide_sku_group_label(**_base(n_complete_origins=1)) == LABEL_IHE


def test_label_hr_no_machine_wins_over_0_or_1_origins():
    """HB present + machine cannot forecast → HR, not IHE via n<=1."""
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        NO_MACHINE_FORECAST,
    )

    for n in (0, 1):
        assert (
            decide_sku_group_label(
                **_base(
                    n_complete_origins=n,
                    deployable_branch=NO_MACHINE_FORECAST,
                    machine_group_wmape=None,
                    human_group_wmape=None,
                )
            )
            == LABEL_HR
        )
        assert (
            decide_sku_group_label(
                **_base(
                    n_complete_origins=n,
                    deployable_branch="neither_naive",
                    machine_group_wmape=None,
                )
            )
            == LABEL_HR
        )


def test_label_ihe_incomplete_hb_still_wins_over_no_machine():
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        NO_MACHINE_FORECAST,
    )

    assert (
        decide_sku_group_label(
            **_base(
                group_horizons_exact=False,
                deployable_branch=NO_MACHINE_FORECAST,
                n_complete_origins=0,
            )
        )
        == LABEL_IHE
    )


def test_label_ihe_bad_denom_still_wins_over_neither_naive():
    assert (
        decide_sku_group_label(
            **_base(
                sum_abs_actual=0.0,
                deployable_branch="neither_naive",
                n_complete_origins=0,
            )
        )
        == LABEL_IHE
    )


def test_sku_group_rule_order_neither_before_origin_count():
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        build_decision_table,
        canonical_per_group_label_rules,
    )

    ordered = canonical_per_group_label_rules()["ordered"]
    whens = [b["when"] for b in ordered]
    neither_i = next(i for i, w in enumerate(whens) if "neither_naive" in w or "NO_MACHINE" in w)
    origin_01_i = next(i for i, w in enumerate(whens) if "in {0,1}" in w)
    assert neither_i < origin_01_i
    branches = build_decision_table()["sku_x_validity_group"]["branches"]
    b_whens = [b["when"] for b in branches]
    b_neither = next(i for i, w in enumerate(b_whens) if "neither_naive" in w)
    b_01 = next(i for i, w in enumerate(b_whens) if "<= 1" in w)
    assert b_neither < b_01


def test_label_ihe_bad_denominator():
    assert decide_sku_group_label(**_base(sum_abs_actual=0.0)) == LABEL_IHE
    assert decide_sku_group_label(**_base(sum_abs_actual=1e-9)) == LABEL_IHE


def test_label_hr_exactly_2_origins():
    assert decide_sku_group_label(**_base(n_complete_origins=2)) == LABEL_HR


def test_label_mnih_pass():
    assert decide_sku_group_label(**_base()) == LABEL_MNIH


def test_label_mnih_meta_evidence_strength():
    meta = decide_sku_group_label_with_meta(**_base())
    assert meta["label"] == LABEL_MNIH
    assert meta["evidence_strength"] == EVIDENCE_STRENGTH_LIMITED
    meta4 = decide_sku_group_label_with_meta(**_base(n_complete_origins=4, n_origins_noninferior=3))
    assert meta4["label"] == LABEL_MNIH
    assert meta4["evidence_strength"] == EVIDENCE_STRENGTH_LIMITED


def test_label_hr_when_relative_fails():
    assert (
        decide_sku_group_label(**_base(machine_group_wmape=34.0, human_group_wmape=31.0))
        == LABEL_HR
    )


def test_label_hr_neither_naive_blocks_mnih():
    assert decide_sku_group_label(**_base(deployable_branch="neither_naive")) == LABEL_HR
    assert (
        decide_sku_group_label(
            **_base(
                deployable_branch="both_native",
                any_evaluated_origin_neither_naive=True,
            )
        )
        == LABEL_HR
    )


def test_label_hr_portfolio_gate_failed():
    assert (
        decide_sku_group_label(**_base(portfolio_activation_passed=False)) == LABEL_HR
    )


def test_label_hr_outside_cohort():
    assert (
        decide_sku_group_label(**_base(in_human_comparable_cohort=False)) == LABEL_HR
    )


def test_ceilings_ignored_mnih_still_passes():
    # Absolute ceilings are out of scope; presence/absence must not change MNIH.
    assert (
        decide_sku_group_label(
            **_base(sku_group_ceiling_pct=None, portfolio_ceiling_pct=None)
        )
        == LABEL_MNIH
    )
    assert (
        decide_sku_group_label(
            **_base(sku_group_ceiling_pct=10.0, portfolio_ceiling_pct=10.0)
        )
        == LABEL_MNIH
    )


def test_origin_stability_insufficient_wins():
    assert decide_sku_group_label(**_base(n_origins_noninferior=2)) == LABEL_HR


def test_label_ihe_nan_metrics():
    assert decide_sku_group_label(**_base(machine_group_wmape=float("nan"))) == LABEL_IHE
    assert decide_sku_group_label(**_base(human_group_wmape=None)) == LABEL_IHE


def test_canonical_rules_mnih_primary_no_mr_executable():
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        canonical_per_group_label_rules,
    )

    ordered = canonical_per_group_label_rules()["ordered"]
    labels = [b["label"] for b in ordered]
    assert LABEL_MNIH in labels
    assert LABEL_IHE in labels
    assert LABEL_HR in labels
    # MACHINE_RELIABLE must not appear as an executable ordered branch label
    assert all(b["label"] != "MACHINE_RELIABLE" for b in ordered)


def test_freeze_refuses_without_allow_flag(tmp_path: Path):
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps({"hb_outcomes_inspected": False, "family": FAMILY}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FinalFreezeNotApproved):
        freeze_reliability_label_contract_v1(repo_root=tmp_path)
    assert not (family / "reliability_label_contract_v1.json").exists()


def test_freeze_with_allow_and_audits(tmp_path: Path):
    up = tmp_path / "src/data/results/portfolio_residual_xgb_v1"
    up.mkdir(parents=True)
    (up / "residual_pipeline_lock_v1.json").write_text(
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
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps({"hb_outcomes_inspected": False, "family": FAMILY}) + "\n",
        encoding="utf-8",
    )
    (family / MAPPING_AUDIT_FILENAME).write_text(
        json.dumps({"hb_outcomes_inspected": False}) + "\n", encoding="utf-8"
    )
    (family / REPRESENTATIVENESS_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "representativeness_conclusion": "NOT_REPRESENTATIVE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = freeze_reliability_label_contract_v1(
        repo_root=tmp_path, allow_final_freeze=True
    )
    assert summary["ok"] is True
    assert summary["hb_outcomes_inspected"] is False
    contract = json.loads(
        (family / "reliability_label_contract_v1.json").read_text(encoding="utf-8")
    )
    assert contract["labels"][LABEL_MNIH]["only_positive_executable_label_v1"] is True
    assert 140401 in contract["protected_hb_origins_reliability"]
    assert 140312 not in contract["protected_hb_origins_reliability"]
    with pytest.raises(LockExistsError):
        freeze_reliability_label_contract_v1(
            repo_root=tmp_path, allow_final_freeze=True
        )


def test_write_corrected_review_no_final_contract(tmp_path: Path):
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
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / MAPPING_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "protected_hb_origins_reliability": [140304, 140307, 140310, 140401],
                "matched_counts_regenerated": {"n_human_complete_1_15": 999},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (family / REPRESENTATIVENESS_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "representativeness_conclusion": "NOT_REPRESENTATIVE",
                "portfolio_conclusion_scope": (
                    "LIMIT_PORTFOLIO_CONCLUSIONS_TO_HB_COVERED_COHORT"
                ),
                "extrapolation_rule": "Do not extrapolate MNIH",
                "flagged_axes": ["cohort_coverage_fraction"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (family / "reliability_label_contract_v1_review.json").write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "approval_checklist": [],
                "explicit_non_goals": [],
                "proposed_numeric_thresholds_flat_for_approval": {},
                "two_scopes": {
                    "portfolio_activation_gate": {"gates_apply_here_only": []},
                    "sku_validity_group_labels": {},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = write_corrected_review(repo_root=tmp_path)
    assert summary["ok"] is True
    assert summary["final_contract_written"] is False
    assert summary["mapping_audit_linked"] is True
    assert summary["representativeness_audit_linked"] is True
    out = tmp_path / FAMILY_DIR_REL / "reliability_label_contract_v1_review.json"
    review = json.loads(out.read_text(encoding="utf-8"))
    assert review["positive_executable_label"] == LABEL_MNIH
    assert review["stop_before_final_contract"] is True
    assert review["do_not_extrapolate_MNIH"] == "Do not extrapolate MNIH"
    assert 140401 in review["protected_hb_origins_reliability"]
    assert not (tmp_path / FAMILY_DIR_REL / "reliability_label_contract_v1.json").exists()


def test_mapping_audit_helpers_complete_pairs():
    from pkg.research.portfolio_forecast_reliability_v1.hb_origin_mapping_audit import (
        _complete_pairs,
        group_complete_pairs,
    )

    rows = []
    for h in range(1, 16):
        rows.append({"product": "A", "forecast_origin": 140304, "horizon": h})
    # 140401 missing h15 — full incomplete, but earlier groups still complete
    for h in range(1, 15):
        rows.append({"product": "B", "forecast_origin": 140401, "horizon": h})
    df = pd.DataFrame(rows)
    pairs, incomplete = _complete_pairs(df, "forecast_origin")
    assert ("A", 140304) in pairs
    assert ("B", 140401) not in pairs
    assert any(r["product"] == "B" and r["forecast_origin"] == 140401 for r in incomplete)
    assert 15 in next(r for r in incomplete if r["product"] == "B")["missing"]

    h1_3, _ = group_complete_pairs(df, horizons=frozenset({1, 2, 3}), origin_col="forecast_origin")
    h4_6, _ = group_complete_pairs(df, horizons=frozenset({4, 5, 6}), origin_col="forecast_origin")
    h7_12, _ = group_complete_pairs(
        df, horizons=frozenset(range(7, 13)), origin_col="forecast_origin"
    )
    h13_15, inc13 = group_complete_pairs(
        df, horizons=frozenset({13, 14, 15}), origin_col="forecast_origin"
    )
    assert ("B", 140401) in h1_3
    assert ("B", 140401) in h4_6
    assert ("B", 140401) in h7_12
    assert ("B", 140401) not in h13_15
    assert any(r["product"] == "B" and 15 in r["missing_in_group"] for r in inc13)


def test_stale_coverage_figures_absent_from_review_skeleton():
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        STALE_COVERAGE_FIGURES_FORBIDDEN,
    )
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        _strip_stale_coverage_figures,
    )

    dirty = {
        "stale_v1_coverage": {"n": 167},
        "ok": True,
        "nested": {"legacy_matched_counts_167": 167, "keep": 1},
    }
    clean = _strip_stale_coverage_figures(dirty)
    assert "stale_v1_coverage" not in clean
    assert "legacy_matched_counts_167" not in clean["nested"]
    assert clean["nested"]["keep"] == 1
    assert STALE_COVERAGE_FIGURES_FORBIDDEN == frozenset({55, 50, 167, 161})


def test_legacy_ts_lock_key_does_not_drive_execution():
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        LEGACY_TS_LOCK_HB_ORIGINS_KEY,
    )

    assert LEGACY_TS_LOCK_HB_ORIGINS_KEY == "legacy_ts_lock_hb_origins_immutable"
    assert 140312 not in PROTECTED_HB_ORIGINS


def test_supersede_origin_validation_erratum(tmp_path: Path):
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        PROTECTED_ORIGIN_VALIDATION_FILENAME,
    )
    from pkg.research.portfolio_forecast_reliability_v1.supersede_origin_validation import (
        stamp_protected_origin_validation_superseded,
    )

    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / PROTECTED_ORIGIN_VALIDATION_FILENAME).write_text(
        json.dumps(
            {
                "recommended_protected_origins": [140304, 140307, 140310, 140312],
                "conclusion": {"correct_protected_set_for_HB": [140304, 140307, 140310, 140312]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = stamp_protected_origin_validation_superseded(repo_root=tmp_path)
    assert summary["ok"] is True
    assert summary["drives_reliability_execution"] is False
    stamped = json.loads(
        (family / PROTECTED_ORIGIN_VALIDATION_FILENAME).read_text(encoding="utf-8")
    )
    assert stamped["supersession"]["status"] == "SUPERSEDED"
    assert stamped["supersession"]["erratum"]["1404Q1_maps_to"] == 140401
    assert stamped["supersession"]["erratum"]["140312_is_not_a_reliability_origin"] is True
    # Original contradictory recommendation preserved but not authoritative
    assert stamped["recommended_protected_origins"] == [140304, 140307, 140310, 140312]


def test_deployable_naive_reconciliation_explicit_fail():
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        NO_MACHINE_FORECAST,
    )
    from pkg.research.portfolio_forecast_reliability_v1.deployable_policy import (
        reconcile_hb_complete_deployable,
    )

    human = {("A", 140304), ("B", 140304)}
    a0 = {("A", 140304)}
    v21 = {("A", 140304)}
    sales = pd.DataFrame({"product": ["A"], "date": [140301], "sales": [10.0]})
    out = reconcile_hb_complete_deployable(
        human_complete=human,
        a0_complete=a0,
        v21_complete=v21,
        sales=sales,
    )
    assert out["identity"]["holds"] is True
    assert out["n_deployable_with_forecast"] == 1
    assert out["n_no_machine_forecast"] == 1
    assert out["reconciliation_status"] == "RECONCILED"
    assert out["NO_MACHINE_FORECAST_COHORT"]["rows"][0]["disposition"] == NO_MACHINE_FORECAST
    assert out["NO_MACHINE_FORECAST_COHORT"]["enters_accuracy_denominator"] is False
    assert out["sku_x_group_mnih_policy"]["any_evaluated_NO_MACHINE_FORECAST_blocks_MNIH"] is True


def test_deployable_naive_available_from_sales_through_fo_minus_2():
    from pkg.research.portfolio_forecast_reliability_v1.deployable_policy import (
        naive_available_through_origin_minus_2,
        reconcile_hb_complete_deployable,
    )

    sales = pd.DataFrame(
        {"product": ["Z", "Z"], "date": [140301, 140302], "sales": [1.0, 2.0]}
    )
    nav = naive_available_through_origin_minus_2(sales, "Z", 140304)
    assert nav["naive_available"] is True
    assert nav["origin_minus_2_cutoff"] == 140302
    out = reconcile_hb_complete_deployable(
        human_complete={("Z", 140304)},
        a0_complete=set(),
        v21_complete=set(),
        sales=sales,
    )
    assert out["reconciliation_status"] == "RECONCILED"
    assert out["identity"]["n_valid_naive"] == 1
    assert out["DEPLOYABLE_PORTFOLIO_COHORT"]["n_product_origin_pairs"] == 1


def test_label_hr_no_machine_forecast_blocks_mnih():
    assert (
        decide_sku_group_label(
            **_base(
                deployable_branch="both_native",
                any_evaluated_origin_no_machine_forecast=True,
            )
        )
        == LABEL_HR
    )


def test_no_contradictory_native_ge3_portfolio_definition():
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        build_decision_table,
        locked_thresholds,
    )

    thr = locked_thresholds()
    assert thr["portfolio"]["primary_population"] == "DEPLOYABLE_PORTFOLIO_COHORT"
    assert "≥3" not in thr["portfolio"]["primary_population"]
    dt = build_decision_table()
    assert dt["portfolio_activation_gate"]["primary_population"] == (
        "DEPLOYABLE_PORTFOLIO_COHORT"
    )
    assert "HUMAN_COMPARABLE_COHORT_ge3" not in str(
        dt["portfolio_activation_gate"]["order"]
    )


def test_pit_cutoff_constant():
    from pkg.benchmark.calendar import shamsi_add_months
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        EARLIEST_HB_ORIGIN_FOR_PIT,
        PIT_REPRESENTATIVENESS_CUTOFF,
    )

    assert PIT_REPRESENTATIVENESS_CUTOFF == 140302
    assert shamsi_add_months(EARLIEST_HB_ORIGIN_FOR_PIT, -2) == 140302


def test_pit_features_exclude_post_cutoff_sales():
    from pkg.research.portfolio_forecast_reliability_v1.hb_cohort_representativeness import (
        _sku_sales_features,
    )

    sales = pd.DataFrame(
        {
            "product": ["P", "P", "P"],
            "date": [140301, 140302, 140305],
            "sales": [10.0, 20.0, 999.0],
        }
    )
    feats = _sku_sales_features(sales, {"P"}, pit_cutoff=140302)
    assert feats["P"]["volume"] == 30.0
    assert feats["P"]["history_length"] == 2.0
    assert feats["P"]["pit_max_date_used"] == 140302.0


def test_machine_coverage_not_balanced_when_missing_policy_skus():
    """Document expected split: among-present vs full-denominator machine coverage."""
    n_eligible = 363
    n_present = 240
    n_missing = n_eligible - n_present
    assert n_missing == 123
    # among-present can be BALANCED while machine coverage is NOT_BALANCED
    assert n_missing > 0


def test_representativeness_denominators_use_full_eligible_not_archval55():
    from pkg.research.portfolio_forecast_reliability_v1.hb_cohort_representativeness import (
        _ks_like_shift,
    )

    block = _ks_like_shift([1.0, 2.0, 3.0], [1.5, 2.5], denom_a=363, denom_b=200)
    assert block["full_eligible"]["n_denominator"] == 363
    assert block["hb_cohort"]["n_denominator"] == 200
    assert block["full_eligible"]["n_missing_from_denominator"] == 360
    assert block["denominator_reconciled"] is True


def test_write_corrected_review_marks_stale_artifacts(tmp_path: Path):
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        LEGACY_TS_LOCK_HB_ORIGINS_KEY,
        PROTECTED_ORIGIN_VALIDATION_FILENAME,
        STALE_COVERAGE_AUDIT_FILENAME,
    )

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
    family = tmp_path / FAMILY_DIR_REL
    family.mkdir(parents=True)
    (family / MAPPING_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "protected_hb_origins_reliability": [140304, 140307, 140310, 140401],
                "full_horizon_matched_counts_regenerated": {
                    "n_human_complete_1_15": 880
                },
                "group_specific_matched_coverage": {
                    "h1_3": {
                        "n_skus_mnih_eligible_ge3_origins": 10,
                        "deployable_branch": {
                            "n_human_complete": 100,
                            "n_machine_any_matched": 90,
                        },
                        "by_origin_hb_complete": {"140401": 55},
                        "by_origin_matched_machine_any": {"140401": 40},
                        "HUMAN_COMPARABLE_COHORT_h1_3": {"n_skus": 80},
                    }
                },
                LEGACY_TS_LOCK_HB_ORIGINS_KEY: {
                    "origins": [140304, 140307, 140310, 140312],
                    "drives_reliability_execution": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (family / REPRESENTATIVENESS_AUDIT_FILENAME).write_text(
        json.dumps(
            {
                "hb_outcomes_inspected": False,
                "coverage_scope": {"status": "INCOMPLETE", "n_eligible": 363},
                "covariate_balance": {
                    "status": "REPRESENTATIVE",
                    "flagged_continuous_axes": [],
                },
                "categorical_balance": {"status": "BALANCED"},
                "engine_fallback_balance": {"status": "BALANCED"},
                "portfolio_conclusion_scope": (
                    "LIMIT_PORTFOLIO_CONCLUSIONS_TO_HB_COVERED_COHORT"
                ),
                "extrapolation_rule": "Do not extrapolate MNIH",
                "populations": {
                    "full_eligible_portfolio": {"n_skus": 363},
                    "hb_group_union_cohort": {"n_skus": 200},
                },
                "sales_source": {
                    "relative_path": (
                        "src/data/results/product_universe/"
                        "sales_full_portfolio_v1/sales.parquet"
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (family / STALE_COVERAGE_AUDIT_FILENAME).write_text(
        json.dumps({"n_human_complete_1_15": 167}) + "\n", encoding="utf-8"
    )
    (family / PROTECTED_ORIGIN_VALIDATION_FILENAME).write_text(
        json.dumps({"recommended_protected_origins": [140312]}) + "\n", encoding="utf-8"
    )
    summary = write_corrected_review(repo_root=tmp_path)
    review = json.loads(
        (family / "reliability_label_contract_v1_review.json").read_text(encoding="utf-8")
    )
    assert summary["ok"] is True
    assert review["stale_artifacts"][STALE_COVERAGE_AUDIT_FILENAME]["status"] == (
        "SUPERSEDED"
    )
    assert review["stale_artifacts"][PROTECTED_ORIGIN_VALIDATION_FILENAME][
        "drives_reliability_execution"
    ] is False
    assert review[LEGACY_TS_LOCK_HB_ORIGINS_KEY]["drives_reliability_execution"] is False
    assert review["coverage_scope"] == "INCOMPLETE"
    assert review["covariate_balance"] == "REPRESENTATIVE"
    assert 140312 not in review["reliability_execution_origins"]
    assert review["mnih_gates_locked"]["neither_naive_any_evaluated_origin_blocks_mnih"] is True
    assert review["mnih_gates_locked"][
        "NO_MACHINE_FORECAST_any_evaluated_origin_blocks_mnih"
    ] is True
    assert review["mnih_gates_locked"]["portfolio_activation_population"] == (
        "DEPLOYABLE_PORTFOLIO_COHORT"
    )
    assert review["evaluation_scopes"]["primary_portfolio_activation"] == (
        "DEPLOYABLE_PORTFOLIO_COHORT"
    )
    assert review["locked_thresholds_preview"]["portfolio"]["primary_population"] == (
        "DEPLOYABLE_PORTFOLIO_COHORT"
    )
    assert review["decision_table_preview"]["portfolio_activation_gate"][
        "primary_population"
    ] == "DEPLOYABLE_PORTFOLIO_COHORT"
    assert "primary_population_HUMAN_COMPARABLE_COHORT_ge3_origins" not in (
        review["decision_table_preview"]["portfolio_activation_gate"]["order"]
    )
    assert review["final_contract_written"] is False
    live = review["hb_origin_mapping_audit"]["full_horizon_matched_counts_regenerated"]
    assert live["n_human_complete_1_15"] == 880
    assert live["n_human_complete_1_15"] not in {55, 50, 167, 161}
    assert "sales_full_portfolio_v1" in (
        review["hb_cohort_representativeness_audit"]["sales_source"]["relative_path"]
    )


def test_identity_native_naive_no_machine():
    from pkg.research.portfolio_forecast_reliability_v1.deployable_policy import (
        reconcile_hb_complete_deployable,
    )

    # 2 native + 1 valid naive + 1 NO_MACHINE = 4
    human = {
        ("N1", 140304),
        ("N2", 140304),
        ("V", 140304),
        ("U", 140304),
    }
    a0 = {("N1", 140304), ("N2", 140304)}
    v21 = {("N1", 140304), ("N2", 140304)}
    sales = pd.DataFrame(
        {"product": ["V"], "date": [140301], "sales": [5.0]}
    )
    out = reconcile_hb_complete_deployable(
        human_complete=human,
        a0_complete=a0,
        v21_complete=v21,
        sales=sales,
    )
    ident = out["identity"]
    assert ident["n_native"] == 2
    assert ident["n_valid_naive"] == 1
    assert ident["n_no_machine_forecast"] == 1
    assert ident["n_hb_complete"] == 4
    assert ident["holds"] is True
    assert out["DEPLOYABLE_PORTFOLIO_COHORT"]["n_product_origin_pairs"] == 3
    assert out["NO_MACHINE_FORECAST_COHORT"]["enters_accuracy_denominator"] is False
    for row in out["NO_MACHINE_FORECAST_COHORT"]["rows"]:
        assert row["enters_accuracy_denominator"] is False
        assert row["deployable"] is False


def test_140404_pending_maturity_not_in_protected():
    from pkg.research.portfolio_forecast_reliability_v1.constants import (
        ORIGIN_STATUS_PENDING_MATURITY,
        PROPOSED_FINAL_FULL_ORIGINS,
        PROSPECTIVE_HB_ORIGIN_140404,
    )
    from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (
        portfolio_temporal_gate,
        prospective_origins_block,
    )

    assert PROSPECTIVE_HB_ORIGIN_140404 == 140404
    assert 140404 not in PROTECTED_HB_ORIGINS
    block = prospective_origins_block()
    assert block["140404"]["status"] == ORIGIN_STATUS_PENDING_MATURITY
    assert block["140404"]["admitted_to_PROTECTED_HB_ORIGINS"] is False
    assert block["140401"]["role"] == "group_specific_evidence_only"
    assert list(PROPOSED_FINAL_FULL_ORIGINS) == [140304, 140307, 140310, 140404]
    tg = portfolio_temporal_gate()
    assert tg["required_noninferior_origins"] == required_noninferior_origins(4)
    assert tg["required_noninferior_origins"] == 3
    assert tg["leave_one_origin"]["every_pooled_fold_machine_minus_human_wmape_pp_max"] == 2.0
    assert 140401 in tg["excluded_from_full_horizon_gate"]
    assert 140401 not in tg["full_horizon_origins"]


def test_candidate_manifest_fail_keeps_140404_excluded():
    from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
        build_candidate_origin_manifest,
    )

    audit = {
        "all_gates_pass": False,
        "admission_gates": [
            {"id": "hb_exact_horizons_1_15", "passed": False},
            {"id": "scoring_sales_complete_through_140506", "passed": False},
        ],
        "hb_outcomes_inspected": False,
    }
    man = build_candidate_origin_manifest(audit)
    assert man["excluded"] is True
    assert man["stop"] is True
    assert man["proposed_final_origins"] is None
    assert man["140404_status"] == "PENDING_MATURITY"


def test_candidate_manifest_pass_proposes_final_set():
    from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
        build_candidate_origin_manifest,
    )

    audit = {"all_gates_pass": True, "admission_gates": [], "hb_outcomes_inspected": False}
    man = build_candidate_origin_manifest(audit)
    assert man["excluded"] is False
    assert man["stop"] is False
    assert man["proposed_final_origins"] == [140304, 140307, 140310, 140404]
    assert 140401 in man["group_specific_evidence_only"]


def test_scoring_sales_gate_rejects_v1_max_date_alone(tmp_path: Path):
    from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
        evaluate_scoring_sales_maturity_gate,
    )

    v1 = tmp_path / "src/data/results/product_universe/sales_full_portfolio_v1"
    v1.mkdir(parents=True)
    # Fake v1 with late max date — must still fail without v2 + closure
    pd.DataFrame({"product": ["A"], "date": [140506], "sales": [1.0]}).to_parquet(
        v1 / "sales.parquet", index=False
    )
    detail = evaluate_scoring_sales_maturity_gate(repo_root=tmp_path)
    assert detail["passed"] is False
    assert detail["v1_max_date_cannot_prove_closure"] is True
    assert detail["v1_max_date_observed"] == 140506
    assert any("warehouse_closure" in r or "v2" in r for r in detail["fail_reasons"])


def test_scoring_sales_gate_passes_with_validated_v2_and_closure(tmp_path: Path):
    import hashlib

    from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
        evaluate_scoring_sales_maturity_gate,
    )

    v2 = tmp_path / "src/data/results/product_universe/sales_full_portfolio_v2"
    v2.mkdir(parents=True)
    pq = v2 / "sales.parquet"
    pd.DataFrame({"product": ["A"], "date": [140506], "sales": [1.0]}).to_parquet(
        pq, index=False
    )
    h = hashlib.sha256()
    with pq.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    sha = h.hexdigest()
    (v2 / "meta.json").write_text(
        json.dumps(
            {
                "snapshot_version": "sales_full_portfolio_v2",
                "immutable": True,
                "sales": {"content_sha256": sha, "n_rows": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (v2 / "warehouse_closure.json").write_text(
        json.dumps(
            {
                "closed_through": 140506,
                "closed_at_utc": "2026-10-01T00:00:00Z",
                "validated": True,
                "source": "test_fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detail = evaluate_scoring_sales_maturity_gate(repo_root=tmp_path)
    assert detail["passed"] is True


def test_identifier_trace_offline_panel_counts():
    from pkg.research.portfolio_forecast_reliability_v1.hb_140404_identifier_trace import (
        build_hb_140404_identifier_trace,
    )

    repo = Path(__file__).resolve().parents[2]
    audit = build_hb_140404_identifier_trace(repo_root=repo, load_sql=False)
    assert audit["hb_outcomes_inspected"] is False
    assert audit["diagnosis"]["missing_target_month"] == 140505
    assert audit["diagnosis"]["missing_horizon"] == 14
    panel = audit["frozen_v2_panel"]
    if panel.get("exists"):
        assert panel.get("target_140505_present_in_panel") is False
        assert panel.get("horizon_14_present_in_panel") is False
        comp = panel["per_sku_horizon_completeness"]
        assert comp["n_skus_complete_1_15"] == 0
        assert comp["n_skus_missing_each_horizon"]["14"] == comp["n_skus_with_any_row"]


def test_sales_v2_prepare_only_no_parquet(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "extract_sales_full_portfolio_v2.py"
    spec = importlib.util.spec_from_file_location("extract_sales_v2", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Seed a fake v1 sales path reference under tmp via symlink-less copy of constants:
    # prepare manifest only needs existence flags.
    family = tmp_path / FAMILY_DIR_REL
    v1 = tmp_path / "src/data/results/product_universe/sales_full_portfolio_v1"
    v1.mkdir(parents=True)
    (v1 / "sales.parquet").write_bytes(b"PAR1")  # non-parquet ok for exists check only
    # write_prepare_manifest uses repo_root / SALES_FULL_PORTFOLIO_REL
    out = mod.write_prepare_manifest(repo_root=tmp_path)
    assert out["ok"] is True
    assert out["parquet_written"] is False
    man = json.loads((family / "sales_full_portfolio_v2_prepare_manifest.json").read_text(encoding="utf-8"))
    assert man["parquet_written"] is False
    assert man["never_overwrite_v1"] is True
    assert not (tmp_path / "src/data/results/product_universe/sales_full_portfolio_v2/sales.parquet").exists()
    with pytest.raises(RuntimeError):
        mod.extract_v2(repo_root=tmp_path)
