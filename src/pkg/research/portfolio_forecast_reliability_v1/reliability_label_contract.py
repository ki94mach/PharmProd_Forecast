"""Reliability label contract v1 — MNIH-primary review; freeze only later.

Family: portfolio_forecast_reliability_v1 (not residual-XGB).
Does not inspect HB forecast/outcome columns.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pkg.portfolio_ts_lock.constants import HORIZON_GROUPS
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.portfolio_ts_lock.production_contract import (
    file_sha256,
    write_immutable_json,
)
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    CANDIDATE_ORIGIN_MANIFEST_FILENAME,
    CONTRACT_FILENAME,
    CONTRACT_VERSION,
    DEPLOYABLE_PORTFOLIO_COHORT,
    EVIDENCE_STRENGTH_LIMITED,
    FAMILY,
    FAMILY_DIR_REL,
    GROUP_SPECIFIC_ONLY_ORIGIN,
    HUMAN_COMPARABLE_COHORT,
    LABEL_HR,
    LABEL_IHE,
    LABEL_MNIH,
    LABEL_MR_OUT_OF_SCOPE,
    LEGACY_TS_LOCK_HB_ORIGINS_KEY,
    MAPPING_AUDIT_FILENAME,
    MATURITY_AUDIT_FILENAME,
    NO_MACHINE_FORECAST,
    NO_MACHINE_FORECAST_COHORT,
    ORIGIN_STATUS_PENDING_MATURITY,
    PROPOSED_FINAL_FULL_ORIGINS,
    PROSPECTIVE_HB_ORIGIN_140404,
    PROSPECTIVE_HB_QUARTER_140404,
    PROSPECTIVE_MACHINE_TRAIN_END,
    PROSPECTIVE_SCORING_SALES_THROUGH,
    PROTECTED_HB_ORIGINS,
    PROTECTED_ORIGIN_VALIDATION_FILENAME,
    REPRESENTATIVENESS_AUDIT_FILENAME,
    REVIEW_FILENAME,
    HB_FORECAST_COVERAGE_AUDIT_FILENAME,
    HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME,
    HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME,
    HB_SCORING_PATH_PREPARE_FILENAME,
    IDENTIFIER_TRACE_FILENAME,
    SALES_FULL_PORTFOLIO_REL,
    SALES_FULL_PORTFOLIO_V2_REL,
    SALES_V2_PREPARE_MANIFEST_FILENAME,
    STALE_COVERAGE_AUDIT_FILENAME,
    STALE_COVERAGE_FIGURES_FORBIDDEN,
    UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL,
    VALIDITY_GROUPS,
    VPS_140404_COMMANDS_FILENAME,
    WMAPE_DENOM_EPS,
    WMAPE_DENOM_RULE,
)

# Back-compat aliases used by older imports/tests during transition.
LABEL_IE = LABEL_IHE  # INSUFFICIENT_HUMAN_EVIDENCE
LABEL_MR = LABEL_MR_OUT_OF_SCOPE  # not an executable v1 label

GROUP_ORDER: tuple[str, ...] = ("h1_3", "h4_6", "h7_12", "h13_15")
GROUP_WIDTH: dict[str, int] = {
    "h1_3": 3,
    "h4_6": 3,
    "h7_12": 6,
    "h13_15": 3,
}

WMAPE_DENOM_SCOPES: tuple[str, ...] = (
    "portfolio",
    "sku_x_validity_group",
    "sku_x_origin",
)


def required_noninferior_origins(n_complete: int) -> int:
    """ceil(0.75 × complete_origins)."""
    if n_complete <= 0:
        return 0
    return int(math.ceil(0.75 * float(n_complete)))


def prospective_140404_admission_gates() -> list[dict[str, Any]]:
    """Preregistered admission gates for origin 140404 (identifier / policy only)."""
    return [
        {
            "id": "hb_version_quarter_maps_to_140404",
            "rule": (
                f"HB Version→quarter mapping for {PROSPECTIVE_HB_QUARTER_140404} "
                f"gives origin {PROSPECTIVE_HB_ORIGIN_140404}"
            ),
        },
        {
            "id": "hb_exact_horizons_1_15",
            "rule": "HB contains exact horizons {1..15} for product×origin 140404",
        },
        {
            "id": "scoring_sales_complete_through_140506",
            "rule": (
                f"Validated sales_full_portfolio_v2 snapshot exists with explicit "
                f"warehouse_closure.closed_through>={PROSPECTIVE_SCORING_SALES_THROUGH}; "
                "v1 max_date alone cannot prove warehouse closure"
            ),
        },
        {
            "id": "machine_training_ends_at_140402",
            "rule": (
                f"Machine training ends at {PROSPECTIVE_MACHINE_TRAIN_END} "
                "(origin−2; frozen sales_full_portfolio_v1)"
            ),
        },
        {
            "id": "never_renumber_or_silent_drop",
            "rule": "No horizons are renumbered or silently dropped",
        },
    ]


def prospective_origins_block() -> dict[str, Any]:
    """Review/contract prospective origin registry (140404 not yet admitted)."""
    return {
        str(PROSPECTIVE_HB_ORIGIN_140404): {
            "status": ORIGIN_STATUS_PENDING_MATURITY,
            "quarter": PROSPECTIVE_HB_QUARTER_140404,
            "admission_gates": prospective_140404_admission_gates(),
            "admitted_to_PROTECTED_HB_ORIGINS": False,
            "in_current_PROTECTED_HB_ORIGINS": (
                PROSPECTIVE_HB_ORIGIN_140404 in PROTECTED_HB_ORIGINS
            ),
            "scoring_sales_through": PROSPECTIVE_SCORING_SALES_THROUGH,
            "machine_train_end": PROSPECTIVE_MACHINE_TRAIN_END,
            "machine_sales_source": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "scoring_sales_source_after_close": SALES_FULL_PORTFOLIO_V2_REL.as_posix(),
        },
        str(GROUP_SPECIFIC_ONLY_ORIGIN): {
            "role": "group_specific_evidence_only",
            "note": (
                "140401 remains in current PROTECTED_HB_ORIGINS for group-level "
                "labels; excluded from proposed full-horizon portfolio temporal "
                "gate once 140404 is admitted."
            ),
            "in_current_PROTECTED_HB_ORIGINS": True,
            "in_proposed_final_full_origins": False,
        },
        "proposed_final_full_origins_after_admission": list(PROPOSED_FINAL_FULL_ORIGINS),
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
    }


def portfolio_temporal_gate() -> dict[str, Any]:
    """Generic portfolio leave-one-origin temporal gate (preregistered; not scored)."""
    n_full = len(PROPOSED_FINAL_FULL_ORIGINS)
    return {
        "status": "PREREGISTERED_NOT_EXECUTED",
        "hb_outcomes_inspected": False,
        "n_full_origins": n_full,
        "full_horizon_origins": list(PROPOSED_FINAL_FULL_ORIGINS),
        "excluded_from_full_horizon_gate": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "required_noninferior_origins_rule": "ceil(0.75 * n_full_origins)",
        "required_noninferior_origins": required_noninferior_origins(n_full),
        "with_four_full_origins_require": "3/4",
        "leave_one_origin": {
            "every_pooled_fold_machine_minus_human_wmape_pp_max": 2.0,
            "rule": (
                "Every leave-one-origin pooled result must satisfy "
                "WMAPE(machine)−WMAPE(human) <= 2.0 pp"
            ),
        },
        "note": (
            "Do not open HB outcomes in this prepare step. "
            "140401 is group-specific evidence only for the full-origin temporal gate."
        ),
    }


def wmape_denominator_valid(sum_abs_actual: float) -> bool:
    """valid_wmape_denominator := sum(abs(actual)) > 1e-9 and finite."""
    try:
        v = float(sum_abs_actual)
    except (TypeError, ValueError):
        return False
    if v != v or v == float("inf") or v == float("-inf"):
        return False
    return v > WMAPE_DENOM_EPS


def evidence_strength_for_n_origins(n_complete: int) -> str | None:
    """LIMITED_3_OR_4_ORIGINS when n in {3,4}; None otherwise."""
    if n_complete in (3, 4):
        return EVIDENCE_STRENGTH_LIMITED
    return None


def mnih_label_meaning() -> dict[str, Any]:
    return {
        "meaning": (
            "Passed the preregistered noninferiority rule on available origins "
            "(not absolute reliability)."
        ),
        "evidence_strength_when_n_in_3_or_4": EVIDENCE_STRENGTH_LIMITED,
        "requires_native_HUMAN_COMPARABLE_COHORT_group": True,
        "blocked_if_any_evaluated_origin_neither_naive": True,
        "blocked_if_any_evaluated_origin_NO_MACHINE_FORECAST": True,
        "only_positive_executable_label_v1": True,
        "portfolio_gate_population": DEPLOYABLE_PORTFOLIO_COHORT,
        "not_portfolio_gate_population": (
            f"{HUMAN_COMPARABLE_COHORT} with ≥3 origins (SKU×group MNIH only)"
        ),
    }


def future_out_of_scope_absolute_ceilings() -> dict[str, Any]:
    return {
        "MACHINE_RELIABLE": LABEL_MR_OUT_OF_SCOPE,
        "portfolio_machine_wmape_ceiling_pct": None,
        "sku_group_machine_wmape_ceiling_pct": None,
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "note": (
            "Absolute WMAPE ceilings and MACHINE_RELIABLE are not executable v1 "
            "activation gates. They may return in a later contract revision."
        ),
    }


def canonical_per_group_label_rules() -> dict[str, Any]:
    """Ordered SKU×group label rules — MNIH is the only positive executable label.

    Precedence (must hold):
      1. missing/incomplete HB evidence → IHE
      2. invalid denominator → IHE
      3. any neither_naive or NO_MACHINE_FORECAST → HR
         (even with 0–1 native origins or absent machine WMAPE)
      4. non-computable metrics on native path → IHE
      5. 0–1 native matched origins → IHE
      6. 2 native matched origins → HR
      7. ≥3 → evaluate MNIH gates
    """
    return {
        "precedence_note": (
            "neither_naive / NO_MACHINE_FORECAST → HUMAN_REVIEW_REQUIRED "
            "before 0–1 native-origin → INSUFFICIENT_HUMAN_EVIDENCE; "
            "HB evidence with no machine forecast must not become IHE"
        ),
        "ordered": [
            {
                "when": "group_horizons_incomplete OR missing_HB_evidence",
                "label": LABEL_IHE,
            },
            {
                "when": "wmape_denominator_invalid (Σ|actual| <= 1e-9 or non-finite)",
                "label": LABEL_IHE,
            },
            {
                "when": (
                    "any evaluated origin for SKU×group has deployable_branch="
                    "neither_naive OR disposition=NO_MACHINE_FORECAST"
                ),
                "label": LABEL_HR,
                "blocks": LABEL_MNIH,
                "note": (
                    "Wins over 0–1 native-origin IHE and over absent machine WMAPE; "
                    "HB present + machine cannot forecast → HUMAN_REVIEW_REQUIRED"
                ),
            },
            {
                "when": "metrics_noncomputable (native path only)",
                "label": LABEL_IHE,
            },
            {
                "when": "n_complete_matched_native_origins_in_group in {0,1}",
                "label": LABEL_IHE,
            },
            {
                "when": "n_complete_matched_native_origins_in_group == 2",
                "label": LABEL_HR,
            },
            {
                "when": (
                    "SKU in native HUMAN_COMPARABLE_COHORT_<group> "
                    "AND n_complete_matched_native_origins_in_group >= 3 "
                    "AND wmape_denominator_valid "
                    "AND WMAPE(machine)-WMAPE(human) <= 2.0 pp "
                    "AND n_origins_noninferior >= ceil(0.75 * n_complete_origins) "
                    "AND portfolio_activation_gate_passed "
                    "AND no evaluated origin is neither_naive or NO_MACHINE_FORECAST"
                ),
                "label": LABEL_MNIH,
                "evidence_strength": EVIDENCE_STRENGTH_LIMITED,
                "evidence_strength_applies_when": "n_complete in {3,4}",
            },
            {
                "when": "otherwise (failed relative/portfolio gates)",
                "label": LABEL_HR,
            },
        ],
        "not_in_executable_v1": future_out_of_scope_absolute_ceilings(),
    }


def apply_review_structure_corrections(review: dict[str, Any]) -> None:
    """Structural corrections for MNIH-primary review (in place)."""
    two = review.setdefault("two_scopes", {})
    port = two.setdefault("portfolio_activation_gate", {})
    gates = list(port.get("gates_apply_here_only") or [])
    required_gates = [
        "complete_1_15_required_never_silent_drop",
        "primary_population_DEPLOYABLE_PORTFOLIO_COHORT",
        "exclude_NO_MACHINE_FORECAST_from_accuracy_denominators",
        "wmape_noninferiority_le_2pp",
        "rmse_p95_p99_le_1_02x",
        "max_ae_ratio",
        "sku_cluster_bootstrap",
        "drop_top3_remaining_noninferior",
        "wmape_denominator_must_be_valid",
    ]
    for g in required_gates:
        if g not in gates:
            gates.append(g)
    # Remove contradictory native≥3 portfolio population gate if present
    gates = [
        g
        for g in gates
        if g
        not in {
            "primary_population_skus_ge3_complete_matched_origins",
            "primary_population_HUMAN_COMPARABLE_COHORT_ge3_origins",
        }
    ]
    if "p99_ae_ratio" in gates and "max_ae_ratio" in gates:
        gates = [x for x in gates if x != "max_ae_ratio"]
        gates.insert(gates.index("p99_ae_ratio") + 1, "max_ae_ratio")
    port["gates_apply_here_only"] = gates
    port["primary_population"] = DEPLOYABLE_PORTFOLIO_COHORT
    port["primary_population_definition"] = (
        "HB-complete product×origins with an actual frozen-policy forecast "
        "(native or valid naive_last_value); excludes NO_MACHINE_FORECAST"
    )
    port["not_primary_population"] = (
        f"{HUMAN_COMPARABLE_COHORT} with ≥3 origins — SKU×group MNIH only"
    )
    port["sensitivity_population"] = (
        "all complete matched product×origin cases + leave-one-origin "
        "(still excluding NO_MACHINE_FORECAST from accuracy denominators)"
    )
    port["temporal_gate"] = portfolio_temporal_gate()
    if "temporal_leave_one_origin_every_fold_wmape_le_2pp" not in gates:
        gates.append("temporal_leave_one_origin_every_fold_wmape_le_2pp")
    if "temporal_required_noninferior_origins_ceil_0_75_n_full" not in gates:
        gates.append("temporal_required_noninferior_origins_ceil_0_75_n_full")
    port["gates_apply_here_only"] = gates
    port["absolute_wmape_ceilings"] = "OUT_OF_EXECUTABLE_V1_SCOPE"

    sku = two.setdefault("sku_validity_group_labels", {})
    sku["per_group_label_rule"] = canonical_per_group_label_rules()
    thr = sku.setdefault("proposed_numeric_thresholds", {})
    thr["sku_group_machine_wmape_ceiling_pct"] = {
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "value": None,
    }

    flat = review.setdefault("proposed_numeric_thresholds_flat_for_approval", {})
    flat["sku_group.sku_group_machine_wmape_ceiling_pct"] = {
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "value": None,
    }
    flat["portfolio.portfolio_machine_wmape_ceiling_pct"] = {
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "value": None,
    }

    review["future_out_of_scope"] = future_out_of_scope_absolute_ceilings()
    review["planning_ceilings_pending"] = {
        "status": "NOT_REQUIRED_FOR_MNIH_V1",
        "note": (
            "Absolute ceilings are out of executable v1 scope. "
            "Do not block MNIH review on Planning ceilings."
        ),
    }

    labels = review.setdefault("labels", {})
    labels[LABEL_MNIH] = mnih_label_meaning()
    labels[LABEL_HR] = {
        "meaning": (
            "Any evaluated neither_naive / NO_MACHINE_FORECAST (even with 0–1 "
            "native origins), two native matched origins, or relative/portfolio "
            "failure. Precedence: machine-unavailable → HR before 0–1 → IHE."
        )
    }
    labels[LABEL_IHE] = {
        "meaning": (
            "Missing/incomplete HB evidence, invalid WMAPE denominator "
            "(Σ|actual| <= 1e-9 or non-finite), non-computable metrics when a "
            "machine forecast exists, or 0–1 native matched origins after "
            "neither_naive / NO_MACHINE_FORECAST have been ruled out."
        )
    }
    if LABEL_MR_OUT_OF_SCOPE in labels:
        labels[LABEL_MR_OUT_OF_SCOPE] = {
            "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
            **future_out_of_scope_absolute_ceilings(),
        }

    checklist = list(review.get("approval_checklist") or [])
    for item in (
        "Confirm MACHINE_NONINFERIOR_TO_HUMAN is the only positive executable label",
        "Confirm portfolio gate population is DEPLOYABLE_PORTFOLIO_COHORT (not native≥3)",
        "Confirm HUMAN_COMPARABLE_COHORT_<group> native≥3 is SKU×group MNIH only",
        "Confirm rule order: incomplete HB / bad denom → IHE; then neither_naive/NO_MACHINE_FORECAST → HR; then 0–1→IHE, 2→HR, ≥3 MNIH-eligible",
        "Confirm neither_naive or NO_MACHINE_FORECAST → HUMAN_REVIEW_REQUIRED even when native origin count is 0–1",
        "Confirm NO_MACHINE_FORECAST never enters accuracy denominators",
        "Confirm 615 native + 224 valid naive + 41 NO_MACHINE_FORECAST = 880",
        "Confirm validity_range_prefix uses MNIH (not MACHINE_RELIABLE)",
        "Confirm WMAPE denom applies separately at portfolio, SKU×group, SKU×origin",
        "Confirm protected HB origins include 140401 (not 140312) for reliability joins",
        "Confirm 140404 is PENDING_MATURITY until all admission gates pass",
        "Confirm portfolio temporal gate: ceil(0.75*n_full) and every LOO fold ≤2 pp WMAPE",
        "Confirm 140401 is group-specific evidence only for proposed full-horizon set",
        "Confirm representativeness audit limits portfolio conclusions if non-representative",
        "Confirm no HB outcomes opened; no router trained; no final contract yet",
    ):
        if item not in checklist:
            checklist.append(item)
    # Drop contradictory checklist items
    checklist = [
        c
        for c in checklist
        if "HUMAN_COMPARABLE_COHORT gate for MNIH" not in c
        or "SKU×group" in c
        or "native≥3" in c
    ]
    review["approval_checklist"] = checklist


def validity_range_prefix(group_labels: Sequence[str]) -> dict[str, Any]:
    """Longest contiguous MACHINE_NONINFERIOR_TO_HUMAN prefix starting at h1_3."""
    prefix: list[str] = []
    for name, lab in zip(GROUP_ORDER, group_labels):
        if lab != LABEL_MNIH:
            break
        prefix.append(name)
    human_from = GROUP_ORDER[len(prefix)] if len(prefix) < len(GROUP_ORDER) else None
    return {
        "machine_noninferior_groups": prefix,
        # legacy key kept for readers; mirrors MNIH prefix
        "machine_reliable_groups": prefix,
        "n_groups": len(prefix),
        "human_review_from_group": human_from,
        "positive_label": LABEL_MNIH,
        "display_range": (
            "empty"
            if not prefix
            else (
                "h1–3"
                if prefix == ["h1_3"]
                else "h1–6"
                if prefix == ["h1_3", "h4_6"]
                else "h1–12"
                if prefix == ["h1_3", "h4_6", "h7_12"]
                else "h1–15"
                if len(prefix) == 4
                else "+".join(prefix)
            )
        ),
    }


def decide_sku_group_label(
    *,
    n_complete_origins: int,
    group_horizons_exact: bool,
    sum_abs_actual: float,
    machine_group_wmape: float | None,
    human_group_wmape: float | None,
    n_origins_noninferior: int,
    portfolio_activation_passed: bool,
    deployable_branch: str,
    in_human_comparable_cohort: bool = True,
    any_evaluated_origin_neither_naive: bool | None = None,
    any_evaluated_origin_no_machine_forecast: bool = False,
    # Deprecated kwargs retained for call-site compatibility; ignored for gates.
    sku_group_ceiling_pct: float | None = None,
    portfolio_ceiling_pct: float | None = None,
    allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling: bool = True,
) -> str:
    """Apply SKU×validity-group decision table (MNIH-primary; no HB I/O).

    Precedence: incomplete HB / invalid denom → IHE; then neither_naive /
    NO_MACHINE_FORECAST → HR (before 0–1 native-origin IHE); then origin-count
    and MNIH gates.
    """
    del sku_group_ceiling_pct, portfolio_ceiling_pct
    del allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling

    # 1. Missing / incomplete HB evidence
    if not group_horizons_exact:
        return LABEL_IHE

    # 2. Invalid denominator (actuals side; independent of machine availability)
    if not wmape_denominator_valid(sum_abs_actual):
        return LABEL_IHE

    neither = bool(any_evaluated_origin_neither_naive)
    if any_evaluated_origin_neither_naive is None:
        neither = deployable_branch == "neither_naive"
    else:
        neither = bool(any_evaluated_origin_neither_naive) or (
            deployable_branch == "neither_naive"
        )
    no_machine = bool(any_evaluated_origin_no_machine_forecast) or (
        deployable_branch == NO_MACHINE_FORECAST
    )

    # 3. Machine cannot forecast → HR even with 0–1 native origins / absent WMAPE
    if neither or no_machine:
        return LABEL_HR

    # 2b. Non-computable outcomes when a machine forecast is expected
    if machine_group_wmape is None or human_group_wmape is None:
        return LABEL_IHE
    if machine_group_wmape != machine_group_wmape or human_group_wmape != human_group_wmape:
        return LABEL_IHE

    # 4–5. Native origin-count gates
    if n_complete_origins <= 1:
        return LABEL_IHE
    if n_complete_origins == 2:
        return LABEL_HR

    # 6. ≥3 → evaluate MNIH relative/portfolio gates
    if not in_human_comparable_cohort:
        return LABEL_HR

    rel_ok = (machine_group_wmape - human_group_wmape) <= 2.0
    stab_ok = n_origins_noninferior >= required_noninferior_origins(n_complete_origins)
    port_ok = bool(portfolio_activation_passed)

    if rel_ok and stab_ok and port_ok:
        return LABEL_MNIH
    return LABEL_HR


def decide_sku_group_label_with_meta(**kwargs: Any) -> dict[str, Any]:
    """Label plus evidence_strength metadata."""
    label = decide_sku_group_label(**kwargs)
    n = int(kwargs.get("n_complete_origins") or 0)
    meta: dict[str, Any] = {"label": label, "evidence_strength": None}
    if label == LABEL_MNIH:
        meta["evidence_strength"] = evidence_strength_for_n_origins(n)
        meta["meaning"] = mnih_label_meaning()["meaning"]
    return meta


def locked_thresholds() -> dict[str, Any]:
    """Structural/relative thresholds only (no absolute ceilings)."""
    return {
        "wmape_denominator": {
            "rule": WMAPE_DENOM_RULE,
            "eps": WMAPE_DENOM_EPS,
            "applies_separately_to_scopes": list(WMAPE_DENOM_SCOPES),
        },
        "portfolio": {
            "primary_population": DEPLOYABLE_PORTFOLIO_COHORT,
            "primary_population_definition": (
                "HB-complete product×origins with an actual frozen-policy forecast "
                "(native or valid naive_last_value)"
            ),
            "excludes_from_accuracy_denominators": NO_MACHINE_FORECAST_COHORT,
            "not_primary_population": (
                f"{HUMAN_COMPARABLE_COHORT} with ≥3 origins (SKU×group MNIH only)"
            ),
            "sensitivity_population": (
                "all complete matched product×origin + leave-one-origin "
                f"(exclude {NO_MACHINE_FORECAST} from accuracy denominators)"
            ),
            "temporal_gate": portfolio_temporal_gate(),
            "min_skus_structural_coverage_note": (
                "Structural min-SKU coverage may still be reported; portfolio "
                f"activation population is {DEPLOYABLE_PORTFOLIO_COHORT}, not native≥3"
            ),
            "machine_vs_human_noninferiority_wmape_margin_pp": 2.0,
            "machine_vs_human_noninferiority_rmse_ratio_max": 1.02,
            "machine_vs_human_noninferiority_p95_ae_ratio_max": 1.02,
            "machine_vs_human_noninferiority_p99_ae_ratio_max": 1.02,
            "tail_risk_max_ae_ratio_max": 1.10,
            "sku_cluster_bootstrap_B": 2000,
            "sku_cluster_bootstrap_seed": 42,
            "sku_cluster_bootstrap_frac_machine_noninferior_ge": 0.90,
            "drop_top_k_machine_advantage_skus": 3,
            "drop_top_k_rank_metric": "sum_ae_human_minus_sum_ae_machine",
            "drop_top_k_remaining_noninferiority_required": True,
            "wmape_denominator_rule": WMAPE_DENOM_RULE,
            "absolute_wmape_ceiling": "OUT_OF_EXECUTABLE_V1_SCOPE",
        },
        "sku_group": {
            "min_complete_origins_per_group_for_mnih": 3,
            "cohort_for_mnih": f"native {HUMAN_COMPARABLE_COHORT}_<group>",
            "origin_count_labels": {
                "0_or_1": LABEL_IHE,
                "2": LABEL_HR,
                "ge_3": "eligible_for_locked_MNIH_rules",
                "applies_after": "neither_naive_or_NO_MACHINE_FORECAST_ruled_out",
            },
            "machine_vs_human_noninferiority_wmape_margin_pp": 2.0,
            "origin_stability_frac": 0.75,
            "origin_stability_rule": "required_noninferior_origins=ceil(0.75*complete_origins)",
            "wmape_denominator_rule": WMAPE_DENOM_RULE,
            "neither_naive_blocks_mnih": True,
            "NO_MACHINE_FORECAST_blocks_mnih": True,
            "evidence_strength_when_mnih_and_n_in_3_or_4": EVIDENCE_STRENGTH_LIMITED,
            "absolute_wmape_ceiling": "OUT_OF_EXECUTABLE_V1_SCOPE",
        },
        "sku_x_origin": {
            "wmape_denominator_rule": WMAPE_DENOM_RULE,
            "note": (
                "Origin-stability per-origin WMAPE uses the same denominator rule "
                f"independently. {NO_MACHINE_FORECAST} origins never enter accuracy "
                "denominators."
            ),
        },
    }


def upstream_residual_pipeline_evidence(repo_root: Path) -> dict[str, Any]:
    """Record residual pipeline lock as upstream evidence: production stays FROZEN_TS."""
    path = repo_root / UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL
    evidence: dict[str, Any] = {
        "relative_path": UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL.as_posix(),
        "role": "upstream_evidence_only",
        "meaning": (
            "Residual XGBoost was rejected (KEEP_FROZEN_TS). "
            "This labeling contract applies to production FROZEN_TS, not residual-XGB."
        ),
        "exists": path.is_file(),
        "sha256": file_sha256(path) if path.is_file() else None,
        "decision": None,
        "production_pipeline": None,
        "selected_pipeline": None,
    }
    if path.is_file():
        lock = json.loads(path.read_text(encoding="utf-8"))
        evidence["decision"] = lock.get("decision")
        evidence["production_pipeline"] = lock.get("production_pipeline")
        evidence["selected_pipeline"] = lock.get("selected_pipeline")
        evidence["residual_xgb_promoted"] = False
        evidence["lock_decision_confirms_FROZEN_TS"] = (
            lock.get("decision") == "KEEP_FROZEN_TS"
            and lock.get("production_pipeline") == "FROZEN_TS"
            and lock.get("selected_pipeline") == "FROZEN_TS"
        )
    return evidence


def build_decision_table() -> dict[str, Any]:
    return {
        "portfolio_activation_gate": {
            "order": [
                "complete_1_15_required_never_silent_drop",
                "primary_population_DEPLOYABLE_PORTFOLIO_COHORT",
                "exclude_NO_MACHINE_FORECAST_from_accuracy_denominators",
                "valid_wmape_denominator_per_portfolio_scope",
                "wmape_noninferiority_le_2pp",
                "rmse_p95_p99_le_1_02x",
                "max_ae_le_1_10x",
                "sku_cluster_bootstrap_frac_ge_0_90",
                "drop_top3_by_sum_ae_machine_advantage_remaining_noninferior",
                "temporal_leave_one_origin_every_fold_wmape_le_2pp",
                "temporal_required_noninferior_origins_ceil_0_75_n_full",
            ],
            "if_fails": "no_MACHINE_NONINFERIOR_TO_HUMAN_anywhere",
            "primary_population": DEPLOYABLE_PORTFOLIO_COHORT,
            "not_primary_population": (
                f"{HUMAN_COMPARABLE_COHORT} with ≥3 origins — SKU×group MNIH only"
            ),
            "sensitivity_population": (
                "all complete matched product×origin + leave-one-origin "
                f"(exclude {NO_MACHINE_FORECAST})"
            ),
            "temporal_gate": portfolio_temporal_gate(),
            "wmape_denominator": {"rule": WMAPE_DENOM_RULE, "scope": "portfolio"},
            "drop_top_k": {
                "k": 3,
                "rank": "ΣAE(human)−ΣAE(machine)",
                "not": "mean_AE",
            },
            "absolute_ceilings": "OUT_OF_EXECUTABLE_V1_SCOPE",
        },
        "sku_x_validity_group": {
            "group_order": list(GROUP_ORDER),
            "wmape_denominator": {
                "rule": WMAPE_DENOM_RULE,
                "scope": "sku_x_validity_group",
            },
            "branches": [
                {"when": "group_horizons_not_exact OR missing_HB_evidence", "label": LABEL_IHE},
                {
                    "when": f"not ({WMAPE_DENOM_RULE}) at sku×group scope",
                    "label": LABEL_IHE,
                },
                {
                    "when": "any_evaluated_origin_neither_naive",
                    "label": LABEL_HR,
                    "blocks": LABEL_MNIH,
                },
                {
                    "when": "any_evaluated_origin_NO_MACHINE_FORECAST",
                    "label": LABEL_HR,
                    "blocks": LABEL_MNIH,
                    "note": "before n_complete_origins<=1 IHE",
                },
                {"when": "metrics_noncomputable (native path)", "label": LABEL_IHE},
                {"when": "n_complete_native_origins <= 1", "label": LABEL_IHE},
                {"when": "n_complete_native_origins == 2", "label": LABEL_HR},
                {
                    "when": (
                        "native HUMAN_COMPARABLE_COHORT_<group> AND n>=3 AND denom_ok "
                        "AND (machine-human)<=2pp AND origin_stability "
                        "AND portfolio_gate_passed AND no neither_naive "
                        "AND no NO_MACHINE_FORECAST"
                    ),
                    "label": LABEL_MNIH,
                    "evidence_strength": EVIDENCE_STRENGTH_LIMITED,
                },
                {"when": "otherwise", "label": LABEL_HR},
            ],
            "rule_order": [
                "incomplete_HB_or_invalid_denom → IHE",
                "neither_naive_or_NO_MACHINE_FORECAST → HR",
                "0_or_1_native_origins → IHE",
                "2_native_origins → HR",
                "ge3 → evaluate_MNIH_gates",
            ],
            "origin_stability": {
                "formula": "ceil(0.75 * n_complete_origins)",
                "n3": 3,
                "n4": 3,
                "per_origin_wmape_denominator": WMAPE_DENOM_RULE,
            },
            "validity_range": (
                "longest contiguous MACHINE_NONINFERIOR_TO_HUMAN prefix from h1_3"
            ),
            "neither_naive": "blocks_MACHINE_NONINFERIOR_TO_HUMAN_assign_HUMAN_REVIEW_REQUIRED",
            "NO_MACHINE_FORECAST": (
                "blocks_MACHINE_NONINFERIOR_TO_HUMAN_assign_HUMAN_REVIEW_REQUIRED;"
                "never_enters_accuracy_denominator"
            ),
            "incomplete_later_group_does_not_discard_earlier": True,
        },
        "sku_x_origin": {
            "wmape_denominator": {
                "rule": WMAPE_DENOM_RULE,
                "scope": "sku_x_origin",
                "applies_to": "each origin WMAPE used in origin-stability counts",
            }
        },
    }


def build_reliability_label_contract_v1(
    *,
    repo_root: Path,
    review_sha256: str | None = None,
    mapping_audit_sha256: str | None = None,
    representativeness_audit_sha256: str | None = None,
) -> dict[str, Any]:
    thr = locked_thresholds()
    validity_groups = [
        {
            "name": name,
            "horizons": sorted(hs),
            "width": GROUP_WIDTH[name],
            "display": {
                "h1_3": "h1–3",
                "h4_6": "h4–6",
                "h7_12": "h7–12",
                "h13_15": "h13–15",
            }[name],
        }
        for name, hs in HORIZON_GROUPS
    ]
    upstream = upstream_residual_pipeline_evidence(repo_root)

    return {
        "contract_version": CONTRACT_VERSION,
        "family": FAMILY,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hb_outcomes_inspected": False,
        "immutable": True,
        "fail_if_exists": True,
        "router_trained": False,
        "protected_hb_origins_reliability": sorted(PROTECTED_HB_ORIGINS),
        "machine_policy": "FROZEN_TS = A0_HYBRID_POLICY",
        "deployable_branches": [
            "both_native",
            "v21_only",
            "a0_only",
            "neither_naive",
        ],
        "validity_groups": validity_groups,
        "labels": {
            LABEL_MNIH: mnih_label_meaning(),
            LABEL_HR: {},
            LABEL_IHE: {
                "includes": [
                    "incomplete_group_horizons",
                    "n_origins_0_or_1",
                    f"invalid_wmape_denominator_not_{WMAPE_DENOM_RULE}",
                ]
            },
        },
        "future_out_of_scope": future_out_of_scope_absolute_ceilings(),
        "locked_thresholds": thr,
        "decision_table": build_decision_table(),
        "upstream_evidence": {"residual_pipeline_lock_v1": upstream},
        "source_review": {"filename": REVIEW_FILENAME, "sha256": review_sha256},
        "hb_origin_mapping_audit_sha256": mapping_audit_sha256,
        "hb_cohort_representativeness_audit_sha256": representativeness_audit_sha256,
        "approved_changes_applied": [
            "family_renamed_to_portfolio_forecast_reliability_v1",
            "MNIH_only_positive_executable_label",
            "HUMAN_COMPARABLE_COHORT_gate",
            "origin_count_0_1_IHE_2_HR_ge3_MNIH_eligible",
            "keep_140401_drop_140312_for_reliability_joins",
            "neither_naive_blocks_MNIH",
            "valid_wmape_denominator_per_scope",
            "validity_range_MNIH_prefix",
            "evidence_strength_LIMITED_3_OR_4_ORIGINS",
            "absolute_ceilings_out_of_executable_v1",
            "upstream_residual_pipeline_lock_confirms_FROZEN_TS",
        ],
    }


def _strip_stale_coverage_figures(obj: Any) -> Any:
    """Remove keys/values that embed superseded 55/50/167/161 coverage figures."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(
                s in kl
                for s in (
                    "167",
                    "161",
                    "archval55",
                    "n_skus_with_at_least_3_complete_origins_v1",
                    "hb_reliability_coverage_audit_v1_counts",
                )
            ):
                continue
            if k in {
                "stale_v1_coverage",
                "legacy_matched_counts_167",
                "coverage_audit_v1_counts",
            }:
                continue
            out[k] = _strip_stale_coverage_figures(v)
        return out
    if isinstance(obj, list):
        return [_strip_stale_coverage_figures(x) for x in obj]
    return obj


def write_corrected_review(
    *,
    repo_root: Path,
    source_review_path: Path | None = None,
) -> dict[str, Any]:
    """Write MNIH-primary review-only artifact. Does NOT write the immutable contract."""
    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    family.mkdir(parents=True, exist_ok=True)
    out_path = family / REVIEW_FILENAME
    mapping_path = family / MAPPING_AUDIT_FILENAME
    rep_path = family / REPRESENTATIVENESS_AUDIT_FILENAME
    prot_val_path = family / PROTECTED_ORIGIN_VALIDATION_FILENAME
    stale_cov_path = family / STALE_COVERAGE_AUDIT_FILENAME

    # Always rebuild from a clean skeleton so stale 55/50/167/161 cannot linger.
    review: dict[str, Any] = {}
    if source_review_path is not None and Path(source_review_path).is_file():
        prior = json.loads(Path(source_review_path).read_text(encoding="utf-8"))
        # Keep only non-coverage structural keys if useful
        for keep in (
            "approval_checklist",
            "explicit_non_goals",
            "two_scopes",
            "proposed_numeric_thresholds_flat_for_approval",
        ):
            if keep in prior:
                review[keep] = prior[keep]
    review = _strip_stale_coverage_figures(review)

    upstream = upstream_residual_pipeline_evidence(repo_root)

    mapping_ref: dict[str, Any] = {"exists": mapping_path.is_file()}
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping_ref["relative_path"] = (
            FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME
        ).as_posix()
        mapping_ref["sha256"] = file_sha256(mapping_path)
        mapping_ref["protected_hb_origins_reliability"] = mapping.get(
            "protected_hb_origins_reliability"
        )
        mapping_ref["full_horizon_matched_counts_regenerated"] = {
            k: (mapping.get("full_horizon_matched_counts_regenerated") or {}).get(k)
            for k in (
                "definition_human",
                "definition_matched_machine_any",
                "role",
                "n_human_complete_1_15",
                "n_machine_any_matched_complete_1_15",
                "n_both_native_matched_complete_1_15",
                "n_neither_naive_human_complete",
                "n_portfolio_activation_deployable",
                "by_branch_human_complete",
                "sku_depth_machine_any_matched",
            )
        }
        mapping_ref["deployable_frozen_policy_reconciliation"] = {
            "reconciliation_status": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("reconciliation_status"),
            "identity": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("identity"),
            "deployable_coverage_pct": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("deployable_coverage_pct"),
            "n_hb_complete_product_origins_required": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("n_hb_complete_product_origins_required"),
            "n_deployable_with_forecast": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("n_deployable_with_forecast"),
            "n_no_machine_forecast": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("n_no_machine_forecast"),
            DEPLOYABLE_PORTFOLIO_COHORT: mapping.get(DEPLOYABLE_PORTFOLIO_COHORT),
            NO_MACHINE_FORECAST_COHORT: mapping.get(NO_MACHINE_FORECAST_COHORT),
            "sku_x_group_mnih_policy": (
                mapping.get("deployable_frozen_policy_reconciliation") or {}
            ).get("sku_x_group_mnih_policy"),
        }
        # Document supersession without embedding forbidden integers as live counts
        mapping_ref["stale_coverage_superseded"] = {
            "status": "SUPERSEDED",
            "artifact": STALE_COVERAGE_AUDIT_FILENAME,
            "forbidden_stale_integers": sorted(STALE_COVERAGE_FIGURES_FORBIDDEN),
            "note": (
                "Do not reuse Archval55/140312 coverage integers "
                f"{sorted(STALE_COVERAGE_FIGURES_FORBIDDEN)}; "
                "use regenerated full-horizon and group-specific counts above."
            ),
        }
        mapping_ref["group_specific_matched_coverage_summary"] = {
            g: {
                "n_skus_mnih_eligible_ge3_origins": block.get(
                    "n_skus_mnih_eligible_ge3_origins"
                ),
                "n_human_complete": (block.get("deployable_branch") or {}).get(
                    "n_human_complete"
                ),
                "n_machine_any_matched": (block.get("deployable_branch") or {}).get(
                    "n_machine_any_matched"
                ),
                "by_origin_hb_complete": block.get("by_origin_hb_complete"),
                "by_origin_matched_machine_any": block.get(
                    "by_origin_matched_machine_any"
                ),
                f"{HUMAN_COMPARABLE_COHORT}_{g}_n_skus": (
                    block.get(f"{HUMAN_COMPARABLE_COHORT}_{g}") or {}
                ).get("n_skus"),
            }
            for g, block in (mapping.get("group_specific_matched_coverage") or {}).items()
        }
        mapping_ref[LEGACY_TS_LOCK_HB_ORIGINS_KEY] = mapping.get(
            LEGACY_TS_LOCK_HB_ORIGINS_KEY
        )
        mapping_ref["postlock_context"] = mapping.get("postlock_140401_140402_context")
        mapping_ref["hb_outcomes_inspected"] = mapping.get(
            "hb_outcomes_inspected", False
        )

    rep_ref: dict[str, Any] = {"exists": rep_path.is_file()}
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rep_ref["relative_path"] = (
            FAMILY_DIR_REL / REPRESENTATIVENESS_AUDIT_FILENAME
        ).as_posix()
        rep_ref["sha256"] = file_sha256(rep_path)
        rep_ref["coverage_scope"] = rep.get("coverage_scope")
        rep_ref["covariate_balance"] = {
            "status": (rep.get("covariate_balance") or {}).get("status"),
            "flagged_continuous_axes": (rep.get("covariate_balance") or {}).get(
                "flagged_continuous_axes"
            ),
            "point_in_time_cutoff": (rep.get("covariate_balance") or {}).get(
                "point_in_time_cutoff"
            ),
        }
        rep_ref["categorical_balance"] = {
            "status": (rep.get("categorical_balance") or {}).get("status")
        }
        rep_ref["native_fallback_balance_among_present"] = {
            "status": (rep.get("native_fallback_balance_among_present") or {}).get(
                "status"
            )
        }
        rep_ref["machine_coverage_balance"] = rep.get("machine_coverage_balance")
        rep_ref["point_in_time"] = rep.get("point_in_time")
        rep_ref["portfolio_conclusion_scope"] = rep.get("portfolio_conclusion_scope")
        rep_ref["extrapolation_rule"] = rep.get("extrapolation_rule")
        rep_ref["sales_source"] = rep.get("sales_source")
        rep_ref["n_eligible"] = (
            (rep.get("populations") or {}).get("full_eligible_portfolio") or {}
        ).get("n_skus")
        rep_ref["n_hb_cohort"] = (
            (rep.get("populations") or {}).get("hb_group_union_cohort") or {}
        ).get("n_skus")
        rep_ref["invalid_prior_archval55_audit"] = rep.get(
            "invalid_prior_archval55_audit"
        )

    review["family"] = FAMILY
    review["artifact_name"] = REVIEW_FILENAME
    review["contract_status"] = "REVIEW_ONLY_MNIH_PRIMARY"
    review["contract_version_proposed"] = CONTRACT_VERSION
    review["final_contract_written"] = False
    review["final_contract_filename"] = None
    review["stop_before_final_contract"] = True
    review["hb_outcomes_inspected"] = False
    review["router_trained"] = False
    review["positive_executable_label"] = LABEL_MNIH
    review["protected_hb_origins_reliability"] = sorted(PROTECTED_HB_ORIGINS)
    review["reliability_execution_origins"] = sorted(PROTECTED_HB_ORIGINS)
    review["prospective_origins"] = prospective_origins_block()
    review["portfolio_temporal_gate"] = portfolio_temporal_gate()
    review[LEGACY_TS_LOCK_HB_ORIGINS_KEY] = {
        "note": (
            "Immutable TS-lock holdout list (may still include 140312). "
            "Must not drive reliability execution."
        ),
        "drives_reliability_execution": False,
        "source": "pkg.portfolio_ts_lock.constants.HB_ORIGINS",
    }
    review["stale_artifacts"] = {
        STALE_COVERAGE_AUDIT_FILENAME: {
            "status": "SUPERSEDED",
            "exists": stale_cov_path.is_file(),
            "drives_reliability_execution": False,
            "forbidden_figures": sorted(STALE_COVERAGE_FIGURES_FORBIDDEN),
            "superseded_by": (FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME).as_posix(),
        },
        PROTECTED_ORIGIN_VALIDATION_FILENAME: {
            "status": "SUPERSEDED",
            "exists": prot_val_path.is_file(),
            "drives_reliability_execution": False,
            "drives_review_contract": False,
            "superseded_by": (FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME).as_posix(),
            "erratum": "1404Q1→140401; 140312 is not a reliability origin",
        },
    }
    review["HUMAN_COMPARABLE_COHORT"] = {
        "definition": (
            "Native-matched complete machine∩HB SKUs — SKU×validity-group MNIH only"
        ),
        "role": "sku_x_group_MNIH_only",
        "not_portfolio_activation_population": True,
    }
    review[DEPLOYABLE_PORTFOLIO_COHORT] = {
        "definition": (
            "HB-complete product×origins with an actual frozen-policy forecast, "
            "including valid naive_last_value fallbacks"
        ),
        "role": "portfolio_activation_gate_population",
        "enters_accuracy_denominator": True,
        "n_product_origin_pairs": (
            (mapping_ref.get("deployable_frozen_policy_reconciliation") or {}).get(
                "n_deployable_with_forecast"
            )
            if mapping_ref.get("exists")
            else None
        ),
        "identity": (
            (mapping_ref.get("deployable_frozen_policy_reconciliation") or {}).get(
                "identity"
            )
            if mapping_ref.get("exists")
            else None
        ),
        "deployable_coverage_pct": (
            (mapping_ref.get("deployable_frozen_policy_reconciliation") or {}).get(
                "deployable_coverage_pct"
            )
            if mapping_ref.get("exists")
            else None
        ),
    }
    review[NO_MACHINE_FORECAST_COHORT] = {
        "definition": (
            "HB-complete without native V2.1/A0 or sales through origin−2; "
            "coverage failures; never zero-filled; excluded from accuracy denominators"
        ),
        "enters_accuracy_denominator": False,
        "n_product_origin_pairs": (
            (mapping_ref.get("deployable_frozen_policy_reconciliation") or {}).get(
                "n_no_machine_forecast"
            )
            if mapping_ref.get("exists")
            else None
        ),
        "n_skus": (
            ((mapping_ref.get(NO_MACHINE_FORECAST_COHORT) or {}) or {}).get("n_skus")
            if mapping_ref.get("exists")
            else None
        ),
        "skus": (
            ((mapping_ref.get(NO_MACHINE_FORECAST_COHORT) or {}) or {}).get("skus")
            if mapping_ref.get("exists")
            else None
        ),
        "reported_as": "coverage_failures",
    }
    review["HUMAN_COMPARABLE_COHORT_by_group"] = {
        f"{HUMAN_COMPARABLE_COHORT}_{g}": {
            "horizons": sorted(hs),
            "role": "sku_validity_group_label_origin_counts",
            "note": (
                "Group completeness does not require full h1–15; "
                "140401 partial horizons may still contribute. "
                "Native comparisons with ≥3 group-complete origins may qualify for MNIH."
            ),
        }
        for g, hs in VALIDITY_GROUPS
    }
    review["origin_count_label_map"] = {
        "scope": "per_SKU_x_validity_group_using_group_specific_complete_matched_origins",
        "applies_after": "neither_naive_or_NO_MACHINE_FORECAST_ruled_out",
        "0_or_1": LABEL_IHE,
        "2": LABEL_HR,
        "ge_3": "eligible_for_locked_MNIH_rules_native_only",
        "note": (
            "Origin-count map is for the native path only. "
            "neither_naive / NO_MACHINE_FORECAST → HUMAN_REVIEW_REQUIRED first."
        ),
    }
    review["mnih_gates_locked"] = {
        "min_complete_group_specific_origins": 3,
        "pooled_machine_minus_human_wmape_pp_max": 2.0,
        "origin_consistency": "ceil(0.75 * n_complete)",
        "wmape_denominator_scopes": list(WMAPE_DENOM_SCOPES),
        "portfolio_relative_gates_required": True,
        "portfolio_activation_population": DEPLOYABLE_PORTFOLIO_COHORT,
        "neither_naive_any_evaluated_origin_blocks_mnih": True,
        "NO_MACHINE_FORECAST_any_evaluated_origin_blocks_mnih": True,
        "evidence_strength_when_n_in_3_or_4": EVIDENCE_STRENGTH_LIMITED,
    }
    review["evaluation_scopes"] = {
        "primary_portfolio_activation": DEPLOYABLE_PORTFOLIO_COHORT,
        "sku_labels": (
            f"native {HUMAN_COMPARABLE_COHORT}_<group>; "
            f"neither_naive or {NO_MACHINE_FORECAST} → HUMAN_REVIEW_REQUIRED; "
            "missing/incomplete HB → INSUFFICIENT_HUMAN_EVIDENCE"
        ),
        "sensitivity": (
            "all complete matched product×origin + leave-one-origin "
            f"(exclude {NO_MACHINE_FORECAST} from accuracy denominators)"
        ),
        "portfolio_temporal_gate": "PREREGISTERED_NOT_EXECUTED",
        "never_silent_discard": True,
        "never_renumber_horizons": True,
        "full_h1_15_must_not_discard_earlier_complete_group": True,
        "never_zero_fill_NO_MACHINE_FORECAST": True,
    }
    # Link prepared 140404 extension artifacts when present (additive; never mutate locks).
    for key, fname in (
        ("hb_140404_maturity_audit", MATURITY_AUDIT_FILENAME),
        ("candidate_origin_manifest", CANDIDATE_ORIGIN_MANIFEST_FILENAME),
        ("sales_full_portfolio_v2_prepare", SALES_V2_PREPARE_MANIFEST_FILENAME),
        ("vps_140404_commands", VPS_140404_COMMANDS_FILENAME),
        ("hb_140404_identifier_trace", IDENTIFIER_TRACE_FILENAME),
        ("hb_reliability_forecast_coverage_audit", HB_FORECAST_COVERAGE_AUDIT_FILENAME),
        ("hb_reliability_candidate_manifest", HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME),
        ("hb_reliability_scoring_path_prepare", HB_SCORING_PATH_PREPARE_FILENAME),
        (
            "hb_reliability_quantity_retrieval_prepare",
            HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME,
        ),
    ):
        p = family / fname
        block: dict[str, Any] = {
            "exists": p.is_file(),
            "relative_path": (FAMILY_DIR_REL / fname).as_posix(),
        }
        if p.is_file():
            block["sha256"] = file_sha256(p)
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if key == "candidate_origin_manifest":
                block["140404_status"] = payload.get("140404_status")
                block["excluded"] = payload.get("excluded")
                block["proposed_final_origins"] = payload.get("proposed_final_origins")
                block["stop"] = payload.get("stop")
            elif key == "hb_140404_maturity_audit":
                block["all_gates_pass"] = payload.get("all_gates_pass")
                block["hb_outcomes_inspected"] = payload.get(
                    "hb_outcomes_inspected", False
                )
            elif key == "sales_full_portfolio_v2_prepare":
                block["status"] = payload.get("status")
                block["parquet_written"] = payload.get("parquet_written", False)
            elif key == "hb_140404_identifier_trace":
                block["hb_outcomes_inspected"] = payload.get(
                    "hb_outcomes_inspected", False
                )
                block["missing_target_month"] = (payload.get("diagnosis") or {}).get(
                    "missing_target_month"
                )
                block["n_skus_complete_1_15"] = (
                    (payload.get("frozen_v2_panel") or {})
                    .get("per_sku_horizon_completeness")
                    or {}
                ).get("n_skus_complete_1_15")
            elif key == "hb_reliability_forecast_coverage_audit":
                prov = payload.get("provenance") or {}
                block["publication_timing_status"] = prov.get(
                    "publication_timing_status"
                )
                gates = payload.get("gates") or {}
                block["machine_ready_status"] = (gates.get("machine_ready") or {}).get(
                    "status"
                )
                block["hb_forecast_coverage_status"] = (
                    gates.get("hb_forecast_coverage") or {}
                ).get("status")
            elif key == "hb_reliability_candidate_manifest":
                gates = payload.get("gates") or {}
                block["machine_ready_status"] = (gates.get("machine_ready") or {}).get(
                    "status"
                )
                block["excluded_from_contract_freeze"] = payload.get(
                    "excluded_from_contract_freeze"
                )
            elif key == "hb_reliability_quantity_retrieval_prepare":
                block["status"] = payload.get("status")
                block["hb_quantities_fetched"] = payload.get(
                    "hb_quantities_fetched", False
                )
                block["assert_exact_key_agreement"] = (
                    payload.get("quantity_query") or {}
                ).get("assert_exact_key_agreement")
        review[key] = block
    review["wmape_denominator"] = {
        "rule": WMAPE_DENOM_RULE,
        "eps": WMAPE_DENOM_EPS,
        "definition": "valid_wmape_denominator := sum(abs(actual)) > 1e-9",
        "applies_separately_to_scopes": list(WMAPE_DENOM_SCOPES),
        "status": "APPROVED",
    }
    review["pipeline_context"] = {
        "FROZEN_TS_equals": "A0_HYBRID_POLICY",
        "production_pipeline_locked": "FROZEN_TS",
        "residual_xgb_promoted": False,
        "median_benchmark_only": True,
        "upstream_residual_pipeline_lock_v1": upstream,
    }
    review["upstream_evidence"] = {"residual_pipeline_lock_v1": upstream}
    review["hb_origin_mapping_audit"] = mapping_ref
    review["hb_cohort_representativeness_audit"] = rep_ref
    if rep_ref.get("exists"):
        review["portfolio_conclusion_limit"] = rep_ref.get(
            "portfolio_conclusion_scope"
        )
        review["do_not_extrapolate_MNIH"] = rep_ref.get("extrapolation_rule")
        review["coverage_scope"] = (rep_ref.get("coverage_scope") or {}).get("status")
        review["covariate_balance"] = (rep_ref.get("covariate_balance") or {}).get(
            "status"
        )
        review["native_fallback_balance_among_present"] = (
            rep_ref.get("native_fallback_balance_among_present") or {}
        ).get("status")
        review["machine_coverage_balance"] = (
            rep_ref.get("machine_coverage_balance") or {}
        ).get("status")
        review["point_in_time_representativeness_cutoff"] = (
            rep_ref.get("point_in_time") or {}
        ).get("cutoff")
    review["labels"] = {
        LABEL_MNIH: mnih_label_meaning(),
        LABEL_HR: {
            "meaning": (
                "Any evaluated neither_naive / NO_MACHINE_FORECAST (even with 0–1 "
                "native origins), two native matched origins, or relative/portfolio "
                "failure."
            )
        },
        LABEL_IHE: {
            "meaning": (
                "Missing/incomplete HB evidence, invalid WMAPE denominator, "
                "non-computable metrics on the native path, or 0–1 native matched "
                "origins after neither_naive / NO_MACHINE_FORECAST ruled out."
            )
        },
    }
    review["future_out_of_scope"] = future_out_of_scope_absolute_ceilings()
    review["decision_table_preview"] = build_decision_table()
    review["locked_thresholds_preview"] = locked_thresholds()

    flat = review.setdefault("proposed_numeric_thresholds_flat_for_approval", {})
    if not isinstance(flat, dict):
        flat = {}
    flat = _strip_stale_coverage_figures(flat)
    flat["wmape_denominator.rule"] = {
        "status": "APPROVED",
        "unit": "predicate",
        "value": WMAPE_DENOM_RULE,
        "eps": WMAPE_DENOM_EPS,
        "applies_separately_to_scopes": list(WMAPE_DENOM_SCOPES),
    }
    flat["portfolio.portfolio_machine_wmape_ceiling_pct"] = {
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "value": None,
    }
    flat["sku_group.sku_group_machine_wmape_ceiling_pct"] = {
        "status": "OUT_OF_EXECUTABLE_V1_SCOPE",
        "value": None,
    }
    review["proposed_numeric_thresholds_flat_for_approval"] = flat

    non_goals = list(review.get("explicit_non_goals") or [])
    for g in (
        "Do not write the final reliability_label_contract_v1.json in this step",
        "Do not open or score HB outcomes",
        "Do not assign SKU labels yet",
        "Do not train a router",
        "Do not mutate ts_architecture_lock.json or production contract",
        "Do not attribute this labeling contract to portfolio_residual_xgb_v1",
        "Do not use hb_reliability_coverage_audit_v1.json or 55/50/167/161 figures",
        "Do not use hb_protected_origin_validation_v1.json for reliability origins",
        "Do not use Archval55 sales for full-portfolio representativeness",
        "Do not use post-PIT-cutoff (after 140302) sales for representativeness features",
        "Do not treat among-present native/fallback balance as full eligible machine coverage",
        "Do not zero-fill NO_MACHINE_FORECAST or treat it as a deployable naive forecast",
        "Do not use HUMAN_COMPARABLE_COHORT with ≥3 origins as the portfolio activation population",
    ):
        if g not in non_goals:
            non_goals.append(g)
    review["explicit_non_goals"] = non_goals

    apply_review_structure_corrections(review)
    review["corrected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Final guard: live coverage counts must not equal superseded Archval55 figures
    live = (
        (review.get("hb_origin_mapping_audit") or {}).get(
            "full_horizon_matched_counts_regenerated"
        )
        or {}
    )
    for key in (
        "n_human_complete_1_15",
        "n_machine_any_matched_complete_1_15",
        "n_both_native_matched_complete_1_15",
    ):
        val = live.get(key)
        if val in STALE_COVERAGE_FIGURES_FORBIDDEN:
            raise ValueError(f"Stale live coverage count for {key}={val}")
    depth = live.get("sku_depth_machine_any_matched") or {}
    if depth.get("n_skus_with_at_least_3_complete_origins") in STALE_COVERAGE_FIGURES_FORBIDDEN:
        raise ValueError("Stale live n_skus_ge3 leaked into review")
    if depth.get("n_skus") in STALE_COVERAGE_FIGURES_FORBIDDEN:
        raise ValueError("Stale live n_skus leaked into review")

    # Guard: no contradictory native≥3 portfolio primary population
    port_pop = (
        (review.get("locked_thresholds_preview") or {})
        .get("portfolio", {})
        .get("primary_population")
    )
    if port_pop != DEPLOYABLE_PORTFOLIO_COHORT:
        raise ValueError(
            f"portfolio primary_population must be {DEPLOYABLE_PORTFOLIO_COHORT}, "
            f"got {port_pop!r}"
        )
    dt_order = (
        (review.get("decision_table_preview") or {})
        .get("portfolio_activation_gate", {})
        .get("order")
        or []
    )
    if any("HUMAN_COMPARABLE_COHORT_ge3" in str(x) for x in dt_order):
        raise ValueError("contradictory native≥3 portfolio order gate remains")
    if any(
        "primary_population_skus_ge3" in str(x) for x in dt_order
    ):
        raise ValueError("contradictory ge3 SKU portfolio order gate remains")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_path)

    recon = (
        (review.get("hb_origin_mapping_audit") or {}).get(
            "deployable_frozen_policy_reconciliation"
        )
        or {}
    )
    return {
        "ok": True,
        "review_path": out_path.as_posix(),
        "review_sha256": file_sha256(out_path),
        "family": FAMILY,
        "final_contract_written": False,
        "hb_outcomes_inspected": False,
        "mapping_audit_linked": bool(mapping_ref.get("exists")),
        "representativeness_audit_linked": bool(rep_ref.get("exists")),
        "wmape_denominator_rule": WMAPE_DENOM_RULE,
        "coverage_scope": review.get("coverage_scope"),
        "covariate_balance": review.get("covariate_balance"),
        "deployable_identity": recon.get("identity"),
        "n_deployable_with_forecast": recon.get("n_deployable_with_forecast"),
        "n_no_machine_forecast": recon.get("n_no_machine_forecast"),
        "deployable_coverage_pct": recon.get("deployable_coverage_pct"),
        "upstream_evidence": upstream,
    }


class FinalFreezeNotApproved(ValueError):
    """Raised when freeze is attempted before Planning/process approval."""


# Back-compat name
PlanningCeilingsRequired = FinalFreezeNotApproved


def freeze_reliability_label_contract_v1(
    *,
    repo_root: Path,
    allow_final_freeze: bool = False,
    # Deprecated — absolute ceilings out of executable v1; ignored.
    portfolio_machine_wmape_ceiling_pct: float | None = None,
    sku_group_machine_wmape_ceiling_pct: float | None = None,
    allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling: bool = True,
) -> dict[str, Any]:
    """Create immutable contract only when explicitly allowed (not this implementation step)."""
    del portfolio_machine_wmape_ceiling_pct
    del sku_group_machine_wmape_ceiling_pct
    del allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling

    if not allow_final_freeze:
        raise FinalFreezeNotApproved(
            "Refuse freeze: final reliability_label_contract_v1.json is not approved "
            "in this step. Keep reliability_label_contract_v1_review.json until "
            "Planning explicitly allows final freeze (allow_final_freeze=True)."
        )

    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    out_path = family / CONTRACT_FILENAME
    review_path = family / REVIEW_FILENAME
    mapping_path = family / MAPPING_AUDIT_FILENAME
    rep_path = family / REPRESENTATIVENESS_AUDIT_FILENAME

    if out_path.exists() or out_path.with_suffix(out_path.suffix + ".sha256").exists():
        raise LockExistsError(f"Immutable contract already exists: {out_path}")
    if not review_path.is_file():
        raise FileNotFoundError(f"Missing review artifact: {review_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing mapping audit: {mapping_path}")
    if not rep_path.is_file():
        raise FileNotFoundError(f"Missing representativeness audit: {rep_path}")

    review_sha = file_sha256(review_path)
    map_sha = file_sha256(mapping_path)
    rep_sha = file_sha256(rep_path)
    payload = build_reliability_label_contract_v1(
        repo_root=repo_root,
        review_sha256=review_sha,
        mapping_audit_sha256=map_sha,
        representativeness_audit_sha256=rep_sha,
    )
    digest = write_immutable_json(out_path, payload)

    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["final_contract_written"] = True
    review["final_contract_filename"] = CONTRACT_FILENAME
    review["final_contract_sha256"] = digest
    review["stop_before_final_contract"] = False
    review["contract_status"] = "FINAL_CONTRACT_FROZEN"
    review["hb_outcomes_inspected"] = False
    tmp = review_path.with_suffix(review_path.suffix + ".tmp")
    tmp.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(review_path)

    return {
        "ok": True,
        "contract_path": out_path.as_posix(),
        "contract_sha256": digest,
        "family": FAMILY,
        "hb_outcomes_inspected": False,
        "decision_table": payload["decision_table"],
        "locked_thresholds": payload["locked_thresholds"],
        "upstream_evidence": payload["upstream_evidence"],
    }
