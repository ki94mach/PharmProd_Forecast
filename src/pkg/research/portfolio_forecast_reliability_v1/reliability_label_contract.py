"""Reliability label contract v1 — review now; immutable freeze only after Planning ceilings.

Family: portfolio_forecast_reliability_v1 (not residual-XGB).
Does not inspect HB forecast/outcome columns.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pkg.portfolio_ts_lock.constants import HB_ORIGINS, HORIZON_GROUPS
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.portfolio_ts_lock.production_contract import (
    file_sha256,
    write_immutable_json,
)
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
    ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY,
    CONTRACT_FILENAME,
    CONTRACT_VERSION,
    FAMILY,
    FAMILY_DIR_REL,
    REVIEW_FILENAME,
    UPSTREAM_RESIDUAL_PIPELINE_LOCK_REL,
    WMAPE_DENOM_EPS,
    WMAPE_DENOM_RULE,
)

LABEL_MR = "MACHINE_RELIABLE"
LABEL_MNIH = "MACHINE_NONINFERIOR_TO_HUMAN"
LABEL_HR = "HUMAN_REVIEW_REQUIRED"
LABEL_IE = "INSUFFICIENT_EVIDENCE"

GROUP_ORDER: tuple[str, ...] = ("h1_3", "h4_6", "h7_12", "h13_15")
GROUP_WIDTH: dict[str, int] = {
    "h1_3": 3,
    "h4_6": 3,
    "h7_12": 6,
    "h13_15": 3,
}

# Scopes that must each apply the WMAPE denominator rule independently.
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


def wmape_denominator_valid(sum_abs_actual: float) -> bool:
    """valid_wmape_denominator := sum(abs(actual)) > 1e-9 and finite."""
    try:
        v = float(sum_abs_actual)
    except (TypeError, ValueError):
        return False
    if v != v or v == float("inf") or v == float("-inf"):
        return False
    return v > WMAPE_DENOM_EPS


def segment_specific_ceiling_contract() -> dict[str, Any]:
    """Required Planning definitions if segment-specific ceilings are chosen."""
    return {
        "allowed": True,
        "proposed_map": None,
        "status": "REQUIRES_PLANNING_DEFINITION_BEFORE_OUTCOMES",
        "must_define_before_opening_outcomes": True,
        "when": (
            "If demand segments need different tolerances, Planning may approve a map "
            "segment_id -> sku_group_machine_wmape_ceiling_pct instead of (or in addition "
            "to) a single global SKU×group ceiling."
        ),
        "segment_assignment": {
            "required_if_map_used": True,
            "definition_required": (
                "Planning must define how each SKU maps to a segment_id before any "
                "HB outcomes are opened."
            ),
            "assignment_artifact": None,
            "stability": (
                "Segment assignment must be frozen and deterministic; no post-hoc "
                "reassignment that uses outcomes."
            ),
        },
        "ceiling_precedence": [
            (
                "If a segment-specific map is approved and the SKU has an assigned "
                "segment with a ceiling in the map → use that segment ceiling"
            ),
            (
                "Else if global sku_group_machine_wmape_ceiling_pct is approved → "
                "use the global ceiling"
            ),
            "Else → no absolute SKU×group ceiling is available for that SKU×group",
        ],
        "missing_segment_behavior": {
            "when_sku_unassigned_or_segment_absent_from_map": (
                "Do not invent a ceiling. MACHINE_RELIABLE is blocked for that "
                "SKU×group. Label INSUFFICIENT_EVIDENCE if the gap is treated as "
                "non-computable absolute gate; otherwise HUMAN_REVIEW_REQUIRED when "
                "n_complete_origins >= 2."
            ),
            "MACHINE_NONINFERIOR_TO_HUMAN": (
                f"Only if {ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}=true AND relative "
                "non-inferiority + stability rules pass; else do not use MNIH."
            ),
        },
        "note": (
            "If Planning chooses segment-specific ceilings, define the segment "
            "assignment, ceiling precedence, and missing-segment behavior before "
            "opening outcomes."
        ),
    }


def canonical_per_group_label_rules() -> dict[str, Any]:
    """Ordered SKU×group label rules (includes MNIH branch; switch defaults false)."""
    return {
        "ordered": [
            {
                "when": "n_complete_origins_in_group < 2",
                "label": LABEL_IE,
            },
            {
                "when": "wmape_denominator_invalid (Σ|actual| <= 1e-9 or non-finite)",
                "label": LABEL_IE,
            },
            {
                "when": "n_complete_origins_in_group == 2",
                "label": LABEL_HR,
            },
            {
                "when": (
                    "n_complete_origins_in_group >= 3 "
                    "AND wmape_denominator_valid "
                    "AND machine_group_wmape <= approved_sku_group_wmape_ceiling "
                    "AND WMAPE(machine)-WMAPE(human) <= 2.0 pp "
                    "AND n_origins_noninferior >= ceil(0.75 * n_complete_origins) "
                    "AND portfolio_activation_gate_passed "
                    "AND deployable_branch != neither_naive "
                    "AND Planning has approved absolute SKU×group ceiling "
                    "(global or segment-specific per ceiling_precedence)"
                ),
                "label": LABEL_MR,
            },
            {
                "when": (
                    f"{ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}=true "
                    "AND n_complete_origins_in_group >= 3 "
                    "AND wmape_denominator_valid "
                    "AND WMAPE(machine)-WMAPE(human) <= 2.0 pp "
                    "AND n_origins_noninferior >= ceil(0.75 * n_complete_origins) "
                    "AND portfolio_activation_gate_passed "
                    "AND deployable_branch != neither_naive "
                    "AND sku_group absolute ceiling unavailable"
                ),
                "label": LABEL_MNIH,
                "switch_default": ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
                "note": (
                    "Canonical ordered branch even though the switch defaults to false."
                ),
            },
            {
                "when": "otherwise (including neither_naive, failed absolute/relative gates)",
                "label": LABEL_HR,
            },
        ],
        "label_naming_contingency": {
            ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY: ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
            "if_sku_group_absolute_ceiling_cannot_be_supplied": (
                "Do not use MACHINE_RELIABLE unless an absolute ceiling is approved. "
                f"MACHINE_NONINFERIOR_TO_HUMAN applies only when "
                f"{ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}=true and relative "
                "non-inferiority + origin-count/stability rules pass."
            ),
        },
    }


def apply_review_structure_corrections(review: dict[str, Any]) -> None:
    """Four structural corrections to the review artifact (in place)."""
    two = review.setdefault("two_scopes", {})
    port = two.setdefault("portfolio_activation_gate", {})
    gates = list(port.get("gates_apply_here_only") or [])
    if "max_ae_ratio" not in gates:
        # Place with the other AE-ratio gates when present; else append before denom.
        if "p99_ae_ratio" in gates:
            i = gates.index("p99_ae_ratio") + 1
            gates.insert(i, "max_ae_ratio")
        elif "wmape_denominator_must_be_valid" in gates:
            gates.insert(gates.index("wmape_denominator_must_be_valid"), "max_ae_ratio")
        else:
            gates.append("max_ae_ratio")
    port["gates_apply_here_only"] = gates

    sku = two.setdefault("sku_validity_group_labels", {})
    sku["per_group_label_rule"] = canonical_per_group_label_rules()

    thr = sku.setdefault("proposed_numeric_thresholds", {})
    ceil = thr.setdefault("sku_group_machine_wmape_ceiling_pct", {})
    if not isinstance(ceil, dict):
        ceil = {}
        thr["sku_group_machine_wmape_ceiling_pct"] = ceil
    ceil["segment_specific_ceilings"] = segment_specific_ceiling_contract()

    flat = review.setdefault("proposed_numeric_thresholds_flat_for_approval", {})
    flat_key = "sku_group.sku_group_machine_wmape_ceiling_pct"
    if flat_key in flat and isinstance(flat[flat_key], dict):
        flat[flat_key]["segment_specific_ceilings"] = segment_specific_ceiling_contract()

    pending = review.setdefault("planning_ceilings_pending", {})
    pending["segment_specific_sku_group_ceilings"] = segment_specific_ceiling_contract()
    pending["note"] = (
        "Immutable reliability_label_contract_v1.json is written only after Planning "
        "supplies absolute ceilings. If Planning chooses segment-specific ceilings, "
        "define segment assignment, ceiling precedence, and missing-segment behavior "
        "before opening outcomes."
    )

    labels = review.setdefault("labels", {})
    if LABEL_IE in labels and isinstance(labels[LABEL_IE], dict):
        labels[LABEL_IE]["meaning"] = (
            "Fewer than 2 complete origins in the group, incomplete/missing group "
            "coverage, invalid WMAPE denominator (Σ|actual| <= 1e-9 or non-finite) "
            "at the relevant scope, or non-computable metrics."
        )

    checklist = list(review.get("approval_checklist") or [])
    for item in (
        "Confirm gates_apply_here_only includes max_ae_ratio",
        "Confirm per_group_label_rule ordered list includes MACHINE_NONINFERIOR_TO_HUMAN branch (switch default false)",
        "If segment-specific ceilings: define assignment, precedence, and missing-segment behavior before opening outcomes",
    ):
        if item not in checklist:
            checklist.append(item)
    review["approval_checklist"] = checklist


def validity_range_prefix(group_labels: Sequence[str]) -> dict[str, Any]:
    """Longest contiguous MACHINE_RELIABLE prefix starting at h1_3."""
    prefix: list[str] = []
    for name, lab in zip(GROUP_ORDER, group_labels):
        if lab != LABEL_MR:
            break
        prefix.append(name)
    human_from = GROUP_ORDER[len(prefix)] if len(prefix) < len(GROUP_ORDER) else None
    return {
        "machine_reliable_groups": prefix,
        "n_groups": len(prefix),
        "human_review_from_group": human_from,
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
    sku_group_ceiling_pct: float | None,
    portfolio_ceiling_pct: float | None,
    allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling: bool = (
        ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT
    ),
) -> str:
    """Apply SKU×validity-group decision table (no HB I/O)."""
    contingency = bool(allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling)

    if not group_horizons_exact:
        return LABEL_IE
    if n_complete_origins < 2:
        return LABEL_IE
    if not wmape_denominator_valid(sum_abs_actual):
        return LABEL_IE
    if machine_group_wmape is None or human_group_wmape is None:
        return LABEL_IE
    if machine_group_wmape != machine_group_wmape or human_group_wmape != human_group_wmape:
        return LABEL_IE

    if n_complete_origins == 2:
        return LABEL_HR

    rel_ok = (machine_group_wmape - human_group_wmape) <= 2.0
    stab_ok = n_origins_noninferior >= required_noninferior_origins(n_complete_origins)
    branch_ok = deployable_branch != "neither_naive"
    port_ok = bool(portfolio_activation_passed)

    if not (rel_ok and stab_ok and branch_ok and port_ok):
        return LABEL_HR

    if sku_group_ceiling_pct is None:
        if contingency:
            return LABEL_MNIH
        return LABEL_HR

    abs_ok = machine_group_wmape <= float(sku_group_ceiling_pct)
    if portfolio_ceiling_pct is None:
        if contingency:
            return LABEL_MNIH
        return LABEL_HR

    if abs_ok:
        return LABEL_MR
    return LABEL_HR


def locked_thresholds(
    *,
    allow_mnih: bool = ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
    portfolio_ceiling: float | None = None,
    sku_group_ceiling: float | None = None,
) -> dict[str, Any]:
    """Structural/relative thresholds; Planning ceilings only when supplied."""
    return {
        "wmape_denominator": {
            "rule": WMAPE_DENOM_RULE,
            "eps": WMAPE_DENOM_EPS,
            "applies_separately_to_scopes": list(WMAPE_DENOM_SCOPES),
        },
        ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY: allow_mnih,
        "portfolio": {
            "min_skus_with_at_least_3_complete_origins": 40,
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
            "portfolio_machine_wmape_ceiling_pct": portfolio_ceiling,
            "portfolio_machine_wmape_ceiling_status": (
                "APPROVED"
                if portfolio_ceiling is not None
                else "REQUIRES_PLANNING_VALUE"
            ),
        },
        "sku_group": {
            "min_complete_origins_per_group": 3,
            "machine_vs_human_noninferiority_wmape_margin_pp": 2.0,
            "origin_stability_frac": 0.75,
            "origin_stability_rule": "required_noninferior_origins=ceil(0.75*complete_origins)",
            "wmape_denominator_rule": WMAPE_DENOM_RULE,
            "sku_group_machine_wmape_ceiling_pct": sku_group_ceiling,
            "sku_group_machine_wmape_ceiling_status": (
                "APPROVED" if sku_group_ceiling is not None else "REQUIRES_PLANNING_VALUE"
            ),
            "segment_specific_sku_group_ceilings": None,
        },
        "sku_x_origin": {
            "wmape_denominator_rule": WMAPE_DENOM_RULE,
            "note": "Origin-stability per-origin WMAPE uses the same denominator rule independently.",
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


def build_decision_table(
    *,
    allow_mnih: bool = ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
) -> dict[str, Any]:
    mnih_when = (
        "same as MR relative rules but sku_ceiling unavailable "
        f"AND {ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}=true"
    )
    return {
        "portfolio_activation_gate": {
            "order": [
                "complete_1_15_required_never_silent_drop",
                "coverage_distribution_min_skus_ge3_origins",
                "valid_wmape_denominator_per_portfolio_scope",
                "portfolio_machine_wmape_ceiling_pct_when_set",
                "wmape_noninferiority_le_2pp",
                "rmse_p95_p99_le_1_02x",
                "max_ae_le_1_10x",
                "sku_cluster_bootstrap_frac_ge_0_90",
                "drop_top3_by_sum_ae_machine_advantage_remaining_noninferior",
            ],
            "if_fails": "no_MACHINE_RELIABLE_anywhere",
            "primary_population": "deployable_frozen_policy_vs_human_budget",
            "diagnostic_population": "common_native_vs_human_budget",
            "wmape_denominator": {
                "rule": WMAPE_DENOM_RULE,
                "scope": "portfolio",
            },
            "drop_top_k": {
                "k": 3,
                "rank": "ΣAE(human)−ΣAE(machine)",
                "not": "mean_AE",
            },
        },
        "sku_x_validity_group": {
            "group_order": list(GROUP_ORDER),
            "wmape_denominator": {
                "rule": WMAPE_DENOM_RULE,
                "scope": "sku_x_validity_group",
            },
            ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY: allow_mnih,
            "branches": [
                {"when": "group_horizons_not_exact", "label": LABEL_IE},
                {"when": "n_complete_origins < 2", "label": LABEL_IE},
                {
                    "when": f"not ({WMAPE_DENOM_RULE}) at sku×group scope",
                    "label": LABEL_IE,
                },
                {"when": "metrics_noncomputable", "label": LABEL_IE},
                {"when": "n_complete_origins == 2", "label": LABEL_HR},
                {
                    "when": (
                        "n>=3 AND denom_ok AND machine_wmape<=sku_ceiling "
                        "AND (machine-human)<=2pp AND origin_stability "
                        "AND portfolio_gate_passed AND branch!=neither_naive "
                        "AND both_ceilings_approved"
                    ),
                    "label": LABEL_MR,
                },
                {"when": mnih_when, "label": LABEL_MNIH},
                {"when": "otherwise", "label": LABEL_HR},
            ],
            "origin_stability": {
                "formula": "ceil(0.75 * n_complete_origins)",
                "n3": 3,
                "n4": 3,
                "per_origin_wmape_denominator": WMAPE_DENOM_RULE,
            },
            "validity_range": (
                "longest contiguous MACHINE_RELIABLE prefix from h1_3; "
                "MR,IE → range h1–3; human review from h4"
            ),
            "neither_naive": "cannot_receive_MACHINE_RELIABLE",
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
    coverage_audit_sha256: str | None = None,
    portfolio_ceiling_pct: float,
    sku_group_ceiling_pct: float,
    allow_mnih: bool = ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
) -> dict[str, Any]:
    thr = locked_thresholds(
        allow_mnih=allow_mnih,
        portfolio_ceiling=float(portfolio_ceiling_pct),
        sku_group_ceiling=float(sku_group_ceiling_pct),
    )
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
        "protected_hb_origins": sorted(HB_ORIGINS),
        "machine_policy": "FROZEN_TS = A0_HYBRID_POLICY",
        "deployable_branches": [
            "both_native",
            "v21_only",
            "a0_only",
            "neither_naive",
        ],
        "validity_groups": validity_groups,
        "labels": {
            LABEL_MR: {
                "requires_portfolio_and_sku_group_absolute_ceilings": True,
                "blocked_if_neither_naive": True,
                "blocked_if_portfolio_gate_fails": True,
            },
            LABEL_MNIH: {
                "enabled_only_when": ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY,
                "default_enabled": ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
                "proves": "relative_noninferiority_only",
            },
            LABEL_HR: {},
            LABEL_IE: {
                "includes": [
                    "incomplete_group_horizons",
                    "n_origins_lt_2",
                    f"invalid_wmape_denominator_not_{WMAPE_DENOM_RULE}",
                ]
            },
        },
        "locked_thresholds": thr,
        "planning_ceilings": {
            "portfolio_machine_wmape_ceiling_pct": float(portfolio_ceiling_pct),
            "sku_group_machine_wmape_ceiling_pct": float(sku_group_ceiling_pct),
            "segment_specific_sku_group_ceilings": None,
            "owner": "Planning",
            "MACHINE_RELIABLE_blocked_until_both_set": False,
        },
        "decision_table": build_decision_table(allow_mnih=allow_mnih),
        "upstream_evidence": {
            "residual_pipeline_lock_v1": upstream,
        },
        "source_review": {
            "filename": REVIEW_FILENAME,
            "sha256": review_sha256,
        },
        "coverage_audit_sha256": coverage_audit_sha256,
        "approved_changes_applied": [
            "family_renamed_to_portfolio_forecast_reliability_v1",
            "portfolio_vs_sku_two_scope_split",
            "drop_top3_by_sum_ae_machine_advantage",
            "coverage_distribution_min_40_skus_ge3_origins",
            "origin_stability_ceil_0_75",
            "sku_group_absolute_wmape_ceiling_requirement",
            "valid_wmape_denominator_sum_abs_actual_gt_1e-9_per_scope",
            "validity_range_MR_IE_is_h1_3_not_empty",
            f"{ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}_default_false",
            "planning_ceilings_required_before_immutable_freeze",
            "upstream_residual_pipeline_lock_confirms_FROZEN_TS",
        ],
    }


def _update_wmape_denom_entries(node: Any) -> None:
    """Rewrite WMAPE denominator definitions in the review tree."""
    if isinstance(node, dict):
        if "wmape_denominator" in str(node.keys()).lower() or any(
            "wmape_denominator" in str(k) for k in node
        ):
            pass
        for key, child in list(node.items()):
            if key in {
                "wmape_denominator_must_be_valid",
                "valid_wmape_denominator",
            } or (
                isinstance(child, dict)
                and "denominator" in key.lower()
                and "wmape" in key.lower()
            ):
                if isinstance(child, dict):
                    child["rule"] = WMAPE_DENOM_RULE
                    child["eps"] = WMAPE_DENOM_EPS
                    child["definition"] = (
                        "valid_wmape_denominator := sum(abs(actual)) > 1e-9. "
                        "Apply separately to every portfolio, SKU×group, and "
                        "SKU×origin calculation. If invalid → INSUFFICIENT_EVIDENCE "
                        "(or portfolio activation non-computable)."
                    )
                    child["applies_separately_to_scopes"] = list(WMAPE_DENOM_SCOPES)
                    child["value"] = True
                    if str(child.get("status", "")).startswith("PROPOSED") or child.get(
                        "status"
                    ) in {"APPROVED", "REQUIRED_REPORT_EXPLICIT"}:
                        child["status"] = "APPROVED"
                continue
            if isinstance(child, dict) and child.get("definition") and (
                "Σ|actual|" in str(child.get("definition", ""))
                or "sum(|actual|)" in str(child.get("definition", "")).lower()
                or "denominator" in key.lower()
            ):
                if "wmape" in key.lower() or "denominator" in str(
                    child.get("definition", "")
                ).lower():
                    child["definition"] = (
                        "valid_wmape_denominator := sum(abs(actual)) > 1e-9. "
                        "Checked independently at portfolio, SKU×group, and SKU×origin scopes."
                    )
                    child["rule"] = WMAPE_DENOM_RULE
                    child["eps"] = WMAPE_DENOM_EPS
            _update_wmape_denom_entries(child)
    elif isinstance(node, list):
        for item in node:
            _update_wmape_denom_entries(item)


def write_corrected_review(
    *,
    repo_root: Path,
    source_review_path: Path | None = None,
) -> dict[str, Any]:
    """Write corrected review-only artifact under portfolio_forecast_reliability_v1.

    Does NOT write the immutable final contract.
    """
    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    family.mkdir(parents=True, exist_ok=True)
    out_path = family / REVIEW_FILENAME

    if source_review_path is None:
        candidates = [
            repo_root
            / "src/data/results/portfolio_residual_xgb_v1"
            / REVIEW_FILENAME,
            out_path,
        ]
        source_review_path = next((p for p in candidates if p.is_file()), None)
    if source_review_path is None or not Path(source_review_path).is_file():
        raise FileNotFoundError("No source reliability_label_contract_v1_review.json")

    review = json.loads(Path(source_review_path).read_text(encoding="utf-8"))
    upstream = upstream_residual_pipeline_evidence(repo_root)

    review["family"] = FAMILY
    review["artifact_name"] = REVIEW_FILENAME
    review["contract_status"] = "REVIEW_ONLY_PENDING_PLANNING_CEILINGS"
    review["contract_version_proposed"] = CONTRACT_VERSION
    review["final_contract_written"] = False
    review["final_contract_filename"] = None
    review["stop_before_final_contract"] = True
    review["hb_outcomes_inspected"] = False
    review[ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] = ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT
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
    review["planning_ceilings_pending"] = {
        "portfolio_machine_wmape_ceiling_pct": None,
        "sku_group_machine_wmape_ceiling_pct": None,
        "segment_specific_sku_group_ceilings": None,
        "owner": "Planning",
        "note": (
            "Immutable reliability_label_contract_v1.json is written only after "
            "Planning supplies both portfolio_machine_wmape_ceiling_pct and "
            "sku_group_machine_wmape_ceiling_pct."
        ),
    }
    review["labels"] = review.get("labels") or {}
    if LABEL_IE in review["labels"]:
        review["labels"][LABEL_IE]["meaning"] = (
            "Fewer than 2 complete origins in the group, incomplete/missing group "
            "coverage, invalid WMAPE denominator (Σ|actual| <= 1e-9 or non-finite) "
            "at the relevant scope, or non-computable metrics."
        )
    if LABEL_MNIH in review["labels"]:
        review["labels"][LABEL_MNIH][ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] = (
            ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT
        )
        review["labels"][LABEL_MNIH]["not_default"] = True
        review["labels"][LABEL_MNIH]["enabled_only_when_switch_true"] = True

    _update_wmape_denom_entries(review)

    # Flat ledger: ensure denom + contingency entries exist
    flat = review.get("proposed_numeric_thresholds_flat_for_approval") or {}
    flat["wmape_denominator.rule"] = {
        "status": "APPROVED",
        "unit": "predicate",
        "value": WMAPE_DENOM_RULE,
        "eps": WMAPE_DENOM_EPS,
        "applies_separately_to_scopes": list(WMAPE_DENOM_SCOPES),
    }
    flat[f"sku_group.{ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}"] = {
        "status": "APPROVED",
        "unit": "boolean",
        "value": ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
        "default": ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
        "note": (
            "When false (default), missing sku_group ceiling → HUMAN_REVIEW_REQUIRED, "
            "not MACHINE_NONINFERIOR_TO_HUMAN."
        ),
    }
    for key, entry in list(flat.items()):
        if not isinstance(entry, dict):
            continue
        if "wmape_denominator" in key:
            entry["rule"] = WMAPE_DENOM_RULE
            entry["eps"] = WMAPE_DENOM_EPS
            entry["definition"] = (
                "valid_wmape_denominator := sum(abs(actual)) > 1e-9; "
                "apply separately at portfolio, SKU×group, and SKU×origin scopes."
            )
            if entry.get("status", "").startswith("PROPOSED") or entry.get("status") in {
                "APPROVED"
            }:
                entry["status"] = "APPROVED"
        # Keep Planning ceilings pending
        if "ceiling" in key and entry.get("value") is None:
            entry["status"] = "REQUIRES_PLANNING_VALUE"
    review["proposed_numeric_thresholds_flat_for_approval"] = flat

    checklist = list(review.get("approval_checklist") or [])
    extras = [
        "Confirm family=portfolio_forecast_reliability_v1 (not residual-XGB)",
        "Confirm valid_wmape_denominator := sum(abs(actual)) > 1e-9 at portfolio, SKU×group, and SKU×origin scopes",
        f"Confirm {ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY}=false (default)",
        "Confirm upstream residual_pipeline_lock_v1 shows KEEP_FROZEN_TS / production=FROZEN_TS",
        "Do not write immutable reliability_label_contract_v1.json until Planning supplies both ceilings",
    ]
    for e in extras:
        if e not in checklist:
            checklist.append(e)
    # Fix stale checklist item about writing final contract
    checklist = [
        c
        for c in checklist
        if "no final contract yet" not in c.lower()
        or "Do not write immutable" in c
    ]
    if "Confirm no HB outcomes opened; no router trained; no final contract yet" not in checklist:
        checklist.append(
            "Confirm no HB outcomes opened; no router trained; no final contract yet"
        )
    review["approval_checklist"] = checklist

    non_goals = list(review.get("explicit_non_goals") or [])
    for g in (
        "Do not write the final reliability_label_contract_v1.json until Planning ceilings are supplied",
        "Do not attribute this labeling contract to portfolio_residual_xgb_v1",
    ):
        if g not in non_goals:
            non_goals.append(g)
    # Replace older "Do not write the final ... yet" if present
    non_goals = [
        g
        for g in non_goals
        if g
        != "Do not write the final reliability_label_contract_v1.json yet"
    ]
    review["explicit_non_goals"] = non_goals

    review["decision_table_preview"] = build_decision_table(
        allow_mnih=ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT
    )
    apply_review_structure_corrections(review)
    review["corrected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_path)

    # Pointer in old residual family dir (mutable note; not a contract)
    old_dir = repo_root / "src/data/results/portfolio_residual_xgb_v1"
    if old_dir.is_dir():
        pointer = {
            "moved_to_family": FAMILY,
            "review_path": (FAMILY_DIR_REL / REVIEW_FILENAME).as_posix(),
            "reason": (
                "Residual XGBoost was rejected; reliability labeling belongs to "
                "portfolio_forecast_reliability_v1. Premature immutable contract removed."
            ),
            "hb_outcomes_inspected": False,
            "upstream_residual_pipeline_lock_v1": upstream,
        }
        ptr_path = old_dir / "reliability_label_contract_MOVED.json"
        ptr_path.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stale = old_dir / REVIEW_FILENAME
        if stale.is_file():
            stale.unlink()

    return {
        "ok": True,
        "review_path": out_path.as_posix(),
        "review_sha256": file_sha256(out_path),
        "family": FAMILY,
        "final_contract_written": False,
        "hb_outcomes_inspected": False,
        ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY: ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT,
        "wmape_denominator_rule": WMAPE_DENOM_RULE,
        "upstream_evidence": upstream,
    }


class PlanningCeilingsRequired(ValueError):
    """Raised when freeze is attempted without both Planning ceilings."""


def freeze_reliability_label_contract_v1(
    *,
    repo_root: Path,
    portfolio_machine_wmape_ceiling_pct: float | None = None,
    sku_group_machine_wmape_ceiling_pct: float | None = None,
    allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling: bool = (
        ALLOW_MNIH_WITHOUT_SKU_CEILING_DEFAULT
    ),
) -> dict[str, Any]:
    """Atomically create immutable contract ONLY after Planning ceilings are supplied."""
    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    out_path = family / CONTRACT_FILENAME
    review_path = family / REVIEW_FILENAME
    cov_path = family / "hb_reliability_coverage_audit_v1.json"

    if portfolio_machine_wmape_ceiling_pct is None or sku_group_machine_wmape_ceiling_pct is None:
        raise PlanningCeilingsRequired(
            "Refuse freeze: Planning must supply both "
            "portfolio_machine_wmape_ceiling_pct and sku_group_machine_wmape_ceiling_pct "
            "before writing immutable reliability_label_contract_v1.json. "
            "Keep reliability_label_contract_v1_review.json until then."
        )

    if out_path.exists() or out_path.with_suffix(out_path.suffix + ".sha256").exists():
        raise LockExistsError(f"Immutable contract already exists: {out_path}")

    if not review_path.is_file():
        raise FileNotFoundError(f"Missing review artifact: {review_path}")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["planning_ceilings_pending"] = {
        "portfolio_machine_wmape_ceiling_pct": float(portfolio_machine_wmape_ceiling_pct),
        "sku_group_machine_wmape_ceiling_pct": float(sku_group_machine_wmape_ceiling_pct),
        "segment_specific_sku_group_ceilings": None,
        "owner": "Planning",
        "status": "SUPPLIED",
    }
    review["contract_status"] = "PLANNING_CEILINGS_SUPPLIED_READY_TO_FREEZE"
    review[ALLOW_MNIH_WITHOUT_SKU_CEILING_KEY] = bool(
        allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling
    )
    review["hb_outcomes_inspected"] = False
    tmp = review_path.with_suffix(review_path.suffix + ".tmp")
    tmp.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(review_path)

    review_sha = file_sha256(review_path)
    cov_sha = file_sha256(cov_path) if cov_path.is_file() else None

    payload = build_reliability_label_contract_v1(
        repo_root=repo_root,
        review_sha256=review_sha,
        coverage_audit_sha256=cov_sha,
        portfolio_ceiling_pct=float(portfolio_machine_wmape_ceiling_pct),
        sku_group_ceiling_pct=float(sku_group_machine_wmape_ceiling_pct),
        allow_mnih=bool(allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling),
    )
    digest = write_immutable_json(out_path, payload)

    # Mark review after successful freeze
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
        "planning_ceilings": payload["planning_ceilings"],
        "decision_table": payload["decision_table"],
        "locked_thresholds": payload["locked_thresholds"],
        "upstream_evidence": payload["upstream_evidence"],
    }
