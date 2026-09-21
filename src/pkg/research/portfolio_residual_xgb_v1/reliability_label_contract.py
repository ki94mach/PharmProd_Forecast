"""Immutable reliability_label_contract_v1 — labels only; no HB outcomes.

Absolute WMAPE ceilings remain Planning-owned (null until supplied).
Relative/structural thresholds locked from the approved review.
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
from pkg.research.portfolio_residual_xgb_v1.constants import FAMILY
from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import FAMILY_DIR_REL

CONTRACT_VERSION = "reliability_label_contract_v1"
CONTRACT_FILENAME = "reliability_label_contract_v1.json"
REVIEW_FILENAME = "reliability_label_contract_v1_review.json"

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


def required_noninferior_origins(n_complete: int) -> int:
    """ceil(0.75 × complete_origins)."""
    if n_complete <= 0:
        return 0
    return int(math.ceil(0.75 * float(n_complete)))


def wmape_denominator_valid(sum_abs_actual: float) -> bool:
    return sum_abs_actual == sum_abs_actual and float(sum_abs_actual) > 0.0


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
    allow_machine_noninferior_without_sku_ceiling: bool = False,
) -> str:
    """Apply locked SKU×validity-group decision table (no HB I/O)."""
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

    # n_complete >= 3
    rel_ok = (machine_group_wmape - human_group_wmape) <= 2.0
    stab_ok = n_origins_noninferior >= required_noninferior_origins(n_complete_origins)
    branch_ok = deployable_branch != "neither_naive"
    port_ok = bool(portfolio_activation_passed)

    if not (rel_ok and stab_ok and branch_ok and port_ok):
        return LABEL_HR

    # Absolute SKU ceiling required for MACHINE_RELIABLE
    if sku_group_ceiling_pct is None:
        if allow_machine_noninferior_without_sku_ceiling:
            return LABEL_MNIH
        return LABEL_HR

    abs_ok = machine_group_wmape <= float(sku_group_ceiling_pct)
    # Portfolio ceiling must also be approved for MR semantics (gate already passed
    # implies portfolio absolute check when ceiling set; if null, MR blocked).
    if portfolio_ceiling_pct is None:
        if allow_machine_noninferior_without_sku_ceiling:
            return LABEL_MNIH
        return LABEL_HR

    if abs_ok:
        return LABEL_MR
    return LABEL_HR


def locked_thresholds() -> dict[str, Any]:
    """Explicitly approved numeric thresholds (Planning ceilings remain null)."""
    return {
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
            "wmape_denominator_must_be_valid": True,
            "portfolio_machine_wmape_ceiling_pct": None,
            "portfolio_machine_wmape_ceiling_status": "REQUIRES_PLANNING_VALUE",
        },
        "sku_group": {
            "min_complete_origins_per_group": 3,
            "machine_vs_human_noninferiority_wmape_margin_pp": 2.0,
            "origin_stability_frac": 0.75,
            "origin_stability_rule": "required_noninferior_origins=ceil(0.75*complete_origins)",
            "wmape_denominator_must_be_valid": True,
            "sku_group_machine_wmape_ceiling_pct": None,
            "sku_group_machine_wmape_ceiling_status": "REQUIRES_PLANNING_VALUE",
            "segment_specific_sku_group_ceilings": None,
        },
    }


def build_reliability_label_contract_v1(
    *,
    review_sha256: str | None = None,
    coverage_audit_sha256: str | None = None,
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

    decision_table = {
        "portfolio_activation_gate": {
            "order": [
                "complete_1_15_required_never_silent_drop",
                "coverage_distribution_min_skus_ge3_origins",
                "valid_wmape_denominator",
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
            "drop_top_k": {
                "k": 3,
                "rank": "ΣAE(human)−ΣAE(machine)",
                "not": "mean_AE",
            },
        },
        "sku_x_validity_group": {
            "group_order": list(GROUP_ORDER),
            "branches": [
                {"when": "group_horizons_not_exact", "label": LABEL_IE},
                {"when": "n_complete_origins < 2", "label": LABEL_IE},
                {"when": "wmape_denominator_invalid", "label": LABEL_IE},
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
                {
                    "when": (
                        "same as MR relative rules but sku_ceiling unavailable "
                        "AND contingency enabled"
                    ),
                    "label": LABEL_MNIH,
                },
                {"when": "otherwise", "label": LABEL_HR},
            ],
            "origin_stability": {
                "formula": "ceil(0.75 * n_complete_origins)",
                "n3": 3,
                "n4": 3,
            },
            "validity_range": (
                "longest contiguous MACHINE_RELIABLE prefix from h1_3; "
                "MR,IE → range h1–3; human review from h4"
            ),
            "neither_naive": "cannot_receive_MACHINE_RELIABLE",
            "incomplete_later_group_does_not_discard_earlier": True,
        },
    }

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
                "use_when": "sku_group absolute ceiling cannot be supplied",
                "proves": "relative_noninferiority_only",
            },
            LABEL_HR: {},
            LABEL_IE: {
                "includes": [
                    "incomplete_group_horizons",
                    "n_origins_lt_2",
                    "invalid_wmape_denominator",
                ]
            },
        },
        "locked_thresholds": thr,
        "planning_ceilings": {
            "portfolio_machine_wmape_ceiling_pct": None,
            "sku_group_machine_wmape_ceiling_pct": None,
            "segment_specific_sku_group_ceilings": None,
            "owner": "Planning",
            "MACHINE_RELIABLE_blocked_until_both_set": True,
        },
        "decision_table": decision_table,
        "source_review": {
            "filename": REVIEW_FILENAME,
            "sha256": review_sha256,
        },
        "coverage_audit_sha256": coverage_audit_sha256,
        "approved_changes_applied": [
            "portfolio_vs_sku_two_scope_split",
            "drop_top3_by_sum_ae_machine_advantage",
            "coverage_distribution_min_40_skus_ge3_origins",
            "origin_stability_ceil_0_75",
            "sku_group_absolute_wmape_ceiling_requirement",
            "valid_wmape_denominator_or_INSUFFICIENT_EVIDENCE",
            "validity_range_MR_IE_is_h1_3_not_empty",
            "MACHINE_NONINFERIOR_TO_HUMAN_contingency_if_no_sku_ceiling",
            "planning_ceilings_remain_null_until_Planning_supplies_values",
        ],
    }


def _promote_threshold_entry(key: str, entry: dict[str, Any]) -> None:
    """Promote proposed structural/relative thresholds; leave Planning ceilings null."""
    st = str(entry.get("status", ""))
    is_planning_ceiling = (
        "ceiling" in key.lower()
        and (
            entry.get("value") is None
            or "PLANNING" in st.upper()
            or "REQUIRED_FROM_PLANNING" in st
        )
    )
    if is_planning_ceiling or "REQUIRES_PLANNING" in st:
        entry["status"] = "REQUIRES_PLANNING_VALUE"
        return
    if st.startswith("PROPOSED") or st == "REQUIRED_REPORT_EXPLICIT":
        if "OPTIONAL" in st and "min_unique" in key:
            entry["status"] = "INFORMATIONAL_NO_HARD_FLOOR"
        else:
            entry["status"] = "APPROVED"


def _apply_approved_threshold_statuses(node: Any, path: str = "") -> None:
    """Walk review JSON and promote only explicitly approved numeric thresholds."""
    if isinstance(node, dict):
        # Flat approval ledger
        if path.endswith("proposed_numeric_thresholds_flat_for_approval") or path.endswith(
            "proposed_numeric_thresholds"
        ):
            for key, entry in node.items():
                if isinstance(entry, dict) and "status" in entry:
                    _promote_threshold_entry(key, entry)
            return
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(child, dict) and "status" in child and (
                "proposed" in key.lower()
                or key.endswith("_pct")
                or "threshold" in key.lower()
                or "ceiling" in key.lower()
                or key
                in {
                    "min_skus_with_at_least_3_complete_origins",
                    "min_unique_matched_skus",
                    "report_skus_by_complete_origin_count",
                    "drop_top_k_machine_advantage_skus",
                    "drop_top_k_remaining_noninferiority_required",
                    "machine_vs_human_noninferiority_wmape_margin_pp",
                    "machine_vs_human_noninferiority_rmse_ratio_max",
                    "machine_vs_human_noninferiority_p95_ae_ratio_max",
                    "machine_vs_human_noninferiority_p99_ae_ratio_max",
                    "tail_risk_max_ae_ratio_max",
                    "sku_cluster_bootstrap_B",
                    "sku_cluster_bootstrap_seed",
                    "sku_cluster_bootstrap_frac_machine_noninferior_ge",
                    "wmape_denominator_must_be_valid",
                    "min_complete_origins_per_group",
                    "origin_stability_min_frac_origins_machine_noninferior",
                    "portfolio_machine_wmape_ceiling_pct",
                    "sku_group_machine_wmape_ceiling_pct",
                }
            ):
                _promote_threshold_entry(key, child)
            else:
                _apply_approved_threshold_statuses(child, child_path)
    elif isinstance(node, list):
        for item in node:
            _apply_approved_threshold_statuses(item, path)


def freeze_reliability_label_contract_v1(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    out_path = family / CONTRACT_FILENAME
    review_path = family / REVIEW_FILENAME
    cov_path = family / "hb_reliability_coverage_audit_v1.json"

    if out_path.exists() or out_path.with_suffix(out_path.suffix + ".sha256").exists():
        raise LockExistsError(f"Immutable contract already exists: {out_path}")

    review_sha = file_sha256(review_path) if review_path.is_file() else None
    cov_sha = file_sha256(cov_path) if cov_path.is_file() else None

    # Mark approved threshold statuses on the review artifact (mutable review only).
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["contract_status"] = "THRESHOLDS_LOCKED_PENDING_PLANNING_CEILINGS"
        review["final_contract_written"] = True
        review["final_contract_filename"] = CONTRACT_FILENAME
        review["stop_before_final_contract"] = False
        review["hb_outcomes_inspected"] = False
        _apply_approved_threshold_statuses(review)
        review["planning_ceilings_pending"] = {
            "portfolio_machine_wmape_ceiling_pct": None,
            "sku_group_machine_wmape_ceiling_pct": None,
            "segment_specific_sku_group_ceilings": None,
            "owner": "Planning",
            "note": (
                "Both portfolio and SKU×group absolute ceilings must be approved "
                "to retain MACHINE_RELIABLE. Portfolio ceiling stays separate from "
                "SKU×group ceiling."
            ),
        }
        tmp = review_path.with_suffix(review_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(review_path)
        review_sha = file_sha256(review_path)

    payload = build_reliability_label_contract_v1(
        review_sha256=review_sha,
        coverage_audit_sha256=cov_sha,
    )
    digest = write_immutable_json(out_path, payload)
    return {
        "ok": True,
        "contract_path": out_path.as_posix(),
        "contract_sha256": digest,
        "hb_outcomes_inspected": False,
        "planning_ceilings": payload["planning_ceilings"],
        "decision_table": payload["decision_table"],
        "locked_thresholds": payload["locked_thresholds"],
    }
