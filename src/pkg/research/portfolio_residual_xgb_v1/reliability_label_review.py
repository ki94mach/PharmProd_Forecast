"""HB coverage audit (identifier columns only) + reliability label contract review draft.

Does not read HB forecast/outcome columns. Does not train a router.
Does not write the final reliability contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.portfolio_ts_lock.constants import HB_ORIGINS, HORIZON_GROUPS
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import FAMILY_DIR_REL

ID_COLS = ("product", "forecast_origin", "horizon", "target_date")
REQUIRED_HORIZONS = frozenset(range(1, 16))
BUDGET_SAFE_COLS = ("product", "budget_origin", "horizon", "target_date")

VALIDITY_GROUPS: tuple[dict[str, Any], ...] = (
    {"name": "h1_3", "horizons": (1, 2, 3), "display": "h1–3", "width": 3},
    {"name": "h4_6", "horizons": (4, 5, 6), "display": "h4–6", "width": 3},
    {"name": "h7_12", "horizons": tuple(range(7, 13)), "display": "h7–12", "width": 6},
    {"name": "h13_15", "horizons": (13, 14, 15), "display": "h13–15", "width": 3},
)

DEPLOYABLE_BRANCHES = (
    "both_native",
    "v21_only",
    "a0_only",
    "neither_naive",
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _complete_pairs(
    df: pd.DataFrame, origin_col: str = "forecast_origin"
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    pairs: set[tuple[str, int]] = set()
    incomplete: list[dict[str, Any]] = []
    for (product, origin), g in df.groupby(
        [df["product"].astype(str), df[origin_col].astype(int)], sort=False
    ):
        hs = {int(h) for h in g["horizon"].tolist()}
        if hs == set(REQUIRED_HORIZONS):
            pairs.add((str(product), int(origin)))
        else:
            incomplete.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "horizons_present": sorted(hs),
                    "missing": sorted(set(REQUIRED_HORIZONS) - hs),
                    "extra": sorted(hs - set(REQUIRED_HORIZONS)),
                    "n_rows": int(len(g)),
                }
            )
    return pairs, incomplete


def _group_complete_pairs(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    origin_col: str = "forecast_origin",
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """Exact completeness for one validity group (all horizons in group present)."""
    need = set(horizons)
    pairs: set[tuple[str, int]] = set()
    incomplete: list[dict[str, Any]] = []
    for (product, origin), g in df.groupby(
        [df["product"].astype(str), df[origin_col].astype(int)], sort=False
    ):
        hs = {int(h) for h in g["horizon"].tolist()}
        present = hs & need
        if present == need:
            pairs.add((str(product), int(origin)))
        elif present:
            # partial group — explicit incomplete, never silent
            incomplete.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "horizons_present_in_group": sorted(present),
                    "missing_in_group": sorted(need - present),
                }
            )
    return pairs, incomplete


def audit_id_panel(path: Path, *, engine: str) -> dict[str, Any]:
    df = pd.read_parquet(path, columns=list(ID_COLS))
    hb = sorted(HB_ORIGINS)
    df = df.loc[df["forecast_origin"].astype(int).isin(hb)].copy()
    pairs, incomplete = _complete_pairs(df)
    by_origin: dict[str, Any] = {}
    for o in hb:
        g = df.loc[df["forecast_origin"].astype(int) == o]
        n_prod = int(g["product"].astype(str).nunique())
        n_complete = sum(1 for _p, oo in pairs if oo == o)
        n_inc = sum(1 for r in incomplete if r["forecast_origin"] == o)
        by_origin[str(o)] = {
            "n_products_with_any_row": n_prod,
            "n_product_origin_complete_1_15": n_complete,
            "n_product_origin_incomplete": n_inc,
            "n_id_rows": int(len(g)),
        }
    return {
        "engine": engine,
        "relative_path": path.as_posix(),
        "sha256": file_sha256(path),
        "columns_read": list(ID_COLS),
        "forecast_or_outcome_columns_read": [],
        "by_origin": by_origin,
        "n_complete_product_origin": len(pairs),
        "n_incomplete_groups": len(incomplete),
        "incomplete_groups": incomplete,
        "complete_horizons_hard_fail_policy": (
            "Any product×origin missing exact horizons {1..15} is incomplete for "
            "portfolio comparison; never silently drop incomplete groups."
        ),
    }


def audit_budget_universe(path: Path) -> dict[str, Any]:
    cols_available = pd.read_parquet(path, columns=None).columns.tolist()
    safe = [c for c in BUDGET_SAFE_COLS if c in cols_available]
    forbidden_present = [
        c
        for c in cols_available
        if c
        in {
            "budget_forecast",
            "sales",
            "budget_residual",
            "actual",
            "forecast",
            "raw_forecast",
        }
    ]
    df = pd.read_parquet(path, columns=safe)
    origin_col = "budget_origin" if "budget_origin" in df.columns else "forecast_origin"
    hb = sorted(HB_ORIGINS)
    df = df.loc[df[origin_col].astype(int).isin(hb)].copy()
    work = df.rename(columns={origin_col: "forecast_origin"})
    pairs, incomplete = _complete_pairs(work, origin_col="forecast_origin")
    by_origin: dict[str, Any] = {}
    for o in hb:
        g = work.loc[work["forecast_origin"].astype(int) == o]
        by_origin[str(o)] = {
            "n_products_with_any_row": int(g["product"].astype(str).nunique()),
            "n_product_origin_complete_1_15": sum(1 for _p, oo in pairs if oo == o),
            "n_product_origin_incomplete": sum(
                1 for r in incomplete if r["forecast_origin"] == o
            ),
            "n_id_rows": int(len(g)),
        }
    return {
        "relative_path": path.as_posix(),
        "sha256": file_sha256(path),
        "columns_read": safe,
        "outcome_columns_present_but_not_read": forbidden_present,
        "by_origin": by_origin,
        "n_complete_product_origin": len(pairs),
        "n_incomplete_groups": len(incomplete),
        "incomplete_groups": incomplete,
        "note": "coverage/availability only; HB outcomes not inspected",
    }


def _load_hb_id_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=list(ID_COLS))
    return df.loc[df["forecast_origin"].astype(int).isin(sorted(HB_ORIGINS))].copy()


def _load_budget_hb_frame(path: Path) -> pd.DataFrame:
    cols_available = pd.read_parquet(path, columns=None).columns.tolist()
    safe = [c for c in BUDGET_SAFE_COLS if c in cols_available]
    df = pd.read_parquet(path, columns=safe)
    ocol = "budget_origin" if "budget_origin" in df.columns else "forecast_origin"
    df = df.loc[df[ocol].astype(int).isin(sorted(HB_ORIGINS))].copy()
    return df.rename(columns={ocol: "forecast_origin"})


def _branch_counts(
    human: set[tuple[str, int]],
    v21: set[tuple[str, int]],
    a0: set[tuple[str, int]],
) -> dict[str, Any]:
    both = v21 & a0
    v21_only = v21 - a0
    a0_only = a0 - v21
    neither = human - (v21 | a0)
    by_branch_matched = {
        "both_native": len(human & both),
        "v21_only": len(human & v21_only),
        "a0_only": len(human & a0_only),
        "neither_naive": len(neither),
    }
    common_native_matched = human & both

    # Per-SKU complete-origin counts on deployable matched (human complete 1–15)
    sku_origin_counts: dict[str, int] = {}
    for product, _o in human:
        sku_origin_counts[product] = sku_origin_counts.get(product, 0) + 1
    hist = {1: 0, 2: 0, 3: 0, 4: 0}
    for n in sku_origin_counts.values():
        if n in hist:
            hist[n] += 1
        elif n > 4:
            # should not happen with 4 HB origins; record under 4+
            hist[4] += 1
    n_skus_ge3 = sum(1 for n in sku_origin_counts.values() if n >= 3)

    return {
        "n_human_complete_1_15": len(human),
        "n_v21_complete_1_15": len(v21),
        "n_a0_complete_1_15": len(a0),
        "n_both_native_complete_1_15": len(both),
        "deployable_policy_matched_complete_1_15": {
            "definition": (
                "Human Budget complete {1..15} product×origins, classified by "
                "deployable frozen-policy branch (ID coverage only)."
            ),
            "n_total": len(human),
            "by_branch": by_branch_matched,
            "by_origin_total": {
                str(o): sum(1 for _p, oo in human if oo == o) for o in sorted(HB_ORIGINS)
            },
            "by_origin_by_branch": {
                str(o): {
                    "both_native": sum(1 for p, oo in (human & both) if oo == o),
                    "v21_only": sum(1 for p, oo in (human & v21_only) if oo == o),
                    "a0_only": sum(1 for p, oo in (human & a0_only) if oo == o),
                    "neither_naive": sum(1 for p, oo in neither if oo == o),
                }
                for o in sorted(HB_ORIGINS)
            },
            "note": (
                "neither_naive remains in portfolio activation comparison when human "
                "coverage is complete, but cannot receive MACHINE_RELIABLE at SKU×group."
            ),
        },
        "common_native_matched_complete_1_15": {
            "definition": (
                "Diagnostic subset: human complete ∩ both_native (V2.1 and A0 complete)."
            ),
            "n_total": len(common_native_matched),
            "by_origin": {
                str(o): sum(1 for _p, oo in common_native_matched if oo == o)
                for o in sorted(HB_ORIGINS)
            },
            "role": "diagnostics_only_not_primary_activation_gate",
        },
        "coverage_distribution_deployable_matched": {
            "definition": (
                "Among unique SKUs with at least one complete-1–15 matched human "
                "product×origin under deployable policy population."
            ),
            "n_unique_matched_skus": len(sku_origin_counts),
            "n_skus_with_1_complete_origin": hist[1],
            "n_skus_with_2_complete_origins": hist[2],
            "n_skus_with_3_complete_origins": hist[3],
            "n_skus_with_4_complete_origins": hist[4],
            "n_skus_with_at_least_3_complete_origins": n_skus_ge3,
        },
    }


def build_coverage_audit(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    v21 = (
        repo_root
        / "src/data/results/product_universe/portfolio_v21_backfill_v2_export/forecasts_panel.parquet"
    )
    a0 = (
        repo_root
        / "src/data/results/product_universe/portfolio_a0_backfill_v2_export/forecasts_panel.parquet"
    )
    budget = repo_root / "src/data/benchmarks/v1/budget_universe.parquet"

    v21_audit = audit_id_panel(v21, engine="v2.1")
    a0_audit = audit_id_panel(a0, engine="a0")
    budget_audit = audit_budget_universe(budget)

    v21_df = _load_hb_id_frame(v21)
    a0_df = _load_hb_id_frame(a0)
    bud_df = _load_budget_hb_frame(budget)

    cv, _ = _complete_pairs(v21_df)
    ca, _ = _complete_pairs(a0_df)
    cb, _ = _complete_pairs(bud_df)

    branch = _branch_counts(cb, cv, ca)

    # Per validity-group exact completeness (human vs engines) — explicit, not silent
    group_coverage: dict[str, Any] = {}
    for vg in VALIDITY_GROUPS:
        hs = tuple(vg["horizons"])
        hv, h_inc = _group_complete_pairs(bud_df, horizons=hs)
        vv, _ = _group_complete_pairs(v21_df, horizons=hs)
        av, _ = _group_complete_pairs(a0_df, horizons=hs)
        both_g = vv & av
        group_coverage[vg["name"]] = {
            "display": vg["display"],
            "horizons": list(hs),
            "width": int(vg["width"]),
            "expected_rows_if_n_origins": {
                "formula": "n_complete_origins_in_group * group_width",
                "example_n_origins_3": 3 * int(vg["width"]),
            },
            "n_human_group_complete": len(hv),
            "n_human_group_partial_incomplete": len(h_inc),
            "human_partial_incomplete_sample": h_inc[:10],
            "n_deployable_matched_group_complete": {
                "both_native": len(hv & both_g),
                "v21_only": len(hv & (vv - av)),
                "a0_only": len(hv & (av - vv)),
                "neither_naive": len(hv - (vv | av)),
                "total_human_group_complete": len(hv),
            },
            "n_common_native_matched_group_complete": len(hv & both_g),
            "rule": (
                "Exact group completeness required for group labels; an incomplete "
                "later group does not discard valid earlier groups; missing/"
                "incomplete groups → INSUFFICIENT_EVIDENCE for that group only."
            ),
        }

    return {
        "audit_version": "hb_reliability_coverage_audit_v1",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "protected_origins": sorted(HB_ORIGINS),
        "required_horizons": sorted(REQUIRED_HORIZONS),
        "columns_policy": "identifier_and_coverage_only",
        "deployable_policy_branches": list(DEPLOYABLE_BRANCHES),
        "v21_panel": v21_audit,
        "a0_panel": a0_audit,
        "budget_universe": budget_audit,
        "matched_counts": branch,
        "validity_group_coverage": group_coverage,
        "hard_fail_incomplete_1_15_for_portfolio_comparison": True,
        "silently_drop_incomplete_groups": False,
        "router_trained": False,
    }


def build_reliability_label_contract_review(
    *,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Two-scope proposed contract for approval — every numeric threshold explicit."""
    dep = coverage["matched_counts"]["deployable_policy_matched_complete_1_15"]
    com = coverage["matched_counts"]["common_native_matched_complete_1_15"]
    dist = coverage["matched_counts"]["coverage_distribution_deployable_matched"]

    portfolio_activation_gate = {
        "scope": "portfolio",
        "name": "portfolio_activation_gate",
        "primary_population": "deployable_frozen_policy_vs_human_budget",
        "diagnostic_population": "common_native_vs_human_budget",
        "machine_policy": "FROZEN_TS = A0_HYBRID_POLICY",
        "semantic_note": {
            "relative_gates_prove": "non_inferiority_to_human_not_absolute_reliability",
            "example": (
                "Machine WMAPE 60% vs human 59% satisfies a 2 pp relative margin but "
                "may still be operationally unacceptable."
            ),
            "label_MACHINE_RELIABLE_requires": [
                "portfolio_machine_wmape_ceiling_pct",
                "sku_group_machine_wmape_ceiling_pct",
            ],
            "if_sku_group_ceiling_unavailable": "use_MACHINE_NONINFERIOR_TO_HUMAN",
            "absolute_gate_owner": "Planning",
        },
        "deployable_branches": {
            "both_native": "0.5*V2.1 + 0.5*A0 seed mean",
            "v21_only": "V2.1",
            "a0_only": "complete A0 seed mean",
            "neither_naive": "production naive_last_value",
        },
        "eligibility": {
            "require_complete_horizons_1_to_15": True,
            "silently_drop_incomplete": False,
            "include_branches_in_primary": list(DEPLOYABLE_BRANCHES),
            "coverage_n_deployable_matched": dep["n_total"],
            "coverage_n_common_native_matched_diagnostic": com["n_total"],
        },
        "coverage_distribution_gates_before_activation": {
            "applies_before": "relative_and_absolute_accuracy_gates",
            "measured_on": "deployable_matched_complete_1_15",
            "observed": dist,
            "proposed_numeric_thresholds": {
                "min_unique_matched_skus": {
                    "value": None,
                    "unit": "count_of_skus",
                    "status": "PROPOSED_FOR_APPROVAL_OPTIONAL_OR_SET_BY_PLANNING",
                    "observed": dist["n_unique_matched_skus"],
                    "note": (
                        "Report unique matched SKUs. Set a hard floor only if Planning "
                        "requires one; otherwise informational with the ≥3-origin floor "
                        "as the binding distribution gate."
                    ),
                },
                "report_skus_by_complete_origin_count": {
                    "value": {
                        "1": dist["n_skus_with_1_complete_origin"],
                        "2": dist["n_skus_with_2_complete_origins"],
                        "3": dist["n_skus_with_3_complete_origins"],
                        "4": dist["n_skus_with_4_complete_origins"],
                    },
                    "unit": "counts_by_n_complete_origins",
                    "status": "REQUIRED_REPORT_EXPLICIT",
                    "definition": "Must be reported before portfolio activation.",
                },
                "min_skus_with_at_least_3_complete_origins": {
                    "value": 40,
                    "unit": "count_of_skus",
                    "status": "PROPOSED_FOR_APPROVAL",
                    "observed": dist["n_skus_with_at_least_3_complete_origins"],
                    "rationale": (
                        "Reasonable proposed floor so activation is not driven by a thin "
                        "multi-origin SKU subset. Explicit Planning approval required."
                    ),
                },
            },
        },
        "gates_apply_here_only": [
            "coverage_distribution",
            "portfolio_machine_wmape_ceiling_pct",
            "wmape_noninferiority",
            "rmse_ratio",
            "p95_ae_ratio",
            "p99_ae_ratio",
            "max_ae_ratio",
            "sku_cluster_bootstrap",
            "drop_top_k_machine_advantage",
            "wmape_denominator_must_be_valid",
        ],
        "if_portfolio_gate_fails": {
            "effect": "No SKU may receive MACHINE_RELIABLE for any validity group",
            "sku_labels_still_allowed": ["HUMAN_REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE"],
        },
        "proposed_numeric_thresholds": {
            "portfolio_machine_wmape_ceiling_pct": {
                "value": None,
                "unit": "portfolio_wmape_percent",
                "definition": (
                    "On the deployable matched set: WMAPE(machine) <= ceiling_pct. "
                    "Portfolio absolute-accuracy gate; separate from SKU×group ceiling. "
                    "Value must be supplied/approved by Planning — do not invent."
                ),
                "status": "PROPOSED_FOR_APPROVAL_VALUE_REQUIRED_FROM_PLANNING",
                "owner": "Planning",
                "required_to_finalize_MACHINE_RELIABLE_contract": True,
            },
            "machine_vs_human_noninferiority_wmape_margin_pp": {
                "value": 2.0,
                "unit": "percentage_points",
                "definition": "WMAPE(machine) - WMAPE(human) <= 2.0 on deployable matched set",
                "proves": "non_inferiority_only",
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "machine_vs_human_noninferiority_rmse_ratio_max": {
                "value": 1.02,
                "unit": "ratio",
                "definition": "RMSE(machine) <= 1.02 * RMSE(human)",
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "machine_vs_human_noninferiority_p95_ae_ratio_max": {
                "value": 1.02,
                "unit": "ratio",
                "definition": "p95_AE(machine) <= 1.02 * p95_AE(human)",
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "machine_vs_human_noninferiority_p99_ae_ratio_max": {
                "value": 1.02,
                "unit": "ratio",
                "definition": "p99_AE(machine) <= 1.02 * p99_AE(human)",
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "tail_risk_max_ae_ratio_max": {
                "value": 1.10,
                "unit": "ratio",
                "definition": "max_AE(machine) <= 1.10 * max_AE(human) on portfolio matched set",
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "sku_cluster_bootstrap_B": {
                "value": 2000,
                "unit": "replicates",
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "sku_cluster_bootstrap_seed": {
                "value": 42,
                "unit": "rng_seed",
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "sku_cluster_bootstrap_frac_machine_noninferior_ge": {
                "value": 0.90,
                "unit": "fraction",
                "definition": (
                    "P(WMAPE(machine)-WMAPE(human) <= 2.0 pp) >= 0.90 under SKU-cluster bootstrap"
                ),
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "drop_top_k_machine_advantage_skus": {
                "value": 3,
                "unit": "count_of_skus",
                "definition": (
                    "Drop the top-3 SKUs ranked by total AE advantage "
                    "ΣAE(human) − ΣAE(machine) (largest positive = strongest machine "
                    "advantage). Remaining set must still satisfy WMAPE non-inferiority. "
                    "NOT mean AE; NOT human-advantage SKUs. Matches pooled WMAPE contribution."
                ),
                "rank_metric": "sum_ae_human_minus_sum_ae_machine",
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "drop_top_k_remaining_noninferiority_required": {
                "value": True,
                "unit": "boolean",
                "status": "PROPOSED_FOR_APPROVAL",
                "scope": "portfolio_activation_gate_only",
            },
            "wmape_denominator_must_be_valid": {
                "value": True,
                "unit": "boolean",
                "definition": (
                    "Σ|actual| on the scored set must be finite and > 0. "
                    "If denominator invalid → gate non-computable "
                    "(portfolio activation fails / INSUFFICIENT_EVIDENCE as applicable)."
                ),
                "status": "PROPOSED_FOR_APPROVAL",
            },
        },
        "common_native_diagnostics": {
            "role": "report_separately_do_not_drive_activation",
            "n_matched_complete_1_15": com["n_total"],
            "note": "Common-native results are diagnostics only.",
        },
    }

    sku_validity_group_labels = {
        "scope": "sku_x_validity_group",
        "prerequisite": (
            "portfolio_activation_gate must pass before any MACHINE_RELIABLE label; "
            "if it fails, SKU×group labels may only be HUMAN_REVIEW_REQUIRED or "
            "INSUFFICIENT_EVIDENCE."
        ),
        "validity_groups": [
            {
                "name": g["name"],
                "display": g["display"],
                "horizons": list(g["horizons"]),
                "width": g["width"],
                "expected_rows_formula": "n_complete_origins_in_group * group_width",
            }
            for g in VALIDITY_GROUPS
        ],
        "evaluate_groups_separately": True,
        "do_not_apply_at_sku_group_scope": [
            "sku_cluster_bootstrap",
            "p95_ae_ratio",
            "p99_ae_ratio",
            "max_ae_ratio",
            "drop_top_k",
            "portfolio_machine_wmape_ceiling_pct",
        ],
        "deployable_branch_rules": {
            "both_native": "eligible_for_MACHINE_RELIABLE_if_gates_pass",
            "v21_only": "eligible_for_MACHINE_RELIABLE_if_gates_pass",
            "a0_only": "eligible_for_MACHINE_RELIABLE_if_gates_pass",
            "neither_naive": "cannot_receive_MACHINE_RELIABLE",
        },
        "proposed_numeric_thresholds": {
            "sku_group_machine_wmape_ceiling_pct": {
                "value": None,
                "unit": "group_wmape_percent",
                "definition": (
                    "Within the SKU×validity-group matched set: "
                    "machine_group_wmape <= approved_sku_group_wmape_ceiling. "
                    "Separate from portfolio_machine_wmape_ceiling_pct. "
                    "Required to assign MACHINE_RELIABLE. If Planning cannot supply "
                    "any absolute SKU ceiling, scientifically accurate label name is "
                    "MACHINE_NONINFERIOR_TO_HUMAN (not MACHINE_RELIABLE)."
                ),
                "status": "PROPOSED_FOR_APPROVAL_VALUE_REQUIRED_FROM_PLANNING",
                "owner": "Planning",
                "required_to_use_label_MACHINE_RELIABLE": True,
                "segment_specific_ceilings": {
                    "allowed": True,
                    "when": (
                        "If demand segments need different tolerances, Planning may "
                        "approve a map segment_id -> sku_group_machine_wmape_ceiling_pct "
                        "instead of a single global SKU×group ceiling."
                    ),
                    "must_define_before_opening_outcomes": True,
                    "segment_assignment": {
                        "required_if_map_used": True,
                        "definition_required": (
                            "Planning must define how each SKU maps to a segment_id "
                            "before any HB outcomes are opened."
                        ),
                        "assignment_artifact": None,
                        "stability": (
                            "Segment assignment must be frozen and deterministic; "
                            "no post-hoc reassignment that uses outcomes."
                        ),
                    },
                    "ceiling_precedence": [
                        (
                            "If a segment-specific map is approved and the SKU has an "
                            "assigned segment with a ceiling in the map → use that "
                            "segment ceiling"
                        ),
                        (
                            "Else if global sku_group_machine_wmape_ceiling_pct is "
                            "approved → use the global ceiling"
                        ),
                        (
                            "Else → no absolute SKU×group ceiling is available for "
                            "that SKU×group"
                        ),
                    ],
                    "missing_segment_behavior": {
                        "when_sku_unassigned_or_segment_absent_from_map": (
                            "Do not invent a ceiling. MACHINE_RELIABLE is blocked. "
                            "Label INSUFFICIENT_EVIDENCE if treated as non-computable "
                            "absolute gate; otherwise HUMAN_REVIEW_REQUIRED when "
                            "n_complete_origins >= 2."
                        ),
                        "MACHINE_NONINFERIOR_TO_HUMAN": (
                            "Only if allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling"
                            "=true AND relative rules pass; else do not use MNIH."
                        ),
                    },
                    "proposed_map": None,
                    "status": "REQUIRES_PLANNING_DEFINITION_BEFORE_OUTCOMES",
                    "note": (
                        "If Planning chooses segment-specific ceilings, define the "
                        "segment assignment, ceiling precedence, and missing-segment "
                        "behavior before opening outcomes."
                    ),
                },
            },
            "min_complete_origins_per_group": {
                "value": 3,
                "unit": "count_of_origins",
                "status": "PROPOSED_FOR_APPROVAL",
                "replaces": "min_matched_complete_rows_per_validity_group=30 (removed)",
                "expected_rows": {
                    "h1_3": "n_origins * 3",
                    "h4_6": "n_origins * 3",
                    "h7_12": "n_origins * 6",
                    "h13_15": "n_origins * 3",
                },
            },
            "machine_vs_human_noninferiority_wmape_margin_pp": {
                "value": 2.0,
                "unit": "percentage_points",
                "definition": (
                    "Within the validity group: WMAPE(machine) - WMAPE(human) <= 2.0"
                ),
                "proves": "non_inferiority_only",
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "origin_stability_min_frac_origins_machine_noninferior": {
                "value": 0.75,
                "unit": "fraction",
                "definition": (
                    "required_noninferior_origins = ceil(0.75 × complete_origins). "
                    "SKU passes origin stability iff count of origins with "
                    "group WMAPE(machine)-WMAPE(human) <= 2.0 pp "
                    ">= required_noninferior_origins."
                ),
                "worked_examples": {
                    "complete_origins_3": {
                        "required_noninferior_origins": 3,
                        "calc": "ceil(0.75*3)=ceil(2.25)=3",
                    },
                    "complete_origins_4": {
                        "required_noninferior_origins": 3,
                        "calc": "ceil(0.75*4)=ceil(3.0)=3",
                    },
                    "complete_origins_2": {
                        "note": "Label is HUMAN_REVIEW_REQUIRED by origin-count rule; "
                        "stability formula not used for MACHINE_RELIABLE",
                    },
                },
                "status": "PROPOSED_FOR_APPROVAL",
            },
            "wmape_denominator_must_be_valid": {
                "value": True,
                "unit": "boolean",
                "definition": (
                    "For any group WMAPE used in labeling: valid_wmape_denominator := "
                    "sum(abs(actual)) > 1e-9 (finite). Invalid if Σ|actual| <= 1e-9 or "
                    "non-finite → INSUFFICIENT_EVIDENCE for that SKU×group."
                ),
                "status": "PROPOSED_FOR_APPROVAL",
            },
        },
        "per_group_label_rule": {
            "ordered": [
                {
                    "when": "n_complete_origins_in_group < 2",
                    "label": "INSUFFICIENT_EVIDENCE",
                },
                {
                    "when": "wmape_denominator_invalid (Σ|actual| <= 1e-9 or non-finite)",
                    "label": "INSUFFICIENT_EVIDENCE",
                },
                {
                    "when": "n_complete_origins_in_group == 2",
                    "label": "HUMAN_REVIEW_REQUIRED",
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
                    "label": "MACHINE_RELIABLE",
                },
                {
                    "when": (
                        "allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling=true "
                        "AND n_complete_origins_in_group >= 3 "
                        "AND wmape_denominator_valid "
                        "AND WMAPE(machine)-WMAPE(human) <= 2.0 pp "
                        "AND n_origins_noninferior >= ceil(0.75 * n_complete_origins) "
                        "AND portfolio_activation_gate_passed "
                        "AND deployable_branch != neither_naive "
                        "AND sku_group absolute ceiling unavailable"
                    ),
                    "label": "MACHINE_NONINFERIOR_TO_HUMAN",
                    "switch_default": False,
                    "note": (
                        "Canonical ordered branch even though the switch defaults to false."
                    ),
                },
                {
                    "when": "otherwise (including neither_naive, failed absolute/relative gates)",
                    "label": "HUMAN_REVIEW_REQUIRED",
                },
            ],
            "label_naming_contingency": {
                "allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling": False,
                "if_sku_group_absolute_ceiling_cannot_be_supplied": (
                    "Do not use MACHINE_RELIABLE unless an absolute ceiling is approved. "
                    "MACHINE_NONINFERIOR_TO_HUMAN applies only when "
                    "allow_MACHINE_NONINFERIOR_TO_HUMAN_without_sku_ceiling=true and "
                    "relative non-inferiority + origin-count/stability rules pass."
                ),
            },
        },        "sku_validity_range": {
            "definition": (
                "Longest contiguous MACHINE_RELIABLE prefix of validity groups "
                "starting at h1_3 (then h4_6, h7_12, h13_15). Human review required "
                "beyond that range."
            ),
            "group_order": ["h1_3", "h4_6", "h7_12", "h13_15"],
            "examples": [
                {
                    "labels": ["MACHINE_RELIABLE", "MACHINE_RELIABLE", "HUMAN_REVIEW_REQUIRED", "MACHINE_RELIABLE"],
                    "validity_range": "h1–6 (h1_3 + h4_6)",
                    "human_review_from": "h7_12 onward",
                },
                {
                    "labels": ["MACHINE_RELIABLE", "INSUFFICIENT_EVIDENCE", "..."],
                    "validity_range": "h1–3",
                    "human_review_from": "h4 onward",
                    "note": "Prefix stops at first non-MACHINE_RELIABLE; range is h1_3, not empty.",
                },
            ],
        },
        "incomplete_coverage_rules": {
            "portfolio_comparison": "complete horizons 1–15 required; never silent drop",
            "group_labels": "exact group completeness required for that group",
            "later_incomplete_does_not_discard_earlier": True,
            "missing_or_incomplete_group_label": "INSUFFICIENT_EVIDENCE",
            "invalid_wmape_denominator_label": "INSUFFICIENT_EVIDENCE",
            "observed_budget_incomplete_1_15": coverage["budget_universe"][
                "incomplete_groups"
            ],
        },
    }

    labels = {
        "MACHINE_RELIABLE": {
            "meaning": (
                "Deployable frozen policy may be used without mandatory human review "
                "for this SKU within the contiguous validity-range prefix. Requires "
                "Planning-approved absolute ceilings at BOTH portfolio and SKU×group "
                "scopes, plus relative non-inferiority — relative gates alone are not enough."
            ),
            "blocked_if_portfolio_activation_gate_fails": True,
            "blocked_if_deployable_branch_is": ["neither_naive"],
            "requires_absolute_ceilings": [
                "portfolio_machine_wmape_ceiling_pct",
                "sku_group_machine_wmape_ceiling_pct",
            ],
            "if_sku_group_ceiling_unavailable": (
                "Do not assign MACHINE_RELIABLE; use MACHINE_NONINFERIOR_TO_HUMAN instead "
                "when only relative non-inferiority is proven."
            ),
        },
        "MACHINE_NONINFERIOR_TO_HUMAN": {
            "meaning": (
                "Contingency label only if Planning cannot supply an absolute SKU×group "
                "WMAPE ceiling. Means relative non-inferiority to Human Budget within "
                "the locked margin/stability rules — not absolute operational reliability."
            ),
            "use_when": "sku_group_machine_wmape_ceiling_pct cannot be approved",
            "not_default": True,
        },
        "HUMAN_REVIEW_REQUIRED": {
            "meaning": (
                "Enough complete origins for a reviewable claim, but not MACHINE_RELIABLE; "
                "or portfolio gate failed; or beyond the MACHINE_RELIABLE validity prefix."
            ),
        },
        "INSUFFICIENT_EVIDENCE": {
            "meaning": (
                "Fewer than 2 complete origins in the group, incomplete/missing group "
                "coverage, invalid WMAPE denominator (Σ|actual| <= 1e-9 or non-finite), "
                "or non-computable metrics."
            ),
        },
    }

    planning_ceilings_pending = {
        "portfolio_machine_wmape_ceiling_pct": None,
        "sku_group_machine_wmape_ceiling_pct": None,
        "segment_specific_sku_group_ceilings": None,
        "owner": "Planning",
        "note": (
            "Both portfolio and SKU×group absolute ceilings must be approved to retain "
            "MACHINE_RELIABLE. Portfolio ceiling stays separate from SKU×group ceiling. "
            "If Planning chooses segment-specific ceilings, define segment assignment, "
            "ceiling precedence, and missing-segment behavior before opening outcomes."
        ),
    }

    all_thresholds: dict[str, Any] = {}
    for k, v in portfolio_activation_gate["coverage_distribution_gates_before_activation"][
        "proposed_numeric_thresholds"
    ].items():
        all_thresholds[f"portfolio.coverage_distribution.{k}"] = v
    for k, v in portfolio_activation_gate["proposed_numeric_thresholds"].items():
        all_thresholds[f"portfolio.{k}"] = v
    for k, v in sku_validity_group_labels["proposed_numeric_thresholds"].items():
        all_thresholds[f"sku_group.{k}"] = v

    return {
        "contract_status": "REVIEW_ONLY_PENDING_APPROVAL",
        "contract_version_proposed": "reliability_label_contract_v1",
        "artifact_name": "reliability_label_contract_v1_review.json",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "router_trained": False,
        "stop_before_final_contract": True,
        "pipeline_context": {
            "production_pipeline_locked": "FROZEN_TS",
            "FROZEN_TS_equals": "A0_HYBRID_POLICY",
            "residual_xgb_promoted": False,
            "median_benchmark_only": True,
        },
        "protected_hb_origins": sorted(HB_ORIGINS),
        "planning_ceilings_pending": planning_ceilings_pending,
        "two_scopes": {
            "portfolio_activation_gate": portfolio_activation_gate,
            "sku_validity_group_labels": sku_validity_group_labels,
        },
        "labels": labels,
        "proposed_numeric_thresholds_flat_for_approval": all_thresholds,
        "coverage_audit_summary": {
            "hb_outcomes_inspected": False,
            "n_deployable_matched_complete_1_15": dep["n_total"],
            "n_deployable_by_branch": dep["by_branch"],
            "n_common_native_matched_complete_1_15_diagnostic": com["n_total"],
            "coverage_distribution": dist,
            "budget_incomplete_1_15_groups_count": coverage["budget_universe"][
                "n_incomplete_groups"
            ],
            "v21_incomplete_1_15_groups_count": coverage["v21_panel"][
                "n_incomplete_groups"
            ],
            "a0_incomplete_1_15_groups_count": coverage["a0_panel"][
                "n_incomplete_groups"
            ],
        },
        "approval_checklist": [
            "Planning: approve portfolio_machine_wmape_ceiling_pct (null until supplied)",
            "Planning: approve sku_group_machine_wmape_ceiling_pct (null until supplied); or segment-specific map",
            "If SKU×group absolute ceiling cannot be supplied: switch label to MACHINE_NONINFERIOR_TO_HUMAN",
            "Approve coverage-distribution floor min_skus_with_at_least_3_complete_origins=40 (or revise)",
            "Approve portfolio relative gates (WMAPE 2pp, RMSE/p95/p99 1.02, maxAE 1.10, bootstrap 2000/42/0.90)",
            "Confirm drop-top-3 ranks by ΣAE(human)−ΣAE(machine) machine-advantage (not mean AE)",
            "Approve SKU×group: min 3 origins, 2pp WMAPE, ceil(0.75×n) stability, valid WMAPE denominator",
            "Confirm validity example: MR,IE → range h1–3; human review from h4",
            "Confirm neither_naive cannot receive MACHINE_RELIABLE; common-native diagnostics-only",
            "Confirm no HB outcomes opened; no router trained; no final contract yet",
        ],
        "explicit_non_goals": [
            "Do not train a reliability router in this step",
            "Do not invent portfolio or SKU×group absolute WMAPE ceilings — Planning must supply them",
            "Do not open HB forecast/outcome columns beyond identifier/coverage",
            "Do not write the final reliability_label_contract_v1.json yet",
            "Do not apply SKU-cluster bootstrap / p95 / p99 / max-AE / drop-top-3 at SKU×group scope",
            "Do not call a label MACHINE_RELIABLE if only relative non-inferiority is proven",
        ],
    }


def write_review_artifacts(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    family = repo_root / FAMILY_DIR_REL
    family.mkdir(parents=True, exist_ok=True)

    coverage = build_coverage_audit(repo_root)
    review = build_reliability_label_contract_review(coverage=coverage)

    cov_path = family / "hb_reliability_coverage_audit_v1.json"
    rev_path = family / "reliability_label_contract_v1_review.json"

    cov_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rev_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dep = coverage["matched_counts"]["deployable_policy_matched_complete_1_15"]
    com = coverage["matched_counts"]["common_native_matched_complete_1_15"]
    dist = coverage["matched_counts"]["coverage_distribution_deployable_matched"]
    return {
        "ok": True,
        "hb_outcomes_inspected": False,
        "router_trained": False,
        "final_contract_written": False,
        "coverage_audit": cov_path.as_posix(),
        "review_contract": rev_path.as_posix(),
        "n_deployable_matched_complete_1_15": dep["n_total"],
        "n_deployable_by_branch": dep["by_branch"],
        "n_common_native_matched_diagnostic": com["n_total"],
        "coverage_distribution": dist,
        "budget_incomplete_groups": coverage["budget_universe"]["n_incomplete_groups"],
        "planning_ceilings_pending": {
            "portfolio_machine_wmape_ceiling_pct": None,
            "sku_group_machine_wmape_ceiling_pct": None,
        },
    }
