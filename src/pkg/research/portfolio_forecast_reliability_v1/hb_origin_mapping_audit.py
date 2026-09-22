"""Identifier-only HB Version→origin mapping audit (keep 140401; group coverage).

Never opens HB forecast qty / sales / machine error columns for scoring.
Does not mutate portfolio_ts_lock.HB_ORIGINS.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.budget_origin_v2 import completeness_by_origin
from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import HB_ORIGINS as LOCK_HB_ORIGINS
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    A0_PANEL_REL,
    BUDGET_UNIVERSE_V2_REL,
    DEPLOYABLE_PORTFOLIO_COHORT,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    HUMAN_COMPARABLE_COHORT,
    ID_COLS_HB,
    ID_COLS_MACHINE,
    LEGACY_TS_LOCK_HB_ORIGINS_KEY,
    MAPPING_AUDIT_FILENAME,
    NO_MACHINE_FORECAST,
    NO_MACHINE_FORECAST_COHORT,
    POSTLOCK_REL,
    PROTECTED_HB_ORIGINS,
    REQUIRED_HORIZONS,
    SALES_FULL_PORTFOLIO_REL,
    STALE_COVERAGE_AUDIT_FILENAME,
    STALE_COVERAGE_FIGURES_FORBIDDEN,
    V21_PANEL_REL,
    VALIDITY_GROUPS,
)
from pkg.research.portfolio_forecast_reliability_v1.deployable_policy import (
    reconcile_hb_complete_deployable,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def complete_pairs_full(
    df: pd.DataFrame, origin_col: str
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """Exact {1..15} completeness; incompletes always recorded (never silent drop)."""
    pairs: set[tuple[str, int]] = set()
    incomplete: list[dict[str, Any]] = []
    need = set(REQUIRED_HORIZONS)
    for (product, origin), g in df.groupby(
        [df["product"].astype(str), df[origin_col].astype(int)], sort=False
    ):
        hs = {int(h) for h in pd.to_numeric(g["horizon"], errors="coerce").dropna()}
        if hs == need:
            pairs.add((str(product), int(origin)))
        else:
            incomplete.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "horizons_present": sorted(hs),
                    "missing": sorted(need - hs),
                    "extra": sorted(hs - need),
                    "n_rows": int(len(g)),
                }
            )
    return pairs, incomplete


# Back-compat alias used by tests / representativeness
_complete_pairs = complete_pairs_full


def group_complete_pairs(
    df: pd.DataFrame,
    *,
    horizons: frozenset[int] | set[int],
    origin_col: str,
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """Exact completeness for one validity group; does not require full h1–15.

    Never renumbers horizons. Partial group membership is recorded as incomplete
    for that group only — earlier complete groups are unaffected.
    """
    need = set(horizons)
    pairs: set[tuple[str, int]] = set()
    incomplete: list[dict[str, Any]] = []
    for (product, origin), g in df.groupby(
        [df["product"].astype(str), df[origin_col].astype(int)], sort=False
    ):
        hs = {int(h) for h in pd.to_numeric(g["horizon"], errors="coerce").dropna()}
        present = hs & need
        if present == need:
            pairs.add((str(product), int(origin)))
        elif present:
            incomplete.append(
                {
                    "product": str(product),
                    "forecast_origin": int(origin),
                    "horizons_present_in_group": sorted(present),
                    "missing_in_group": sorted(need - present),
                    "n_rows": int(len(g)),
                }
            )
        # else: no group horizons at all → not an incomplete partial; absent
    return pairs, incomplete


def _load_hb_id(path: Path) -> pd.DataFrame:
    available = pd.read_parquet(path, columns=None).columns.tolist()
    forbidden_present = [c for c in available if c in FORBIDDEN_OUTCOME_COLS]
    safe = [c for c in ID_COLS_HB if c in available]
    for req in ("product", "budget_origin", "horizon", "target_date"):
        if req not in safe:
            raise ValueError(f"HB panel missing required id column {req}: {path}")
    df = pd.read_parquet(path, columns=safe)
    assert not any(c in df.columns for c in FORBIDDEN_OUTCOME_COLS)
    df.attrs["forbidden_present_but_not_read"] = forbidden_present
    df.attrs["columns_read"] = safe
    return df


def _load_machine_id(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(ID_COLS_MACHINE))


def _by_origin_stats(
    pairs: set[tuple[str, int]],
    incomplete: list[dict[str, Any]],
    df: pd.DataFrame,
    origin_col: str,
    origins: list[int],
    *,
    complete_key: str = "n_product_origin_complete_1_15",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for o in origins:
        g = df.loc[pd.to_numeric(df[origin_col], errors="coerce") == o]
        out[str(o)] = {
            "n_products_with_any_row": int(g["product"].astype(str).nunique())
            if len(g)
            else 0,
            complete_key: sum(1 for _p, oo in pairs if oo == o),
            "n_product_origin_incomplete": sum(
                1 for r in incomplete if r["forecast_origin"] == o
            ),
            "n_id_rows": int(len(g)),
        }
    return out


def _sku_origin_histogram(pairs: set[tuple[str, int]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for product, _o in pairs:
        counts[product] += 1
    hist = {str(k): sum(1 for n in counts.values() if n == k) for k in (1, 2, 3, 4)}
    hist["gt4"] = sum(1 for n in counts.values() if n > 4)
    return {
        "n_skus": len(counts),
        "n_product_origin_pairs": len(pairs),
        "sku_complete_origin_count_histogram": hist,
        "n_skus_with_at_least_3_complete_origins": sum(
            1 for n in counts.values() if n >= 3
        ),
        "n_skus_with_0_or_1_complete_origins_among_covered": sum(
            1 for n in counts.values() if n <= 1
        ),
        "n_skus_with_exactly_2_complete_origins": sum(
            1 for n in counts.values() if n == 2
        ),
        "n_skus_mnih_eligible_ge3": sum(1 for n in counts.values() if n >= 3),
    }


def _branch_breakdown(
    human: set[tuple[str, int]],
    a0: set[tuple[str, int]],
    v21: set[tuple[str, int]],
    origins: list[int],
) -> dict[str, Any]:
    both = a0 & v21
    v21_only = v21 - a0
    a0_only = a0 - v21
    neither = human - (a0 | v21)
    by_branch = {
        "both_native": len(human & both),
        "v21_only": len(human & v21_only),
        "a0_only": len(human & a0_only),
        "neither_naive": len(neither),
    }
    by_origin_by_branch: dict[str, Any] = {}
    for o in origins:
        by_origin_by_branch[str(o)] = {
            "both_native": sum(1 for p, oo in (human & both) if oo == o),
            "v21_only": sum(1 for p, oo in (human & v21_only) if oo == o),
            "a0_only": sum(1 for p, oo in (human & a0_only) if oo == o),
            "neither_naive": sum(1 for p, oo in neither if oo == o),
            "hb_complete": sum(1 for _p, oo in human if oo == o),
            "machine_any_matched": sum(
                1 for p, oo in (human & (a0 | v21)) if oo == o
            ),
        }
    return {
        "by_branch": by_branch,
        "by_origin_by_branch": by_origin_by_branch,
        "n_human_complete": len(human),
        "n_machine_any_matched": len(human & (a0 | v21)),
        "n_both_native_matched": len(human & both),
        "n_neither_naive": len(neither),
    }


def _group_coverage_block(
    *,
    group_name: str,
    horizons: frozenset[int],
    hb_df: pd.DataFrame,
    a0_df: pd.DataFrame,
    v21_df: pd.DataFrame,
    origins: list[int],
) -> dict[str, Any]:
    hb_pairs, hb_inc = group_complete_pairs(
        hb_df, horizons=horizons, origin_col="forecast_origin"
    )
    a0_pairs, _ = group_complete_pairs(
        a0_df, horizons=horizons, origin_col="forecast_origin"
    )
    v21_pairs, _ = group_complete_pairs(
        v21_df, horizons=horizons, origin_col="forecast_origin"
    )
    matched = hb_pairs & (a0_pairs | v21_pairs)
    branches = _branch_breakdown(hb_pairs, a0_pairs, v21_pairs, origins)
    depth = _sku_origin_histogram(matched)
    cohort_name = f"{HUMAN_COMPARABLE_COHORT}_{group_name}"
    cohort_skus = sorted({p for p, _o in matched})
    mnih_eligible = sorted(
        p
        for p, n in Counter(p for p, _o in matched).items()
        if n >= 3
    )
    return {
        "group": group_name,
        "horizons": sorted(horizons),
        "never_requires_full_h1_15": True,
        "never_renumber_horizons": True,
        "incomplete_groups_recorded": True,
        "n_incomplete_partial_groups": len(hb_inc),
        "incomplete_partial_groups_sample": hb_inc[:40],
        "by_origin_hb_complete": {
            str(o): sum(1 for _p, oo in hb_pairs if oo == o) for o in origins
        },
        "by_origin_matched_machine_any": {
            str(o): sum(1 for _p, oo in matched if oo == o) for o in origins
        },
        "deployable_branch": branches,
        "sku_origin_depth_matched": depth,
        "n_skus_mnih_eligible_ge3_origins": len(mnih_eligible),
        cohort_name: {
            "definition": (
                f"SKUs with matched complete machine and HB forecasts for group "
                f"{group_name} horizons {sorted(horizons)} (not full h1–15)"
            ),
            "n_skus": len(cohort_skus),
            "n_product_origin_pairs": len(matched),
            "n_skus_mnih_eligible_ge3": len(mnih_eligible),
            "only_this_cohort_may_receive_MNIH_for_group": True,
        },
    }


def build_hb_origin_mapping_audit(*, repo_root: Path) -> dict[str, Any]:
    """Build identifier-only mapping + full + group-specific coverage."""
    repo_root = Path(repo_root).resolve()
    hb_path = repo_root / BUDGET_UNIVERSE_V2_REL
    a0_path = repo_root / A0_PANEL_REL
    v21_path = repo_root / V21_PANEL_REL
    postlock_path = repo_root / POSTLOCK_REL
    stale_path = repo_root / FAMILY_DIR_REL / STALE_COVERAGE_AUDIT_FILENAME

    if not hb_path.is_file():
        raise FileNotFoundError(
            f"Missing v2 budget universe (required for 140401 remap): {hb_path}"
        )

    hb = _load_hb_id(hb_path)
    protected = sorted(PROTECTED_HB_ORIGINS)

    qcol = (
        "source_budget_quarter" if "source_budget_quarter" in hb.columns else None
    )
    origin_for_1404q1: list[int] = []
    n_target_140312_under_1404q1 = 0
    if qcol is not None:
        q1404 = hb.loc[hb[qcol].astype(str).str.upper() == "1404Q1"]
        origin_for_1404q1 = (
            sorted(int(x) for x in q1404["budget_origin"].unique())
            if len(q1404)
            else []
        )
        n_target_140312_under_1404q1 = int(
            ((q1404["target_date"].astype(int) == 140312).sum()) if len(q1404) else 0
        )

    all_origins = (
        sorted(int(x) for x in hb["budget_origin"].unique()) if len(hb) else []
    )

    hb_prot = hb.loc[
        pd.to_numeric(hb["budget_origin"], errors="coerce").isin(protected)
    ].copy()
    work = hb_prot.rename(columns={"budget_origin": "forecast_origin"})
    human_pairs, human_incomplete = complete_pairs_full(work, "forecast_origin")

    machine: dict[str, Any] = {}
    machine_frames: dict[str, pd.DataFrame] = {}
    machine_pairs: dict[str, set[tuple[str, int]]] = {}
    for label, path in (("a0", a0_path), ("v21", v21_path)):
        if not path.is_file():
            machine[label] = {"exists": False, "path": path.as_posix()}
            machine_pairs[label] = set()
            machine_frames[label] = pd.DataFrame(
                columns=list(ID_COLS_MACHINE)
            )
            continue
        mdf = _load_machine_id(path)
        mdf_p = mdf.loc[
            pd.to_numeric(mdf["forecast_origin"], errors="coerce").isin(protected)
        ].copy()
        pairs, incomplete = complete_pairs_full(mdf_p, "forecast_origin")
        machine_pairs[label] = pairs
        machine_frames[label] = mdf_p
        machine[label] = {
            "exists": True,
            "relative_path": path.as_posix(),
            "sha256": file_sha256(path),
            "columns_read": list(ID_COLS_MACHINE),
            "forecast_or_outcome_columns_read": [],
            "by_origin": _by_origin_stats(
                pairs, incomplete, mdf_p, "forecast_origin", protected
            ),
            "n_complete_product_origin": len(pairs),
            "n_incomplete_groups": len(incomplete),
            "incomplete_groups_sample": incomplete[:50],
            "incomplete_groups_truncated": len(incomplete) > 50,
        }

    a0_p = machine_pairs["a0"]
    v21_p = machine_pairs["v21"]
    branches_full = _branch_breakdown(human_pairs, a0_p, v21_p, protected)

    # Deployable frozen-policy reconciliation for ALL HB-complete pairs (incl. naive)
    sales_path = repo_root / SALES_FULL_PORTFOLIO_REL
    if not sales_path.is_file():
        raise FileNotFoundError(
            f"Sales required to reconcile naive_last_value through origin−2: {sales_path}"
        )
    sales = pd.read_parquet(sales_path, columns=["product", "date", "sales"])
    deployable_recon = reconcile_hb_complete_deployable(
        human_complete=human_pairs,
        a0_complete=a0_p,
        v21_complete=v21_p,
        sales=sales,
    )
    # Native-matched remains MNIH cohort; portfolio gate uses deployable incl. naive
    matched_native = human_pairs & (a0_p | v21_p)
    matched_for_cohort = matched_native
    cohort_skus = sorted({p for p, _o in matched_for_cohort})
    primary_skus = sorted(
        p
        for p, n in Counter(p for p, _o in matched_for_cohort).items()
        if n >= 3
    )
    portfolio_deployable_n = deployable_recon["n_deployable_with_forecast"]

    # Group-specific coverage (retain 140401 partial groups)
    group_coverage: dict[str, Any] = {}
    for gname, ghs in VALIDITY_GROUPS:
        group_coverage[gname] = _group_coverage_block(
            group_name=gname,
            horizons=ghs,
            hb_df=work,
            a0_df=machine_frames["a0"],
            v21_df=machine_frames["v21"],
            origins=protected,
        )

    postlock_note = {
        "path": POSTLOCK_REL.as_posix(),
        "exists": postlock_path.is_file(),
        "machine_outcomes_previously_examined_for_origins": [140401, 140402],
        "hb_forecasts_inspected_in_postlock": False,
        "note": (
            "Machine outcomes for 140401/140402 were examined in post-lock TS "
            "stability. HB forecasts for those origins were not inspected then "
            "and remain unopened in this audit."
        ),
    }
    if postlock_path.is_file():
        postlock = json.loads(postlock_path.read_text(encoding="utf-8"))
        postlock_note["postlock_hb_outcomes_inspected_flag"] = postlock.get(
            "hb_outcomes_inspected"
        )
        postlock_note["sha256"] = file_sha256(postlock_path)

    stale_prior = {
        "artifact": STALE_COVERAGE_AUDIT_FILENAME,
        "status": "SUPERSEDED",
        "superseded_by": MAPPING_AUDIT_FILENAME,
        "drives_reliability_execution": False,
        "discarded_figures": {
            "n_archval55_like_skus": 55,
            "n_skus_with_at_least_3_complete_origins": 50,
            "n_human_complete_1_15": 167,
            "common_native_matched_n_total": 161,
        },
        "forbidden_stale_integers": sorted(STALE_COVERAGE_FIGURES_FORBIDDEN),
        "reason": (
            "Those counts used v1 budget_universe with origin 140312 / Archval55. "
            "After 140312→140401 remap, coverage is regenerated from v2 + machine "
            "panels; do not reuse 55/50/167/161."
        ),
        "stale_file_exists": stale_path.is_file(),
    }

    sku_hist_human = _sku_origin_histogram(human_pairs)
    sku_hist_matched = _sku_origin_histogram(matched_for_cohort)

    audit: dict[str, Any] = {
        "audit_version": "hb_origin_mapping_audit_v1",
        "family": FAMILY,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "columns_read_hb": list(hb.attrs.get("columns_read") or []),
        "outcome_columns_present_but_not_read": list(
            hb.attrs.get("forbidden_present_but_not_read") or []
        ),
        "forecast_or_outcome_columns_read": [],
        "authoritative_mapping": {
            "rule": "Version → version_to_qrt → origin_from_quarter",
            "module": "pkg.benchmark.budget_origin_v2",
            "never": "min(target_date) as budget_origin",
            "horizon": "shamsi_month_diff(target_date, budget_origin)+1; keep 1..15",
            "never_silent_drop": True,
            "never_renumber_horizons": True,
        },
        "remap_140312_to_140401": {
            "1404Q1_budget_origins": origin_for_1404q1,
            "expected_origin": 140401,
            "mapping_ok": origin_for_1404q1 == [140401] or origin_for_1404q1 == [],
            "target_140312_under_1404Q1_in_kept_panel": n_target_140312_under_1404q1,
            "note": (
                "1404Q1 → budget_origin 140401. Target 140312 under that origin is "
                "pre_origin (excluded from 1..15 panel), not a forecast origin id. "
                "Keep 140401 even when full h1–15 matched coverage is limited; "
                "group-level evidence still retains valid horizons."
            ),
            "keep_140401_despite_limited_full_horizon_coverage": True,
            "140312_is_not_a_reliability_origin": True,
        },
        "protected_hb_origins_reliability": protected,
        "reliability_execution_origins": protected,
        LEGACY_TS_LOCK_HB_ORIGINS_KEY: {
            "origins": sorted(LOCK_HB_ORIGINS),
            "drives_reliability_execution": False,
            "role": (
                "Immutable portfolio_ts_lock / production holdout list. "
                "Preserved for lock continuity; must not drive reliability joins "
                "or coverage denominators."
            ),
        },
        "all_distinct_hb_origins_in_v2_panel": all_origins,
        "budget_universe_v2": {
            "relative_path": BUDGET_UNIVERSE_V2_REL.as_posix(),
            "sha256": file_sha256(hb_path),
            "completeness_all_origins": completeness_by_origin(hb),
            "by_origin_protected_full_h1_15": _by_origin_stats(
                human_pairs,
                human_incomplete,
                work,
                "forecast_origin",
                protected,
            ),
            "n_complete_product_origin_protected_full_h1_15": len(human_pairs),
            "n_incomplete_groups_protected_full_h1_15": len(human_incomplete),
            "incomplete_groups_sample": human_incomplete[:80],
            "incomplete_groups_truncated": len(human_incomplete) > 80,
        },
        "machine_panels_id_only": machine,
        "full_horizon_matched_counts_regenerated": {
            "definition_human": (
                "HB product×origin with exact horizons {1..15} on protected origins"
            ),
            "definition_matched_machine_any": (
                "Human complete ∩ (A0 complete ∪ V2.1 complete) on same product×origin "
                "— native-only; used for HUMAN_COMPARABLE_COHORT_<group> / MNIH"
            ),
            "definition_portfolio_activation_deployable": (
                f"{DEPLOYABLE_PORTFOLIO_COHORT}: HB-complete with actual frozen-policy "
                "forecast (native or valid naive_last_value)"
            ),
            "role_native_matched": (
                f"{HUMAN_COMPARABLE_COHORT}_<group> / SKU×group MNIH eligibility only"
            ),
            "role_deployable": (
                f"portfolio_activation_gate_population = {DEPLOYABLE_PORTFOLIO_COHORT}"
            ),
            "not_portfolio_population": (
                f"{HUMAN_COMPARABLE_COHORT} with ≥3 origins"
            ),
            "n_human_complete_1_15": len(human_pairs),
            "n_machine_any_matched_complete_1_15": len(matched_native),
            "n_both_native_matched_complete_1_15": branches_full[
                "n_both_native_matched"
            ],
            "n_native_deployable": deployable_recon["identity"]["n_native"],
            "n_valid_naive_deployable": deployable_recon["identity"]["n_valid_naive"],
            "n_no_machine_forecast": deployable_recon["identity"][
                "n_no_machine_forecast"
            ],
            "identity_equation": deployable_recon["identity"]["equation"],
            "identity_holds": deployable_recon["identity"]["holds"],
            "n_neither_naive_human_complete_legacy_note": (
                "Prefer disposition counts: valid naive vs NO_MACHINE_FORECAST; "
                "legacy _branch_breakdown.neither_naive lumps both"
            ),
            "n_portfolio_activation_deployable": portfolio_deployable_n,
            "identity": deployable_recon.get("identity"),
            "deployable_coverage_pct": deployable_recon.get("deployable_coverage_pct"),
            "by_disposition_among_hb_complete": deployable_recon[
                "by_disposition_among_hb_complete"
            ],
            "by_branch_human_complete": branches_full["by_branch"],
            "by_origin_by_branch": branches_full["by_origin_by_branch"],
            "sku_depth_human_complete": sku_hist_human,
            "sku_depth_machine_any_matched": sku_hist_matched,
            "stale_prior_counts_superseded": stale_prior,
        },
        "deployable_frozen_policy_reconciliation": deployable_recon,
        DEPLOYABLE_PORTFOLIO_COHORT: deployable_recon[DEPLOYABLE_PORTFOLIO_COHORT],
        NO_MACHINE_FORECAST_COHORT: {
            "definition": deployable_recon[NO_MACHINE_FORECAST_COHORT]["definition"],
            "n_product_origin_pairs": deployable_recon[NO_MACHINE_FORECAST_COHORT][
                "n_product_origin_pairs"
            ],
            "n_skus": deployable_recon[NO_MACHINE_FORECAST_COHORT]["n_skus"],
            "skus": deployable_recon[NO_MACHINE_FORECAST_COHORT]["skus"],
            "enters_accuracy_denominator": False,
            "reported_as": "coverage_failures",
        },
        # Back-compat key for earlier review writers
        "matched_counts_regenerated": {
            "n_human_complete_1_15": len(human_pairs),
            "n_machine_any_matched_complete_1_15": len(matched_for_cohort),
            "n_both_native_matched_complete_1_15": branches_full[
                "n_both_native_matched"
            ],
            "stale_prior_counts_discarded": stale_prior,
        },
        "group_specific_matched_coverage": group_coverage,
        "group_vs_full_horizon_policy": {
            "full_h1_15_cohort": (
                f"Portfolio activation gate population = {DEPLOYABLE_PORTFOLIO_COHORT} "
                "(HB-complete with actual frozen-policy forecast; "
                f"not {HUMAN_COMPARABLE_COHORT} with ≥3 origins)"
            ),
            "group_cohorts": (
                "SKU×validity-group labels use HUMAN_COMPARABLE_COHORT_<group> "
                "native origin counts; requiring full h1–15 must not discard a valid "
                "earlier group (e.g. 140401 may contribute to h1_3/h4_6/h7_12 "
                "even when h15 is missing)."
            ),
        },
        HUMAN_COMPARABLE_COHORT: {
            "definition": (
                "SKUs with matched, complete native machine and HB forecasts "
                "(complete {1..15} on both sides for ≥1 protected origin) — "
                "SKU×validity-group MNIH eligibility only (not portfolio gate)"
            ),
            "n_skus": len(cohort_skus),
            "n_product_origin_pairs": len(matched_for_cohort),
            "role": "sku_x_group_MNIH_only",
            "not_portfolio_activation_population": True,
            "primary_portfolio_evaluation": {
                "definition": (
                    f"Portfolio activation gate = {DEPLOYABLE_PORTFOLIO_COHORT} "
                    f"({portfolio_deployable_n} product×origins); "
                    f"{NO_MACHINE_FORECAST_COHORT} excluded from accuracy denominators"
                ),
                "n_product_origin_pairs_deployable": portfolio_deployable_n,
                "deployable_coverage_pct": deployable_recon.get(
                    "deployable_coverage_pct"
                ),
                "n_skus_native_ge3_complete_origins_for_MNIH_context_only": len(
                    primary_skus
                ),
                "deployable_reconciliation_status": deployable_recon[
                    "reconciliation_status"
                ],
            },
            "sensitivity_evaluation": {
                "definition": (
                    "All complete matched product×origin cases, plus leave-one-origin "
                    f"(exclude {NO_MACHINE_FORECAST} from accuracy denominators)"
                ),
                "n_product_origin_pairs": len(matched_for_cohort),
            },
            "only_native_cohort_may_receive_MACHINE_NONINFERIOR_TO_HUMAN": True,
            "neither_naive_or_NO_MACHINE_FORECAST_blocks_MNIH_at_sku_group": True,
        },
        "sales_used_for_naive_availability_only": {
            "relative_path": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "sha256": file_sha256(sales_path),
            "columns_read": ["product", "date", "sales"],
            "purpose": "origin−2 naive_last_value availability for deployable reconciliation",
            "hb_forecast_columns_read": [],
            "earliest_origin_minus_2_example": int(
                shamsi_add_months(min(protected), -2)
            ),
        },
        "postlock_140401_140402_context": postlock_note,
        "never_silent_discard": {
            "policy": (
                "Missing or invalid origins are recorded in incomplete_groups; "
                "never dropped without an audit row. Group incompleteness does not "
                "discard earlier complete groups."
            ),
            "n_incomplete_full_horizon_recorded": len(human_incomplete),
        },
        "supersedes": {
            STALE_COVERAGE_AUDIT_FILENAME: stale_prior,
        },
    }
    return audit


def write_hb_origin_mapping_audit(*, repo_root: Path) -> dict[str, Any]:
    """Write hb_origin_mapping_audit_v1.json under the family results dir."""
    repo_root = Path(repo_root).resolve()
    audit = build_hb_origin_mapping_audit(repo_root=repo_root)
    out_dir = repo_root / FAMILY_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / MAPPING_AUDIT_FILENAME
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out_path)

    # Stamp stale coverage audit as superseded (preserve file; do not delete)
    stale_path = out_dir / STALE_COVERAGE_AUDIT_FILENAME
    if stale_path.is_file():
        try:
            stale = json.loads(stale_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stale = {}
        stale["supersession"] = {
            "status": "SUPERSEDED",
            "superseded_by": (FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME).as_posix(),
            "drives_reliability_execution": False,
            "stamped_at": _utc(),
            "note": (
                "140312-based Archval55 coverage figures (55/50/167/161) must not "
                "drive reliability execution. Use hb_origin_mapping_audit_v1.json."
            ),
        }
        tmp_s = stale_path.with_suffix(stale_path.suffix + ".tmp")
        tmp_s.write_text(
            json.dumps(stale, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp_s.replace(stale_path)

    gcov = audit["group_specific_matched_coverage"]
    return {
        "ok": True,
        "path": out_path.as_posix(),
        "sha256": file_sha256(out_path),
        "hb_outcomes_inspected": False,
        "protected_hb_origins_reliability": audit[
            "protected_hb_origins_reliability"
        ],
        "full_horizon_matched": {
            k: audit["full_horizon_matched_counts_regenerated"][k]
            for k in (
                "n_human_complete_1_15",
                "n_machine_any_matched_complete_1_15",
                "n_both_native_matched_complete_1_15",
                "n_portfolio_activation_deployable",
                "deployable_coverage_pct",
            )
        },
        "identity": audit["full_horizon_matched_counts_regenerated"].get("identity"),
        "deployable_reconciliation_status": audit[
            "deployable_frozen_policy_reconciliation"
        ]["reconciliation_status"],
        "n_no_machine_forecast": audit["deployable_frozen_policy_reconciliation"][
            "n_no_machine_forecast"
        ],
        "n_valid_naive": audit["deployable_frozen_policy_reconciliation"]["identity"][
            "n_valid_naive"
        ],
        "group_mnih_eligible": {
            g: gcov[g]["n_skus_mnih_eligible_ge3_origins"] for g in gcov
        },
        "legacy_ts_lock_drives_execution": False,
    }
