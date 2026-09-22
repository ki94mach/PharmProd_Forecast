"""Pre-outcome representativeness audit: HB cohort vs full eligible portfolio.

Uses sales_full_portfolio_v1 (all eligible SKUs), not Archval55.
Separates coverage_scope from covariate_balance.
Does not open HB forecast qty or score machine-vs-human errors.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    A0_PANEL_REL,
    BUDGET_UNIVERSE_V2_REL,
    EARLIEST_HB_ORIGIN_FOR_PIT,
    ELIGIBLE_UNIVERSE_REL,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    HUMAN_COMPARABLE_COHORT,
    MAPPING_AUDIT_FILENAME,
    PIT_REPRESENTATIVENESS_CUTOFF,
    POLICY_PANEL_REL,
    PROTECTED_HB_ORIGINS,
    REPRESENTATIVENESS_AUDIT_FILENAME,
    REQUIRED_HORIZONS,
    SALES_ARCHVAL55_REL,
    SALES_FULL_PORTFOLIO_REL,
    V21_PANEL_REL,
    VALIDITY_GROUPS,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_origin_mapping_audit import (
    complete_pairs_full,
    group_complete_pairs,
    _load_hb_id,
    _load_machine_id,
)
from pkg.ts_v2.intermittency import intermittency_stats


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and v == v], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "n_missing": None,
            "mean": None,
            "std": None,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _ks_like_shift(
    a: list[float], b: list[float], *, denom_a: int, denom_b: int
) -> dict[str, Any]:
    sa = _summarize(a)
    sb = _summarize(b)
    sa["n_denominator"] = int(denom_a)
    sa["n_missing_from_denominator"] = int(denom_a - sa["n"])
    sb["n_denominator"] = int(denom_b)
    sb["n_missing_from_denominator"] = int(denom_b - sb["n"])
    mean_a, mean_b = sa["mean"], sb["mean"]
    rel = None
    if mean_a is not None and mean_b is not None and abs(mean_a) > 1e-12:
        rel = abs(mean_b - mean_a) / abs(mean_a)
    elif mean_a is not None and mean_b is not None:
        rel = abs(mean_b - mean_a)
    return {
        "full_eligible": sa,
        "hb_cohort": sb,
        "abs_mean_diff": (
            None
            if mean_a is None or mean_b is None
            else abs(float(mean_b) - float(mean_a))
        ),
        "rel_mean_diff": rel,
        "denominator_reconciled": True,
    }


def _sku_sales_features(
    sales: pd.DataFrame,
    products: set[str],
    *,
    pit_cutoff: int,
) -> dict[str, dict[str, float]]:
    """Point-in-time features using only sales with date ≤ pit_cutoff."""
    out: dict[str, dict[str, float]] = {}
    sub = sales.loc[sales["product"].astype(str).isin(products)].copy()
    if sub.empty:
        return out
    sub["date"] = pd.to_numeric(sub["date"], errors="coerce")
    sub["sales"] = pd.to_numeric(sub["sales"], errors="coerce")
    # Hard PIT filter — prohibit target-period / post-cutoff sales
    sub = sub.loc[sub["date"] <= int(pit_cutoff)]
    if sub.empty:
        return out
    for product, g in sub.groupby(sub["product"].astype(str), sort=False):
        g = g.sort_values("date")
        vals = g["sales"]
        stats = intermittency_stats(vals)
        y = vals.to_numpy(dtype=float)
        finite = np.isfinite(y)
        yf = y[finite]
        vol = float(np.std(yf)) if yf.size else float("nan")
        if yf.size >= 3:
            x = np.arange(yf.size, dtype=float)
            slope = float(np.polyfit(x, yf, 1)[0])
        else:
            slope = float("nan")
        out[str(product)] = {
            "history_length": float(finite.sum()),
            "volume": float(np.nansum(y)),
            "intermittency_zero_share": float(stats.zero_month_proportion or 0.0),
            "intermittency_adi": float(stats.average_inter_demand_interval)
            if stats.average_inter_demand_interval is not None
            else float("nan"),
            "volatility_std": vol,
            "trend_slope": slope,
            "pit_max_date_used": float(pd.to_numeric(g["date"], errors="coerce").max()),
        }
    return out


def _load_cohorts(repo_root: Path) -> dict[str, Any]:
    hb = _load_hb_id(repo_root / BUDGET_UNIVERSE_V2_REL)
    protected = sorted(PROTECTED_HB_ORIGINS)
    work = hb.loc[
        pd.to_numeric(hb["budget_origin"], errors="coerce").isin(protected)
    ].rename(columns={"budget_origin": "forecast_origin"})
    human_full, _ = complete_pairs_full(work, "forecast_origin")

    a0 = _load_machine_id(repo_root / A0_PANEL_REL)
    v21 = _load_machine_id(repo_root / V21_PANEL_REL)
    a0_p = a0.loc[
        pd.to_numeric(a0["forecast_origin"], errors="coerce").isin(protected)
    ]
    v21_p = v21.loc[
        pd.to_numeric(v21["forecast_origin"], errors="coerce").isin(protected)
    ]
    a0_full, _ = complete_pairs_full(a0_p, "forecast_origin")
    v21_full, _ = complete_pairs_full(v21_p, "forecast_origin")
    matched_full = human_full & (a0_full | v21_full)
    full_cohort = {p for p, _o in matched_full}
    primary = {
        p for p, n in Counter(p for p, _o in matched_full).items() if n >= 3
    }

    group_cohorts: dict[str, set[str]] = {}
    for gname, ghs in VALIDITY_GROUPS:
        hb_g, _ = group_complete_pairs(
            work, horizons=ghs, origin_col="forecast_origin"
        )
        a0_g, _ = group_complete_pairs(
            a0_p, horizons=ghs, origin_col="forecast_origin"
        )
        v21_g, _ = group_complete_pairs(
            v21_p, horizons=ghs, origin_col="forecast_origin"
        )
        matched_g = hb_g & (a0_g | v21_g)
        group_cohorts[gname] = {p for p, _o in matched_g}

    return {
        "full_horizon_cohort": full_cohort,
        "primary_ge3": primary,
        "group_cohorts": group_cohorts,
        "n_matched_full_pairs": len(matched_full),
        "required_horizons": sorted(REQUIRED_HORIZONS),
    }


def build_hb_cohort_representativeness_audit(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    eligible_path = repo_root / ELIGIBLE_UNIVERSE_REL
    sales_path = repo_root / SALES_FULL_PORTFOLIO_REL
    archval_path = repo_root / SALES_ARCHVAL55_REL
    policy_path = repo_root / POLICY_PANEL_REL
    mapping_path = repo_root / FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME

    if not eligible_path.is_file():
        raise FileNotFoundError(eligible_path)
    if not sales_path.is_file():
        raise FileNotFoundError(
            f"Required full-portfolio sales missing: {sales_path} "
            "(do not fall back to Archval55)"
        )

    eligible = pd.read_csv(eligible_path)
    eligible["product"] = eligible["product"].astype(str)
    full_skus = set(eligible["product"].tolist())
    n_eligible = len(full_skus)
    if n_eligible != 363:
        # Still proceed, but record expected vs actual
        pass

    cohorts = _load_cohorts(repo_root)
    # Primary representativeness comparison uses union of group HB-covered cohorts
    # intersected with eligible, plus report full-horizon separately.
    group_union: set[str] = set()
    for s in cohorts["group_cohorts"].values():
        group_union |= s
    hb_cohort = group_union & full_skus
    full_horizon_in_eligible = cohorts["full_horizon_cohort"] & full_skus
    outside = full_skus - hb_cohort

    sales = pd.read_parquet(sales_path, columns=["product", "date", "sales"])
    sales_products = set(sales["product"].astype(str).unique())
    pit_cutoff = int(PIT_REPRESENTATIVENESS_CUTOFF)
    # Sanity: cutoff == earliest HB origin − 2
    from pkg.benchmark.calendar import shamsi_add_months

    expected_cut = int(shamsi_add_months(EARLIEST_HB_ORIGIN_FOR_PIT, -2))
    if pit_cutoff != expected_cut:
        raise AssertionError(
            f"PIT cutoff {pit_cutoff} != {EARLIEST_HB_ORIGIN_FOR_PIT}-2 ({expected_cut})"
        )
    # Guard: no post-cutoff rows enter features
    n_post_cutoff_rows = int(
        (
            pd.to_numeric(sales["date"], errors="coerce") > pit_cutoff
        ).sum()
    )
    feats = _sku_sales_features(sales, full_skus, pit_cutoff=pit_cutoff)

    metric_names = [
        "history_length",
        "volume",
        "intermittency_zero_share",
        "intermittency_adi",
        "volatility_std",
        "trend_slope",
    ]
    continuous: dict[str, Any] = {}
    covariate_flags: list[str] = []
    for m in metric_names:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for p in full_skus:
            if p in feats:
                v = _safe_float(feats[p][m])
                if v is not None:
                    a_vals.append(v)
        for p in hb_cohort:
            if p in feats:
                v = _safe_float(feats[p][m])
                if v is not None:
                    b_vals.append(v)
        block = _ks_like_shift(
            a_vals, b_vals, denom_a=n_eligible, denom_b=len(hb_cohort)
        )
        continuous[m] = block
        rel = block.get("rel_mean_diff")
        if rel is not None and rel > 0.25:
            covariate_flags.append(m)

    # Reject Archval55-only identity trap
    archval_n = None
    if archval_path.is_file():
        arch = pd.read_parquet(archval_path, columns=["product"])
        archval_n = int(arch["product"].astype(str).nunique())
    invalid_prior = {
        "status": "INVALID_PRIOR_AUDIT",
        "reason": (
            "Prior representativeness compared ~54 identical Archval55 SKUs using "
            "benchmarks/v1/raw/sales.parquet. Rebuilt from sales_full_portfolio_v1 "
            f"with denominator n_eligible={n_eligible}."
        ),
        "archval55_n_products": archval_n,
        "must_not_use_sales_path": SALES_ARCHVAL55_REL.as_posix(),
        "required_sales_path": SALES_FULL_PORTFOLIO_REL.as_posix(),
    }

    cat_cols = [c for c in ("provider", "field") if c in eligible.columns]
    categorical: dict[str, Any] = {}
    cat_flags: list[str] = []
    for col in cat_cols:
        full_vc = (
            eligible[col].astype(str).value_counts(normalize=True).to_dict()
        )
        coh = eligible.loc[eligible["product"].isin(hb_cohort), col]
        coh_vc = (
            coh.astype(str).value_counts(normalize=True).to_dict() if len(coh) else {}
        )
        keys = set(full_vc) | set(coh_vc)
        tvd = 0.5 * sum(abs(full_vc.get(k, 0.0) - coh_vc.get(k, 0.0)) for k in keys)
        categorical[col] = {
            "role": "manufacturer" if col == "provider" else "category",
            "n_full_denominator": n_eligible,
            "n_cohort_denominator": len(hb_cohort),
            "full_eligible_share": full_vc,
            "hb_cohort_share": coh_vc,
            "total_variation_distance": float(tvd),
        }
        if tvd > 0.20:
            cat_flags.append(f"categorical:{col}")

    # Engine / fallback: SEPARATE among-present balance from machine coverage balance
    native_block: dict[str, Any] = {"exists": False}
    native_fallback_flags: list[str] = []
    n_missing_policy = None
    n_present_policy = None
    if policy_path.is_file():
        cols_available = pd.read_parquet(policy_path, columns=None).columns.tolist()
        safe = [
            c
            for c in (
                "product",
                "forecast_origin",
                "fallback_category",
                "v21_ok",
                "a0_ok",
                "joint_availability",
            )
            if c in cols_available
        ]
        forbidden_present = [c for c in cols_available if c in FORBIDDEN_OUTCOME_COLS]
        pol = pd.read_parquet(policy_path, columns=safe)
        pol_products = set(pol["product"].astype(str).unique())
        missing_forecasts = sorted(full_skus - pol_products)
        n_missing_policy = len(missing_forecasts)
        n_present_policy = len(full_skus & pol_products)

        if "fallback_category" in pol.columns:
            prod_fb = pol.groupby(pol["product"].astype(str))[
                "fallback_category"
            ].agg(lambda s: s.mode().iloc[0] if len(s) else "unknown")
        else:
            prod_fb = pd.Series(dtype=object)

        def _fallback_among_present(sku_set: set[str]) -> dict[str, Any]:
            present = prod_fb.loc[prod_fb.index.isin(sku_set)]
            n_present = int(len(present))
            if n_present == 0:
                return {
                    "n_present": 0,
                    "by_category": {},
                    "both_native_rate": None,
                    "fallback_rate_non_both_native": None,
                }
            by_cat = {
                str(k): float(v)
                for k, v in present.value_counts(normalize=True).to_dict().items()
            }
            both = float(by_cat.get("both_native", 0.0))
            return {
                "n_present": n_present,
                "by_category": by_cat,
                "both_native_rate": both,
                "fallback_rate_non_both_native": float(1.0 - both),
            }

        full_among = _fallback_among_present(full_skus)
        coh_among = _fallback_among_present(hb_cohort)
        fb_diff = None
        if (
            full_among["fallback_rate_non_both_native"] is not None
            and coh_among["fallback_rate_non_both_native"] is not None
        ):
            fb_diff = abs(
                float(coh_among["fallback_rate_non_both_native"])
                - float(full_among["fallback_rate_non_both_native"])
            )
            if fb_diff > 0.15:
                native_fallback_flags.append("fallback_rate_among_present")

        native_block = {
            "exists": True,
            "relative_path": POLICY_PANEL_REL.as_posix(),
            "sha256": file_sha256(policy_path),
            "columns_read": safe,
            "forecast_or_outcome_columns_read": [],
            "outcome_columns_present_but_not_read": forbidden_present,
            "n_eligible_denominator": n_eligible,
            "n_eligible_with_policy_rows": n_present_policy,
            "n_eligible_missing_from_policy_panel": n_missing_policy,
            "missing_forecast_products_sample": missing_forecasts[:40],
            "among_present_full_eligible": full_among,
            "among_present_hb_cohort": coh_among,
            "fallback_rate_abs_diff_among_present": fb_diff,
        }

    coverage_frac = len(hb_cohort) / max(n_eligible, 1)
    coverage_scope = {
        "status": "INCOMPLETE" if coverage_frac < 0.70 else "COMPLETE",
        "n_eligible": n_eligible,
        "n_hb_group_union_cohort": len(hb_cohort),
        "n_full_horizon_cohort_in_eligible": len(full_horizon_in_eligible),
        "n_remaining_eligible_outside_hb_cohort": len(outside),
        "coverage_fraction_group_union": float(coverage_frac),
        "threshold_complete_ge": 0.70,
        "note": (
            "Low coverage alone does not imply covariate non-representativeness. "
            "See covariate_balance and machine_coverage_balance separately."
        ),
        "by_group_cohort_n_skus_in_eligible": {
            g: len(s & full_skus) for g, s in cohorts["group_cohorts"].items()
        },
    }

    # Covariate balance independent of coverage
    if covariate_flags:
        covariate_status = "NOT_REPRESENTATIVE"
    else:
        covariate_status = "REPRESENTATIVE"

    if cat_flags:
        categorical_balance = "NOT_BALANCED"
    else:
        categorical_balance = "BALANCED"

    # among-present native/fallback only (does NOT imply overall machine coverage)
    if native_fallback_flags:
        native_fallback_balance = "NOT_BALANCED"
    else:
        native_fallback_balance = (
            "BALANCED" if native_block.get("exists") else "UNKNOWN"
        )

    # machine coverage over ALL eligible — missing policy rows force incompleteness
    if n_missing_policy is None:
        machine_coverage_balance = {
            "status": "UNKNOWN",
            "n_eligible": n_eligible,
        }
    elif n_missing_policy > 0:
        machine_coverage_balance = {
            "status": "NOT_BALANCED",
            "coverage_status": "INCOMPLETE",
            "n_eligible": n_eligible,
            "n_with_machine_policy_rows": n_present_policy,
            "n_missing_machine_policy_rows": n_missing_policy,
            "missing_share": float(n_missing_policy / max(n_eligible, 1)),
            "note": (
                f"{n_missing_policy}/{n_eligible} eligible SKUs lack machine-policy "
                "rows. Do not report overall balance from the present subset alone."
            ),
        }
    else:
        machine_coverage_balance = {
            "status": "BALANCED",
            "coverage_status": "COMPLETE",
            "n_eligible": n_eligible,
            "n_with_machine_policy_rows": n_present_policy,
            "n_missing_machine_policy_rows": 0,
            "missing_share": 0.0,
        }

    # Portfolio conclusion: limit if coverage incomplete OR covariates not representative
    # OR machine coverage incomplete
    machine_incomplete = (
        isinstance(machine_coverage_balance, dict)
        and machine_coverage_balance.get("coverage_status") == "INCOMPLETE"
    )
    if (
        coverage_scope["status"] == "INCOMPLETE"
        or machine_incomplete
        or covariate_status == "NOT_REPRESENTATIVE"
    ):
        portfolio_scope = "LIMIT_PORTFOLIO_CONCLUSIONS_TO_HB_COVERED_COHORT"
        extrapolation = (
            "Do not extrapolate MACHINE_NONINFERIOR_TO_HUMAN to the remaining "
            f"~{len(outside)} eligible SKUs outside HB-covered cohorts. "
            f"coverage_scope={coverage_scope['status']}; "
            f"covariate_balance={covariate_status}; "
            f"machine_coverage_balance="
            f"{machine_coverage_balance.get('status')}; "
            f"native_fallback_balance_among_present={native_fallback_balance}."
        )
    else:
        portfolio_scope = "COHORT_MAY_INFORM_PORTFOLIO_DISCUSSION_WITH_CAVEATS"
        extrapolation = (
            "Coverage and covariates look acceptable on audited axes; still only "
            f"{HUMAN_COMPARABLE_COHORT} / group cohorts may receive MNIH labels."
        )

    mapping_ref: dict[str, Any] = {"exists": mapping_path.is_file()}
    if mapping_path.is_file():
        mapping_ref["relative_path"] = (
            FAMILY_DIR_REL / MAPPING_AUDIT_FILENAME
        ).as_posix()
        mapping_ref["sha256"] = file_sha256(mapping_path)

    feature_definitions = {
        "history_length": "count of finite sales months with date ≤ PIT cutoff",
        "volume": "sum of sales over months with date ≤ PIT cutoff",
        "intermittency_zero_share": "zero-month proportion on PIT-truncated series",
        "intermittency_adi": "average inter-demand interval on PIT-truncated series",
        "volatility_std": "std of finite sales on PIT-truncated series",
        "trend_slope": "linear slope vs time index on PIT-truncated series (≥3 pts)",
        "pit_cutoff": pit_cutoff,
        "earliest_hb_origin": EARLIEST_HB_ORIGIN_FOR_PIT,
        "cutoff_rule": "earliest_HB_origin_minus_2_months",
        "target_period_sales_prohibited": True,
    }

    return {
        "audit_version": "hb_cohort_representativeness_audit_v1",
        "family": FAMILY,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
        "point_in_time": {
            "cutoff": pit_cutoff,
            "earliest_hb_origin": EARLIEST_HB_ORIGIN_FOR_PIT,
            "cutoff_equals_earliest_hb_minus_2": pit_cutoff == expected_cut,
            "n_sales_rows_after_cutoff_in_source_file": n_post_cutoff_rows,
            "feature_definitions": feature_definitions,
            "target_period_sales_used": False,
        },
        "invalid_prior_archval55_audit": invalid_prior,
        "populations": {
            "full_eligible_portfolio": {
                "relative_path": ELIGIBLE_UNIVERSE_REL.as_posix(),
                "n_skus": n_eligible,
                "expected_n_skus": 363,
            },
            "hb_group_union_cohort": {
                "definition": (
                    "Union of HUMAN_COMPARABLE_COHORT_<group> SKUs ∩ eligible"
                ),
                "n_skus": len(hb_cohort),
            },
            HUMAN_COMPARABLE_COHORT + "_full_horizon": {
                "n_skus_in_eligible": len(full_horizon_in_eligible),
                "n_matched_pairs": cohorts["n_matched_full_pairs"],
            },
            "group_cohorts": {
                f"{HUMAN_COMPARABLE_COHORT}_{g}": {
                    "n_skus_total": len(s),
                    "n_skus_in_eligible": len(s & full_skus),
                }
                for g, s in cohorts["group_cohorts"].items()
            },
            "remaining_eligible_outside_hb_cohort": {"n_skus": len(outside)},
        },
        "coverage_scope": coverage_scope,
        "covariate_balance": {
            "status": covariate_status,
            "flagged_continuous_axes": covariate_flags,
            "continuous_features": continuous,
            "thresholds": {"continuous_rel_mean_diff_flag": 0.25},
            "point_in_time_cutoff": pit_cutoff,
            "note": (
                "Independent of coverage_scope. Features use sales date ≤ "
                f"{pit_cutoff} only."
            ),
        },
        "categorical_balance": {
            "status": categorical_balance,
            "flagged_axes": cat_flags,
            "features": categorical,
            "thresholds": {"categorical_tvd_flag": 0.20},
        },
        "native_fallback_balance_among_present": {
            "status": native_fallback_balance,
            "flagged_axes": native_fallback_flags,
            "details": native_block,
            "thresholds": {"fallback_rate_abs_diff_flag": 0.15},
            "note": (
                "Computed only among SKUs with machine-policy rows. "
                "Must not be read as overall machine coverage balance."
            ),
        },
        "machine_coverage_balance": machine_coverage_balance,
        "portfolio_conclusion_scope": portfolio_scope,
        "extrapolation_rule": extrapolation,
        "mapping_audit_ref": mapping_ref,
        "sales_source": {
            "relative_path": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "sha256": file_sha256(sales_path),
            "columns_read": ["product", "date", "sales"],
            "pit_cutoff_applied": pit_cutoff,
            "n_products_in_sales_file": len(sales_products),
            "n_eligible_with_sales_rows_through_pit": len(
                {p for p in full_skus if p in feats}
            ),
            "n_eligible_missing_pit_sales": len(full_skus - set(feats)),
            "not_used": SALES_ARCHVAL55_REL.as_posix(),
        },
        "representativeness_conclusion": covariate_status,
        "flagged_axes": covariate_flags
        + cat_flags
        + native_fallback_flags
        + (
            ["machine_coverage_incomplete"]
            if machine_incomplete
            else []
        ),
    }


def write_hb_cohort_representativeness_audit(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    audit = build_hb_cohort_representativeness_audit(repo_root=repo_root)
    out_dir = repo_root / FAMILY_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / REPRESENTATIVENESS_AUDIT_FILENAME
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(out_path)
    return {
        "ok": True,
        "path": out_path.as_posix(),
        "sha256": file_sha256(out_path),
        "hb_outcomes_inspected": False,
        "coverage_scope": audit["coverage_scope"]["status"],
        "covariate_balance": audit["covariate_balance"]["status"],
        "categorical_balance": audit["categorical_balance"]["status"],
        "native_fallback_balance_among_present": audit[
            "native_fallback_balance_among_present"
        ]["status"],
        "machine_coverage_balance": audit["machine_coverage_balance"].get("status"),
        "machine_coverage_status": audit["machine_coverage_balance"].get(
            "coverage_status"
        ),
        "pit_cutoff": audit["point_in_time"]["cutoff"],
        "portfolio_conclusion_scope": audit["portfolio_conclusion_scope"],
        "n_eligible": audit["populations"]["full_eligible_portfolio"]["n_skus"],
        "n_hb_cohort": audit["populations"]["hb_group_union_cohort"]["n_skus"],
    }
