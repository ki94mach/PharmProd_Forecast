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
    ELIGIBLE_UNIVERSE_REL,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    HUMAN_COMPARABLE_COHORT,
    MAPPING_AUDIT_FILENAME,
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
    sales: pd.DataFrame, products: set[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    sub = sales.loc[sales["product"].astype(str).isin(products)].copy()
    if sub.empty:
        return out
    sub["date"] = pd.to_numeric(sub["date"], errors="coerce")
    sub["sales"] = pd.to_numeric(sub["sales"], errors="coerce")
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
    # Features for ALL eligible SKUs that appear in sales; missing sales = missing feature
    feats = _sku_sales_features(sales, full_skus)

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

    # Engine / fallback vs ALL eligible, accounting for missing forecasts
    native_block: dict[str, Any] = {"exists": False}
    engine_flags: list[str] = []
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

        if "fallback_category" in pol.columns:
            prod_fb = pol.groupby(pol["product"].astype(str))[
                "fallback_category"
            ].agg(lambda s: s.mode().iloc[0] if len(s) else "unknown")
        else:
            prod_fb = pd.Series(dtype=object)

        def _fallback_rates(sku_set: set[str]) -> dict[str, Any]:
            present = prod_fb.loc[prod_fb.index.isin(sku_set)]
            n_set = len(sku_set)
            n_present = int(len(present))
            n_missing = n_set - n_present
            if n_present == 0:
                by_cat: dict[str, float] = {}
                both = 0.0
            else:
                by_cat = {
                    str(k): float(v)
                    for k, v in present.value_counts(normalize=True).to_dict().items()
                }
                both = float(by_cat.get("both_native", 0.0))
            # Rates over full denominator including missing forecasts
            missing_share = n_missing / max(n_set, 1)
            return {
                "n_denominator": n_set,
                "n_with_policy_rows": n_present,
                "n_missing_forecasts": n_missing,
                "missing_forecast_share": float(missing_share),
                "by_category_among_present": by_cat,
                "both_native_rate_among_present": both,
                "fallback_rate_non_both_native_among_present": float(1.0 - both)
                if n_present
                else None,
                # Treat missing forecasts as non-native coverage gap
                "effective_both_native_rate_full_denominator": float(
                    both * (n_present / max(n_set, 1))
                ),
            }

        full_fb = _fallback_rates(full_skus)
        coh_fb = _fallback_rates(hb_cohort)

        def _eng_rate(sku_set: set[str]) -> dict[str, Any]:
            s = pol.loc[pol["product"].astype(str).isin(sku_set)]
            n_set = len(sku_set)
            n_present = int(s["product"].astype(str).nunique()) if len(s) else 0
            if s.empty:
                return {
                    "n_denominator": n_set,
                    "n_with_rows": 0,
                    "n_missing_forecasts": n_set,
                    "v21_ok_rate_among_present": None,
                    "a0_ok_rate_among_present": None,
                    "v21_ok_rate_full_denominator": 0.0,
                    "a0_ok_rate_full_denominator": 0.0,
                }
            v21_m = float(pd.to_numeric(s["v21_ok"], errors="coerce").mean())
            a0_m = float(pd.to_numeric(s["a0_ok"], errors="coerce").mean())
            return {
                "n_denominator": n_set,
                "n_with_rows": n_present,
                "n_missing_forecasts": n_set - n_present,
                "v21_ok_rate_among_present": v21_m,
                "a0_ok_rate_among_present": a0_m,
                "v21_ok_rate_full_denominator": float(
                    v21_m * (n_present / max(n_set, 1))
                ),
                "a0_ok_rate_full_denominator": float(
                    a0_m * (n_present / max(n_set, 1))
                ),
            }

        native_block = {
            "exists": True,
            "relative_path": POLICY_PANEL_REL.as_posix(),
            "sha256": file_sha256(policy_path),
            "columns_read": safe,
            "forecast_or_outcome_columns_read": [],
            "outcome_columns_present_but_not_read": forbidden_present,
            "n_eligible_denominator": n_eligible,
            "n_eligible_missing_from_policy_panel": len(missing_forecasts),
            "missing_forecast_products_sample": missing_forecasts[:40],
            "full_eligible": full_fb,
            "hb_cohort": coh_fb,
            "engine_availability": {
                "full_eligible": _eng_rate(full_skus),
                "hb_cohort": _eng_rate(hb_cohort),
            },
            "fallback_rate_abs_diff_among_present": (
                None
                if full_fb["fallback_rate_non_both_native_among_present"] is None
                or coh_fb["fallback_rate_non_both_native_among_present"] is None
                else abs(
                    float(coh_fb["fallback_rate_non_both_native_among_present"])
                    - float(full_fb["fallback_rate_non_both_native_among_present"])
                )
            ),
        }
        if (
            native_block["fallback_rate_abs_diff_among_present"] is not None
            and native_block["fallback_rate_abs_diff_among_present"] > 0.15
        ):
            engine_flags.append("fallback_rate")

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
            "See covariate_balance separately."
        ),
        "by_group_cohort_n_skus_in_eligible": {
            g: len(s & full_skus) for g, s in cohorts["group_cohorts"].items()
        },
    }

    # Covariate balance independent of coverage
    if covariate_flags or cat_flags:
        covariate_status = "NOT_REPRESENTATIVE"
    else:
        covariate_status = "REPRESENTATIVE"

    if engine_flags:
        engine_balance = "NOT_BALANCED"
    else:
        engine_balance = "BALANCED" if native_block.get("exists") else "UNKNOWN"

    if cat_flags:
        categorical_balance = "NOT_BALANCED"
    else:
        categorical_balance = "BALANCED"

    # Portfolio conclusion: limit if coverage incomplete OR covariates not representative
    if coverage_scope["status"] == "INCOMPLETE":
        portfolio_scope = "LIMIT_PORTFOLIO_CONCLUSIONS_TO_HB_COVERED_COHORT"
        extrapolation = (
            "Do not extrapolate MACHINE_NONINFERIOR_TO_HUMAN to the remaining "
            f"~{len(outside)} eligible SKUs outside HB-covered cohorts. "
            "Coverage is incomplete; this does not by itself prove covariate "
            f"non-representativeness (covariate_balance={covariate_status})."
        )
    elif covariate_status == "NOT_REPRESENTATIVE":
        portfolio_scope = "LIMIT_PORTFOLIO_CONCLUSIONS_TO_HB_COVERED_COHORT"
        extrapolation = (
            "HB cohort covariates are not representative of the full eligible "
            f"portfolio; do not extrapolate MNIH to the remaining ~{len(outside)} SKUs."
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

    return {
        "audit_version": "hb_cohort_representativeness_audit_v1",
        "family": FAMILY,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
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
            "note": (
                "Independent of coverage_scope. Low coverage must not automatically "
                "imply covariate non-representativeness."
            ),
        },
        "categorical_balance": {
            "status": categorical_balance,
            "flagged_axes": cat_flags,
            "features": categorical,
            "thresholds": {"categorical_tvd_flag": 0.20},
        },
        "engine_fallback_balance": {
            "status": engine_balance,
            "flagged_axes": engine_flags,
            "details": native_block,
            "thresholds": {"fallback_rate_abs_diff_flag": 0.15},
        },
        "portfolio_conclusion_scope": portfolio_scope,
        "extrapolation_rule": extrapolation,
        "mapping_audit_ref": mapping_ref,
        "sales_source": {
            "relative_path": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "sha256": file_sha256(sales_path),
            "columns_read": ["product", "date", "sales"],
            "n_products_in_sales_file": len(sales_products),
            "n_eligible_with_sales_rows": len(full_skus & sales_products),
            "n_eligible_missing_sales": len(full_skus - sales_products),
            "not_used": SALES_ARCHVAL55_REL.as_posix(),
        },
        # legacy flat keys for older readers
        "representativeness_conclusion": covariate_status,
        "flagged_axes": covariate_flags + cat_flags + engine_flags,
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
        "engine_fallback_balance": audit["engine_fallback_balance"]["status"],
        "portfolio_conclusion_scope": audit["portfolio_conclusion_scope"],
        "n_eligible": audit["populations"]["full_eligible_portfolio"]["n_skus"],
        "n_hb_cohort": audit["populations"]["hb_group_union_cohort"]["n_skus"],
    }
