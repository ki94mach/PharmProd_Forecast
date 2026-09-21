#!/usr/bin/env python3
"""Portfolio TS architecture lock v1 — audit / evaluate / optional immutable lock.

TS is frozen BEFORE residual-XGBoost research.
Human Budget origins are never scored.
Decision panels use complete product×origin jobs only (15 horizons).

Default: --audit-only (no lock write). Use --write-lock to create ts_architecture_lock.json
(fails if the file already exists).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.universes import load_universe_product_names  # noqa: E402
from pkg.portfolio_ts_lock.audit import (  # noqa: E402
    duplicate_audit,
    file_sha256,
    hb_coverage_metadata,
    horizon_audit,
    status_counts,
)
from pkg.portfolio_ts_lock.bootstrap import (  # noqa: E402
    drop_top_k_hybrid_gainers,
    hierarchical_origin_then_sku_bootstrap,
    leave_one_origin,
    leave_one_sku,
    origin_block_sensitivity,
    sku_cluster_bootstrap,
)
from pkg.portfolio_ts_lock.completeness import (  # noqa: E402
    assert_rows_multiple_of_15,
    audit_product_origin_completeness,
)
from pkg.portfolio_ts_lock.constants import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    HB_ORIGINS,
    HORIZON_GROUPS,
    LOCKED_SCORE_END,
    POLICY_HYBRID,
    POST_HB_ORIGINS,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import (  # noqa: E402
    assert_exported_actual_agreement,
    build_dense_actuals,
)
from pkg.portfolio_ts_lock.freeze import (  # noqa: E402
    horizon_group_wmape_lifts,
    make_recommendation,
    score_frame,
    sku_origin_wins,
)
from pkg.portfolio_ts_lock.lock_io import LockExistsError, write_lock_atomic  # noqa: E402
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb  # noqa: E402
from pkg.portfolio_ts_lock.router_features import build_router_features_sku_origin  # noqa: E402

DEFAULT_OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_ts_architecture_lock_v1"
)
V21_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_v21_backfill_v2_export"
)
A0_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_a0_backfill_v2_export"
)
SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_panel(export_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(export_dir / "forecasts_panel.parquet")


def _mask_primary(df: pd.DataFrame, unseen: set[str]) -> pd.DataFrame:
    return df.loc[
        df["product"].astype(str).isin(unseen)
        & df["forecast_origin"].astype(int).isin(RESIDUAL_DEVELOPMENT_ORIGINS)
    ].copy()


def _mask_original55(df: pd.DataFrame, original: set[str]) -> pd.DataFrame:
    return df.loc[
        df["product"].astype(str).isin(original)
        & df["forecast_origin"].astype(int).isin(RESIDUAL_DEVELOPMENT_ORIGINS)
    ].copy()


def _cohort_reconcile(df: pd.DataFrame, label: str) -> dict[str, Any]:
    return assert_rows_multiple_of_15(df, label)


def _v21_dense_coverage(panel_v21: pd.DataFrame, dense: pd.DataFrame) -> dict[str, Any]:
    """V2 exports often lack actuals; verify dense coverage + agreement if present."""
    agree = assert_exported_actual_agreement(panel_v21, dense)
    p = filter_non_hb(panel_v21)
    p = p.loc[p["model"].astype(str) != "naive_last_value"] if "model" in p.columns else p
    d = dense.rename(columns={"date": "target_date", "actual": "dense_actual"})
    m = p.merge(
        d[["product", "target_date", "dense_actual"]],
        on=["product", "target_date"],
        how="left",
    )
    n = int(len(m))
    n_miss = int(m["dense_actual"].isna().sum())
    return {
        "exported_actual_agreement": agree,
        "n_v21_forecast_rows_non_hb": n,
        "n_missing_dense_for_v21_target": n_miss,
        "dense_coverage_ok": n_miss == 0,
    }


def run_audit_and_eval(
    *,
    out_dir: Path,
    write_lock: bool,
    audit_only: bool,
    bootstrap_b: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "ts_architecture_lock.json"

    sales = pd.read_parquet(SALES)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    dense.to_parquet(out_dir / "dense_actuals.parquet", index=False)

    panel_v21_raw = _load_panel(V21_EXPORT)
    panel_a0_raw = _load_panel(A0_EXPORT)
    panel_v21 = filter_non_hb(panel_v21_raw)
    panel_a0 = filter_non_hb(panel_a0_raw)

    assert not panel_v21["forecast_origin"].astype(int).isin(HB_ORIGINS).any()
    assert not panel_a0["forecast_origin"].astype(int).isin(HB_ORIGINS).any()

    original55 = set(load_universe_product_names("mvp_products"))
    status, incomplete = audit_product_origin_completeness(
        panel_v21=panel_v21_raw,
        panel_a0=panel_a0_raw,
        dense=dense,
        original55=original55,
    )
    status.to_csv(out_dir / "product_origin_completeness.csv", index=False)
    incomplete.to_csv(out_dir / "incomplete_product_origins.csv", index=False)

    agree_a0 = assert_exported_actual_agreement(panel_a0, dense)
    v21_dense = _v21_dense_coverage(panel_v21, dense)

    audit: dict[str, Any] = {
        "sales_sha256": file_sha256(SALES),
        "sales_path": SALES.as_posix(),
        "locked_score_end": LOCKED_SCORE_END,
        "hb_origins_excluded": sorted(HB_ORIGINS),
        "post_hb_origins": sorted(POST_HB_ORIGINS),
        "residual_development": sorted(RESIDUAL_DEVELOPMENT_ORIGINS),
        "v21_status_counts": status_counts(V21_EXPORT / "state.sqlite"),
        "a0_status_counts": status_counts(A0_EXPORT / "state.sqlite"),
        "hb_coverage_v21": hb_coverage_metadata(V21_EXPORT / "state.sqlite", "v2.1"),
        "hb_coverage_a0": hb_coverage_metadata(A0_EXPORT / "state.sqlite", "a0"),
        "horizon_v21": horizon_audit(panel_v21_raw),
        "horizon_a0": horizon_audit(panel_a0_raw),
        "duplicates_v21": duplicate_audit(panel_v21_raw),
        "duplicates_a0": duplicate_audit(panel_a0_raw),
        "actual_agreement_a0": agree_a0,
        "v21_vs_dense": v21_dense,
        "dense_n_rows": int(len(dense)),
        "dense_n_products": int(dense["product"].nunique()),
        "n_product_origins_status": int(len(status)),
        "n_deployable_complete_po": int(status["deployable_complete"].sum()),
        "n_joint_both_native_complete_po": int(status["joint_both_native_complete"].sum()),
        "n_incomplete_po": int((~status["deployable_complete"]).sum()),
        "n_incomplete_csv_rows": int(len(incomplete)),
        "note_ts_before_residual_xgb": (
            "TS architecture is frozen before residual-XGBoost research."
        ),
        "note_joint_vs_fallback": (
            "joint_both_native_complete = both engines complete 15h; "
            "fallback_category on deployable distinguishes policy fill "
            "(v21_only / a0_only / naive) from joint availability."
        ),
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    common, deployable, cat_counts, partial = build_policy_panels(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        completeness_status=status,
    )
    cat_counts.to_csv(out_dir / "audit_fallback_categories.csv", index=False)
    partial.to_csv(out_dir / "partial_product_origins_excluded.csv", index=False)
    common.to_parquet(out_dir / "policy_panel_common_native.parquet", index=False)
    deployable.to_parquet(out_dir / "policy_panel_deployable.parquet", index=False)

    unseen = set(deployable["product"].astype(str).unique()) - original55
    primary_dep = _mask_primary(deployable, unseen)
    primary_nat = _mask_primary(common, unseen)
    evidence_55 = _mask_original55(deployable, original55)

    reconcile = {
        "deployable_all": _cohort_reconcile(deployable, "deployable_all"),
        "common_native_all": _cohort_reconcile(common, "common_native_all"),
        "primary_deployable_unseen_residual": _cohort_reconcile(
            primary_dep, "primary_deployable"
        ),
        "primary_common_native_unseen_residual": _cohort_reconcile(
            primary_nat, "primary_common_native"
        ),
        "original55_residual_deployable": _cohort_reconcile(
            evidence_55, "original55_residual"
        ),
    }
    # Per origin_role / cohort
    role_rows = []
    for (cohort, role), g in deployable.groupby(["cohort", "origin_role"]):
        n_po = int(g.groupby(["product", "forecast_origin"]).ngroups)
        role_rows.append(
            {
                "cohort": cohort,
                "origin_role": role,
                "n_rows": int(len(g)),
                "n_product_origins": n_po,
                "n_product_origins_x_15": n_po * 15,
                "reconciled": int(len(g)) == n_po * 15,
            }
        )
    role_df = pd.DataFrame(role_rows)
    role_df.to_csv(out_dir / "audit_reconcile_cohort_role.csv", index=False)
    (out_dir / "audit_reconcile.json").write_text(
        json.dumps(reconcile, indent=2) + "\n", encoding="utf-8"
    )

    all_reconciled = all(v["reconciled"] for v in reconcile.values()) and (
        role_df.empty or bool(role_df["reconciled"].all())
    ) and (cat_counts.empty or bool(cat_counts["reconciled"].all()))

    if audit_only:
        summary = {
            "mode": "audit_only",
            "all_reconciled": all_reconciled,
            "reconcile": reconcile,
            "fallback_categories": cat_counts.to_dict(orient="records"),
            "n_primary_deployable_rows": int(len(primary_dep)),
            "n_primary_common_native_rows": int(len(primary_nat)),
            "n_original55_residual_rows": int(len(evidence_55)),
            "n_partial_excluded_po": int(len(partial)),
            "lock_written": False,
            "lock_exists": lock_path.is_file(),
        }
        (out_dir / "audit_run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    if not all_reconciled:
        raise SystemExit(
            "Refusing --evaluate: primary counts not reconciled to n_po*15. "
            "Fix completeness first."
        )

    s_v2_d = score_frame(primary_dep["actual"], primary_dep["f_v2_policy"])
    s_hy_d = score_frame(primary_dep["actual"], primary_dep["f_a0_hybrid"])
    s_a0_d = score_frame(
        primary_dep.loc[primary_dep["a0_ok"], "actual"],
        primary_dep.loc[primary_dep["a0_ok"], "f_a0_only"],
    )
    s_v2_n = score_frame(primary_nat["actual"], primary_nat["f_v2_policy"])
    s_hy_n = score_frame(primary_nat["actual"], primary_nat["f_a0_hybrid"])
    s_a0_n = score_frame(primary_nat["actual"], primary_nat["f_a0_only"])
    lift_d = float(s_v2_d["portfolio_wmape_pct"] - s_hy_d["portfolio_wmape_pct"])
    lift_n = float(s_v2_n["portfolio_wmape_pct"] - s_hy_n["portfolio_wmape_pct"])

    wins = sku_origin_wins(primary_dep)
    boot = sku_cluster_bootstrap(primary_dep, B=bootstrap_b, seed=BOOTSTRAP_SEED)
    boot_h = hierarchical_origin_then_sku_bootstrap(
        primary_dep, B=bootstrap_b, seed=BOOTSTRAP_SEED
    )
    origin_sens = origin_block_sensitivity(primary_dep)
    origin_sens.to_csv(out_dir / "robustness_origin_block.csv", index=False)
    leave_one_origin(primary_dep).to_csv(out_dir / "robustness_loo_origin.csv", index=False)
    leave_one_sku(primary_dep).to_csv(out_dir / "robustness_loo_sku.csv", index=False)
    drop3 = drop_top_k_hybrid_gainers(primary_dep, k=3)
    hz_lifts = horizon_group_wmape_lifts(primary_dep, HORIZON_GROUPS)

    # Evidence-only metrics for original 55
    s55_v2 = score_frame(evidence_55["actual"], evidence_55["f_v2_policy"])
    s55_hy = score_frame(evidence_55["actual"], evidence_55["f_a0_hybrid"])

    unexplained = bool(
        not agree_a0.get("ok", True)
        or not v21_dense.get("dense_coverage_ok", True)
    )

    rec = make_recommendation(
        lift_deployable_wmape=lift_d,
        lift_common_native_wmape=lift_n,
        wins=wins,
        bootstrap_frac_positive=float(boot["frac_positive"]),
        drop_top3_lift_positive=bool(drop3["lift_positive"]),
        rmse_hybrid=float(s_hy_d["rmse"]),
        rmse_v2=float(s_v2_d["rmse"]),
        p95_hybrid=float(s_hy_d["p95_ae"]),
        p95_v2=float(s_v2_d["p95_ae"]),
        horizon_group_lifts=hz_lifts,
        unexplained_a0_defect=unexplained,
    )

    metrics = {
        "primary_deployable_v2": s_v2_d,
        "primary_deployable_hybrid": s_hy_d,
        "primary_deployable_a0_only_where_available": s_a0_d,
        "primary_common_native_v2": s_v2_n,
        "primary_common_native_hybrid": s_hy_n,
        "primary_common_native_a0_only": s_a0_n,
        "lift_deployable_wmape_pp": lift_d,
        "lift_common_native_wmape_pp": lift_n,
        "original55_residual_v2": s55_v2,
        "original55_residual_hybrid": s55_hy,
        "wins": wins,
        "sku_cluster_bootstrap": boot,
        "hierarchical_bootstrap": boot_h,
        "drop_top3": drop3,
        "horizon_group_lifts": hz_lifts,
        "recommendation_provisional": rec,
        "reconcile": reconcile,
    }
    (out_dir / "metrics_primary.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )

    router_base = deployable.loc[
        deployable["forecast_origin"].astype(int).isin(
            set(RESIDUAL_DEVELOPMENT_ORIGINS) | set(POST_HB_ORIGINS)
        )
    ]
    router = build_router_features_sku_origin(
        policy_rows=router_base, dense=dense, allow_post_hb_tag=True
    )
    router.to_csv(out_dir / "router_features_sku_origin.csv", index=False)

    lock_payload = {
        "version": "portfolio_ts_architecture_lock_v1",
        "locked_at": _utc(),
        "ts_frozen_before_residual_xgboost": True,
        "post_hb_must_not_change_lock": True,
        "hb_origins_excluded": sorted(HB_ORIGINS),
        "primary_population": {
            "sku_cohort": "previously_unseen_not_mvp55",
            "origins": sorted(RESIDUAL_DEVELOPMENT_ORIGINS),
            "requires_complete_product_origin_15h": True,
        },
        "sales_sha256": audit["sales_sha256"],
        "choice": rec["choice"],
        "frozen_policy": rec["frozen_policy"],
        "hybrid_weights": {"v2.1": 0.5, "a0": 0.5}
        if rec["frozen_policy"] == POLICY_HYBRID
        else None,
        "gates": rec.get("gates"),
        "rationale_bullets": rec.get("rationale_bullets"),
        "lift_deployable_wmape_pp": lift_d,
        "lift_common_native_wmape_pp": lift_n,
        "provisional": not write_lock,
        "exports": {"v21": V21_EXPORT.as_posix(), "a0": A0_EXPORT.as_posix()},
    }

    if write_lock:
        write_lock_atomic(lock_path, lock_payload)
        lock_written = True
    else:
        (out_dir / "recommendation_preview.json").write_text(
            json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8"
        )
        lock_written = False
        if lock_path.is_file():
            # Must not modify existing lock in evaluate mode
            pass

    return {
        "mode": "evaluate" if not write_lock else "write_lock",
        "lock_written": lock_written,
        "lock_exists": lock_path.is_file(),
        "recommendation": rec,
        "lift_deployable_wmape_pp": lift_d,
        "lift_common_native_wmape_pp": lift_n,
        "wins": wins,
        "bootstrap_frac_positive": boot["frac_positive"],
        "drop_top3": drop3,
        "horizon_group_lifts": hz_lifts,
        "reconcile": reconcile,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--write-lock", action="store_true")
    ap.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    args = ap.parse_args()

    if args.write_lock:
        audit_only, write_lock = False, True
    elif args.evaluate:
        audit_only, write_lock = False, False
    else:
        audit_only, write_lock = True, False

    summary = run_audit_and_eval(
        out_dir=args.out_dir,
        write_lock=write_lock,
        audit_only=audit_only,
        bootstrap_b=int(args.bootstrap_b),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
