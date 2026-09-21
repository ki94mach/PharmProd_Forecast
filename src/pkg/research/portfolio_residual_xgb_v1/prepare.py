"""Prepare-only orchestration for portfolio residual-XGBoost v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.universes import load_universe_product_names
from pkg.portfolio_ts_lock.completeness import audit_product_origin_completeness
from pkg.portfolio_ts_lock.constants import (
    HB_ORIGINS,
    LOCKED_SCORE_END,
    POST_HB_ORIGINS,
    RESIDUAL_DEVELOPMENT_ORIGINS,
)
from pkg.portfolio_ts_lock.dense_actuals import build_dense_actuals
from pkg.portfolio_ts_lock.decomposition import resolve_naive_deployable_partials
from pkg.portfolio_ts_lock.lock_io import assert_lock_immutable_read, refuse_lock_mutation
from pkg.portfolio_ts_lock.policies import build_policy_panels, filter_non_hb
from pkg.portfolio_ts_lock.production_contract import (
    file_sha256,
    validate_contract_against_lock,
)
from pkg.research.portfolio_residual_xgb_v1.constants import (
    COMPARATOR_FROZEN_TS,
    COMPARATOR_HUMAN,
    COMPARATOR_HUMAN_XGB,
    COMPARATOR_INTEGRATED_DIAG,
    COMPARATOR_TS_RESID_XGB,
    FAMILY,
    HUMAN_XGB_STATUS,
    MIN_PRIOR_MATURED_ORIGINS,
    MIN_TRAIN_ROWS,
    MIN_TRAIN_SKU_ORIGINS,
)
from pkg.research.portfolio_residual_xgb_v1.contract import (
    build_residual_study_contract,
    write_immutable_json,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    FEATURE_ASOF_RULES,
    TS_RESID_FEATURE_LIST,
    assert_all_features_have_pit_rules,
    pit_feature_schema_manifest,
)
from pkg.research.portfolio_residual_xgb_v1.splits import build_oof_split_manifest


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def audit_hb_availability_budget_origins_only(budget_universe_path: Path) -> dict[str, Any]:
    """Read only budget_origin — never open sales / HB outcomes."""
    bud = pd.read_parquet(budget_universe_path, columns=["budget_origin"])
    origins = sorted({int(x) for x in bud["budget_origin"].dropna().astype(int).unique()})
    residual = sorted(RESIDUAL_DEVELOPMENT_ORIGINS)
    overlap = sorted(set(origins) & set(residual))
    return {
        "budget_origins_present": origins,
        "residual_development_origins": residual,
        "overlap_with_residual_development": overlap,
        "protected_hb_origins": sorted(HB_ORIGINS),
        "human_xgb_status": HUMAN_XGB_STATUS,
        "finding": (
            "No HB vintages exist for residual_development origins. "
            "Earliest HB origins are the four protected holdouts. "
            "Human+XGBoost cannot be independently trained before those origins."
        ),
        "hb_outcomes_inspected": False,
        "columns_read": ["budget_origin"],
    }


def build_job_plan(*, eligible_origins: list[int], n_products: int) -> dict[str, Any]:
    n_eval = len(eligible_origins)
    # One residual-XGB OOF job per eligible origin (prepare counts only).
    ts_resid_jobs = n_eval
    return {
        "executable_comparators": {
            COMPARATOR_HUMAN: {
                "n_jobs": 0,
                "note": "Baseline panel comparison only — no forecast_jobs execute.",
            },
            COMPARATOR_FROZEN_TS: {
                "n_jobs": 0,
                "note": "Uses frozen hybrid panel already produced by TS lock.",
            },
            COMPARATOR_TS_RESID_XGB: {
                "n_jobs": int(ts_resid_jobs),
                "n_eligible_origins": n_eval,
                "eligible_origins": list(eligible_origins),
                "note": "Prepared counts only — do not execute in prepare run.",
            },
        },
        COMPARATOR_HUMAN_XGB: {
            "status": HUMAN_XGB_STATUS,
            "n_jobs": 0,
            "executable": False,
            "note": (
                "Do not create executable jobs. Enabling Human+XGBoost requires "
                "older HB forecasts or an explicitly approved design that sacrifices "
                "part of the four protected HB origins for development."
            ),
        },
        COMPARATOR_INTEGRATED_DIAG: {
            "status": "DIAGNOSTIC_ONLY_NOT_EXECUTABLE",
            "n_jobs": 0,
            "executable": False,
        },
        "n_products_context": int(n_products),
        "total_executable_jobs": int(ts_resid_jobs),
    }


def run_prepare(
    *,
    repo_root: Path,
    out_dir: Path,
    lock_dir: Path,
    sales_path: Path,
    v21_export: Path,
    a0_export: Path,
    budget_universe_path: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_path = lock_dir / "ts_architecture_lock.json"
    contract_path = lock_dir / "ts_architecture_production_contract_v1.json"
    sidecar = contract_path.with_suffix(contract_path.suffix + ".sha256")

    lock_bytes = lock_path.read_bytes()
    contract_bytes = contract_path.read_bytes()
    lock = assert_lock_immutable_read(lock_path)
    contract = json.loads(contract_bytes.decode("utf-8"))
    validate_contract_against_lock(contract, lock_dir=lock_dir, repo_root=repo_root)
    if file_sha256(contract_path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("Production contract sidecar SHA-256 mismatch")

    for p in (lock_path, contract_path):
        try:
            refuse_lock_mutation(p)
        except PermissionError:
            pass

    assert_all_features_have_pit_rules()

    hb_audit = audit_hb_availability_budget_origins_only(budget_universe_path)
    _atomic_write_json(out_dir / "hb_availability_audit.json", hb_audit)

    sales = pd.read_parquet(sales_path)
    sales_sha = file_sha256(sales_path)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    panel_v21 = filter_non_hb(pd.read_parquet(v21_export / "forecasts_panel.parquet"))
    panel_a0 = filter_non_hb(pd.read_parquet(a0_export / "forecasts_panel.parquet"))

    status, _inc = audit_product_origin_completeness(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        original55=set(load_universe_product_names("mvp_products")),
    )
    common, deployable, _cats, partial = build_policy_panels(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        completeness_status=status,
    )
    naive_extra, _nofc = resolve_naive_deployable_partials(partial=partial, dense=dense)
    if not naive_extra.empty:
        deployable = pd.concat([deployable, naive_extra], ignore_index=True)

    residual_panel = deployable.loc[
        deployable["forecast_origin"].astype(int).isin(RESIDUAL_DEVELOPMENT_ORIGINS)
    ].copy()
    if residual_panel["forecast_origin"].astype(int).isin(HB_ORIGINS | POST_HB_ORIGINS).any():
        raise RuntimeError("HB/post-HB leaked into residual_development panel")

    # Ensure target_date column name
    if "target_date" not in residual_panel.columns and "date" in residual_panel.columns:
        residual_panel = residual_panel.rename(columns={"date": "target_date"})

    origin_roles = {
        "residual_development": sorted(RESIDUAL_DEVELOPMENT_ORIGINS),
        "human_budget_final_evaluation": sorted(HB_ORIGINS),
        "post_hb_ts_only_validation": sorted(POST_HB_ORIGINS),
        "rules": {
            "training_forecast_origin_lt_O": True,
            "training_target_date_le_O_minus_2": True,
            "feature_as_of_date_le_O_minus_2": True,
            "never_train_tune_select_on_hb_or_post_hb": True,
            "hb_outcomes_inspected": False,
            "random_row_split": False,
        },
        "minima": {
            "min_prior_matured_origins": MIN_PRIOR_MATURED_ORIGINS,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_train_sku_origins": MIN_TRAIN_SKU_ORIGINS,
        },
    }
    _atomic_write_json(out_dir / "origin_roles.json", origin_roles)

    oof = build_oof_split_manifest(residual_panel)
    _atomic_write_json(out_dir / "oof_split_manifest.json", oof)

    feature_schema = pit_feature_schema_manifest()
    feature_schema["pit_audit"] = {
        "all_features_have_as_of_rules": True,
        "n_features": len(TS_RESID_FEATURE_LIST),
        "feature_names": list(TS_RESID_FEATURE_LIST),
        "rules_registered": sorted(FEATURE_ASOF_RULES.keys()),
    }
    _atomic_write_json(out_dir / "feature_schema.json", feature_schema)

    n_products = int(residual_panel["product"].nunique()) if len(residual_panel) else 0
    job_plan = build_job_plan(
        eligible_origins=list(oof["eligible_evaluation_origins"]),
        n_products=n_products,
    )
    _atomic_write_json(out_dir / "job_plan.json", job_plan)

    leakage = {
        "hb_in_residual_panel": False,
        "post_hb_in_residual_panel": False,
        "pit_predicates_enforced": [
            "training_forecast_origin < O",
            "training_target_date <= O-2",
            "feature_as_of_date <= O-2",
        ],
        "all_clean_quant_features_have_pit_rules": True,
        "random_row_split": False,
        "hb_outcomes_inspected": False,
        "human_xgb_executable_jobs": 0,
        "lock_bytes_unchanged_check": "pending",
        "contract_bytes_unchanged_check": "pending",
    }

    try:
        bud_rel = budget_universe_path.resolve().relative_to(repo_root.resolve())
        bud_path_str = bud_rel.as_posix()
    except ValueError:
        bud_path_str = str(budget_universe_path)
    feature_source_hashes = {
        "v21_forecasts_panel": file_sha256(v21_export / "forecasts_panel.parquet"),
        "a0_forecasts_panel": file_sha256(a0_export / "forecasts_panel.parquet"),
        "budget_universe_path": bud_path_str,
    }

    resid_contract = build_residual_study_contract(
        lock_sha256=file_sha256(lock_path),
        contract_sha256=file_sha256(contract_path),
        sales_sha256=sales_sha,
        feature_source_hashes=feature_source_hashes,
        eligible_evaluation_origins=list(oof["eligible_evaluation_origins"]),
        oof_split_manifest=oof,
    )
    write_immutable_json(out_dir / "residual_study_contract_v1.json", resid_contract)

    deployment_note = {
        "production_contract_dirty_tree": bool(
            (contract.get("git") or {}).get("dirty_tree")
        ),
        "research_still_valid": True,
        "before_deployment_require_additive_release_manifest": {
            "source_tree_hash": True,
            "python_environment": True,
            "tensorflow_version": True,
            "xgboost_version": True,
            "deterministic_settings": True,
        },
        "note": (
            "dirty_tree: true on the production contract does not invalidate research, "
            "but deployment needs an additive release manifest hashing the exact source "
            "tree, Python environment, TensorFlow/XGBoost versions, and deterministic settings."
        ),
    }
    _atomic_write_json(out_dir / "deployment_note.json", deployment_note)

    if lock_path.read_bytes() != lock_bytes:
        raise RuntimeError("Scientific lock mutated during residual prepare — FAIL")
    if contract_path.read_bytes() != contract_bytes:
        raise RuntimeError("Production contract mutated during residual prepare — FAIL")
    leakage["lock_bytes_unchanged_check"] = "pass"
    leakage["contract_bytes_unchanged_check"] = "pass"
    _atomic_write_json(out_dir / "leakage_checklist.json", leakage)

    summary = {
        "family": FAMILY,
        "hb_outcomes_inspected": False,
        "human_xgb_status": HUMAN_XGB_STATUS,
        "origin_class_counts": {
            k: len(v) for k, v in (oof.get("by_class") or {}).items()
        },
        "eligible_evaluation_origins": list(oof["eligible_evaluation_origins"]),
        "job_plan_total_executable_jobs": job_plan["total_executable_jobs"],
        "human_xgb_jobs": 0,
        "residual_study_contract": str(
            (out_dir / "residual_study_contract_v1.json").as_posix()
        ),
        "lock_sha256": resid_contract["ts_architecture"]["lock_sha256"],
        "production_contract_sha256": resid_contract["ts_architecture"][
            "production_contract_sha256"
        ],
        "xgb_params_hash": resid_contract["xgb_params_hash"],
        "feature_list_hash": resid_contract["feature_list_hash"],
        "scientific_lock_choice": lock.get("choice"),
        "prepared_only": True,
        "training_executed": False,
    }
    _atomic_write_json(out_dir / "prepare_summary.json", summary)
    return summary
