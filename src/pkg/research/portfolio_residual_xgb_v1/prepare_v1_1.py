"""Prepare residual-XGBoost v1.1 artifacts without mutating v1 or TS lock files."""

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
    DETERMINISTIC_SEED,
    FAMILY,
    HUMAN_XGB_STATUS,
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
    MIN_PRIOR_MATURED_ORIGINS,
    MIN_TRAIN_ROWS,
    MIN_TRAIN_SKU_ORIGINS,
)
from pkg.research.portfolio_residual_xgb_v1.contract_v1_1 import (
    build_residual_study_contract_v1_1,
    feature_list_hash,
    feature_semantics_hash,
    write_immutable_json,
    xgb_params_hash,
)
from pkg.research.portfolio_residual_xgb_v1.methods import EXECUTABLE_METHODS, method_registry
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    FEATURE_ASOF_RULES,
    TS_RESID_FEATURE_LIST,
    assert_all_features_have_pit_rules,
    assert_no_sales_uses_origin_minus_1,
    pit_feature_schema_manifest,
)
from pkg.research.portfolio_residual_xgb_v1.prepare import (
    audit_hb_availability_budget_origins_only,
)
from pkg.research.portfolio_residual_xgb_v1.promotion_gates import (
    build_promotion_gate_artifact,
)
from pkg.research.portfolio_residual_xgb_v1.splits import build_oof_split_manifest


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Overwrite allowed for v1_1 rebuildable manifests (not immutable contracts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def ensure_immutable_json(path: Path, payload: dict[str, Any]) -> tuple[str, bool]:
    """Create-only write, or reuse existing file if already present.

    Returns (sha256, created_new).
    When the file already exists (e.g. copied to VPS), do not overwrite;
    verify/create sidecar hash and return the existing content hash.
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_file():
        sha = file_sha256(path)
        if sidecar.is_file():
            expected = sidecar.read_text(encoding="utf-8").strip()
            if expected != sha:
                raise RuntimeError(
                    f"Immutable artifact sidecar mismatch: {path} "
                    f"(file={sha} sidecar={expected})"
                )
        else:
            sidecar.write_text(sha + "\n", encoding="utf-8")
        return sha, False
    sha = write_immutable_json(path, payload)
    sidecar.write_text(sha + "\n", encoding="utf-8")
    return sha, True


def build_job_inventory_v1_1(*, eligible_origins: list[int]) -> dict[str, Any]:
    origins = [int(o) for o in eligible_origins]
    smoke_origin = 140107
    if smoke_origin not in origins:
        raise RuntimeError(f"Smoke origin {smoke_origin} not in eligible set {origins}")
    full_jobs = []
    for o in origins:
        full_jobs.append(
            {
                "evaluation_origin": o,
                "methods": list(EXECUTABLE_METHODS),
                "xgb_fit": True,
                "outer_workers": 1,
            }
        )
    return {
        "deterministic_seed": DETERMINISTIC_SEED,
        "outer_workers": 1,
        "note": "One outer worker; XGBoost uses n_jobs=-1 to avoid CPU oversubscription.",
        "HUMAN_XGB": {"status": HUMAN_XGB_STATUS, "n_jobs": 0, "executable": False},
        "methods": list(EXECUTABLE_METHODS),
        "smoke": {
            "run_id": "portfolio_residual_xgb_v1_1_smoke",
            "origins": [smoke_origin],
            "n_origin_jobs": 1,
            "n_xgb_fits": 1,
        },
        "full": {
            "run_id": "portfolio_residual_xgb_v1_1_full",
            "origins": origins,
            "n_origin_jobs": len(origins),
            "n_xgb_fits": len(origins),
            "jobs": full_jobs,
        },
        "per_origin_methods": {
            METHOD_FROZEN_TS: "score only",
            METHOD_MEDIAN: "train-median residual from PIT train rows",
            METHOD_XGB: "fit XGB on PIT train rows; predict test origin",
        },
    }


def hard_fail_forbidden_origins(
    frame: pd.DataFrame,
    *,
    context: str,
    forbidden: frozenset[int],
) -> None:
    fo = frame["forecast_origin"].astype(int)
    bad = sorted(set(fo) & set(forbidden))
    if bad:
        raise RuntimeError(f"Forbidden origins in {context}: {bad}")


def hard_fail_hb(frame: pd.DataFrame, *, context: str) -> None:
    """HB holdouts must never appear with residual training/eval actuals paths."""
    hard_fail_forbidden_origins(frame, context=context, forbidden=HB_ORIGINS)


def hard_fail_hb_post_hb(frame: pd.DataFrame, *, context: str) -> None:
    """Residual train/tune/eval must exclude HB and post-HB."""
    hard_fail_forbidden_origins(
        frame, context=context, forbidden=HB_ORIGINS | POST_HB_ORIGINS
    )


def run_prepare_v1_1(
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

    # Do not touch existing v1 prepare files; only write v1_1-named artifacts.
    v1_contract_path = out_dir / "residual_study_contract_v1.json"
    if not v1_contract_path.is_file():
        raise FileNotFoundError(
            f"Expected existing v1 contract (unchanged reference): {v1_contract_path}"
        )
    v1_sha = file_sha256(v1_contract_path)
    v1_bytes = v1_contract_path.read_bytes()

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
    for o in sorted(RESIDUAL_DEVELOPMENT_ORIGINS):
        assert_no_sales_uses_origin_minus_1(o)

    hb_audit = audit_hb_availability_budget_origins_only(budget_universe_path)
    _atomic_write_json(out_dir / "hb_availability_audit_v1_1.json", hb_audit)

    sales = pd.read_parquet(sales_path)
    sales_sha = file_sha256(sales_path)
    dense = build_dense_actuals(sales, through_date=LOCKED_SCORE_END)
    panel_v21 = filter_non_hb(pd.read_parquet(v21_export / "forecasts_panel.parquet"))
    panel_a0 = filter_non_hb(pd.read_parquet(a0_export / "forecasts_panel.parquet"))
    hard_fail_hb(panel_v21, context="v21 panel")
    hard_fail_hb(panel_a0, context="a0 panel")

    status, _inc = audit_product_origin_completeness(
        panel_v21=panel_v21,
        panel_a0=panel_a0,
        dense=dense,
        original55=set(load_universe_product_names("mvp_products")),
    )
    _common, deployable, _cats, partial = build_policy_panels(
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
    hard_fail_hb_post_hb(residual_panel, context="residual_development panel")
    if "target_date" not in residual_panel.columns and "date" in residual_panel.columns:
        residual_panel = residual_panel.rename(columns={"date": "target_date"})

    oof = build_oof_split_manifest(residual_panel)
    _atomic_write_json(out_dir / "oof_split_manifest_v1_1.json", oof)
    eligible = list(oof["eligible_evaluation_origins"])
    if len(eligible) != 7:
        raise RuntimeError(
            f"Expected 7 EVALUABLE origins after corrected PIT; got {eligible}"
        )

    feature_schema = pit_feature_schema_manifest()
    feature_schema["pit_audit"] = {
        "all_features_have_as_of_rules": True,
        "n_features": len(TS_RESID_FEATURE_LIST),
        "sales_never_uses_origin_minus_1": True,
        "last_complete_month": "forecast_origin-2",
        "rules_registered": sorted(FEATURE_ASOF_RULES.keys()),
    }
    _atomic_write_json(out_dir / "feature_schema_v1_1.json", feature_schema)

    inventory = build_job_inventory_v1_1(eligible_origins=eligible)
    _atomic_write_json(out_dir / "job_inventory_v1_1.json", inventory)
    _atomic_write_json(out_dir / "methods_registry_v1_1.json", method_registry())

    origin_roles = {
        "residual_development": sorted(RESIDUAL_DEVELOPMENT_ORIGINS),
        "human_budget_final_evaluation": sorted(HB_ORIGINS),
        "post_hb_ts_only_validation": sorted(POST_HB_ORIGINS),
        "rules": {
            "training_forecast_origin_lt_evaluation_origin": True,
            "training_target_date_le_evaluation_origin_minus_2": True,
            "every_sales_feature_date_le_evaluation_origin_minus_2": True,
            "last_complete_month": "forecast_origin-2",
            "categorical_fit_on_train_only": True,
            "hb_outcomes_inspected": False,
            "HUMAN_XGB": HUMAN_XGB_STATUS,
            "random_row_split": False,
        },
        "minima": {
            "min_prior_matured_origins": MIN_PRIOR_MATURED_ORIGINS,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_train_sku_origins": MIN_TRAIN_SKU_ORIGINS,
        },
        "eligible_evaluation_origins": eligible,
    }
    _atomic_write_json(out_dir / "origin_roles_v1_1.json", origin_roles)

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

    resid_contract = build_residual_study_contract_v1_1(
        superseded_v1_sha256=v1_sha,
        lock_sha256=file_sha256(lock_path),
        contract_sha256=file_sha256(contract_path),
        sales_sha256=sales_sha,
        feature_source_hashes=feature_source_hashes,
        eligible_evaluation_origins=eligible,
        oof_split_manifest=oof,
    )
    contract_v11_path = out_dir / "residual_study_contract_v1_1.json"
    contract_v11_sha, contract_created = ensure_immutable_json(
        contract_v11_path, resid_contract
    )

    gates = build_promotion_gate_artifact(
        contract_v1_1_sha256=contract_v11_sha,
        eligible_evaluation_origins=eligible,
    )
    gates_path = out_dir / "residual_promotion_gates_v1.json"
    gates_sha, gates_created = ensure_immutable_json(gates_path, gates)

    leakage = {
        "version": "v1_1",
        "hb_in_residual_panel": False,
        "post_hb_in_residual_panel": False,
        "pit_predicates_enforced": [
            "training_forecast_origin < evaluation_origin",
            "training_target_date <= evaluation_origin-2",
            "every sales feature date <= evaluation_origin-2",
        ],
        "sales_never_uses_origin_minus_1": True,
        "last_complete_month": "forecast_origin-2",
        "all_clean_quant_features_have_pit_rules": True,
        "categorical_fit_on_train_only": True,
        "random_row_split": False,
        "hb_outcomes_inspected": False,
        "human_xgb_executable_jobs": 0,
        "v1_contract_bytes_unchanged": "pending",
        "lock_bytes_unchanged_check": "pending",
        "production_contract_bytes_unchanged_check": "pending",
    }

    if v1_contract_path.read_bytes() != v1_bytes:
        raise RuntimeError("v1 residual contract mutated — FAIL")
    if lock_path.read_bytes() != lock_bytes:
        raise RuntimeError("Scientific lock mutated — FAIL")
    if contract_path.read_bytes() != contract_bytes:
        raise RuntimeError("Production contract mutated — FAIL")
    leakage["v1_contract_bytes_unchanged"] = "pass"
    leakage["lock_bytes_unchanged_check"] = "pass"
    leakage["production_contract_bytes_unchanged_check"] = "pass"
    _atomic_write_json(out_dir / "leakage_checklist_v1_1.json", leakage)

    summary = {
        "family": FAMILY,
        "contract_version": "residual_study_contract_v1_1",
        "superseded_v1_sha256": v1_sha,
        "residual_study_contract_v1_1_sha256": contract_v11_sha,
        "promotion_gates_sha256": gates_sha,
        "feature_list_hash": feature_list_hash(),
        "feature_semantics_hash": feature_semantics_hash(),
        "xgb_params_hash": xgb_params_hash(),
        "lock_sha256": resid_contract["ts_architecture"]["lock_sha256"],
        "production_contract_sha256": resid_contract["ts_architecture"][
            "production_contract_sha256"
        ],
        "eligible_evaluation_origins": eligible,
        "origin_class_counts": {k: len(v) for k, v in oof["by_class"].items()},
        "job_inventory": {
            "smoke_n_xgb_fits": inventory["smoke"]["n_xgb_fits"],
            "full_n_xgb_fits": inventory["full"]["n_xgb_fits"],
            "outer_workers_note": "Configured per smoke/full YAML (smoke=1; full=4 on 16-core VPS)",
            "HUMAN_XGB_jobs": 0,
        },
        "immutable_artifacts": {
            "residual_study_contract_v1_1_created": contract_created,
            "residual_promotion_gates_v1_created": gates_created,
        },
        "methods": list(EXECUTABLE_METHODS),
        "hb_outcomes_inspected": False,
        "human_xgb_status": HUMAN_XGB_STATUS,
        "scientific_lock_choice": lock.get("choice"),
        "prepared_only": True,
        "training_executed": False,
    }
    _atomic_write_json(out_dir / "prepare_summary_v1_1.json", summary)
    return summary
