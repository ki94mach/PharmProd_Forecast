"""Immutable companion reproducibility manifest for residual_pipeline_lock_v1.

Does not rewrite the pipeline lock. Fail-if-exists write + SHA-256 sidecar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.portfolio_ts_lock.production_contract import (
    file_sha256,
    git_status,
    write_immutable_json,
)
from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import (
    EVAL_DIR_NAME,
    EXPECTED_ORIGINS,
    FAMILY_DIR_REL,
    FULL_RUN_ID,
    LOCK_DIR_REL,
)
from pkg.research.portfolio_residual_xgb_v1.pipeline_lock import LOCK_FILENAME

MANIFEST_FILENAME = "residual_pipeline_lock_v1_reproducibility_manifest.json"
MANIFEST_VERSION = "residual_pipeline_lock_v1_reproducibility_manifest"

# Gate-supporting evaluation_v1 artifacts (bootstrap / origins / horizons / drop-top3 / gates).
GATE_SUPPORTING_EVAL_ARTIFACTS: tuple[str, ...] = (
    "gates_result.json",
    "integrity_report.json",
    "recommendation_preview.json",
    "metrics_overall.csv",
    "metrics_by_origin.csv",
    "metrics_by_horizon_group.csv",
    "robustness_sku_bootstrap.json",
    "robustness_hierarchical_bootstrap.json",
    "robustness_loo_origin.csv",
    "robustness_loo_sku.csv",
    "robustness_drop_top3.json",
    "sku_origin_wins.csv",
    "oof_panel.parquet",
)

# Evaluation entrypoint + helpers used to produce evaluation_v1 / freeze.
CODE_MODULES: tuple[str, ...] = (
    "scripts/analyze_portfolio_residual_xgb_v1_evaluation.py",
    "scripts/freeze_portfolio_residual_pipeline_lock_v1.py",
    "src/pkg/research/portfolio_residual_xgb_v1/evaluation_v1.py",
    "src/pkg/research/portfolio_residual_xgb_v1/pipeline_lock.py",
    "src/pkg/research/portfolio_residual_xgb_v1/evaluate_oof.py",
    "src/pkg/research/portfolio_residual_xgb_v1/methods.py",
    "src/pkg/research/portfolio_residual_xgb_v1/promotion_gates.py",
    "src/pkg/portfolio_ts_lock/bootstrap.py",
    "src/pkg/portfolio_ts_lock/freeze.py",
    "src/pkg/portfolio_ts_lock/production_contract.py",
)


def _hash_file(repo_root: Path, rel: str) -> dict[str, str]:
    path = repo_root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": rel.replace("\\", "/"),
        "sha256": file_sha256(path),
    }


def build_reproducibility_manifest(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    family_dir = repo_root / FAMILY_DIR_REL
    lock_dir = repo_root / LOCK_DIR_REL
    eval_dir = family_dir / EVAL_DIR_NAME
    run_dir = family_dir / "runs" / FULL_RUN_ID
    pipeline_lock = family_dir / LOCK_FILENAME
    pipeline_sidecar = pipeline_lock.with_suffix(pipeline_lock.suffix + ".sha256")
    prod_contract_path = lock_dir / "ts_architecture_production_contract_v1.json"

    if not pipeline_lock.is_file():
        raise FileNotFoundError(f"Pipeline lock missing: {pipeline_lock}")
    if not pipeline_sidecar.is_file():
        raise FileNotFoundError(f"Pipeline lock sidecar missing: {pipeline_sidecar}")

    lock_sha = file_sha256(pipeline_lock)
    sidecar_sha = pipeline_sidecar.read_text(encoding="utf-8").strip()
    if lock_sha != sidecar_sha:
        raise RuntimeError(
            f"Pipeline lock SHA mismatch: file={lock_sha} sidecar={sidecar_sha}"
        )

    prod = json.loads(prod_contract_path.read_text(encoding="utf-8"))
    policy = prod.get("policy_semantics") or {}

    oof_predictions: dict[str, str] = {}
    oof_meta: dict[str, str] = {}
    for o in EXPECTED_ORIGINS:
        pred = run_dir / f"oof_predictions_origin_{o}.parquet"
        meta = run_dir / f"oof_meta_origin_{o}.json"
        if not pred.is_file() or not meta.is_file():
            raise FileNotFoundError(f"Missing OOF artifacts for origin {o}")
        oof_predictions[pred.name] = file_sha256(pred)
        oof_meta[meta.name] = file_sha256(meta)

    gate_artifacts: dict[str, str] = {}
    for name in GATE_SUPPORTING_EVAL_ARTIFACTS:
        p = eval_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing gate-supporting eval artifact: {p}")
        gate_artifacts[name] = file_sha256(p)

    code_hashes: dict[str, str] = {}
    for rel in CODE_MODULES:
        h = _hash_file(repo_root, rel)
        code_hashes[h["relative_path"]] = h["sha256"]

    git = git_status(repo_root)

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hb_outcomes_inspected": False,
        "companion_to": {
            "artifact": LOCK_FILENAME,
            "relative_path": (pipeline_lock.relative_to(repo_root).as_posix()),
            "sha256": lock_sha,
            "sidecar_sha256": sidecar_sha,
            "note": "This manifest does not rewrite the pipeline lock; it is additive.",
        },
        "frozen_ts_definition": {
            "FROZEN_TS": "A0_HYBRID_POLICY",
            "equals": "A0_HYBRID_POLICY",
            "hybrid_weights": {"v2.1": 0.5, "a0": 0.5},
            "policy_semantics": {
                "both_native": policy.get(
                    "both_native", "0.5*V2.1 + 0.5*A0 seed mean"
                ),
                "v2_only": policy.get("v2_only", "V2.1"),
                "a0_only": policy.get("a0_only", "complete A0 seed mean"),
                "neither": policy.get("neither", "production naive_last_value"),
            },
            "production_contract_relative_path": (
                prod_contract_path.relative_to(repo_root).as_posix()
            ),
            "production_contract_sha256": file_sha256(prod_contract_path),
            "ts_architecture_lock_sha256": file_sha256(
                lock_dir / "ts_architecture_lock.json"
            ),
            "explanation": (
                "In residual-XGBoost evaluation, method column FROZEN_TS is the locked "
                "A0_HYBRID_POLICY forecast (f_a0_hybrid anchor): when both V2.1 and A0 "
                "natives exist, use 0.5*V2.1 + 0.5*A0 seed mean; otherwise apply the "
                "fallback semantics above (v2_only / a0_only / neither)."
            ),
        },
        "sku_origin_wins_field_glossary": {
            "note": (
                "sku_origin_wins_summary inherits portfolio_ts_lock helper field names "
                "that are misleading in the residual study context."
            ),
            "hybrid_wins": {
                "means": "TS_PLUS_RESIDUAL_XGB wins",
                "do_not_read_as": "A0 hybrid policy wins",
            },
            "v2_wins": {
                "means": "FROZEN_TS (A0_HYBRID_POLICY) wins",
                "do_not_read_as": "V2.1-only policy wins",
            },
            "ties": "mean AE within TIE_AE_EPS",
            "comparison": (
                "Per (product, forecast_origin), lower mean AE wins between "
                "FROZEN_TS and TS_PLUS_RESIDUAL_XGB after mapping helper columns "
                "f_v2_policy:=FROZEN_TS and f_a0_hybrid:=TS_PLUS_RESIDUAL_XGB on a copy."
            ),
        },
        "raw_oof_prediction_parquet_sha256": oof_predictions,
        "raw_oof_meta_json_sha256": oof_meta,
        "n_oof_prediction_files": len(oof_predictions),
        "n_oof_meta_files": len(oof_meta),
        "gate_supporting_evaluation_artifact_sha256": gate_artifacts,
        "evaluation_code_sha256": code_hashes,
        "git": {
            **git,
            "dirty_tree_note": (
                "A dirty Git tree is a deployment/release concern. It does not "
                "invalidate this research decision. Address dirty-tree hygiene in a "
                "later release manifest; do not reopen residual promotion gates or HB."
            ),
        },
        "process_rules": {
            "does_not_rewrite_pipeline_lock": True,
            "fail_if_exists": True,
            "hb_outcomes_inspected": False,
            "no_refit": True,
        },
    }


def write_reproducibility_manifest(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    family_dir = repo_root / FAMILY_DIR_REL
    out_path = family_dir / MANIFEST_FILENAME
    if out_path.exists() or out_path.with_suffix(out_path.suffix + ".sha256").exists():
        raise LockExistsError(f"Immutable companion manifest already exists: {out_path}")

    payload = build_reproducibility_manifest(repo_root=repo_root)
    # Confirm referenced lock still byte-identical to companion_to.sha256 after build
    lock_path = family_dir / LOCK_FILENAME
    if file_sha256(lock_path) != payload["companion_to"]["sha256"]:
        raise RuntimeError("Pipeline lock changed while building companion manifest")

    digest = write_immutable_json(out_path, payload)
    if file_sha256(lock_path) != payload["companion_to"]["sha256"]:
        raise RuntimeError("Pipeline lock mutated during companion manifest write")

    return {
        "ok": True,
        "manifest_path": out_path.as_posix(),
        "manifest_sha256": digest,
        "residual_pipeline_lock_v1_sha256": payload["companion_to"]["sha256"],
        "hb_outcomes_inspected": False,
    }
