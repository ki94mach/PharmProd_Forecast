"""Freeze residual pipeline decision from evaluation_v1/ (no refit, no HB).

Creates fail-if-exists ``residual_pipeline_lock_v1.json`` + ``.sha256`` sidecar.
"""

from __future__ import annotations

import csv
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
from pkg.research.portfolio_residual_xgb_v1.constants import (
    FAMILY,
    HUMAN_XGB_STATUS,
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
)
from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import (
    EVAL_DIR_NAME,
    EXPECTED_ORIGINS,
    FAMILY_DIR_REL,
    LOCK_DIR_REL,
    REC_INSUFFICIENT,
    REC_KEEP,
    REC_PROMOTE,
)

LOCK_FILENAME = "residual_pipeline_lock_v1.json"
LOCK_VERSION = "residual_pipeline_lock_v1"

PIPELINE_FROZEN_TS = "FROZEN_TS"
PIPELINE_RESIDUAL_XGB = "TS_PLUS_RESIDUAL_XGB"


class FreezeAbortError(RuntimeError):
    """Fail-fast freeze abort (integrity / gates / hashes)."""


def decide_pipeline_from_gates(
    *,
    gates_result: dict[str, Any],
    non_computable_gates: list[str],
) -> tuple[str, str | None]:
    """Return (decision_code, production_pipeline_or_None).

    ``INSUFFICIENT_EVIDENCE`` yields no production choice (None).
    """
    if non_computable_gates or any(v is None for v in gates_result.values()):
        return REC_INSUFFICIENT, None
    vals = list(gates_result.values())
    if not vals:
        return REC_INSUFFICIENT, None
    if all(bool(v) for v in vals):
        return REC_PROMOTE, PIPELINE_RESIDUAL_XGB
    # Any computable gate fails
    return REC_KEEP, PIPELINE_FROZEN_TS


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics_overall(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            out[method] = {
                k: float(row[k])
                for k in row
                if k != "method" and row[k] not in ("", None)
            }
    return out


def snapshot_upstream_hashes(
    *,
    family_dir: Path,
    lock_dir: Path,
    eval_dir: Path,
) -> dict[str, str]:
    """Byte hashes of upstream artifacts that must remain unchanged by freeze."""
    paths = {
        "residual_study_contract_v1_1.json": family_dir
        / "residual_study_contract_v1_1.json",
        "residual_promotion_gates_v1.json": family_dir
        / "residual_promotion_gates_v1.json",
        "prepare_summary_v1_1.json": family_dir / "prepare_summary_v1_1.json",
        "ts_architecture_lock.json": lock_dir / "ts_architecture_lock.json",
        "ts_architecture_production_contract_v1.json": lock_dir
        / "ts_architecture_production_contract_v1.json",
        "integrity_report.json": eval_dir / "integrity_report.json",
        "gates_result.json": eval_dir / "gates_result.json",
        "recommendation_preview.json": eval_dir / "recommendation_preview.json",
        "metrics_overall.csv": eval_dir / "metrics_overall.csv",
        "oof_panel.parquet": eval_dir / "oof_panel.parquet",
    }
    out: dict[str, str] = {}
    for name, p in paths.items():
        if not p.is_file():
            raise FreezeAbortError(f"Missing upstream file for hash snapshot: {p}")
        out[name] = file_sha256(p)
    return out


def verify_upstream_unchanged(
    before: dict[str, str],
    *,
    family_dir: Path,
    lock_dir: Path,
    eval_dir: Path,
) -> None:
    after = snapshot_upstream_hashes(
        family_dir=family_dir, lock_dir=lock_dir, eval_dir=eval_dir
    )
    if before != after:
        diffs = {
            k: {"before": before.get(k), "after": after.get(k)}
            for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)
        }
        raise FreezeAbortError(
            f"Upstream files changed during freeze (must stay byte-identical): {diffs}"
        )


def validate_evaluation_for_freeze(eval_dir: Path, family_dir: Path) -> dict[str, Any]:
    """Fail-fast checks against evaluation_v1/ + live hashes vs prepare/eval."""
    integrity_path = eval_dir / "integrity_report.json"
    gates_path = eval_dir / "gates_result.json"
    preview_path = eval_dir / "recommendation_preview.json"
    metrics_path = eval_dir / "metrics_overall.csv"

    for p in (integrity_path, gates_path, preview_path, metrics_path):
        if not p.is_file():
            raise FreezeAbortError(f"Missing evaluation artifact: {p}")

    integrity = _load_json(integrity_path)
    if not integrity.get("ok"):
        raise FreezeAbortError("integrity_report.ok is not true")

    # Complete horizons must have passed (never silently exclude)
    hz_ok = False
    for c in integrity.get("checks", []):
        if c.get("check") == "complete_horizons_1_to_15":
            hz_ok = bool(c.get("pass"))
            break
    if not hz_ok:
        raise FreezeAbortError(
            "complete_horizons_1_to_15 did not pass — refuse freeze "
            "(never silently exclude incomplete groups)"
        )

    # Live hash checks from integrity detail must still match disk
    prepare = _load_json(family_dir / "prepare_summary_v1_1.json")
    for check_name, path, prep_key in (
        (
            "contract_sha_matches_prepare",
            family_dir / "residual_study_contract_v1_1.json",
            "residual_study_contract_v1_1_sha256",
        ),
        (
            "gates_sha_matches_prepare",
            family_dir / "residual_promotion_gates_v1.json",
            "promotion_gates_sha256",
        ),
    ):
        live = file_sha256(path)
        expected = prepare.get(prep_key)
        if live != expected:
            raise FreezeAbortError(
                f"Hash mismatch for {path.name}: live={live} expected={expected}"
            )
        # Also confirm integrity recorded a pass for this check
        found = next(
            (c for c in integrity.get("checks", []) if c.get("check") == check_name),
            None,
        )
        if found is None or not found.get("pass"):
            raise FreezeAbortError(f"Integrity check {check_name} missing or failed")

    gates_doc = _load_json(gates_path)
    gates_result = gates_doc.get("gates_result") or {}
    non_computable = list(gates_doc.get("non_computable_gates") or [])
    if any(v is None for v in gates_result.values()):
        for k, v in gates_result.items():
            if v is None and k not in non_computable:
                non_computable.append(k)

    preview = _load_json(preview_path)
    if preview.get("hb_outcomes_inspected") is not False:
        raise FreezeAbortError("recommendation_preview.hb_outcomes_inspected must be false")
    if gates_doc.get("hb_outcomes_inspected") is not False:
        raise FreezeAbortError("gates_result.hb_outcomes_inspected must be false")

    metrics = _read_metrics_overall(metrics_path)
    for m in (METHOD_FROZEN_TS, METHOD_MEDIAN, METHOD_XGB):
        if m not in metrics:
            raise FreezeAbortError(f"metrics_overall missing method {m}")

    return {
        "integrity": integrity,
        "gates_doc": gates_doc,
        "gates_result": gates_result,
        "non_computable_gates": non_computable,
        "preview": preview,
        "metrics_overall": metrics,
        "prepare_summary": prepare,
        "evaluation_dir_sha256": {
            "integrity_report.json": file_sha256(integrity_path),
            "gates_result.json": file_sha256(gates_path),
            "recommendation_preview.json": file_sha256(preview_path),
            "metrics_overall.csv": file_sha256(metrics_path),
            "oof_panel.parquet": file_sha256(eval_dir / "oof_panel.parquet"),
        },
    }


def build_pipeline_lock_payload(
    *,
    repo_root: Path,
    family_dir: Path,
    lock_dir: Path,
    eval_dir: Path,
    validated: dict[str, Any],
    decision: str,
    production_pipeline: str,
    upstream_hashes_before: dict[str, str],
) -> dict[str, Any]:
    prepare = validated["prepare_summary"]
    gates_doc = validated["gates_doc"]
    gates_result = validated["gates_result"]
    metrics = validated["metrics_overall"]
    preview = validated["preview"]

    failed_gates = sorted(k for k, v in gates_result.items() if v is False)
    passed_gates = sorted(k for k, v in gates_result.items() if v is True)

    rationale = [
        f"Decision={decision}; production_pipeline={production_pipeline}.",
        f"Pooled WMAPE lift (TS−XGB)={gates_doc.get('lift_wmape_pp_xgb_vs_ts')} pp.",
        f"Origins improved={gates_doc.get('n_origins_improved')}/7.",
        f"Passed gates: {passed_gates}.",
        f"Failed gates: {failed_gates}.",
        f"{METHOD_MEDIAN} remains benchmark-only (not selectable).",
        "HB outcomes not inspected; freeze uses evaluation_v1 only.",
    ]

    return {
        "lock_version": LOCK_VERSION,
        "family": FAMILY,
        "locked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hb_outcomes_inspected": False,
        "HUMAN_XGB": HUMAN_XGB_STATUS,
        "decision": decision,
        "production_pipeline": production_pipeline,
        "selected_pipeline": production_pipeline,
        "median_benchmark_only": True,
        "median_method": METHOD_MEDIAN,
        "not_selected": [METHOD_MEDIAN, METHOD_XGB]
        if production_pipeline == PIPELINE_FROZEN_TS
        else [METHOD_MEDIAN, METHOD_FROZEN_TS],
        "eligible_evaluation_origins": list(EXPECTED_ORIGINS),
        "promotion_gates_pass": bool(gates_doc.get("promotion_gates_pass")),
        "gates_result": dict(gates_result),
        "non_computable_gates": list(validated["non_computable_gates"]),
        "exact_metrics": {
            "lift_wmape_pp_xgb_vs_ts": gates_doc.get("lift_wmape_pp_xgb_vs_ts"),
            "n_origins_improved": gates_doc.get("n_origins_improved"),
            "by_origin": gates_doc.get("by_origin"),
            "horizon_group_lifts": gates_doc.get("horizon_group_lifts"),
            "sku_origin_wins_summary": gates_doc.get("sku_origin_wins_summary"),
            "methods": metrics,
        },
        "evaluation_recommendation_preview": preview.get("recommendation"),
        "hashes": {
            "evaluation_v1": validated["evaluation_dir_sha256"],
            "residual_study_contract_v1_1_sha256": prepare.get(
                "residual_study_contract_v1_1_sha256"
            ),
            "promotion_gates_sha256": prepare.get("promotion_gates_sha256"),
            "lock_sha256": prepare.get("lock_sha256"),
            "production_contract_sha256": prepare.get("production_contract_sha256"),
            "feature_list_hash": prepare.get("feature_list_hash"),
            "feature_semantics_hash": prepare.get("feature_semantics_hash"),
            "xgb_params_hash": prepare.get("xgb_params_hash"),
            "upstream_byte_snapshot_before_freeze": upstream_hashes_before,
        },
        "paths": {
            "evaluation_v1": (eval_dir.relative_to(repo_root).as_posix()),
            "family": (family_dir.relative_to(repo_root).as_posix()),
            "ts_architecture_lock": (
                (lock_dir / "ts_architecture_lock.json").relative_to(repo_root).as_posix()
            ),
        },
        "git": git_status(repo_root),
        "rationale_bullets": rationale,
        "process_rules": {
            "no_refit": True,
            "no_hb_outcomes": True,
            "fail_if_exists": True,
            "never_silently_exclude_incomplete_horizon_groups": True,
            "median_not_selectable": True,
        },
    }


def freeze_residual_pipeline_lock(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate evaluation_v1 and atomically create residual_pipeline_lock_v1.json."""
    repo_root = Path(repo_root).resolve()
    family_dir = repo_root / FAMILY_DIR_REL
    lock_dir = repo_root / LOCK_DIR_REL
    eval_dir = family_dir / EVAL_DIR_NAME
    out_path = family_dir / LOCK_FILENAME

    if out_path.exists() or out_path.with_suffix(out_path.suffix + ".sha256").exists():
        raise LockExistsError(f"Immutable pipeline lock already exists: {out_path}")

    validated = validate_evaluation_for_freeze(eval_dir, family_dir)
    decision, production_pipeline = decide_pipeline_from_gates(
        gates_result=validated["gates_result"],
        non_computable_gates=validated["non_computable_gates"],
    )

    if decision == REC_INSUFFICIENT or production_pipeline is None:
        raise FreezeAbortError(
            "INSUFFICIENT_EVIDENCE: non-computable required gate(s) "
            f"{validated['non_computable_gates']}; write no production choice / no lock"
        )

    upstream_before = snapshot_upstream_hashes(
        family_dir=family_dir, lock_dir=lock_dir, eval_dir=eval_dir
    )

    payload = build_pipeline_lock_payload(
        repo_root=repo_root,
        family_dir=family_dir,
        lock_dir=lock_dir,
        eval_dir=eval_dir,
        validated=validated,
        decision=decision,
        production_pipeline=production_pipeline,
        upstream_hashes_before=upstream_before,
    )

    digest = write_immutable_json(out_path, payload)

    verify_upstream_unchanged(
        upstream_before,
        family_dir=family_dir,
        lock_dir=lock_dir,
        eval_dir=eval_dir,
    )

    # Confirm sidecar and content
    sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
    if file_sha256(out_path) != digest:
        raise FreezeAbortError("Lock content hash mismatch after write")
    if sidecar.read_text(encoding="utf-8").strip() != digest:
        raise FreezeAbortError("Sidecar SHA-256 mismatch")

    return {
        "ok": True,
        "lock_path": str(out_path.as_posix()),
        "lock_sha256": digest,
        "decision": decision,
        "production_pipeline": production_pipeline,
        "hb_outcomes_inspected": False,
        "upstream_unchanged": True,
    }
