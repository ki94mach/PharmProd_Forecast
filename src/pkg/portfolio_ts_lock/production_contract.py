"""Immutable production contract I/O for portfolio TS architecture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pkg.portfolio_ts_lock.lock_io import LockExistsError

REQUIRED_CONTRACT_FIELDS = frozenset(
    {
        "contract_version",
        "created_at",
        "scientific_lock",
        "frozen_choice",
        "policy_semantics",
        "a0_engine",
        "cutoff",
        "horizon",
        "forecast_constraints",
        "sales",
        "sources",
        "git",
        "gate_thresholds",
        "metric_snapshot",
        "hb_origins_excluded",
        "post_hb_must_not_change_lock",
        "known_risks",
        "context_not_decision_thresholds",
    }
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomic create-only write; also write ``{name}.sha256`` sidecar. Returns content hash."""
    path = Path(path)
    if path.exists():
        raise LockExistsError(f"Immutable file already exists: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        raise LockExistsError(f"Immutable sidecar already exists: {sidecar}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = content_sha256(payload)
    body = canonical_json_bytes(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    if file_sha256(path) != digest:
        raise RuntimeError(f"Content hash mismatch after write: {path}")
    return digest


def validate_contract_against_lock(
    contract: dict[str, Any],
    *,
    lock_dir: Path,
    repo_root: Path,
) -> None:
    """Validate required fields, relative lock path, and lock SHA-256."""
    missing = sorted(REQUIRED_CONTRACT_FIELDS - set(contract))
    if missing:
        raise ValueError(f"Contract missing required fields: {missing}")

    ref = contract["scientific_lock"]
    rel = str(ref["relative_path"]).replace("\\", "/")
    if rel.startswith("/") or rel.startswith("..") or "/../" in f"/{rel}/":
        raise ValueError(f"Lock relative_path must be repo-relative without '..': {rel}")
    lock_path = (Path(repo_root) / rel).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    actual = file_sha256(lock_path)
    if actual != ref["sha256"]:
        raise ValueError(
            f"Lock SHA-256 mismatch: contract={ref['sha256']} actual={actual}"
        )
    if contract.get("frozen_choice") != "A0_HYBRID_POLICY":
        raise ValueError("frozen_choice must be A0_HYBRID_POLICY")
    sem = contract["policy_semantics"]
    if sem["both_native"] != "0.5*V2.1 + 0.5*A0 seed mean":
        raise ValueError("both_native semantics mismatch")
    if sem["v2_only"] != "V2.1":
        raise ValueError("v2_only semantics mismatch")
    if sem["a0_only"] != "complete A0 seed mean":
        raise ValueError("a0_only semantics mismatch")
    if sem["neither"] != "production naive_last_value":
        raise ValueError("neither semantics mismatch")


def git_status(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(repo_root), text=True
            ).strip()
        )
        return {"commit": commit, "dirty_tree": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "dirty_tree": None, "error": str(exc)}


def build_production_contract_v1(
    *,
    repo_root: Path,
    lock_dir: Path,
    metrics: dict[str, Any],
    impact: dict[str, Any],
) -> dict[str, Any]:
    """Assemble contract payload (does not write)."""
    repo_root = Path(repo_root).resolve()
    lock_dir = Path(lock_dir).resolve()
    lock_path = lock_dir / "ts_architecture_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = file_sha256(lock_path)
    lock_rel = lock_path.relative_to(repo_root).as_posix()

    sales_rel = "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet"
    sales_path = repo_root / sales_rel
    v21_exp = "src/data/results/product_universe/portfolio_v21_backfill_v2_export"
    a0_exp = "src/data/results/product_universe/portfolio_a0_backfill_v2_export"

    boot = metrics.get("sku_cluster_bootstrap") or {}
    hier = metrics.get("hierarchical_bootstrap") or {}
    pooled = (impact.get("pooled") or {}).get("deployable_residual") or {}
    cohort = impact.get("cohort_deployable") or []
    share_55 = None
    for row in cohort:
        if row.get("cohort") == "original_55":
            share_55 = row.get("share_of_portfolio_actual_volume")

    import pandas as pd

    sales_df = pd.read_parquet(sales_path)

    return {
        "contract_version": "ts_architecture_production_contract_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_lock": {
            "relative_path": lock_rel,
            "sha256": lock_sha,
            "version": lock.get("version"),
            "locked_at": lock.get("locked_at"),
        },
        "frozen_choice": "A0_HYBRID_POLICY",
        "policy_semantics": {
            "both_native": "0.5*V2.1 + 0.5*A0 seed mean",
            "v2_only": "V2.1",
            "a0_only": "complete A0 seed mean",
            "neither": "production naive_last_value",
        },
        "a0_engine": {
            "architecture": "legacy_adaptive_recursive_lstm",
            "seeds": [41, 42, 43],
            "all_three_seeds_required": True,
            "internal_steps": 16,
            "discard_bridge_step": True,
        },
        "cutoff": {
            "train_through": "origin-2",
            "bridge_month": "origin-1",
            "bridge_excluded_from_training": True,
        },
        "horizon": {
            "business_h1_equals_origin": True,
            "n_horizons": 15,
        },
        "forecast_constraints": {
            "non_negative_final_forecasts": True,
        },
        "sales": {
            "relative_path": sales_rel,
            "sha256": file_sha256(sales_path),
            "schema": list(sales_df.columns.astype(str)),
            "n_rows": int(len(sales_df)),
            "last_complete_month": 140505,
            "partial_month": 140506,
            "locked_score_end": 140504,
        },
        "sources": {
            "v2.1": {
                "run_id": "portfolio_v21_backfill_v2",
                "config_relative_path": "configs/forecast_jobs/portfolio_v21_backfill_v2.win.yaml",
                "config_yaml_sha256": file_sha256(
                    repo_root / "configs/forecast_jobs/portfolio_v21_backfill_v2.win.yaml"
                ),
                "config_hash": "3eb2e872407c119d",
                "export_relative_path": v21_exp,
                "forecasts_panel_sha256": file_sha256(
                    repo_root / v21_exp / "forecasts_panel.parquet"
                ),
                "export_summary_sha256": file_sha256(
                    repo_root / v21_exp / "export_summary.json"
                ),
                "state_sqlite_sha256": file_sha256(
                    repo_root / v21_exp / "state.sqlite"
                ),
            },
            "a0": {
                "run_id": "portfolio_a0_backfill_v2",
                "config_relative_path": "configs/forecast_jobs/portfolio_a0_backfill_v2.win.yaml",
                "config_yaml_sha256": file_sha256(
                    repo_root / "configs/forecast_jobs/portfolio_a0_backfill_v2.win.yaml"
                ),
                "config_hash": "77d9dea2942aa0f7",
                "export_relative_path": a0_exp,
                "forecasts_panel_sha256": file_sha256(
                    repo_root / a0_exp / "forecasts_panel.parquet"
                ),
                "forecasts_seed_mean_sha256": file_sha256(
                    repo_root / a0_exp / "forecasts_seed_mean.parquet"
                ),
                "export_summary_sha256": file_sha256(
                    repo_root / a0_exp / "export_summary.json"
                ),
                "state_sqlite_sha256": file_sha256(
                    repo_root / a0_exp / "state.sqlite"
                ),
            },
        },
        "git": git_status(repo_root),
        "gate_thresholds": {
            "unseen_primary_deployable_wmape_lift_gt_0": True,
            "unseen_primary_common_native_wmape_lift_gt_0": True,
            "hybrid_sku_origin_wins_exceed_v2": True,
            "sku_cluster_bootstrap_frac_positive_ge": 0.90,
            "drop_top3_lift_positive": True,
            "rmse_not_more_than_2pct_worse_than_v2": True,
            "p95_ae_not_more_than_2pct_worse_than_v2": True,
            "no_horizon_group_wmape_regression_gt_pp": 2.0,
            "pooled_residual_deployable_wmape_not_worse_than_v2": True,
            "tie_ae_eps": 1e-9,
            "horizon_groups": ["h1_3", "h4_6", "h7_12", "h13_15"],
            "bootstrap_B": 2000,
            "bootstrap_seed": 42,
        },
        "metric_snapshot": {
            "unseen_primary_deployable_lift_wmape_pp": metrics.get(
                "lift_deployable_wmape_pp"
            ),
            "unseen_primary_common_native_lift_wmape_pp": metrics.get(
                "lift_common_native_wmape_pp"
            ),
            "pooled_residual_deployable_lift_wmape_pp": pooled.get("lift_wmape_pp"),
            "pooled_residual_deployable_v2": pooled.get("v2"),
            "pooled_residual_deployable_hybrid": pooled.get("hybrid"),
            "wins": metrics.get("wins"),
            "sku_cluster_bootstrap": boot,
            "hierarchical_origin_then_sku_bootstrap": hier,
            "drop_top3": metrics.get("drop_top3"),
            "horizon_group_lifts": metrics.get("horizon_group_lifts"),
            "original55_residual_v2_wmape_pct": (
                metrics.get("original55_residual_v2") or {}
            ).get("portfolio_wmape_pct"),
            "original55_residual_hybrid_wmape_pct": (
                metrics.get("original55_residual_hybrid") or {}
            ).get("portfolio_wmape_pct"),
        },
        "hb_origins_excluded": [140304, 140307, 140310, 140312],
        "post_hb_must_not_change_lock": True,
        "known_risks": [
            {
                "id": "original55_wmape_regression",
                "detail": (
                    "original-55 hybrid WMAPE 30.04% versus V2 28.23% "
                    "on residual_development deployable panel"
                ),
            },
            {
                "id": "original55_by_origin",
                "detail": "hybrid worse on 7/9 original-55 residual origins",
            },
            {
                "id": "pooled_bias_sign_flip",
                "detail": "pooled residual deployable bias changes from +123 to -338",
            },
            {
                "id": "original55_concentrated_regressors",
                "detail": (
                    "original-55 regressions concentrated in influential products "
                    "(e.g. Cinnatropin family); loss_pattern=concentrated"
                ),
            },
            {
                "id": "pooled_residual_still_positive",
                "detail": "pooled residual deployable improvement remains positive at +1.12 pp WMAPE",
            },
        ],
        "context_not_decision_thresholds": {
            "note": (
                "Descriptive context only — not used as a freeze decision threshold."
            ),
            "original55_share_of_residual_deployable_abs_actual_volume": share_55,
        },
        "ts_frozen_before_residual_xgboost": True,
    }
