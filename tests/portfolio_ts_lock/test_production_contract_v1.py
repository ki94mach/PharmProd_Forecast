"""Tests for production contract immutability and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pkg.portfolio_ts_lock.lock_io import LockExistsError, write_lock_atomic
from pkg.portfolio_ts_lock.production_contract import (
    REQUIRED_CONTRACT_FIELDS,
    content_sha256,
    file_sha256,
    validate_contract_against_lock,
    write_immutable_json,
)


def _minimal_contract(lock_rel: str, lock_sha: str) -> dict:
    return {
        "contract_version": "ts_architecture_production_contract_v1",
        "created_at": "2026-09-21T00:00:00Z",
        "scientific_lock": {
            "relative_path": lock_rel,
            "sha256": lock_sha,
            "version": "portfolio_ts_architecture_lock_v1",
            "locked_at": "2026-09-21T00:00:00Z",
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
        "horizon": {"business_h1_equals_origin": True, "n_horizons": 15},
        "forecast_constraints": {"non_negative_final_forecasts": True},
        "sales": {
            "relative_path": "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet",
            "sha256": "abc",
            "schema": ["product", "date", "sales"],
            "n_rows": 1,
            "last_complete_month": 140505,
            "partial_month": 140506,
            "locked_score_end": 140504,
        },
        "sources": {},
        "git": {"commit": "x", "dirty_tree": False},
        "gate_thresholds": {},
        "metric_snapshot": {},
        "hb_origins_excluded": [140304, 140307, 140310, 140312],
        "post_hb_must_not_change_lock": True,
        "known_risks": [],
        "context_not_decision_thresholds": {
            "note": "Descriptive context only — not used as a freeze decision threshold.",
            "original55_share_of_residual_deployable_abs_actual_volume": 0.4923,
        },
    }


def test_required_fields_and_fallback_semantics(tmp_path: Path):
    lock = tmp_path / "ts_architecture_lock.json"
    write_lock_atomic(lock, {"choice": "A0_HYBRID_POLICY", "version": "v1"})
    lock_sha = file_sha256(lock)
    # place under fake repo
    repo = tmp_path / "repo"
    dest = repo / "src" / "data" / "lock"
    dest.mkdir(parents=True)
    lock2 = dest / "ts_architecture_lock.json"
    lock2.write_bytes(lock.read_bytes())
    rel = "src/data/lock/ts_architecture_lock.json"
    c = _minimal_contract(rel, file_sha256(lock2))
    assert REQUIRED_CONTRACT_FIELDS <= set(c)
    validate_contract_against_lock(c, lock_dir=dest, repo_root=repo)

    bad = dict(c)
    bad["policy_semantics"] = dict(c["policy_semantics"])
    bad["policy_semantics"]["neither"] = "zero"
    with pytest.raises(ValueError, match="neither"):
        validate_contract_against_lock(bad, lock_dir=dest, repo_root=repo)


def test_lock_reference_hash_validation(tmp_path: Path):
    repo = tmp_path / "repo"
    dest = repo / "src" / "data" / "lock"
    dest.mkdir(parents=True)
    lock = dest / "ts_architecture_lock.json"
    write_lock_atomic(lock, {"choice": "A0_HYBRID_POLICY"})
    rel = "src/data/lock/ts_architecture_lock.json"
    c = _minimal_contract(rel, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_contract_against_lock(c, lock_dir=dest, repo_root=repo)


def test_relative_path_rejects_parent(tmp_path: Path):
    repo = tmp_path / "repo"
    dest = repo / "src" / "data" / "lock"
    dest.mkdir(parents=True)
    lock = dest / "ts_architecture_lock.json"
    write_lock_atomic(lock, {"choice": "A0_HYBRID_POLICY"})
    c = _minimal_contract("../secrets/lock.json", file_sha256(lock))
    with pytest.raises(ValueError, match="relative"):
        validate_contract_against_lock(c, lock_dir=dest, repo_root=repo)


def test_contract_overwrite_prevention(tmp_path: Path):
    path = tmp_path / "ts_architecture_production_contract_v1.json"
    payload = {"a": 1}
    write_immutable_json(path, payload)
    with pytest.raises(LockExistsError):
        write_immutable_json(path, {"a": 2})
    side = path.with_suffix(path.suffix + ".sha256")
    assert side.read_text(encoding="utf-8").strip() == content_sha256(
        json.loads(path.read_text(encoding="utf-8"))
    )
