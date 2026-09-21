"""Tests for smoke acceptance gate (no training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkg.research.portfolio_residual_xgb_v1.methods import (
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import HYBRID_ANCHOR
from pkg.research.portfolio_residual_xgb_v1.smoke_acceptance import (
    ACCEPTANCE_FILENAME,
    EXPECTED_N_TEST,
    EXPECTED_N_TRAIN,
    SMOKE_ORIGIN,
    assert_smoke_accepted,
    run_smoke_acceptance,
    verify_resume_no_duplicate,
)


def _write_minimal_results_tree(tmp: Path, repo_style: Path) -> tuple[Path, Path]:
    """Create results_dir + lock_dir with hashes matching a prepare_summary."""
    results = tmp / "results"
    lock = tmp / "lock"
    results.mkdir()
    lock.mkdir()
    (results / "runs" / "portfolio_residual_xgb_v1_1_smoke").mkdir(parents=True)

    lock_path = lock / "ts_architecture_lock.json"
    prod_path = lock / "ts_architecture_production_contract_v1.json"
    lock_path.write_text('{"choice":"A0_HYBRID_POLICY"}\n', encoding="utf-8")
    prod_path.write_text('{"frozen_choice":"A0_HYBRID_POLICY"}\n', encoding="utf-8")

    from pkg.portfolio_ts_lock.production_contract import file_sha256

    lock_sha = file_sha256(lock_path)
    prod_sha = file_sha256(prod_path)

    contract = results / "residual_study_contract_v1_1.json"
    gates = results / "residual_promotion_gates_v1.json"
    contract.write_text('{"contract_version":"residual_study_contract_v1_1"}\n')
    gates.write_text('{"gate_version":"residual_promotion_gates_v1"}\n')
    v11_sha = file_sha256(contract)
    gates_sha = file_sha256(gates)

    summary = {
        "lock_sha256": lock_sha,
        "production_contract_sha256": prod_sha,
        "residual_study_contract_v1_1_sha256": v11_sha,
        "promotion_gates_sha256": gates_sha,
    }
    (results / "prepare_summary_v1_1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "origins": [
            {
                "evaluation_origin": SMOKE_ORIGIN,
                "n_train_rows": EXPECTED_N_TRAIN,
                "n_test_rows": EXPECTED_N_TEST,
            }
        ]
    }
    (results / "oof_split_manifest_v1_1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results, lock


def _synthetic_pred(n: int = EXPECTED_N_TEST) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "product": [f"P{i%10}" for i in range(n)],
            "forecast_origin": [SMOKE_ORIGIN] * n,
            "horizon": [(i % 15) + 1 for i in range(n)],
            "target_date": [140107 + (i % 15) for i in range(n)],
            "actual": rng.random(n) * 100,
            HYBRID_ANCHOR: rng.random(n) * 100,
            METHOD_FROZEN_TS: rng.random(n) * 100,
            METHOD_MEDIAN: rng.random(n) * 100,
            METHOD_XGB: rng.random(n) * 100,
        }
    )


def test_assert_smoke_accepted_blocks_without_file(tmp_path: Path):
    with pytest.raises(RuntimeError, match="missing smoke acceptance"):
        assert_smoke_accepted(tmp_path)


def test_acceptance_fails_without_predictions(tmp_path: Path):
    results, lock = _write_minimal_results_tree(tmp_path, tmp_path)
    report = run_smoke_acceptance(
        repo_root=tmp_path, results_dir=results, lock_dir=lock, require_predictions=True
    )
    assert report["passed"] is False
    assert not (results / ACCEPTANCE_FILENAME).exists()


def test_acceptance_passes_on_synthetic_smoke(tmp_path: Path):
    results, lock = _write_minimal_results_tree(tmp_path, tmp_path)
    run_dir = results / "runs" / "portfolio_residual_xgb_v1_1_smoke"
    pred = _synthetic_pred()
    # unique keys
    pred["product"] = [f"SKU{i}" for i in range(len(pred))]
    pred_path = run_dir / f"oof_predictions_origin_{SMOKE_ORIGIN}.parquet"
    pred.to_parquet(pred_path, index=False)
    meta = {
        "evaluation_origin": SMOKE_ORIGIN,
        "n_train_rows": EXPECTED_N_TRAIN,
        "n_test_rows": EXPECTED_N_TEST,
        "max_sales_feature_date_train": 140105,
        "max_sales_feature_date_test": 140105,
        "hb_outcomes_inspected": False,
    }
    (run_dir / f"oof_meta_origin_{SMOKE_ORIGIN}.json").write_text(
        json.dumps(meta) + "\n", encoding="utf-8"
    )
    report = run_smoke_acceptance(
        repo_root=tmp_path, results_dir=results, lock_dir=lock
    )
    assert report["passed"] is True, json.dumps(report, indent=2)
    assert (results / ACCEPTANCE_FILENAME).is_file()
    assert report["promotion_gates_affected"] is False
    assert report["parameters_tuned"] is False
    # gates untouched
    assert_smoke_accepted(results)


def test_resume_verify_exactly_one(tmp_path: Path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    pred = run_dir / "oof_predictions_origin_140107.parquet"
    meta = run_dir / "oof_meta_origin_140107.json"
    pred.write_bytes(b"abc")
    meta.write_text("{}")
    from pkg.portfolio_ts_lock.production_contract import file_sha256

    info = verify_resume_no_duplicate(
        pred_path=pred, meta_path=meta, pred_sha_before=file_sha256(pred)
    )
    assert info["exactly_one_each"] is True
    assert info["pred_sha_unchanged"] is True
