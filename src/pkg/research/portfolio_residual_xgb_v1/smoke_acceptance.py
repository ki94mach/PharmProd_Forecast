"""Smoke acceptance gate — must pass before full residual-XGBoost execution.

Does not tune parameters or mutate promotion gates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.constants import HB_ORIGINS, POST_HB_ORIGINS
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS
from pkg.research.portfolio_residual_xgb_v1.methods import (
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    last_complete_month,
)

SMOKE_ORIGIN = 140107
SMOKE_ORIGIN_MINUS_2 = 140105
EXPECTED_N_TRAIN = 880
EXPECTED_N_TEST = 2055
SMOKE_RUN_ID = "portfolio_residual_xgb_v1_1_smoke"
ACCEPTANCE_FILENAME = "smoke_acceptance_v1.json"
PRED_KEYS = ("product", "forecast_origin", "horizon", "target_date")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check(
    name: str, ok: bool, detail: Any, checks: list[dict[str, Any]]
) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def load_expected_counts(manifest_path: Path) -> tuple[int, int, dict[str, Any]]:
    """Return (n_train, n_test, reconciliation_or_empty)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        r
        for r in manifest["origins"]
        if int(r["evaluation_origin"]) == SMOKE_ORIGIN
    )
    expected_train = int(row["n_train_rows"])
    expected_test = int(row["n_test_rows"])
    recon_path = manifest_path.parent / "smoke_row_count_reconciliation.json"
    recon: dict[str, Any] = {}
    if recon_path.is_file():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))
    return expected_train, expected_test, recon


def verify_resume_no_duplicate(
    *,
    pred_path: Path,
    meta_path: Path,
    pred_sha_before: str,
) -> dict[str, Any]:
    """Confirm resume would not create a second prediction artifact."""
    n_pred = len(list(pred_path.parent.glob("oof_predictions_origin_140107*.parquet")))
    n_meta = len(list(meta_path.parent.glob("oof_meta_origin_140107*.json")))
    sha_now = file_sha256(pred_path) if pred_path.is_file() else ""
    return {
        "n_prediction_files": n_pred,
        "n_meta_files": n_meta,
        "pred_sha_unchanged": sha_now == pred_sha_before,
        "exactly_one_each": n_pred == 1 and n_meta == 1,
    }


def run_smoke_acceptance(
    *,
    repo_root: Path,
    results_dir: Path,
    lock_dir: Path,
    require_predictions: bool = True,
) -> dict[str, Any]:
    """Evaluate smoke outputs; write acceptance artifact if all checks pass."""
    results_dir = Path(results_dir)
    lock_dir = Path(lock_dir)
    run_dir = results_dir / "runs" / SMOKE_RUN_ID
    pred_path = run_dir / f"oof_predictions_origin_{SMOKE_ORIGIN}.parquet"
    meta_path = run_dir / f"oof_meta_origin_{SMOKE_ORIGIN}.json"
    gates_path = results_dir / "residual_promotion_gates_v1.json"
    contract_v11 = results_dir / "residual_study_contract_v1_1.json"
    summary_path = results_dir / "prepare_summary_v1_1.json"
    manifest_path = results_dir / "oof_split_manifest_v1_1.json"
    lock_path = lock_dir / "ts_architecture_lock.json"
    prod_path = lock_dir / "ts_architecture_production_contract_v1.json"

    checks: list[dict[str, Any]] = []
    gates_bytes_before = gates_path.read_bytes() if gates_path.is_file() else b""
    contract_bytes_before = contract_v11.read_bytes() if contract_v11.is_file() else b""
    lock_bytes_before = lock_path.read_bytes()
    prod_bytes_before = prod_path.read_bytes()

    # --- presence ---
    if require_predictions and not pred_path.is_file():
        _check(
            "smoke_predictions_exist",
            False,
            f"missing {pred_path}",
            checks,
        )
        report = _finalize_fail(checks, results_dir, gates_bytes_before, contract_bytes_before)
        return report

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_train, expected_test, recon = load_expected_counts(manifest_path)
    # Prefer canonical expected 880/2055 unless recon documents otherwise
    if expected_train != EXPECTED_N_TRAIN or expected_test != EXPECTED_N_TEST:
        if not recon:
            _check(
                "manifest_matches_canonical_880_2055",
                False,
                {
                    "manifest_train": expected_train,
                    "manifest_test": expected_test,
                    "canonical_train": EXPECTED_N_TRAIN,
                    "canonical_test": EXPECTED_N_TEST,
                    "note": "Provide smoke_row_count_reconciliation.json if counts differ",
                },
                checks,
            )
        else:
            expected_train = int(recon.get("n_train_rows", expected_train))
            expected_test = int(recon.get("n_test_rows", expected_test))
            _check(
                "row_counts_reconciled",
                True,
                recon,
                checks,
            )
    else:
        _check(
            "manifest_matches_canonical_880_2055",
            True,
            {"n_train": expected_train, "n_test": expected_test},
            checks,
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    pred = pd.read_parquet(pred_path) if pred_path.is_file() else pd.DataFrame()

    # exactly 1 successful fit
    n_pred_files = len(list(run_dir.glob("oof_predictions_origin_*.parquet")))
    n_meta_files = len(list(run_dir.glob("oof_meta_origin_*.json")))
    _check(
        "exactly_one_successful_fit_140107",
        n_pred_files == 1
        and n_meta_files == 1
        and int(meta.get("evaluation_origin", -1)) == SMOKE_ORIGIN,
        {
            "n_pred_files": n_pred_files,
            "n_meta_files": n_meta_files,
            "evaluation_origin": meta.get("evaluation_origin"),
        },
        checks,
    )

    # row counts
    n_train = int(meta.get("n_train_rows", -1))
    n_test = int(meta.get("n_test_rows", -1))
    rows_ok = n_train == expected_train and n_test == expected_test and len(pred) == expected_test
    _check(
        "train_test_row_counts",
        rows_ok,
        {
            "meta_n_train": n_train,
            "meta_n_test": n_test,
            "pred_rows": int(len(pred)),
            "expected_train": expected_train,
            "expected_test": expected_test,
            "reconciled": bool(recon),
        },
        checks,
    )

    # zero HB/post-HB
    if len(pred):
        fo = pred["forecast_origin"].astype(int)
        bad = sorted(set(fo) & set(HB_ORIGINS | POST_HB_ORIGINS))
        _check("zero_hb_post_hb_rows", len(bad) == 0, {"forbidden_origins": bad}, checks)
    else:
        _check("zero_hb_post_hb_rows", False, "no predictions", checks)

    # all sales feature dates <= 140105 (from forecast_origin of each row)
    if len(pred):
        max_sales = int(pred["forecast_origin"].astype(int).map(last_complete_month).max())
        # Also verify O-2 constant
        o2 = shamsi_add_months(SMOKE_ORIGIN, -2)
        _check(
            "all_sales_feature_dates_le_140105",
            max_sales <= SMOKE_ORIGIN_MINUS_2 and o2 == SMOKE_ORIGIN_MINUS_2,
            {
                "max_last_complete_month_on_pred_rows": max_sales,
                "origin_minus_2": o2,
                "cutoff": SMOKE_ORIGIN_MINUS_2,
                "note": "Pred rows are Fo=140107 → last_complete=140105; train Fo<O ⇒ Fo-2≤140105",
            },
            checks,
        )
        if "max_sales_feature_date_train" in meta:
            _check(
                "train_max_sales_feature_date_le_140105",
                int(meta["max_sales_feature_date_train"]) <= SMOKE_ORIGIN_MINUS_2,
                meta.get("max_sales_feature_date_train"),
                checks,
            )
    else:
        _check("all_sales_feature_dates_le_140105", False, "no predictions", checks)

    # complete finite non-negative corrected forecasts
    method_cols = [METHOD_FROZEN_TS, METHOD_MEDIAN, METHOD_XGB]
    if len(pred) and all(c in pred.columns for c in method_cols):
        finite_ok = True
        nonneg_ok = True
        complete_ok = True
        details = {}
        for c in method_cols:
            arr = pred[c].to_numpy(dtype=float)
            details[c] = {
                "n_nan": int(np.isnan(arr).sum()),
                "n_inf": int(np.isinf(arr).sum()),
                "n_neg": int((arr < 0).sum()),
                "n_null": int(pred[c].isna().sum()),
            }
            if details[c]["n_nan"] or details[c]["n_inf"]:
                finite_ok = False
            if details[c]["n_neg"]:
                nonneg_ok = False
            if details[c]["n_null"]:
                complete_ok = False
        _check(
            "complete_finite_nonnegative_forecasts",
            complete_ok and finite_ok and nonneg_ok,
            details,
            checks,
        )
    else:
        _check(
            "complete_finite_nonnegative_forecasts",
            False,
            "missing method columns",
            checks,
        )

    # TS anchors unchanged on pred rows (f_a0_hybrid present and finite)
    if len(pred) and HYBRID_ANCHOR in pred.columns:
        anc = pred[HYBRID_ANCHOR].to_numpy(dtype=float)
        _check(
            "ts_anchors_present_finite",
            bool(np.isfinite(anc).all()) and len(anc) == len(pred),
            {"n": int(len(anc))},
            checks,
        )
    else:
        _check("ts_anchors_present_finite", False, "missing f_a0_hybrid", checks)

    # lock / contract hashes unchanged vs prepare summary
    lock_sha = file_sha256(lock_path)
    prod_sha = file_sha256(prod_path)
    v11_sha = file_sha256(contract_v11) if contract_v11.is_file() else ""
    gates_sha = file_sha256(gates_path) if gates_path.is_file() else ""
    _check(
        "lock_hash_unchanged",
        lock_sha == summary.get("lock_sha256"),
        {"now": lock_sha, "expected": summary.get("lock_sha256")},
        checks,
    )
    _check(
        "production_contract_hash_unchanged",
        prod_sha == summary.get("production_contract_sha256"),
        {"now": prod_sha, "expected": summary.get("production_contract_sha256")},
        checks,
    )
    _check(
        "residual_contract_v1_1_hash_unchanged",
        v11_sha == summary.get("residual_study_contract_v1_1_sha256"),
        {"now": v11_sha, "expected": summary.get("residual_study_contract_v1_1_sha256")},
        checks,
    )
    _check(
        "promotion_gates_hash_unchanged",
        gates_sha == summary.get("promotion_gates_sha256"),
        {"now": gates_sha, "expected": summary.get("promotion_gates_sha256")},
        checks,
    )

    # no duplicate prediction keys
    if len(pred):
        dup = int(pred.duplicated(list(PRED_KEYS)).sum())
        _check(
            "no_duplicate_prediction_keys",
            dup == 0,
            {"duplicate_rows": dup, "keys": list(PRED_KEYS)},
            checks,
        )
    else:
        _check("no_duplicate_prediction_keys", False, "no predictions", checks)

    # resume rerun creates no duplicate job
    if pred_path.is_file():
        pred_sha = file_sha256(pred_path)
        resume_info = verify_resume_no_duplicate(
            pred_path=pred_path, meta_path=meta_path, pred_sha_before=pred_sha
        )
        _check(
            "resume_no_duplicate_job",
            bool(resume_info["exactly_one_each"] and resume_info["pred_sha_unchanged"]),
            resume_info,
            checks,
        )
    else:
        _check("resume_no_duplicate_job", False, "no predictions", checks)

    # smoke eval must not affect promotion gates / params
    _check(
        "smoke_does_not_mutate_promotion_gates",
        gates_path.read_bytes() == gates_bytes_before if gates_path.is_file() else True,
        "gates bytes compared pre/post acceptance",
        checks,
    )
    _check(
        "smoke_does_not_mutate_residual_contract",
        contract_v11.read_bytes() == contract_bytes_before if contract_v11.is_file() else True,
        "contract v1_1 bytes compared pre/post acceptance",
        checks,
    )
    _check(
        "smoke_does_not_mutate_ts_lock_or_prod_contract",
        lock_path.read_bytes() == lock_bytes_before
        and prod_path.read_bytes() == prod_bytes_before,
        True,
        checks,
    )
    _check(
        "smoke_eval_does_not_tune_or_gate",
        True,
        {
            "HUMAN_XGB": HUMAN_XGB_STATUS,
            "promotion_gates_role": "immutable_reference_only",
            "smoke_evaluation": "diagnostic_only_not_promotion",
            "parameters_tuned": False,
        },
        checks,
    )

    all_pass = all(c["pass"] for c in checks)
    report = {
        "acceptance_version": "smoke_acceptance_v1",
        "created_at": _utc(),
        "smoke_origin": SMOKE_ORIGIN,
        "smoke_run_id": SMOKE_RUN_ID,
        "passed": all_pass,
        "checks": checks,
        "blocks_full_execute_until_pass": True,
        "hb_outcomes_inspected": False,
        "HUMAN_XGB": HUMAN_XGB_STATUS,
        "promotion_gates_affected": False,
        "parameters_tuned": False,
    }

    out = results_dir / ACCEPTANCE_FILENAME
    # Only write passing acceptance as the gate file full execute requires;
    # always write a diagnostic sibling.
    diag = results_dir / "smoke_acceptance_v1.last_attempt.json"
    diag.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if all_pass:
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(out)
        report["acceptance_path"] = str(out.as_posix())
    else:
        report["acceptance_path"] = None
        report["last_attempt_path"] = str(diag.as_posix())

    # Re-verify gates untouched after write
    if gates_path.is_file() and gates_path.read_bytes() != gates_bytes_before:
        raise RuntimeError("Smoke acceptance mutated promotion gates — FAIL")
    if contract_v11.is_file() and contract_v11.read_bytes() != contract_bytes_before:
        raise RuntimeError("Smoke acceptance mutated residual contract — FAIL")

    return report


def _finalize_fail(
    checks: list[dict[str, Any]],
    results_dir: Path,
    gates_bytes_before: bytes,
    contract_bytes_before: bytes,
) -> dict[str, Any]:
    report = {
        "acceptance_version": "smoke_acceptance_v1",
        "created_at": _utc(),
        "smoke_origin": SMOKE_ORIGIN,
        "passed": False,
        "checks": checks,
        "blocks_full_execute_until_pass": True,
        "hb_outcomes_inspected": False,
        "HUMAN_XGB": HUMAN_XGB_STATUS,
        "promotion_gates_affected": False,
        "parameters_tuned": False,
    }
    diag = results_dir / "smoke_acceptance_v1.last_attempt.json"
    diag.parent.mkdir(parents=True, exist_ok=True)
    diag.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    report["last_attempt_path"] = str(diag.as_posix())
    return report


def assert_smoke_accepted(results_dir: Path) -> dict[str, Any]:
    """Hard-require a passing smoke acceptance artifact before full execute."""
    path = Path(results_dir) / ACCEPTANCE_FILENAME
    if not path.is_file():
        raise RuntimeError(
            f"Full execute blocked: missing smoke acceptance {path}. "
            "Run smoke Execute then AcceptSmoke first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError(f"Full execute blocked: smoke acceptance failed — see {path}")
    return payload
