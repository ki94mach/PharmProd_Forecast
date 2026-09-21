"""Create-only residual study contract (freeze research before opening HB)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pkg.benchmark.config import CLEAN_QUANT_FEATURES, XGB_PARAMS
from pkg.portfolio_ts_lock.lock_io import LockExistsError
from pkg.research.portfolio_residual_xgb_v1.constants import (
    CLIP_FORECAST_NONNEGATIVE,
    COMPARATOR_HUMAN_XGB,
    CONTRACT_VERSION,
    FAMILY,
    FILLNA_SALES_LAGS,
    HUMAN_XGB_STATUS,
    MIN_PRIOR_MATURED_ORIGINS,
    MIN_TRAIN_ROWS,
    MIN_TRAIN_SKU_ORIGINS,
)
from pkg.research.portfolio_residual_xgb_v1.pit_features import (
    HYBRID_ANCHOR,
    TS_RESID_FEATURE_LIST,
    pit_feature_schema_manifest,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_bytes(text.encode("utf-8"))


def xgb_params_hash() -> str:
    return _sha256_json(dict(XGB_PARAMS))


def feature_list_hash(features: list[str] | None = None) -> str:
    feats = list(features) if features is not None else list(TS_RESID_FEATURE_LIST)
    return _sha256_json(feats)


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise LockExistsError(f"Residual study contract already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def build_residual_study_contract(
    *,
    lock_sha256: str,
    contract_sha256: str,
    sales_sha256: str,
    feature_source_hashes: dict[str, str],
    eligible_evaluation_origins: list[int],
    oof_split_manifest: dict[str, Any],
) -> dict[str, Any]:
    schema = pit_feature_schema_manifest()
    return {
        "contract_version": CONTRACT_VERSION,
        "family": FAMILY,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hb_outcomes_inspected": False,
        "ts_architecture": {
            "lock_sha256": lock_sha256,
            "production_contract_sha256": contract_sha256,
            "frozen_choice": "A0_HYBRID_POLICY",
            "note": "TS lock/contract referenced read-only; never mutated by residual prep.",
        },
        "sales_sha256": sales_sha256,
        "feature_source_hashes": feature_source_hashes,
        "xgb_params": dict(XGB_PARAMS),
        "xgb_params_hash": xgb_params_hash(),
        "feature_list": list(TS_RESID_FEATURE_LIST),
        "feature_list_hash": feature_list_hash(),
        "clean_quant_features": list(CLEAN_QUANT_FEATURES),
        "anchor": HYBRID_ANCHOR,
        "feature_schema": schema,
        "split_rules": {
            "method": "expanding_window_by_origin",
            "random_row_split": False,
            "pit": {
                "training_forecast_origin_lt_O": True,
                "training_target_date_le_O_minus_2": True,
                "feature_as_of_date_le_O_minus_2": True,
            },
            "minima": {
                "min_prior_matured_origins": MIN_PRIOR_MATURED_ORIGINS,
                "min_train_rows": MIN_TRAIN_ROWS,
                "min_train_sku_origins": MIN_TRAIN_SKU_ORIGINS,
            },
            "origin_classes": [
                "WARMUP_NOT_EVALUABLE",
                "INSUFFICIENT_TRAINING_DATA",
                "EVALUABLE",
            ],
            "eligible_evaluation_origins": list(eligible_evaluation_origins),
            "oof_split_manifest_summary": {
                "by_class": oof_split_manifest.get("by_class"),
                "n_origins": len(oof_split_manifest.get("origins", [])),
            },
        },
        "missing_value_and_clipping": {
            "sales_lag_fillna": FILLNA_SALES_LAGS,
            "clip_forecast_nonnegative": CLIP_FORECAST_NONNEGATIVE,
            "prediction": "max(0, anchor + resid_hat)",
        },
        "comparators": {
            "executable": [
                "HUMAN_BUDGET",
                "FROZEN_TS",
                "FROZEN_TS_PLUS_RESIDUAL_XGB",
            ],
            COMPARATOR_HUMAN_XGB: HUMAN_XGB_STATUS,
            "integrated": "DIAGNOSTIC_ONLY_NOT_EXECUTABLE",
        },
        "promotion_criteria": {
            "note": (
                "Freeze residual model + threshold via this contract (or follow-on lock) "
                "before opening any HB outcomes. TS postlock CAUTION blocks direct "
                "hybrid promotion pending final pipeline validation."
            ),
            "require_residual_model_frozen_before_hb": True,
            "require_hb_outcomes_inspected_false_until_residual_freeze": True,
            "human_xgb_requires_separate_approved_design": True,
        },
        "process_rule": (
            "After residual development, freeze residual model and threshold before "
            "opening any HB outcomes."
        ),
    }
