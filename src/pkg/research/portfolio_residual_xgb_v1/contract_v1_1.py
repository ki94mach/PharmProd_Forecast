"""Additive residual_study_contract_v1_1 (supersedes v1 by hash; does not rewrite v1)."""

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
    CONTRACT_VERSION_V1_1,
    DETERMINISTIC_SEED,
    FAMILY,
    FILLNA_SALES_LAGS,
    HUMAN_XGB_STATUS,
    METHOD_FROZEN_TS,
    METHOD_MEDIAN,
    METHOD_XGB,
    MIN_PRIOR_MATURED_ORIGINS,
    MIN_TRAIN_ROWS,
    MIN_TRAIN_SKU_ORIGINS,
)
from pkg.research.portfolio_residual_xgb_v1.methods import EXECUTABLE_METHODS
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


def feature_semantics_hash() -> str:
    schema = pit_feature_schema_manifest()
    return _sha256_json(schema["production_semantics"])


def write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise LockExistsError(f"Immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return _sha256_bytes(path.read_bytes())


def build_residual_study_contract_v1_1(
    *,
    superseded_v1_sha256: str,
    lock_sha256: str,
    contract_sha256: str,
    sales_sha256: str,
    feature_source_hashes: dict[str, str],
    eligible_evaluation_origins: list[int],
    oof_split_manifest: dict[str, Any],
) -> dict[str, Any]:
    schema = pit_feature_schema_manifest()
    return {
        "contract_version": CONTRACT_VERSION_V1_1,
        "family": FAMILY,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": {
            "contract_version": "residual_study_contract_v1",
            "sha256": superseded_v1_sha256,
            "reason": (
                "Correct production PIT sales semantics: last_complete_month=origin-2; "
                "no sales feature may use origin-1 or later."
            ),
        },
        "hb_outcomes_inspected": False,
        "HUMAN_XGB": HUMAN_XGB_STATUS,
        "ts_architecture": {
            "lock_sha256": lock_sha256,
            "production_contract_sha256": contract_sha256,
            "frozen_choice": "A0_HYBRID_POLICY",
            "note": "TS lock/contract referenced read-only; never mutated.",
        },
        "sales_sha256": sales_sha256,
        "feature_source_hashes": feature_source_hashes,
        "xgb_params": dict(XGB_PARAMS),
        "xgb_params_hash": xgb_params_hash(),
        "feature_list": list(TS_RESID_FEATURE_LIST),
        "feature_list_hash": feature_list_hash(),
        "feature_semantics_hash": feature_semantics_hash(),
        "clean_quant_features": list(CLEAN_QUANT_FEATURES),
        "anchor": HYBRID_ANCHOR,
        "feature_schema": schema,
        "deterministic_seed": DETERMINISTIC_SEED,
        "split_rules": {
            "method": "expanding_window_by_origin",
            "random_row_split": False,
            "pit": {
                "training_forecast_origin_lt_evaluation_origin": True,
                "training_target_date_le_evaluation_origin_minus_2": True,
                "every_sales_feature_date_le_evaluation_origin_minus_2": True,
            },
            "minima": {
                "min_prior_matured_origins": MIN_PRIOR_MATURED_ORIGINS,
                "min_train_rows": MIN_TRAIN_ROWS,
                "min_train_sku_origins": MIN_TRAIN_SKU_ORIGINS,
            },
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
        "methods": {
            "executable": list(EXECUTABLE_METHODS),
            METHOD_FROZEN_TS: "frozen hybrid panel",
            METHOD_MEDIAN: "train-median residual correction",
            METHOD_XGB: "XGBoost residual correction",
            "HUMAN_XGB": HUMAN_XGB_STATUS,
        },
        "categorical_encoding": {
            "fit_on": "training_rows_only",
            "unknown_token": "__UNKNOWN__",
            "deterministic": True,
        },
        "process_rule": (
            "After residual development, freeze residual model and threshold via "
            "promotion gates before opening any HB outcomes."
        ),
    }
