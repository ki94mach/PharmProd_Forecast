"""Immutable experiment-level metadata for historical backfills.

Persists ``manifest.json`` under ``data/backfills/{experiment_id}/{engine}/``.
The scientific ``config_hash`` excludes runtime provenance (timestamps, host,
package versions, worker count) so resume works across machines; those fields
are still recorded in the manifest for auditability.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pkg.benchmark.backfill_runner.state import (
    BackfillStateError,
    compute_config_hash,
    resolve_git_commit,
)
from pkg.benchmark.dataset import file_sha256
from pkg.benchmark.universes import universes_dir
from pkg.benchmark.vintages import vintages_dir

MANIFEST_FILENAME = "manifest.json"


class ExperimentManifestError(BackfillStateError):
    """Immutable manifest / config-hash conflict."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return file_sha256(path)


def resolve_manifest_paths(name: str, *, kind: str) -> tuple[Path, Path]:
    """Return ``(csv_path, meta_path)`` for a universe or vintage stem."""
    stem = str(name).strip()
    if stem.endswith(".csv"):
        stem = stem[:-4]
    base = universes_dir() if kind == "universe" else vintages_dir()
    return base / f"{stem}.csv", base / f"{stem}.meta.json"


def make_experiment_id(vintage_name: str, universe_name: str = "mvp_products") -> str:
    """Human-facing experiment id (engine is a subdirectory, not part of the id).

    Examples::

        vintage=ts_backfill_1401Q1_1405Q2, universe=mvp_products
          -> ts_mvp_backfill_1401Q1_1405Q2
    """
    stem = str(vintage_name).strip()
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.startswith("ts_backfill_"):
        span = stem[len("ts_backfill_") :]
    else:
        span = stem
    uni = str(universe_name).strip()
    if uni.endswith(".csv"):
        uni = uni[:-4]
    if uni == "mvp_products":
        return f"ts_mvp_backfill_{span}"
    return f"ts_{uni}_backfill_{span}"


def experiment_engine_dir(
    output_root: Path, experiment_id: str, engine: str
) -> Path:
    """``{root}/{experiment_id}/{engine}/`` — never mixed with production CSVs."""
    return Path(output_root) / str(experiment_id) / str(engine)


def _serialize_value(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def collect_package_versions(names: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def collect_environment_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or "",
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os_name": os.name,
        "cpu_count_logical": os.cpu_count(),
        "env_overrides": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
            if os.environ.get(k)
        },
    }


def engine_model_configuration(engine: str) -> dict[str, Any]:
    """Capture model knobs for the active engine (no GPU/LSTM in this step)."""
    key = str(engine).strip().lower()
    if key == "v2":
        from pkg.ts_v2.config import DEFAULT_CONFIG

        cfg = _serialize_value(DEFAULT_CONFIG)
        return {
            "engine": "v2",
            "ts_forecast_config": cfg,
            "notes": "V2 uses pkg.ts_v2.config.DEFAULT_CONFIG unless overridden.",
        }
    if key == "v1":
        return {
            "engine": "v1",
            "adapter": "pkg.benchmark.backfill_runner.engines.v1_adapter.V1ForecastEngine",
            "notes": (
                "Legacy SalesForecast path; internal +62100 date offset, "
                "model_selection, predict, redistribute_smoothing preserved."
            ),
            "legacy_behaviors": {
                "date_offset": 62100,
                "smoothing": "redistribute_smoothing",
                "nonnegative": "replace_negative_sales (inside V1)",
            },
        }
    if key == "dummy":
        return {
            "engine": "dummy",
            "notes": "Deterministic constant-level engine for orchestration tests.",
        }
    return {"engine": key, "notes": "engine-specific configuration not registered"}


def engine_cv_configuration(engine: str) -> dict[str, Any]:
    key = str(engine).strip().lower()
    if key == "v2":
        from pkg.ts_v2.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
        return {
            "scheme": "expanding_window_backtest",
            "origin_discovery": (
                "pkg.ts_v2.backtest_origins.discover_origins — monthly origins "
                "from first+min_train_months through last observed month"
            ),
            "train_rule": "date < origin",
            "forecast_horizon": int(cfg.forecast_horizon),
            "min_train_months": int(cfg.min_train_months),
            "min_selection_origins": int(cfg.min_selection_origins),
            "min_selection_predictions": int(cfg.min_selection_predictions),
            "selection_metric": cfg.selection_metric,
        }
    if key == "v1":
        return {
            "scheme": "v1_legacy_model_selection",
            "notes": (
                "V1 uses internal 80/20-style rolling selection inside "
                "SalesForecast.model_selection(); not the V2 multi-origin CV."
            ),
        }
    return {"scheme": "none", "engine": key}


def engine_selection_strategy(engine: str) -> dict[str, Any]:
    key = str(engine).strip().lower()
    if key == "v2":
        from pkg.ts_v2.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
        return {
            "selection_strategy": cfg.selection_strategy,
            "selection_metric": cfg.selection_metric,
            "selection_tie_tolerance": cfg.selection_tie_tolerance,
            "selection_simplicity_order": list(cfg.selection_simplicity_order),
            "ensemble_top_k": int(cfg.ensemble_top_k),
        }
    if key == "v1":
        return {
            "selection_strategy": "v1_best_model_rmse",
            "notes": "Legacy SalesForecast.model_selection best_model_type",
        }
    return {"selection_strategy": "n/a", "engine": key}


def engine_nonnegative_policy(engine: str) -> dict[str, Any]:
    key = str(engine).strip().lower()
    if key == "v2":
        from pkg.ts_v2.config import DEFAULT_CONFIG

        enabled = bool(DEFAULT_CONFIG.nonnegative_forecasts)
        return {
            "enabled": enabled,
            "policy": "max(raw, 0)" if enabled else "none",
            "applied_in": "pkg.ts_v2.postprocess.apply_final_constraints",
        }
    if key == "v1":
        return {
            "enabled": True,
            "policy": "replace_negative_sales (legacy V1)",
            "applied_in": "pkg.forecast.SalesForecast",
        }
    return {"enabled": False, "policy": "none", "engine": key}


def engine_smoothing_policy(engine: str) -> dict[str, Any]:
    key = str(engine).strip().lower()
    if key == "v2":
        return {
            "enabled": False,
            "policy": "none",
            "notes": (
                "V2 forbids redistribute_smoothing; see "
                "pkg.ts_v2.postprocess.V2_FORBIDDEN_POSTPROCESS_NAMES"
            ),
        }
    if key == "v1":
        return {
            "enabled": True,
            "policy": "redistribute_smoothing",
            "applied_in": "SalesForecast.redistribute_smoothing",
        }
    return {"enabled": False, "policy": "none", "engine": key}


def engine_random_seeds(engine: str) -> dict[str, Any]:
    """Document seeds where relevant; most classical TS models are deterministic."""
    key = str(engine).strip().lower()
    if key == "v2":
        return {
            "numpy": None,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "prophet": None,
            "notes": (
                "V2 baseline classical models have no explicit RNG seed; "
                "GPU/LSTM not used in this backfill step."
            ),
        }
    return {
        "numpy": None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "notes": "No explicit seeds configured for this engine.",
    }


def build_scientific_config(
    *,
    engine: str,
    vintage_name: str,
    universe_name: str,
    vintage_sha256: str,
    universe_sha256: str,
    horizon: int,
    model_configuration: Mapping[str, Any],
    cv_configuration: Mapping[str, Any],
    selection_strategy: Mapping[str, Any],
    nonnegative_policy: Mapping[str, Any],
    smoothing_policy: Mapping[str, Any],
    random_seeds: Mapping[str, Any],
) -> dict[str, Any]:
    """Fields that enter ``config_hash`` (reproducibility-critical)."""
    return {
        "engine_version": str(engine),
        "vintage_manifest": str(vintage_name),
        "vintage_manifest_sha256": str(vintage_sha256),
        "universe_manifest": str(universe_name),
        "universe_manifest_sha256": str(universe_sha256),
        "horizon": int(horizon),
        "training_cutoff_rule": "date < forecast_origin",
        "model_configuration": _serialize_value(model_configuration),
        "cv_configuration": _serialize_value(cv_configuration),
        "selection_strategy": _serialize_value(selection_strategy),
        "nonnegative_policy": _serialize_value(nonnegative_policy),
        "smoothing_policy": _serialize_value(smoothing_policy),
        "random_seeds": _serialize_value(random_seeds),
    }


def build_experiment_manifest(
    *,
    experiment_id: str,
    engine: str,
    vintage_name: str,
    universe_name: str,
    horizon: int = 15,
    git_commit: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the full immutable experiment manifest document."""
    vintage_csv, vintage_meta = resolve_manifest_paths(vintage_name, kind="vintage")
    universe_csv, universe_meta = resolve_manifest_paths(universe_name, kind="universe")
    vintage_sha = _sha256_file(vintage_csv)
    universe_sha = _sha256_file(universe_csv)

    model_cfg = engine_model_configuration(engine)
    cv_cfg = engine_cv_configuration(engine)
    selection = engine_selection_strategy(engine)
    nonnegative = engine_nonnegative_policy(engine)
    smoothing = engine_smoothing_policy(engine)
    seeds = engine_random_seeds(engine)

    scientific = build_scientific_config(
        engine=engine,
        vintage_name=vintage_name,
        universe_name=universe_name,
        vintage_sha256=vintage_sha,
        universe_sha256=universe_sha,
        horizon=horizon,
        model_configuration=model_cfg,
        cv_configuration=cv_cfg,
        selection_strategy=selection,
        nonnegative_policy=nonnegative,
        smoothing_policy=smoothing,
        random_seeds=seeds,
    )
    config_hash = compute_config_hash(scientific)
    commit = git_commit if git_commit is not None else resolve_git_commit()
    created = created_at or _utc_stamp()

    return {
        "schema_version": 1,
        "experiment_id": str(experiment_id),
        "engine_version": str(engine),
        "config_hash": config_hash,
        "created_at": created,
        "git_commit": commit,
        "universe": {
            "manifest": str(universe_name),
            "csv_path": str(universe_csv),
            "meta_path": str(universe_meta) if universe_meta.exists() else None,
            "sha256": universe_sha,
        },
        "vintages": {
            "manifest": str(vintage_name),
            "csv_path": str(vintage_csv),
            "meta_path": str(vintage_meta) if vintage_meta.exists() else None,
            "sha256": vintage_sha,
        },
        "forecast_horizon": int(horizon),
        "model_configuration": model_cfg,
        "cv_configuration": cv_cfg,
        "selection_strategy": selection,
        "nonnegative_policy": nonnegative,
        "smoothing_policy": smoothing,
        "random_seeds": seeds,
        "training_cutoff_rule": "date < forecast_origin",
        "scientific_config": scientific,
        "environment": collect_environment_info(),
        "python_version": sys.version.split()[0],
        "package_versions": collect_package_versions(
            (
                "numpy",
                "pandas",
                "scipy",
                "statsmodels",
                "sklearn",
                "xgboost",
                "prophet",
                "pmdarima",
            )
        ),
        "output_layout": {
            "root_relative": f"data/backfills/{experiment_id}/{engine}/",
            "manifest": MANIFEST_FILENAME,
            "state_db": "state.sqlite",
            "forecasts": "forecasts/",
            "backtests": "backtests/",
            "logs": "logs/",
            "note": (
                "Historical backfill artifacts only — not production forecast CSVs."
            ),
        },
    }


def load_manifest(experiment_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(experiment_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_immutable_manifest(
    experiment_dir: Path,
    manifest: Mapping[str, Any],
    *,
    allow_create: bool = True,
) -> dict[str, Any]:
    """Write ``manifest.json`` once; refuse conflicting config hashes.

    If a manifest already exists with the same ``config_hash``, return it
    unchanged. If ``config_hash`` differs, raise
    :class:`ExperimentManifestError` so two configs cannot silently share a
    completed experiment directory.
    """
    from pkg.benchmark.backfill_runner.store import atomic_write_json

    exp = Path(experiment_dir)
    exp.mkdir(parents=True, exist_ok=True)
    path = exp / MANIFEST_FILENAME
    incoming_hash = str(manifest.get("config_hash", ""))
    if not incoming_hash:
        raise ExperimentManifestError("manifest missing config_hash")

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_hash = str(existing.get("config_hash", ""))
        if existing_hash != incoming_hash:
            raise ExperimentManifestError(
                f"Refusing to write into {exp}: existing config_hash="
                f"{existing_hash!r} differs from incoming {incoming_hash!r}. "
                "Use a distinct experiment_id (or a fresh engine subdirectory) "
                "for a different configuration."
            )
        # Same hash — immutable; do not rewrite provenance fields.
        return existing

    if not allow_create:
        raise ExperimentManifestError(f"manifest missing at {path}")

    atomic_write_json(path, dict(manifest))
    return dict(manifest)
