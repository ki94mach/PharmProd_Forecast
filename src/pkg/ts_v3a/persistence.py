"""Immutable V3A screening-experiment persistence.

Layout (not production/backfill outputs)::

    src/data/ts_v3a/screening/{experiment_id}/
        manifest.json
        oof_predictions.parquet
        fold_metadata.parquet
        architecture_metrics.csv
        horizon_metrics.csv
        seed_metrics.csv
        failures.csv
        .complete

Incomplete / checkpoint experiments live under::

    src/data/ts_v3a/screening/.incomplete/{experiment_id}/

Completed experiments are never overwritten or appended to with a different
configuration. ``config_hash`` covers scientific settings only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.metrics import horizon_bias, horizon_mae, horizon_rmse, horizon_wmape
from pkg.ts_v3a.architectures import ArchitectureName, coerce_architecture_name
from pkg.ts_v3a.backtest import STATUS_OK, NeuralOuterBacktestResult
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig

V3A_VERSION = "v3a"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete"
CHECKPOINT_FILENAME = "checkpoint.json"
COMPLETE_MARKER = ".complete"
MANIFEST_FILENAME = "manifest.json"

OOF_PREDICTIONS_NAME = "oof_predictions.parquet"
FOLD_METADATA_NAME = "fold_metadata.parquet"
ARCHITECTURE_METRICS_NAME = "architecture_metrics.csv"
HORIZON_METRICS_NAME = "horizon_metrics.csv"
SEED_METRICS_NAME = "seed_metrics.csv"
FAILURES_NAME = "failures.csv"

ARTIFACT_FILES = (
    MANIFEST_FILENAME,
    OOF_PREDICTIONS_NAME,
    FOLD_METADATA_NAME,
    ARCHITECTURE_METRICS_NAME,
    HORIZON_METRICS_NAME,
    SEED_METRICS_NAME,
    FAILURES_NAME,
)

FAILURE_COLUMNS = (
    "product",
    "architecture",
    "seed",
    "origin",
    "status",
    "unavailable_reason",
    "error_type",
    "error_message",
)

HORIZON_METRICS_COLUMNS = (
    "product",
    "architecture",
    "horizon",
    "mae",
    "rmse",
    "bias",
    "wmape",
)


class ExperimentPersistenceError(Exception):
    """Base error for V3A screening persistence."""


class ExperimentImmutableError(ExperimentPersistenceError):
    """Attempt to mutate or overwrite a completed experiment."""


class ExperimentConfigConflictError(ExperimentPersistenceError):
    """Stored config_hash differs from the requested scientific configuration."""


class ExperimentCheckpointError(ExperimentPersistenceError):
    """Invalid incomplete/checkpoint state."""


def default_screening_root() -> Path:
    """Canonical ``src/data/ts_v3a/screening`` (created on demand)."""
    # src/pkg/ts_v3a/persistence.py -> parents[2] = src/
    root = Path(__file__).resolve().parents[2] / "data" / "ts_v3a" / "screening"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_experiment_id(*, created_at: Optional[datetime] = None) -> str:
    """Unique experiment id: ``YYYYMMDDTHHMMSSZ_<8 hex>``."""
    ts = created_at or datetime.now(timezone.utc)
    stamp = ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def incomplete_experiment_dir(base_dir: Path, experiment_id: str) -> Path:
    return Path(base_dir) / ".incomplete" / str(experiment_id)


def complete_experiment_dir(base_dir: Path, experiment_id: str) -> Path:
    return Path(base_dir) / str(experiment_id)


def is_complete_experiment(experiment_dir: Path) -> bool:
    """True when ``experiment_dir`` is a finalized immutable screening run."""
    path = Path(experiment_dir)
    if (path / COMPLETE_MARKER).is_file():
        return True
    manifest_path = path / MANIFEST_FILENAME
    if manifest_path.is_file():
        return _read_json_if_exists(manifest_path).get("status") == COMPLETE_STATUS
    return False


def get_git_commit() -> Optional[str]:
    """Best-effort ``git rev-parse HEAD`` from repo root."""
    try:
        root = Path(__file__).resolve().parents[3]
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = proc.stdout.strip()
        return commit or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _serialize_value(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, ArchitectureName):
        return obj.value
    return obj


def neural_config_to_dict(config: NeuralExperimentConfig) -> dict[str, Any]:
    """JSON-serializable NeuralExperimentConfig snapshot."""
    out: dict[str, Any] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, tuple):
            out[f.name] = list(val)
        else:
            out[f.name] = val
    return out


def v2_config_to_dict(config: TSForecastConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, tuple):
            out[f.name] = list(val)
        else:
            out[f.name] = val
    return out


def _normalize_architectures(
    architectures: Optional[Sequence[Union[str, ArchitectureName]]],
) -> list[str]:
    if architectures is None:
        from pkg.ts_v3a.model_factory import IMPLEMENTED_ARCHITECTURES

        return [a.value for a in IMPLEMENTED_ARCHITECTURES]
    return [coerce_architecture_name(a).value for a in architectures]


def _origins_payload(origins: Optional[Sequence[int]]) -> Any:
    if origins is None:
        return "auto"
    return [int(o) for o in origins]


def screening_config_payload(
    *,
    neural_config: NeuralExperimentConfig,
    v2_config: TSForecastConfig,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    origins: Optional[Sequence[int]] = None,
    product_universe_id: Optional[str] = None,
    product_universe_hash: Optional[str] = None,
    min_successful_seeds: Optional[int] = None,
) -> dict[str, Any]:
    """Scientific configuration only (no timestamps / host / package versions)."""
    cfg = neural_config
    seed_list = list(int(s) for s in (seeds if seeds is not None else cfg.random_seeds))
    min_ok = int(
        min_successful_seeds if min_successful_seeds is not None else cfg.min_successful_seeds
    )
    arch_names = _normalize_architectures(architectures)
    return {
        "v3a_version": V3A_VERSION,
        "architectures": arch_names,
        "neural_experiment_config": neural_config_to_dict(cfg),
        "seeds": seed_list,
        "min_successful_seeds": min_ok,
        "v2_cv_configuration": {
            "forecast_horizon": int(v2_config.forecast_horizon),
            "min_train_months": int(v2_config.min_train_months),
            "activity_start_min_sales": v2_config.activity_start_min_sales,
            "missing_month_policy": v2_config.missing_month_policy,
            "nonnegative_forecasts": bool(v2_config.nonnegative_forecasts),
        },
        "origins": _origins_payload(origins),
        "product_universe_id": product_universe_id,
        "product_universe_hash": product_universe_hash,
        "eligibility_configuration": {
            "min_internal_train_windows": int(cfg.min_internal_train_windows),
            "min_internal_validation_windows": int(cfg.min_internal_validation_windows),
            "validation_fraction": float(cfg.validation_fraction),
            "min_successful_seeds": min_ok,
        },
        "scaling_method": cfg.scaling_method,
        "loss": cfg.loss,
        "optimizer": cfg.optimizer,
        "learning_rate": float(cfg.learning_rate),
    }


def screening_config_hash(
    *,
    neural_config: NeuralExperimentConfig,
    v2_config: TSForecastConfig,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    origins: Optional[Sequence[int]] = None,
    product_universe_id: Optional[str] = None,
    product_universe_hash: Optional[str] = None,
    min_successful_seeds: Optional[int] = None,
) -> str:
    """Stable short hash of the scientific screening configuration."""
    payload = screening_config_payload(
        neural_config=neural_config,
        v2_config=v2_config,
        architectures=architectures,
        seeds=seeds,
        origins=origins,
        product_universe_id=product_universe_id,
        product_universe_hash=product_universe_hash,
        min_successful_seeds=min_successful_seeds,
    )
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def collect_package_versions(names: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def _tensorflow_version() -> Optional[str]:
    try:
        import tensorflow as tf

        return str(tf.__version__)
    except Exception:
        return None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class ScreeningExperimentCheckpoint:
    """In-progress screening experiment; must match ``config_hash`` on resume."""

    experiment_id: str
    config_hash: str
    created_at: str
    experiment_dir: Path
    neural_config: NeuralExperimentConfig
    v2_config: TSForecastConfig
    architectures: tuple[str, ...]
    seeds: tuple[int, ...]
    origins: Optional[tuple[int, ...]]
    product_universe_id: Optional[str]
    product_universe_hash: Optional[str]
    min_successful_seeds: int

    @property
    def is_complete(self) -> bool:
        return is_complete_experiment(self.experiment_dir)


def assert_compatible_config(
    experiment_dir_or_manifest: Union[Path, Mapping[str, Any]],
    *,
    neural_config: NeuralExperimentConfig,
    v2_config: TSForecastConfig,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    origins: Optional[Sequence[int]] = None,
    product_universe_id: Optional[str] = None,
    product_universe_hash: Optional[str] = None,
    min_successful_seeds: Optional[int] = None,
) -> str:
    """Raise if stored ``config_hash`` differs from the requested configuration."""
    if isinstance(experiment_dir_or_manifest, Mapping):
        stored = dict(experiment_dir_or_manifest)
    else:
        path = Path(experiment_dir_or_manifest)
        manifest_path = path / MANIFEST_FILENAME
        checkpoint_path = path / CHECKPOINT_FILENAME
        if manifest_path.is_file():
            stored = _read_json_if_exists(manifest_path)
        elif checkpoint_path.is_file():
            stored = _read_json_if_exists(checkpoint_path)
        else:
            raise ExperimentPersistenceError(
                f"no manifest or checkpoint at {path}"
            )
    expected = screening_config_hash(
        neural_config=neural_config,
        v2_config=v2_config,
        architectures=architectures,
        seeds=seeds,
        origins=origins,
        product_universe_id=product_universe_id,
        product_universe_hash=product_universe_hash,
        min_successful_seeds=min_successful_seeds,
    )
    got = stored.get("config_hash")
    if got != expected:
        raise ExperimentConfigConflictError(
            f"config_hash mismatch: stored={got!r}, requested={expected!r}; "
            "refusing to append results generated with a different configuration"
        )
    return expected


def build_failures_dataframe(fold_metadata: pd.DataFrame) -> pd.DataFrame:
    """Fold rows with ``status != ok``; empty frame keeps failure headers."""
    empty = pd.DataFrame(columns=list(FAILURE_COLUMNS))
    if fold_metadata is None or fold_metadata.empty:
        return empty
    bad = fold_metadata.loc[fold_metadata["status"] != STATUS_OK].copy()
    if bad.empty:
        return empty
    out = pd.DataFrame(
        {
            "product": bad["product"].astype(str),
            "architecture": bad["architecture"].astype(str),
            "seed": bad["seed"],
            "origin": bad["origin"],
            "status": bad["status"].astype(str),
            "unavailable_reason": bad.get("unavailable_reason"),
            "error_type": bad.get("error_type"),
            "error_message": bad.get("error_message"),
        }
    )
    return out[list(FAILURE_COLUMNS)]


def build_horizon_metrics_dataframe(
    ensemble_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Long-form per-horizon metrics from ensemble OOF predictions."""
    empty = pd.DataFrame(columns=list(HORIZON_METRICS_COLUMNS))
    if ensemble_predictions is None or ensemble_predictions.empty:
        return empty
    rows: list[dict[str, Any]] = []
    keys = (
        ensemble_predictions[["product", "architecture"]]
        .drop_duplicates()
        .sort_values(["product", "architecture"])
    )
    for _, key in keys.iterrows():
        product = str(key["product"])
        architecture = str(key["architecture"])
        sub = ensemble_predictions.loc[
            (ensemble_predictions["product"] == product)
            & (ensemble_predictions["architecture"] == architecture)
        ]
        h_mae = horizon_mae(sub)
        h_rmse = horizon_rmse(sub)
        h_bias = horizon_bias(sub)
        h_wmape = horizon_wmape(sub)
        horizons = sorted(
            set(h_mae.index.astype(int).tolist())
            | set(h_rmse.index.astype(int).tolist())
            | set(h_bias.index.astype(int).tolist())
            | set(h_wmape.index.astype(int).tolist())
        )
        for h in horizons:
            rows.append(
                {
                    "product": product,
                    "architecture": architecture,
                    "horizon": int(h),
                    "mae": float(h_mae[h]) if h in h_mae.index else float("nan"),
                    "rmse": float(h_rmse[h]) if h in h_rmse.index else float("nan"),
                    "bias": float(h_bias[h]) if h in h_bias.index else float("nan"),
                    "wmape": float(h_wmape[h]) if h in h_wmape.index else float("nan"),
                }
            )
    return (
        pd.DataFrame(rows, columns=list(HORIZON_METRICS_COLUMNS))
        if rows
        else empty
    )


def build_screening_manifest(
    checkpoint: ScreeningExperimentCheckpoint,
    *,
    status: str,
    completed_at: Optional[str] = None,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Full audit manifest (includes provenance outside config_hash)."""
    cfg = checkpoint.neural_config
    arch_configs = {
        name: neural_config_to_dict(cfg.with_architecture(name))
        for name in checkpoint.architectures
    }
    return {
        "experiment_id": checkpoint.experiment_id,
        "created_at": checkpoint.created_at,
        "completed_at": completed_at,
        "status": str(status),
        "git_commit": git_commit if git_commit is not None else get_git_commit(),
        "v3a_version": V3A_VERSION,
        "config_hash": checkpoint.config_hash,
        "architectures": list(checkpoint.architectures),
        "architecture_configs": arch_configs,
        "seeds": list(checkpoint.seeds),
        "forecast_horizon": int(checkpoint.v2_config.forecast_horizon),
        "cv_configuration": {
            **v2_config_to_dict(checkpoint.v2_config),
            "origins": (
                list(checkpoint.origins)
                if checkpoint.origins is not None
                else "auto"
            ),
        },
        "eligibility_configuration": {
            "min_internal_train_windows": int(cfg.min_internal_train_windows),
            "min_internal_validation_windows": int(cfg.min_internal_validation_windows),
            "validation_fraction": float(cfg.validation_fraction),
            "min_successful_seeds": int(checkpoint.min_successful_seeds),
        },
        "scaling_method": cfg.scaling_method,
        "loss": cfg.loss,
        "optimizer": cfg.optimizer,
        "product_universe_id": checkpoint.product_universe_id,
        "product_universe_hash": checkpoint.product_universe_hash,
        "origins": (
            list(checkpoint.origins) if checkpoint.origins is not None else "auto"
        ),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "tensorflow_version": _tensorflow_version(),
        "package_versions": collect_package_versions(
            ("pandas", "numpy", "keras", "pyarrow", "sklearn")
        ),
        "artifact_files": list(ARTIFACT_FILES),
    }


def _write_result_artifacts(run_dir: Path, result: NeuralOuterBacktestResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    preds = result.predictions
    if preds is None:
        preds = pd.DataFrame()
    fold = result.fold_metadata
    if fold is None:
        fold = pd.DataFrame()
    arch_metrics = result.ensemble_metrics
    if arch_metrics is None or arch_metrics.empty:
        arch_metrics = result.metrics if result.metrics is not None else pd.DataFrame()
    seed_metrics = (
        result.seed_metrics if result.seed_metrics is not None else pd.DataFrame()
    )
    ens_preds = (
        result.ensemble_predictions
        if result.ensemble_predictions is not None
        else pd.DataFrame()
    )

    preds.to_parquet(run_dir / OOF_PREDICTIONS_NAME, index=False)
    fold.to_parquet(run_dir / FOLD_METADATA_NAME, index=False)
    arch_metrics.to_csv(run_dir / ARCHITECTURE_METRICS_NAME, index=False)
    build_horizon_metrics_dataframe(ens_preds).to_csv(
        run_dir / HORIZON_METRICS_NAME, index=False
    )
    seed_metrics.to_csv(run_dir / SEED_METRICS_NAME, index=False)
    build_failures_dataframe(fold).to_csv(run_dir / FAILURES_NAME, index=False)


def begin_screening_experiment(
    *,
    neural_config: Optional[NeuralExperimentConfig] = None,
    v2_config: Optional[TSForecastConfig] = None,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    origins: Optional[Sequence[int]] = None,
    product_universe_id: Optional[str] = None,
    product_universe_hash: Optional[str] = None,
    min_successful_seeds: Optional[int] = None,
    base_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    resume: bool = False,
) -> ScreeningExperimentCheckpoint:
    """Start or resume an incomplete screening experiment checkpoint."""
    cfg = neural_config or DEFAULT_CONFIG
    v2_cfg = v2_config or V2_DEFAULT_CONFIG
    root = Path(base_dir) if base_dir is not None else default_screening_root()
    root.mkdir(parents=True, exist_ok=True)

    arch_names = tuple(_normalize_architectures(architectures))
    seed_list = tuple(
        int(s) for s in (seeds if seeds is not None else cfg.random_seeds)
    )
    min_ok = int(
        min_successful_seeds if min_successful_seeds is not None else cfg.min_successful_seeds
    )
    origin_tuple = (
        tuple(int(o) for o in origins) if origins is not None else None
    )
    created = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created_iso = created.strftime("%Y-%m-%dT%H:%M:%SZ")
    eid = experiment_id or new_experiment_id(created_at=created)
    cfg_hash = screening_config_hash(
        neural_config=cfg,
        v2_config=v2_cfg,
        architectures=arch_names,
        seeds=seed_list,
        origins=origin_tuple,
        product_universe_id=product_universe_id,
        product_universe_hash=product_universe_hash,
        min_successful_seeds=min_ok,
    )

    complete_dir = complete_experiment_dir(root, eid)
    if complete_dir.exists() and is_complete_experiment(complete_dir):
        raise ExperimentImmutableError(
            f"experiment {eid!r} already completed at {complete_dir}; cannot restart"
        )

    inc_dir = incomplete_experiment_dir(root, eid)
    checkpoint_path = inc_dir / CHECKPOINT_FILENAME

    if resume:
        if not checkpoint_path.is_file():
            raise ExperimentCheckpointError(
                f"cannot resume: missing checkpoint at {checkpoint_path}"
            )
        stored = _read_json_if_exists(checkpoint_path)
        if stored.get("config_hash") != cfg_hash:
            raise ExperimentConfigConflictError(
                f"config_hash mismatch for experiment {eid!r}: "
                f"stored={stored.get('config_hash')!r}, requested={cfg_hash!r}"
            )
        return ScreeningExperimentCheckpoint(
            experiment_id=eid,
            config_hash=cfg_hash,
            created_at=str(stored.get("created_at", created_iso)),
            experiment_dir=inc_dir,
            neural_config=cfg,
            v2_config=v2_cfg,
            architectures=arch_names,
            seeds=seed_list,
            origins=origin_tuple,
            product_universe_id=product_universe_id,
            product_universe_hash=product_universe_hash,
            min_successful_seeds=min_ok,
        )

    if inc_dir.exists() and any(inc_dir.iterdir()):
        # Existing incomplete with different hash → conflict; same hash needs resume.
        existing = _read_json_if_exists(checkpoint_path)
        if existing.get("config_hash") and existing.get("config_hash") != cfg_hash:
            raise ExperimentConfigConflictError(
                f"incomplete experiment {eid!r} exists with config_hash="
                f"{existing.get('config_hash')!r}; requested={cfg_hash!r}"
            )
        raise ExperimentCheckpointError(
            f"incomplete experiment directory already exists: {inc_dir}; use resume=True"
        )

    inc_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "experiment_id": eid,
        "config_hash": cfg_hash,
        "created_at": created_iso,
        "status": INCOMPLETE_STATUS,
        "v3a_version": V3A_VERSION,
        "architectures": list(arch_names),
        "seeds": list(seed_list),
        "origins": list(origin_tuple) if origin_tuple is not None else "auto",
        "min_successful_seeds": min_ok,
        "product_universe_id": product_universe_id,
        "product_universe_hash": product_universe_hash,
    }
    _write_json(checkpoint_path, checkpoint_payload)
    return ScreeningExperimentCheckpoint(
        experiment_id=eid,
        config_hash=cfg_hash,
        created_at=created_iso,
        experiment_dir=inc_dir,
        neural_config=cfg,
        v2_config=v2_cfg,
        architectures=arch_names,
        seeds=seed_list,
        origins=origin_tuple,
        product_universe_id=product_universe_id,
        product_universe_hash=product_universe_hash,
        min_successful_seeds=min_ok,
    )


def write_screening_checkpoint(
    checkpoint: ScreeningExperimentCheckpoint,
    result: NeuralOuterBacktestResult,
    *,
    completed_products: Optional[Sequence[str]] = None,
) -> None:
    """Write screening artifacts into the incomplete experiment directory."""
    if checkpoint.is_complete:
        raise ExperimentImmutableError(
            f"experiment {checkpoint.experiment_id!r} is already complete"
        )
    root = checkpoint.experiment_dir.parents[1]
    complete_dir = complete_experiment_dir(root, checkpoint.experiment_id)
    if complete_dir.exists() and is_complete_experiment(complete_dir):
        raise ExperimentImmutableError(
            f"completed experiment exists for {checkpoint.experiment_id!r}; "
            "refusing checkpoint write"
        )
    assert_compatible_config(
        checkpoint.experiment_dir,
        neural_config=checkpoint.neural_config,
        v2_config=checkpoint.v2_config,
        architectures=checkpoint.architectures,
        seeds=checkpoint.seeds,
        origins=checkpoint.origins,
        product_universe_id=checkpoint.product_universe_id,
        product_universe_hash=checkpoint.product_universe_hash,
        min_successful_seeds=checkpoint.min_successful_seeds,
    )
    _write_result_artifacts(checkpoint.experiment_dir, result)
    # Incomplete manifest snapshot (status incomplete; no .complete).
    manifest = build_screening_manifest(
        checkpoint,
        status=INCOMPLETE_STATUS,
        completed_at=None,
        git_commit=get_git_commit(),
    )
    _write_json(checkpoint.experiment_dir / MANIFEST_FILENAME, manifest)

    # Progress marker for per-SKU resume (also mirrors fold_metadata products).
    checkpoint_path = checkpoint.experiment_dir / CHECKPOINT_FILENAME
    stored = _read_json_if_exists(checkpoint_path)
    if not stored:
        stored = {
            "experiment_id": checkpoint.experiment_id,
            "config_hash": checkpoint.config_hash,
            "created_at": checkpoint.created_at,
            "status": INCOMPLETE_STATUS,
            "v3a_version": V3A_VERSION,
        }
    if completed_products is not None:
        stored["completed_products"] = [str(p) for p in completed_products]
    elif result.fold_metadata is not None and not result.fold_metadata.empty:
        stored["completed_products"] = sorted(
            result.fold_metadata["product"].astype(str).unique().tolist()
        )
    stored["status"] = INCOMPLETE_STATUS
    stored["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json(checkpoint_path, stored)


def list_completed_products(experiment_dir: Path) -> list[str]:
    """Products already checkpointed under an incomplete experiment directory."""
    exp = Path(experiment_dir)
    stored = _read_json_if_exists(exp / CHECKPOINT_FILENAME)
    raw = stored.get("completed_products")
    if isinstance(raw, list) and raw:
        return [str(p) for p in raw]
    fold_path = exp / FOLD_METADATA_NAME
    if fold_path.is_file():
        fold = pd.read_parquet(fold_path)
        if not fold.empty and "product" in fold.columns:
            return sorted(fold["product"].astype(str).unique().tolist())
    return []


def load_incomplete_screening_result(
    experiment_dir: Path,
) -> NeuralOuterBacktestResult:
    """Reload a partial NeuralOuterBacktestResult from an incomplete checkpoint."""
    exp = Path(experiment_dir)
    oof_path = exp / OOF_PREDICTIONS_NAME
    fold_path = exp / FOLD_METADATA_NAME
    if not oof_path.is_file() or not fold_path.is_file():
        return NeuralOuterBacktestResult(
            predictions=pd.DataFrame(),
            fold_metadata=pd.DataFrame(),
            metrics=pd.DataFrame(),
            ensemble_predictions=pd.DataFrame(),
            ensemble_origin_status=pd.DataFrame(),
            seed_metrics=pd.DataFrame(),
            ensemble_metrics=pd.DataFrame(),
            stability=pd.DataFrame(),
        )
    preds = pd.read_parquet(oof_path)
    fold = pd.read_parquet(fold_path)
    # Metrics CSVs are optional for resume; reassemble happens after next product.
    return NeuralOuterBacktestResult(
        predictions=preds,
        fold_metadata=fold,
        metrics=pd.DataFrame(),
        ensemble_predictions=pd.DataFrame(),
        ensemble_origin_status=pd.DataFrame(),
        seed_metrics=pd.DataFrame(),
        ensemble_metrics=pd.DataFrame(),
        stability=pd.DataFrame(),
    )


def finalize_screening_experiment(
    checkpoint: ScreeningExperimentCheckpoint,
    result: NeuralOuterBacktestResult,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    """Write artifacts and atomically promote incomplete → immutable completed."""
    root = (
        Path(base_dir)
        if base_dir is not None
        else checkpoint.experiment_dir.parents[1]
    )
    complete_dir = complete_experiment_dir(root, checkpoint.experiment_id)
    if complete_dir.exists():
        if is_complete_experiment(complete_dir):
            raise ExperimentImmutableError(
                f"experiment {checkpoint.experiment_id!r} already finalized at "
                f"{complete_dir}"
            )
        raise ExperimentCheckpointError(
            f"cannot finalize: non-complete path exists at {complete_dir}"
        )

    write_screening_checkpoint(checkpoint, result)

    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_screening_manifest(
        checkpoint,
        status=COMPLETE_STATUS,
        completed_at=completed_at,
        git_commit=get_git_commit(),
    )
    _write_json(checkpoint.experiment_dir / MANIFEST_FILENAME, manifest)
    (checkpoint.experiment_dir / COMPLETE_MARKER).write_text("", encoding="utf-8")
    # Resume checkpoints apply only under .incomplete/; drop stale incomplete marker.
    checkpoint_path = checkpoint.experiment_dir / CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        checkpoint_path.unlink()

    complete_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(checkpoint.experiment_dir, complete_dir)
    return complete_dir


def persist_completed_screening_experiment(
    result: NeuralOuterBacktestResult,
    *,
    neural_config: Optional[NeuralExperimentConfig] = None,
    v2_config: Optional[TSForecastConfig] = None,
    architectures: Optional[Sequence[Union[str, ArchitectureName]]] = None,
    seeds: Optional[Sequence[int]] = None,
    origins: Optional[Sequence[int]] = None,
    product_universe_id: Optional[str] = None,
    product_universe_hash: Optional[str] = None,
    min_successful_seeds: Optional[int] = None,
    base_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> Path:
    """Begin, write, and finalize a screening experiment in one step."""
    checkpoint = begin_screening_experiment(
        neural_config=neural_config,
        v2_config=v2_config,
        architectures=architectures,
        seeds=seeds,
        origins=origins,
        product_universe_id=product_universe_id,
        product_universe_hash=product_universe_hash,
        min_successful_seeds=min_successful_seeds,
        base_dir=base_dir,
        experiment_id=experiment_id,
        created_at=created_at,
    )
    return finalize_screening_experiment(
        checkpoint, result, base_dir=base_dir
    )


@dataclass
class LoadedScreeningExperiment:
    """Completed screening experiment loaded from disk."""

    experiment_id: str
    experiment_dir: Path
    manifest: dict[str, Any]
    oof_predictions: pd.DataFrame
    fold_metadata: pd.DataFrame
    architecture_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    seed_metrics: pd.DataFrame
    failures: pd.DataFrame


def load_screening_experiment(
    experiment_id: str,
    *,
    base_dir: Optional[Path] = None,
) -> LoadedScreeningExperiment:
    """Load a completed screening experiment; requires ``.complete``."""
    root = Path(base_dir) if base_dir is not None else default_screening_root()
    exp_dir = complete_experiment_dir(root, experiment_id)
    if not exp_dir.is_dir():
        raise ExperimentPersistenceError(f"experiment directory not found: {exp_dir}")
    if not is_complete_experiment(exp_dir):
        raise ExperimentPersistenceError(
            f"experiment {experiment_id!r} is not marked complete at {exp_dir}"
        )
    manifest = _read_json_if_exists(exp_dir / MANIFEST_FILENAME)
    return LoadedScreeningExperiment(
        experiment_id=str(experiment_id),
        experiment_dir=exp_dir,
        manifest=manifest,
        oof_predictions=pd.read_parquet(exp_dir / OOF_PREDICTIONS_NAME),
        fold_metadata=pd.read_parquet(exp_dir / FOLD_METADATA_NAME),
        architecture_metrics=pd.read_csv(exp_dir / ARCHITECTURE_METRICS_NAME),
        horizon_metrics=pd.read_csv(exp_dir / HORIZON_METRICS_NAME),
        seed_metrics=pd.read_csv(exp_dir / SEED_METRICS_NAME),
        failures=pd.read_csv(exp_dir / FAILURES_NAME),
    )


__all__ = [
    "ARTIFACT_FILES",
    "COMPLETE_MARKER",
    "COMPLETE_STATUS",
    "INCOMPLETE_STATUS",
    "MANIFEST_FILENAME",
    "V3A_VERSION",
    "ExperimentCheckpointError",
    "ExperimentConfigConflictError",
    "ExperimentImmutableError",
    "ExperimentPersistenceError",
    "LoadedScreeningExperiment",
    "ScreeningExperimentCheckpoint",
    "assert_compatible_config",
    "begin_screening_experiment",
    "build_failures_dataframe",
    "build_horizon_metrics_dataframe",
    "build_screening_manifest",
    "complete_experiment_dir",
    "default_screening_root",
    "finalize_screening_experiment",
    "get_git_commit",
    "incomplete_experiment_dir",
    "is_complete_experiment",
    "list_completed_products",
    "load_incomplete_screening_result",
    "load_screening_experiment",
    "new_experiment_id",
    "persist_completed_screening_experiment",
    "screening_config_hash",
    "screening_config_payload",
    "write_screening_checkpoint",
]
