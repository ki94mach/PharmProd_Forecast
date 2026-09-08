"""Run directory layout and artifact persistence for forecast_jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from pkg.benchmark.backfill_runner.state import JobIdentity
from pkg.benchmark.backfill_runner.types import EngineJobResult
from pkg.forecast_jobs.config import JobRunConfig, config_to_dict

MANIFEST_FILENAME = "manifest.json"
CONFIG_COPY_FILENAME = "config.resolved.json"


def run_dir(output_dir: Path, run_id: str) -> Path:
    return Path(output_dir) / str(run_id)


def job_artifact_dir(run_root: Path, identity: JobIdentity) -> Path:
    return (
        Path(run_root)
        / "jobs"
        / str(identity.engine_version)
        / str(identity.slug)
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def write_resolved_config(run_root: Path, config: JobRunConfig) -> Path:
    path = Path(run_root) / CONFIG_COPY_FILENAME
    atomic_write_json(path, config_to_dict(config))
    return path


def write_manifest(
    run_root: Path,
    *,
    config: JobRunConfig,
    config_hash: str,
    git_commit: str,
    input_snapshot: Mapping[str, Any],
    status: str,
) -> Path:
    payload = {
        "run_id": config.run_id,
        "schema_version": config.schema_version,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "status": status,
        "architectures": list(config.architectures),
        "origins": list(config.origins),
        "seeds": list(config.seeds),
        "horizon": config.horizon,
        "cutoff_policy": config.cutoff_policy.name,
        "output_dir": Path(config.output_dir).as_posix(),
        "input_snapshot": dict(input_snapshot),
        "scientific_config": config.scientific_payload(),
    }
    path = Path(run_root) / MANIFEST_FILENAME
    atomic_write_json(path, payload)
    return path


def read_manifest(run_root: Path) -> Optional[dict[str, Any]]:
    path = Path(run_root) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def persist_job_result(
    run_root: Path,
    identity: JobIdentity,
    result: EngineJobResult,
    *,
    log_payload: Mapping[str, Any],
) -> Path:
    """Write one job's forecast CSV + log JSON under the run directory."""
    out_dir = job_artifact_dir(run_root, identity)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "job.json", dict(log_payload))
    if result.success and result.forecasts is not None and not result.forecasts.empty:
        result.forecasts.to_csv(out_dir / "forecast.csv", index=False)
    elif not result.success:
        atomic_write_json(
            out_dir / "error.json",
            {
                "error_type": result.error_type,
                "error_message": result.error_message,
            },
        )
    return out_dir
