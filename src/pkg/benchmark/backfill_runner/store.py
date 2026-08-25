"""Atomic artifact persistence for historical backfills (SQLite owns job status)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.benchmark.backfill_runner.state import JobIdentity
from pkg.benchmark.backfill_runner.types import EngineJobResult

COMPLETE_MARKER = ".complete"
FORECAST_CSV = "forecast.csv"
RESULT_JSON = "result.json"
LOG_JSON = "job_log.json"


class BackfillStoreError(Exception):
    """Artifact persistence / immutability error."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def default_backfill_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "backfill"
    root.mkdir(parents=True, exist_ok=True)
    return root


class BackfillStore:
    """Per-experiment artifact tree.

    Layout::

        {root}/{experiment_id}/
            backfill.sqlite
            run.lock
            artifacts/{engine}/{quarter}__{product}/
                forecast.csv
                result.json
                job_log.json
                .complete
    """

    def __init__(self, experiment_dir: Path, engine: str):
        self.experiment_dir = Path(experiment_dir)
        self.engine = str(engine)
        self.artifacts_root = self.experiment_dir / "artifacts" / self.engine
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, identity: JobIdentity) -> Path:
        return self.artifacts_root / identity.slug

    def has_complete_artifacts(self, identity: JobIdentity) -> bool:
        job = self.job_dir(identity)
        return (job / COMPLETE_MARKER).exists() and (job / FORECAST_CSV).exists()

    def clear_artifacts(self, identity: JobIdentity) -> None:
        job = self.job_dir(identity)
        if not job.exists():
            return
        for name in (COMPLETE_MARKER, FORECAST_CSV, RESULT_JSON, LOG_JSON, "failure.json"):
            path = job / name
            if path.exists():
                path.unlink()

    def persist_success(
        self,
        identity: JobIdentity,
        result: EngineJobResult,
        log_payload: dict[str, Any],
    ) -> Path:
        if self.has_complete_artifacts(identity):
            raise BackfillStoreError(
                f"Refusing to overwrite completed artifacts: {identity.slug}. "
                "Use --force-job."
            )
        job = self.job_dir(identity)
        job.mkdir(parents=True, exist_ok=True)
        if result.forecasts is None or result.forecasts.empty:
            raise BackfillStoreError(f"successful job missing forecasts: {identity.slug}")

        atomic_write_csv(job / FORECAST_CSV, result.forecasts)
        atomic_write_json(
            job / RESULT_JSON,
            {
                "success": True,
                "job_id": identity.job_id,
                "product_id": identity.product_id,
                "quarter": identity.quarter,
                "forecast_origin": identity.forecast_origin,
                "engine_version": identity.engine_version,
                "config_hash": identity.config_hash,
                "selected_model": result.selected_model,
                "extras": dict(result.extras),
            },
        )
        atomic_write_json(job / LOG_JSON, log_payload)
        atomic_write_text(job / COMPLETE_MARKER, log_payload.get("finished_at", "") + "\n")
        return job

    def persist_failure(
        self,
        identity: JobIdentity,
        log_payload: dict[str, Any],
    ) -> Path:
        job = self.job_dir(identity)
        job.mkdir(parents=True, exist_ok=True)
        # No .complete marker — FAILED jobs are retryable.
        marker = job / COMPLETE_MARKER
        if marker.exists():
            marker.unlink()
        atomic_write_json(job / LOG_JSON, log_payload)
        atomic_write_json(
            job / "failure.json",
            {
                "success": False,
                "job_id": identity.job_id,
                "error_type": log_payload.get("error_type"),
                "error_message": log_payload.get("error_message"),
                "finished_at": log_payload.get("finished_at"),
            },
        )
        return job
