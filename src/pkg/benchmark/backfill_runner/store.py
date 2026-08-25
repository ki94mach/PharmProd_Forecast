"""Atomic job persistence and resume ledger for historical backfills."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from pkg.benchmark.backfill_runner.types import EngineJobResult, JobKey, JobLogRecord

COMPLETE_MARKER = ".complete"
FORECAST_CSV = "forecast.csv"
RESULT_JSON = "result.json"
LOG_JSON = "job_log.json"
STATUS_CSV = "status_ledger.csv"
RUN_META = "run_metadata.json"


class BackfillStoreError(Exception):
    """Persistence / immutability error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
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
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


class BackfillStore:
    """Per-engine output tree with immutable completed jobs.

    Layout::

        {root}/{engine}/
            run_metadata.json
            status_ledger.csv
            jobs/{quarter}__{product}/
                forecast.csv
                result.json
                job_log.json
                .complete
    """

    def __init__(self, root: Path, engine: str):
        self.root = Path(root)
        self.engine = str(engine)
        self.engine_root = self.root / self.engine
        self.jobs_root = self.engine_root / "jobs"
        self.ledger_path = self.engine_root / STATUS_CSV
        self.engine_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, key: JobKey) -> Path:
        safe_product = str(key.product).replace("/", "_").replace("\\", "_")
        return self.jobs_root / f"{key.quarter}__{safe_product}"

    def is_complete(self, key: JobKey) -> bool:
        job = self.job_dir(key)
        marker = job / COMPLETE_MARKER
        forecast = job / FORECAST_CSV
        # Partial CSV alone is NOT completion.
        return marker.exists() and forecast.exists()

    def assert_writable(self, key: JobKey, *, resume: bool) -> None:
        if self.is_complete(key):
            if resume:
                return
            raise BackfillStoreError(
                f"Completed job is immutable: {self.job_dir(key)}. "
                "Use --resume to skip, or a new output root."
            )

    def write_run_metadata(self, meta: dict[str, Any]) -> None:
        path = self.engine_root / RUN_META
        if path.exists():
            # Append-safe: keep first run meta; write sidecar with timestamp.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            atomic_write_json(self.engine_root / f"run_metadata_{stamp}.json", meta)
            return
        atomic_write_json(path, meta)

    def persist_success(
        self,
        key: JobKey,
        result: EngineJobResult,
        log: JobLogRecord,
    ) -> Path:
        if self.is_complete(key):
            raise BackfillStoreError(f"Refusing to overwrite completed job: {key.slug}")
        job = self.job_dir(key)
        job.mkdir(parents=True, exist_ok=True)

        if result.forecasts is None or result.forecasts.empty:
            raise BackfillStoreError(f"successful job missing forecasts: {key.slug}")

        # Write payload first; marker last (completion = marker + forecast).
        atomic_write_csv(job / FORECAST_CSV, result.forecasts)
        atomic_write_json(
            job / RESULT_JSON,
            {
                "success": True,
                "product": result.product,
                "quarter": result.quarter,
                "forecast_origin": result.forecast_origin,
                "selected_model": result.selected_model,
                "extras": dict(result.extras),
            },
        )
        atomic_write_json(job / LOG_JSON, asdict(log))
        atomic_write_text(job / COMPLETE_MARKER, _utc_now() + "\n")
        self._append_ledger(log)
        return job

    def persist_failure(self, key: JobKey, log: JobLogRecord) -> Path:
        job = self.job_dir(key)
        job.mkdir(parents=True, exist_ok=True)
        # Failures are resumable: no .complete marker.
        atomic_write_json(job / LOG_JSON, asdict(log))
        fail_path = job / "failure.json"
        atomic_write_json(
            fail_path,
            {
                "success": False,
                "error_message": log.error_message,
                "logged_at_utc": log.end_time_utc,
            },
        )
        self._append_ledger(log)
        return job

    def completed_keys(self) -> set[tuple[str, str]]:
        """Return set of (quarter, product) with valid completion markers."""
        done: set[tuple[str, str]] = set()
        if not self.jobs_root.exists():
            return done
        for job in self.jobs_root.iterdir():
            if not job.is_dir():
                continue
            if not (job / COMPLETE_MARKER).exists():
                continue
            if not (job / FORECAST_CSV).exists():
                continue
            # Directory name: {quarter}__{product}
            name = job.name
            if "__" not in name:
                continue
            qrt, product = name.split("__", 1)
            done.add((qrt, product))
        return done

    def _append_ledger(self, log: JobLogRecord) -> None:
        row = pd.DataFrame([asdict(log)])
        if self.ledger_path.exists():
            prev = pd.read_csv(self.ledger_path)
            out = pd.concat([prev, row], ignore_index=True)
        else:
            out = row
        atomic_write_csv(self.ledger_path, out)


def default_backfill_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "backfill"
    root.mkdir(parents=True, exist_ok=True)
    return root
