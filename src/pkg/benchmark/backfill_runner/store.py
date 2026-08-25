"""Atomic artifact persistence for historical backfills (SQLite owns job status).

Layout (isolated from production forecast CSVs)::

    data/backfills/{experiment_id}/{engine}/
        manifest.json
        state.sqlite
        run.lock
        forecasts/{quarter}__{product}/
            forecast.csv
            result.json
            .complete
        backtests/{quarter}__{product}/   # optional selection/CV summaries
        logs/{quarter}__{product}/
            job_log.json
            failure.json
        logs/run_meta.json
"""
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
FAILURE_JSON = "failure.json"
BACKTEST_JSON = "backtest_summary.json"


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
    """Historical backfill root — not production forecast export paths."""
    root = Path(__file__).resolve().parents[3] / "data" / "backfills"
    root.mkdir(parents=True, exist_ok=True)
    return root


class BackfillStore:
    """Per-experiment / per-engine artifact tree under ``data/backfills``."""

    def __init__(self, experiment_dir: Path, engine: str):
        self.experiment_dir = Path(experiment_dir)
        self.engine = str(engine)
        self.forecasts_root = self.experiment_dir / "forecasts"
        self.backtests_root = self.experiment_dir / "backtests"
        self.logs_root = self.experiment_dir / "logs"
        for path in (self.forecasts_root, self.backtests_root, self.logs_root):
            path.mkdir(parents=True, exist_ok=True)

    def forecast_dir(self, identity: JobIdentity) -> Path:
        return self.forecasts_root / identity.slug

    def backtest_dir(self, identity: JobIdentity) -> Path:
        return self.backtests_root / identity.slug

    def log_dir(self, identity: JobIdentity) -> Path:
        return self.logs_root / identity.slug

    # Back-compat alias used by older call sites / tests.
    def job_dir(self, identity: JobIdentity) -> Path:
        return self.forecast_dir(identity)

    def has_complete_artifacts(self, identity: JobIdentity) -> bool:
        job = self.forecast_dir(identity)
        return (job / COMPLETE_MARKER).exists() and (job / FORECAST_CSV).exists()

    def clear_artifacts(self, identity: JobIdentity) -> None:
        for root, names in (
            (self.forecast_dir(identity), (COMPLETE_MARKER, FORECAST_CSV, RESULT_JSON)),
            (self.log_dir(identity), (LOG_JSON, FAILURE_JSON)),
            (self.backtest_dir(identity), (BACKTEST_JSON,)),
        ):
            if not root.exists():
                continue
            for name in names:
                path = root / name
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
        forecast_dir = self.forecast_dir(identity)
        log_dir = self.log_dir(identity)
        forecast_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if result.forecasts is None or result.forecasts.empty:
            raise BackfillStoreError(f"successful job missing forecasts: {identity.slug}")

        atomic_write_csv(forecast_dir / FORECAST_CSV, result.forecasts)
        atomic_write_json(
            forecast_dir / RESULT_JSON,
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
        atomic_write_json(log_dir / LOG_JSON, log_payload)
        atomic_write_text(
            forecast_dir / COMPLETE_MARKER,
            log_payload.get("finished_at", "") + "\n",
        )

        # Optional backtest/selection summary when the engine provides one.
        extras = dict(result.extras or {})
        if extras.get("backtest_summary") is not None or extras.get("selection"):
            bt = self.backtest_dir(identity)
            bt.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                bt / BACKTEST_JSON,
                {
                    "job_id": identity.job_id,
                    "product_id": identity.product_id,
                    "quarter": identity.quarter,
                    "selected_model": result.selected_model,
                    "backtest_summary": extras.get("backtest_summary"),
                    "selection": extras.get("selection"),
                    "selected_strategy": extras.get("selected_strategy"),
                },
            )
        return forecast_dir

    def persist_failure(
        self,
        identity: JobIdentity,
        log_payload: dict[str, Any],
    ) -> Path:
        log_dir = self.log_dir(identity)
        log_dir.mkdir(parents=True, exist_ok=True)
        # No .complete marker — FAILED jobs are retryable.
        forecast_dir = self.forecast_dir(identity)
        marker = forecast_dir / COMPLETE_MARKER
        if marker.exists():
            marker.unlink()
        atomic_write_json(log_dir / LOG_JSON, log_payload)
        atomic_write_json(
            log_dir / FAILURE_JSON,
            {
                "success": False,
                "job_id": identity.job_id,
                "error_type": log_payload.get("error_type"),
                "error_message": log_payload.get("error_message"),
                "finished_at": log_payload.get("finished_at"),
            },
        )
        return log_dir
