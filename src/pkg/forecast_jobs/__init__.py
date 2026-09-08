"""Resumable server job runner for V2.1, A0, and A1.

Reads a versioned YAML (or JSON) configuration, validates a frozen sales
Parquet input snapshot, and executes product × origin × architecture jobs with
SQLite checkpoints, an exclusive run lock, and graceful SIGTERM handling.

Usage::

    python -m pkg.forecast_jobs --config configs/forecast_jobs/example_v21_a0_a1.yaml --validate
    python -m pkg.forecast_jobs --config ... --dry-run
    python -m pkg.forecast_jobs --config ... --execute --resume

Outputs land under ``src/data/forecast_jobs/{run_id}/`` (configurable).
"""
from __future__ import annotations

from pkg.forecast_jobs.config import JobRunConfig, load_job_config
from pkg.forecast_jobs.snapshot import SalesInputSnapshot, fingerprint_sales_parquet

__all__ = [
    "JobRunConfig",
    "SalesInputSnapshot",
    "fingerprint_sales_parquet",
    "load_job_config",
]
