"""Generic historical forecast backfill runner (orchestration only).

Durable SQLite checkpoints, exclusive run lock, and engine adapters (v1/v2/dummy).

CLI::

    python -m pkg.benchmark.backfill_runner \\
        --engine v2 \\
        --vintages ts_backfill_1401Q1_1405Q2 \\
        --universe mvp_products \\
        --resume

    python -m pkg.benchmark.backfill_runner ... --status
    python -m pkg.benchmark.backfill_runner ... --retry-failed
"""
from __future__ import annotations

from pkg.benchmark.backfill_runner.runner import (
    BackfillPlan,
    BackfillRunSummary,
    build_backfill_plan,
    enforce_historical_cutoff,
    print_status,
    run_backfill,
)
from pkg.benchmark.backfill_runner.state import (
    JOB_FAILED,
    JOB_PENDING,
    JOB_RUNNING,
    JOB_SUCCESS,
    JobIdentity,
    JobStateStore,
    RunLock,
    RunLockError,
)
from pkg.benchmark.backfill_runner.store import BackfillStore, default_backfill_root
from pkg.benchmark.backfill_runner.types import (
    EngineJobRequest,
    EngineJobResult,
    ForecastEngine,
    JobLogRecord,
)

__all__ = [
    "BackfillPlan",
    "BackfillRunSummary",
    "BackfillStore",
    "EngineJobRequest",
    "EngineJobResult",
    "ForecastEngine",
    "JOB_FAILED",
    "JOB_PENDING",
    "JOB_RUNNING",
    "JOB_SUCCESS",
    "JobIdentity",
    "JobLogRecord",
    "JobStateStore",
    "RunLock",
    "RunLockError",
    "build_backfill_plan",
    "default_backfill_root",
    "enforce_historical_cutoff",
    "print_status",
    "run_backfill",
]
