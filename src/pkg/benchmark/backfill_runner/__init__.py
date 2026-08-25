"""Generic historical forecast backfill runner (orchestration only).

Durable SQLite checkpoints, exclusive run lock, and engine adapters (v1/v2/dummy).
SKU-vintage jobs may run concurrently via ``--workers`` (default 1).

Artifacts::

    data/backfills/{experiment_id}/{engine}/
        manifest.json
        state.sqlite
        forecasts/
        backtests/
        logs/

CLI::

    python -m pkg.benchmark.backfill_runner \\
        --engine v2 \\
        --vintages ts_backfill_1401Q1_1405Q2 \\
        --universe mvp_products \\
        --resume \\
        --workers 4

    python -m pkg.benchmark.backfill_runner ... --status
    python -m pkg.benchmark.backfill_runner ... --retry-failed
    python -m pkg.benchmark.backfill_runner ... --dry-run
"""
from __future__ import annotations

from pkg.benchmark.backfill_runner.manifest import (
    ExperimentManifestError,
    make_experiment_id,
)
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
    "ExperimentManifestError",
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
    "make_experiment_id",
    "print_status",
    "run_backfill",
]
