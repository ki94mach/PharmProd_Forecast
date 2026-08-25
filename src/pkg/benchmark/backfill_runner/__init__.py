"""Generic historical forecast backfill runner (orchestration only).

CLI::

    python -m pkg.benchmark.backfill_runner \\
        --engine v2 \\
        --vintages ts_backfill_1401Q1_1405Q2 \\
        --universe mvp_products \\
        --resume

    python -m pkg.benchmark.backfill_runner --engine dummy --dry-run ...
"""
from __future__ import annotations

from pkg.benchmark.backfill_runner.runner import (
    BackfillPlan,
    BackfillRunSummary,
    build_backfill_plan,
    enforce_historical_cutoff,
    run_backfill,
)
from pkg.benchmark.backfill_runner.store import BackfillStore, default_backfill_root
from pkg.benchmark.backfill_runner.types import (
    EngineJobRequest,
    EngineJobResult,
    ForecastEngine,
    JobKey,
    JobLogRecord,
)

__all__ = [
    "BackfillPlan",
    "BackfillRunSummary",
    "BackfillStore",
    "EngineJobRequest",
    "EngineJobResult",
    "ForecastEngine",
    "JobKey",
    "JobLogRecord",
    "build_backfill_plan",
    "default_backfill_root",
    "enforce_historical_cutoff",
    "run_backfill",
]
