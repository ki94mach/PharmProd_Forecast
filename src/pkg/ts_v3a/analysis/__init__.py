"""Offline analysis of completed V3A screening experiments.

Always use bracket column access (``df["product"]``), never attribute access
like ``df.product`` (pandas shadows ``DataFrame.product``).
"""
from __future__ import annotations

from pkg.ts_v3a.analysis.metrics import (
    architecture_summary,
    history_bucket,
    paired_architecture_comparisons,
    portfolio_wmape,
    slice_summaries,
)
from pkg.ts_v3a.analysis.panel import (
    DEFAULT_VALIDATION_ORIGINS,
    build_validation_panel,
    write_validation_panel,
)
from pkg.ts_v3a.analysis.recigen import diagnose_recigen
from pkg.ts_v3a.analysis.report import write_validation_report

__all__ = [
    "DEFAULT_VALIDATION_ORIGINS",
    "architecture_summary",
    "build_validation_panel",
    "diagnose_recigen",
    "history_bucket",
    "paired_architecture_comparisons",
    "portfolio_wmape",
    "slice_summaries",
    "write_validation_panel",
    "write_validation_report",
]
