"""Versioned forecast benchmark: frozen dataset + ``backtest`` API.

Example::

    from pkg.benchmark import backtest, scoreboard

    result = backtest("ts")           # Analysis B PRIMARY matched WMAPE
    table = scoreboard()              # TS / Human / TS+XGB / Human+XGB / Integrated

Heavy imports (XGBoost, evaluate) are lazy so subpackages such as
``backfill_runner`` / ``universes`` / ``vintages`` can load without the
full research stack (needed on pip-only V2 backfill servers).
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "BacktestResult",
    "BenchmarkDataset",
    "backtest",
    "load_benchmark",
    "scoreboard",
    "wmape",
]


def __getattr__(name: str) -> Any:
    if name in {"BacktestResult", "backtest", "scoreboard", "wmape"}:
        from pkg.benchmark.evaluate import (
            BacktestResult,
            backtest,
            scoreboard,
            wmape,
        )

        mapping = {
            "BacktestResult": BacktestResult,
            "backtest": backtest,
            "scoreboard": scoreboard,
            "wmape": wmape,
        }
        return mapping[name]
    if name in {"BenchmarkDataset", "load_benchmark"}:
        from pkg.benchmark.dataset import BenchmarkDataset, load_benchmark

        return {"BenchmarkDataset": BenchmarkDataset, "load_benchmark": load_benchmark}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
