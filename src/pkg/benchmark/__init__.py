"""Versioned forecast benchmark: frozen dataset + ``backtest`` API.

Example::

    from pkg.benchmark import backtest, scoreboard

    result = backtest("ts")           # Analysis B PRIMARY matched WMAPE
    table = scoreboard()              # TS / Human / TS+XGB / Human+XGB / Integrated
"""
from pkg.benchmark.evaluate import BacktestResult, backtest, scoreboard, wmape
from pkg.benchmark.dataset import BenchmarkDataset, load_benchmark

__all__ = [
    "BacktestResult",
    "BenchmarkDataset",
    "backtest",
    "load_benchmark",
    "scoreboard",
    "wmape",
]
