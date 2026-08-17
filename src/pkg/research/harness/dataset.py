"""In-memory freeze-panel copies. Never writes parquet."""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from pkg.benchmark.dataset import BenchmarkDataset


def resolve_origin_col(df: pd.DataFrame, origin_col: Optional[str] = None) -> str:
    if origin_col is not None:
        return origin_col
    if "origin" in df.columns:
        return "origin"
    if "ts_origin" in df.columns:
        return "ts_origin"
    if "budget_origin" in df.columns:
        return "budget_origin"
    raise ValueError("panel needs origin / ts_origin / budget_origin")


def copy_dataset(ds: BenchmarkDataset) -> BenchmarkDataset:
    return BenchmarkDataset(
        version=ds.version,
        root=ds.root,
        ts_universe=ds.ts_universe.copy(),
        budget_universe=ds.budget_universe.copy(),
        matched_universe=ds.matched_universe.copy(),
        manifest=ds.manifest,
    )


def enrich_dataset(
    ds: BenchmarkDataset,
    fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> BenchmarkDataset:
    """Attach features to copies of all three universes. Does not write to disk."""
    copied = copy_dataset(ds)
    return BenchmarkDataset(
        version=copied.version,
        root=copied.root,
        ts_universe=fn(copied.ts_universe),
        budget_universe=fn(copied.budget_universe),
        matched_universe=fn(copied.matched_universe),
        manifest=copied.manifest,
    )
