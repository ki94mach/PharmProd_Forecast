"""Load frozen benchmark panels (offline; no SQL / results CSVs)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from pkg.benchmark.config import (
    BENCHMARK_VERSION,
    PANEL_FILES,
    PRIMARY_ORIGINS,
    default_benchmark_root,
    manifest_path,
)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Optional[Path] = None) -> dict:
    p = path or manifest_path()
    with p.open(encoding="utf-8") as f:
        return json.load(f)


@dataclass
class BenchmarkDataset:
    """Frozen panels for one benchmark version."""

    version: str
    root: Path
    ts_universe: pd.DataFrame
    budget_universe: pd.DataFrame
    matched_universe: pd.DataFrame
    manifest: dict

    @property
    def primary_origins(self) -> list[int]:
        origins = self.manifest.get("primary_origins", list(PRIMARY_ORIGINS))
        return [int(o) for o in origins]


def load_benchmark(
    root: Optional[Path] = None,
    *,
    version: str = BENCHMARK_VERSION,
    verify_checksums: bool = False,
) -> BenchmarkDataset:
    """Load parquet panels from ``src/data/benchmarks/{version}/``.

    Raises ``FileNotFoundError`` if the freeze has not been built yet.
    """
    base = Path(root) if root is not None else default_benchmark_root()
    if version != BENCHMARK_VERSION and root is None:
        base = default_benchmark_root().parent / version

    missing = [name for name in PANEL_FILES if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Benchmark {version} incomplete under {base}. "
            f"Missing: {missing}. Run: python -m pkg.benchmark.freeze"
        )

    manifest = load_manifest()
    if verify_checksums:
        _assert_checksums(base, manifest)

    ts = pd.read_parquet(base / "ts_universe.parquet")
    bud = pd.read_parquet(base / "budget_universe.parquet")
    matched = pd.read_parquet(base / "matched_universe.parquet")
    return BenchmarkDataset(
        version=version,
        root=base,
        ts_universe=ts,
        budget_universe=bud,
        matched_universe=matched,
        manifest=manifest,
    )


def _assert_checksums(root: Path, manifest: dict) -> None:
    checksums = manifest.get("checksums", {})
    for rel, expected in checksums.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen file for checksum: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {rel}: expected {expected}, got {actual}"
            )


def filter_products(
    df: pd.DataFrame, products: Optional[Sequence[str]]
) -> pd.DataFrame:
    if products is None:
        return df
    prod_set = set(str(p) for p in products)
    return df.loc[df["product"].astype(str).isin(prod_set)].copy()


def horizon_bucket(h: int) -> str:
    from pkg.benchmark.config import HORIZON_BUCKETS

    h = int(h)
    for name, lo, hi in HORIZON_BUCKETS:
        if lo <= h <= hi:
            return name
    return "other"


def prep_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_12", "sales_roll3"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    for c in ["model_enc", "field_enc", "form_enc", "provider_enc"]:
        if c in out.columns:
            out[c] = out[c].fillna(-1)
    return out


def resolve_origins(
    origins: Optional[Iterable[int]],
    dataset: BenchmarkDataset,
    *,
    use_primary: bool = True,
) -> list[int]:
    if origins is not None:
        return sorted(int(o) for o in origins)
    if use_primary:
        return dataset.primary_origins
    return sorted(dataset.matched_universe["origin"].astype(int).unique().tolist())
