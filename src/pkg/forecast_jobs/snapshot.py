"""Frozen sales Parquet fingerprinting and validation (read-only)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

from pkg.benchmark.dataset import file_sha256
from pkg.forecast_jobs.config import SalesConfig, SalesSchemaConfig

SNAPSHOT_FILENAME = "input_snapshot.json"


class SalesSnapshotError(ValueError):
    """Sales input snapshot missing, unreadable, or mismatched."""


@dataclass(frozen=True)
class SalesInputSnapshot:
    """Recorded fingerprint of the frozen sales Parquet input."""

    path: str
    resolved_path: str
    content_sha256: str
    n_rows: int
    size_bytes: int
    mtime_ns: int
    mtime_utc: str
    schema_columns: tuple[str, ...]
    schema_arrow: tuple[str, ...]
    dtypes: Mapping[str, str]
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "resolved_path": self.resolved_path,
            "content_sha256": self.content_sha256,
            "n_rows": int(self.n_rows),
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "mtime_utc": self.mtime_utc,
            "schema_columns": list(self.schema_columns),
            "schema_arrow": list(self.schema_arrow),
            "dtypes": dict(self.dtypes),
            "read_only": bool(self.read_only),
        }


def _utc_from_mtime_ns(mtime_ns: int) -> str:
    dt = datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint_sales_parquet(
    path: Path,
    *,
    schema: Optional[SalesSchemaConfig] = None,
    read_only: bool = True,
    configured_path: Optional[str] = None,
) -> SalesInputSnapshot:
    """Compute path/schema/rows/mtime/content-hash metadata without mutating the file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise SalesSnapshotError(f"sales parquet not found: {resolved}")

    # Read-only intent: refuse writable opens by using metadata + hash only.
    pf = pq.ParquetFile(resolved)
    n_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
    arrow_names = tuple(str(n) for n in pf.schema_arrow.names)
    stat = resolved.stat()
    digest = file_sha256(resolved)

    expected_schema = schema or SalesSchemaConfig()
    missing = [c for c in expected_schema.columns if c not in arrow_names]
    if missing:
        raise SalesSnapshotError(
            f"sales parquet missing columns {missing}; has {list(arrow_names)}"
        )

    return SalesInputSnapshot(
        path=configured_path or Path(path).as_posix(),
        resolved_path=resolved.as_posix(),
        content_sha256=digest,
        n_rows=n_rows,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        mtime_utc=_utc_from_mtime_ns(int(stat.st_mtime_ns)),
        schema_columns=tuple(expected_schema.columns),
        schema_arrow=arrow_names,
        dtypes=dict(expected_schema.dtypes),
        read_only=bool(read_only),
    )


def validate_sales_config(sales: SalesConfig) -> SalesInputSnapshot:
    """Fingerprint the file and enforce optional expected snapshot pins."""
    snap = fingerprint_sales_parquet(
        sales.path,
        schema=sales.schema,
        read_only=sales.read_only,
        configured_path=Path(sales.path).as_posix(),
    )
    exp = sales.expected
    mismatches: list[str] = []
    if exp.content_sha256 and exp.content_sha256.lower() != snap.content_sha256.lower():
        mismatches.append(
            f"content_sha256 expected={exp.content_sha256} actual={snap.content_sha256}"
        )
    if exp.n_rows is not None and int(exp.n_rows) != int(snap.n_rows):
        mismatches.append(f"n_rows expected={exp.n_rows} actual={snap.n_rows}")
    if exp.size_bytes is not None and int(exp.size_bytes) != int(snap.size_bytes):
        mismatches.append(
            f"size_bytes expected={exp.size_bytes} actual={snap.size_bytes}"
        )
    if exp.mtime_ns is not None and int(exp.mtime_ns) != int(snap.mtime_ns):
        mismatches.append(f"mtime_ns expected={exp.mtime_ns} actual={snap.mtime_ns}")
    if mismatches:
        raise SalesSnapshotError(
            "sales input snapshot does not match configured expected values: "
            + "; ".join(mismatches)
        )
    return snap


def assert_snapshot_matches_recorded(
    current: SalesInputSnapshot,
    recorded: Mapping[str, Any],
) -> None:
    """Ensure a resume uses the same frozen sales input as the first run."""
    checks = (
        ("content_sha256", current.content_sha256, recorded.get("content_sha256")),
        ("n_rows", current.n_rows, recorded.get("n_rows")),
        ("size_bytes", current.size_bytes, recorded.get("size_bytes")),
        ("resolved_path", current.resolved_path, recorded.get("resolved_path")),
    )
    problems = []
    for name, actual, expected in checks:
        if expected is None:
            continue
        if str(actual) != str(expected):
            problems.append(f"{name}: recorded={expected!r} current={actual!r}")
    if problems:
        raise SalesSnapshotError(
            "input snapshot changed since run was created: " + "; ".join(problems)
        )


def write_input_snapshot(run_dir: Path, snapshot: SalesInputSnapshot) -> Path:
    path = Path(run_dir) / SNAPSHOT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(snapshot.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)
    return path


def read_input_snapshot(run_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(run_dir) / SNAPSHOT_FILENAME
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SalesSnapshotError(f"invalid snapshot file: {path}")
    return data


def load_sales_frame(
    sales: SalesConfig,
    *,
    snapshot: Optional[SalesInputSnapshot] = None,
) -> pd.DataFrame:
    """Load frozen sales as a DataFrame (read-only path; never writes back)."""
    snap = snapshot or validate_sales_config(sales)
    path = Path(snap.resolved_path)
    df = pd.read_parquet(path)
    for col in sales.schema.columns:
        if col not in df.columns:
            raise SalesSnapshotError(f"loaded frame missing column {col!r}")
    df = df.loc[:, list(sales.schema.columns)].copy()
    df["product"] = df["product"].astype(str)
    df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("int64")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    return df
