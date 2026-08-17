"""Shared markdown/CSV helpers for research family reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd


def fmt_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
    return str(v)[:220]


def md_table(
    df: pd.DataFrame,
    max_rows: int = 20,
    cols: Optional[list[str]] = None,
    *,
    format_cells: bool = True,
) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    sub = df if cols is None else df[[c for c in cols if c in df.columns]]
    sub = sub.head(max_rows)
    names = list(sub.columns)
    lines = [
        "| " + " | ".join(str(c) for c in names) + " |",
        "| " + " | ".join(["---"] * len(names)) + " |",
    ]
    for _, row in sub.iterrows():
        if format_cells:
            cells = [fmt_cell(row[c]) for c in names]
        else:
            cells = [
                "" if pd.isna(row[c]) else str(row[c])[:90] for c in names
            ]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in CSV)_")
    return "\n".join(lines) + "\n"


def read_csv(out_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = out_dir / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def repo_relative(path: Path) -> str:
    """POSIX path relative to the Forecast repo root when possible."""
    root = Path(__file__).resolve().parents[4]
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()
