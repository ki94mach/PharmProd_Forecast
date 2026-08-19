"""Load SQL files from the ``query/`` directory at repository root."""
from __future__ import annotations

from pathlib import Path


def _query_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "query"


def load_sql(name: str) -> str:
    """Read ``query/{name}`` as UTF-8 text."""
    path = _query_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")
