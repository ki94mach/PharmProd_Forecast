"""Immutable lock file I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class LockExistsError(FileExistsError):
    """Raised when attempting to overwrite an existing architecture lock."""


def write_lock_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Create lock only if absent. Never overwrite."""
    path = Path(path)
    if path.exists():
        raise LockExistsError(f"Lock already exists (immutable): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    if not path.is_file():
        raise RuntimeError(f"Lock write failed: {path}")


def assert_lock_immutable_read(path: Path) -> dict[str, Any]:
    """Load lock for post-lock analysis; caller must not write this path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_lock_mutation(path: Path) -> None:
    """Post-lock scripts call this before any write attempt to the lock path."""
    raise PermissionError(
        f"Post-lock analysis must not mutate the architecture lock: {path}"
    )
