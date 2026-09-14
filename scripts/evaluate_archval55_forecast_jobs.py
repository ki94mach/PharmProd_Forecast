#!/usr/bin/env python3
"""Evaluate Archval55 panel (defaults to versioned dirs; never archval25)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval55_export"
)
OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval55_production"
)
TARGET = REPO / "scripts" / "evaluate_archval25_forecast_jobs.py"


def main() -> int:
    # Inject defaults only when caller did not pass --export-dir / --out-dir.
    argv = list(sys.argv[1:])
    if "--export-dir" not in argv:
        argv = ["--export-dir", str(EXPORT), *argv]
    if "--out-dir" not in argv:
        argv = ["--out-dir", str(OUT), *argv]
    sys.argv = [str(TARGET), *argv]
    runpy.run_path(str(TARGET), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
