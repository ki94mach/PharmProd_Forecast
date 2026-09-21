#!/usr/bin/env python
"""Freeze immutable reliability_label_contract_v1.json (fail-if-exists; no HB outcomes)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.portfolio_ts_lock.lock_io import LockExistsError  # noqa: E402
from pkg.research.portfolio_residual_xgb_v1.reliability_label_contract import (  # noqa: E402
    freeze_reliability_label_contract_v1,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    args = ap.parse_args()
    try:
        summary = freeze_reliability_label_contract_v1(repo_root=args.repo_root)
    except LockExistsError as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
