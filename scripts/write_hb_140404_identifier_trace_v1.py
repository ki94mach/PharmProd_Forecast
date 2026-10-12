#!/usr/bin/env python
"""Write identifier-only 140404 / 140505 filter trace (no HB quantities)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_forecast_reliability_v1.hb_140404_identifier_trace import (  # noqa: E402
    write_hb_140404_identifier_trace,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument(
        "--skip-sql",
        action="store_true",
        help="Skip live warehouse load; diagnose from frozen v2 panel + constants only",
    )
    args = ap.parse_args()
    summary = write_hb_140404_identifier_trace(
        repo_root=args.repo_root,
        load_sql=not args.skip_sql,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
