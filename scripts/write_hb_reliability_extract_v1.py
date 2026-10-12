#!/usr/bin/env python
"""Build identifier-only HB reliability extract (no INCOMPLETE_SHAMSI drop).

Outside frozen benchmarks/. No HB quantity scoring, forecasting, or final contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_forecast_reliability_v1.hb_reliability_extract import (  # noqa: E402
    refresh_gates_from_existing_panel,
    run_identifier_extract_and_audits,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument(
        "--refresh-from-panel",
        action="store_true",
        help=(
            "Skip SQL; recompute gates/provenance/prepare artifacts from the "
            "existing identifiers_panel.parquet"
        ),
    )
    args = ap.parse_args()
    if args.refresh_from_panel:
        summary = refresh_gates_from_existing_panel(repo_root=args.repo_root)
    else:
        summary = run_identifier_extract_and_audits(repo_root=args.repo_root)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
