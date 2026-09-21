#!/usr/bin/env python
"""Write corrected reliability_label_contract_v1_review.json (no immutable freeze)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_forecast_reliability_v1.reliability_label_contract import (  # noqa: E402
    write_corrected_review,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument(
        "--source-review",
        type=Path,
        default=None,
        help="Optional path to prior review JSON",
    )
    args = ap.parse_args()
    summary = write_corrected_review(
        repo_root=args.repo_root,
        source_review_path=args.source_review,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
