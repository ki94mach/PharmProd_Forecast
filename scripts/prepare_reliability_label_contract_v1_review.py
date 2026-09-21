#!/usr/bin/env python
"""Write HB coverage audit + reliability_label_contract_v1_review.json (no final contract, no router)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_residual_xgb_v1.reliability_label_review import (  # noqa: E402
    write_review_artifacts,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    args = ap.parse_args()
    summary = write_review_artifacts(repo_root=args.repo_root)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
