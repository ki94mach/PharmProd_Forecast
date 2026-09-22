#!/usr/bin/env python
"""Write pre-outcome hb_cohort_representativeness_audit_v1.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_forecast_reliability_v1.hb_cohort_representativeness import (  # noqa: E402
    write_hb_cohort_representativeness_audit,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    args = ap.parse_args()
    summary = write_hb_cohort_representativeness_audit(repo_root=args.repo_root)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
