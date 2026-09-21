#!/usr/bin/env python
"""CLI: residual-XGBoost v1.1 post-hoc evaluation (no refit, no residual lock)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.research.portfolio_residual_xgb_v1.evaluation_v1 import (  # noqa: E402
    EvaluationOverwriteError,
    run_evaluation,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override evaluation_v1 output directory",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-empty evaluation_v1/",
    )
    args = ap.parse_args()
    try:
        summary = run_evaluation(
            repo_root=args.repo_root,
            force=bool(args.force),
            out_dir=args.out_dir,
        )
    except EvaluationOverwriteError as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
