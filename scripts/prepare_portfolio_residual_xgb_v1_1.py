#!/usr/bin/env python3
"""Prepare residual-XGBoost v1.1 (corrected PIT). Does not train or open HB.

Does not overwrite residual_study_contract_v1.json or TS lock/postlock artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.research.portfolio_residual_xgb_v1.prepare_v1_1 import run_prepare_v1_1  # noqa: E402

DEFAULT_OUT = REPO / "src" / "data" / "results" / "portfolio_residual_xgb_v1"
DEFAULT_LOCK = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_ts_architecture_lock_v1"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK)
    ap.add_argument(
        "--sales",
        type=Path,
        default=REPO
        / "src/data/results/product_universe/sales_full_portfolio_v1/sales.parquet",
    )
    ap.add_argument(
        "--v21-export",
        type=Path,
        default=REPO
        / "src/data/results/product_universe/portfolio_v21_backfill_v2_export",
    )
    ap.add_argument(
        "--a0-export",
        type=Path,
        default=REPO
        / "src/data/results/product_universe/portfolio_a0_backfill_v2_export",
    )
    ap.add_argument(
        "--budget-universe",
        type=Path,
        default=REPO / "src/data/benchmarks/v1/budget_universe.parquet",
    )
    args = ap.parse_args()

    summary = run_prepare_v1_1(
        repo_root=REPO,
        out_dir=Path(args.out_dir),
        lock_dir=Path(args.lock_dir),
        sales_path=Path(args.sales),
        v21_export=Path(args.v21_export),
        a0_export=Path(args.a0_export),
        budget_universe_path=Path(args.budget_universe),
    )
    commands = {
        "prepare_validate": [
            "$env:PYTHONPATH = 'src'",
            "python scripts/prepare_portfolio_residual_xgb_v1_1.py",
            "python -m pytest tests/portfolio_residual_xgb_v1 -q",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action Validate -Profile Smoke",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action DryRun -Profile Smoke",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action DryRun -Profile Full",
        ],
        "vps_execute_resume": [
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action Execute -Profile Smoke",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action Execute -Profile Full",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action Status -Profile Full",
            ".\\deploy\\windows\\PortfolioResidualXgbV1_1.ps1 -Action Evaluate -Profile Full",
        ],
        "note": "Cursor prepares/validates only. Do not train locally. HUMAN_XGB blocked.",
    }
    print(json.dumps({"prepare_summary_v1_1": summary, "commands": commands}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
