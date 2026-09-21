#!/usr/bin/env python3
"""Prepare-only residual-XGBoost study on frozen hybrid TS base.

Does not train models, open HB outcomes, or schedule HUMAN_XGB jobs.
Writes residual_study_contract_v1.json and prepare artifacts under
src/data/results/portfolio_residual_xgb_v1/.
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

from pkg.research.portfolio_residual_xgb_v1.prepare import run_prepare  # noqa: E402

DEFAULT_OUT = REPO / "src" / "data" / "results" / "portfolio_residual_xgb_v1"
DEFAULT_LOCK = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_ts_architecture_lock_v1"
)
DEFAULT_SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)
DEFAULT_V21 = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_v21_backfill_v2_export"
)
DEFAULT_A0 = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_a0_backfill_v2_export"
)
DEFAULT_BUDGET = REPO / "src" / "data" / "benchmarks" / "v1" / "budget_universe.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    ap.add_argument("--v21-export", type=Path, default=DEFAULT_V21)
    ap.add_argument("--a0-export", type=Path, default=DEFAULT_A0)
    ap.add_argument("--budget-universe", type=Path, default=DEFAULT_BUDGET)
    args = ap.parse_args()

    summary = run_prepare(
        repo_root=REPO,
        out_dir=Path(args.out_dir),
        lock_dir=Path(args.lock_dir),
        sales_path=Path(args.sales),
        v21_export=Path(args.v21_export),
        a0_export=Path(args.a0_export),
        budget_universe_path=Path(args.budget_universe),
    )

    commands = {
        "powershell": [
            "$env:PYTHONPATH = 'src'",
            "python scripts/evaluate_portfolio_ts_architecture_postlock_v2.py",
            "python scripts/prepare_portfolio_residual_xgb_v1.py",
            "python -m pytest tests/portfolio_ts_lock tests/portfolio_residual_xgb_v1 -q",
            "python -m pkg.forecast_jobs --config configs/forecast_jobs/portfolio_residual_xgb_v1.win.yaml --validate",
            "python -m pkg.forecast_jobs --config configs/forecast_jobs/portfolio_residual_xgb_v1.win.yaml --dry-run",
            ".\\deploy\\windows\\Prepare-PortfolioResidualXgbV1.ps1 -Validate",
        ],
        "note": (
            "Prepare-only complete. Do not train. HUMAN_XGB is "
            "BLOCKED_NO_DEVELOPMENT_VINTAGES. Freeze residual model before HB outcomes."
        ),
    }
    print(json.dumps({"prepare_summary": summary, "execution_commands": commands}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
