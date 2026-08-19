"""CLI entry point for F3E Step 3 — Controlled Peer Demand Experiment.

Usage:
    python -m pkg.research.evaluate_f3e

Exit codes:
    0 — success
    1 — missing source artifacts
    2 — assertion / gate failure
"""
from __future__ import annotations

import sys
import traceback

from pkg.research.f3e.config import f3e_source_dir, NORMALIZED_MONTHLY_SALES_PARQUET, PRODUCT_PEER_PROFILE_PARQUET
from pkg.research.f3e.evaluate import evaluate_f3e
from pkg.research.f3e.report import write_f3e_report


def main() -> None:
    # Check required Step 1 artifacts exist before starting
    source_dir = f3e_source_dir()
    for fname in (NORMALIZED_MONTHLY_SALES_PARQUET, PRODUCT_PEER_PROFILE_PARQUET):
        p = source_dir / fname
        if not p.exists():
            print(f"[F3E Step 3] ERROR: required source artifact not found: {p}", file=sys.stderr)
            print("[F3E Step 3] Run prepare_f3e (Step 1) first.", file=sys.stderr)
            sys.exit(1)

    print("[F3E Step 3] Starting — Controlled Peer Demand Experiment")
    try:
        out = evaluate_f3e()
        print(f"[F3E Step 3] Evaluation complete. Primary verdict: {out['primary_verdict']}")
        print(f"[F3E Step 3] E2 vs E1 verdict: {out['e2_verdict']}")
    except AssertionError as exc:
        print(f"[F3E Step 3] ASSERTION FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    except Exception as exc:
        print(f"[F3E Step 3] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    print("[F3E Step 3] Writing report ...")
    try:
        report_path = write_f3e_report()
        print(f"[F3E Step 3] Report written to: {report_path}")
    except Exception as exc:
        print(f"[F3E Step 3] Report write failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("[F3E Step 3] Done. STOP — do not start another feature family.")


if __name__ == "__main__":
    main()
