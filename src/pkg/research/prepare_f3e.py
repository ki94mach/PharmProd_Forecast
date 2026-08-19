"""F3E Step 1 CLI.

Usage:
    python -m pkg.research.prepare_f3e

Exit codes:
    0  success — all assertions passed, artifacts written
    1  file not found (missing freeze files)
    2  normalization / exclusion assertion failed — STOP, do not proceed to Step 2
"""
from __future__ import annotations

import sys


def main() -> int:
    from pathlib import Path

    try:
        from pkg.research.f3e.prepare import prepare_peer_demand_source
        from pkg.research.f3e.source_report import write_peer_demand_source_audit
        from pkg.research.f3e.config import f3e_source_dir, docs_dir
    except ImportError as exc:
        print(f"[F3E] ImportError: {exc}", file=sys.stderr)
        return 1

    print("[F3E] Starting Step 1 — Peer Demand Source, Normalization, and Semantic Audit")

    try:
        result = prepare_peer_demand_source()
    except FileNotFoundError as exc:
        print(f"[F3E] ERROR: {exc}", file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"[F3E] STOP — assertion failed: {exc}", file=sys.stderr)
        return 2

    panel = result["panel"]
    mvp_list = result["mvp_list"]
    n_mvp_in_panel = result["n_mvp_in_panel"]
    out_dir: Path = result["out_dir"]

    print(f"[F3E] Peer panel: {len(panel):,} rows, {panel['product'].nunique():,} products, "
          f"{panel['date'].nunique()} months")
    print(f"[F3E] MVP coverage: {n_mvp_in_panel}/{len(mvp_list)} products in peer panel")
    neg = result["negative_sales_report"]
    if not neg.empty and "n_negative_rows" in neg.columns:
        dqty_neg = neg.loc[neg["quantity"] == "monthly_dqty", "n_negative_rows"]
        if not dqty_neg.empty:
            print(f"[F3E] Negative monthly_dqty rows: {int(dqty_neg.iloc[0]):,}")
    print(f"[F3E] Artifacts written to: {out_dir}")

    doc_path = write_peer_demand_source_audit(result)
    print(f"[F3E] Report written to: {doc_path}")
    print("[F3E] Step 1 complete. STOP — do not proceed to Step 2 until report reviewed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
