#!/usr/bin/env python3
"""Summarize FAILED forecast_jobs for one run_id (root-cause triage)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def _repo_root() -> Path:
    env = (os.environ.get("FORECAST_ROOT") or "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def _normalize_error(msg: str | None) -> str:
    if not msg:
        return "(no error_message)"
    text = " ".join(str(msg).split())
    # Drop product-specific prefixes like 'Xacrel': ...
    text = re.sub(r"^'[^']+'\s*:\s*", "", text)
    text = re.sub(r"\b\d{5,6}\b", "<date>", text)
    text = re.sub(r"\bseed=\d+\b", "seed=<n>", text)
    if len(text) > 180:
        text = text[:177] + "..."
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="archval25_v21_a0_a1")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run directory (default: <FORECAST_ROOT>/src/data/forecast_jobs/<run-id>)",
    )
    p.add_argument("--limit", type=int, default=50, help="Max detail rows to print")
    args = p.parse_args()

    run_dir = args.run_dir or (
        _repo_root() / "src" / "data" / "forecast_jobs" / args.run_id
    )
    db = run_dir / "state.sqlite"
    if not db.is_file():
        print(f"missing DB: {db}")
        return 2

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
    print(f"run_dir={run_dir}")
    print("status_counts=" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    failed = list(
        conn.execute(
            """
            SELECT job_id, engine_version, quarter, forecast_origin, product_id,
                   error_type, error_message, runtime_seconds, output_path
            FROM jobs
            WHERE status = 'FAILED'
            ORDER BY engine_version, product_id, forecast_origin
            """
        )
    )
    print(f"failed_jobs={len(failed)}")
    if not failed:
        return 0

    by_type = Counter((r["error_type"] or "None") for r in failed)
    by_engine = Counter(str(r["engine_version"]) for r in failed)
    by_product = Counter(str(r["product_id"]) for r in failed)
    by_norm = Counter(_normalize_error(r["error_message"]) for r in failed)

    print("\n=== By error_type ===")
    for k, v in by_type.most_common():
        print(f"  {v:4d}  {k}")

    print("\n=== By architecture/engine ===")
    for k, v in by_engine.most_common():
        print(f"  {v:4d}  {k}")

    print("\n=== By product ===")
    for k, v in by_product.most_common():
        print(f"  {v:4d}  {k}")

    print("\n=== Normalized error message (root-cause clusters) ===")
    for k, v in by_norm.most_common():
        print(f"  {v:4d}  {k}")

    print(f"\n=== Detail (first {args.limit}) ===")
    for i, r in enumerate(failed[: args.limit], 1):
        err_path = None
        out = r["output_path"]
        if out:
            cand = Path(out) / "error.json"
            if cand.is_file():
                err_path = cand
            else:
                # output_path may be relative to run_dir
                cand2 = run_dir / out / "error.json"
                if cand2.is_file():
                    err_path = cand2
        extra = ""
        if err_path and err_path.is_file():
            try:
                payload = json.loads(err_path.read_text(encoding="utf-8"))
                extra = f" | error.json={payload}"
            except Exception as exc:  # noqa: BLE001
                extra = f" | error.json_read_failed={exc}"
        print(
            f"{i:02d}. eng={r['engine_version']} origin={r['forecast_origin']} "
            f"product={r['product_id']} type={r['error_type']}\n"
            f"    msg={r['error_message']}{extra}"
        )

    # Suggest likely class
    print("\n=== Quick read ===")
    top_msg, top_n = by_norm.most_common(1)[0]
    print(f"Dominant failure pattern ({top_n}/{len(failed)}): {top_msg}")
    if len(by_product) == 1:
        print("All failures are on a single product — likely data/series issue for that SKU.")
    elif by_engine.most_common(1)[0][1] == len(failed):
        eng = by_engine.most_common(1)[0][0]
        print(f"All failures are in engine={eng} — architecture-specific bug/limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
