#!/usr/bin/env python3
"""Export Archval55 A2–A5 late-prod run into a versioned panel.

Writes ``archval55_a2_a5_late_prod_export_v1/`` without mutating archval25/55
V2.1/A0/A1 trees. Reuses ``assemble_panel`` from export_archval55_panel.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from export_archval55_panel import assemble_panel  # noqa: E402

DEFAULT_RUN = REPO / "src" / "data" / "forecast_jobs" / "archval55_a2_a5_late_prod_v1"
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_OUT = RESULTS / "archval55_a2_a5_late_prod_export_v1"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    panel, summary = assemble_panel(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "forecasts_panel.parquet", index=False)
    panel.to_csv(out_dir / "forecasts_panel.csv", index=False)
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    db = run_dir / "state.sqlite"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        status_counts = dict(
            conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        )
        conn.close()
        (out_dir / "status_counts.json").write_text(
            json.dumps(status_counts, indent=2) + "\n", encoding="utf-8"
        )
    (out_dir / "README.md").write_text(
        "# Archval55 A2–A5 late-prod export\n\n"
        f"- Source run: `{run_dir.as_posix()}`\n"
        f"- Rows: {len(panel)}\n"
        f"- Summary: `{summary}`\n"
        "- Built by `scripts/export_archval55_a2_a5_panel.py`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'forecasts_panel.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
