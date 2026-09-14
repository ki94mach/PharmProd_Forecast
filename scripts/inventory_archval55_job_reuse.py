#!/usr/bin/env python3
"""Inventory Archval55 jobs: reusable vs required vs ineligible-fill.

Writes CSVs under
``src/data/results/ts_v3a_architecture_validation/archval55_lock/``.
Does not mutate archval25 artifacts or start fits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from archval55_reuse import (  # noqa: E402
    LOCK_DIR,
    SALES_PATH,
    build_inventory,
    check_reuse_gate,
)


def main() -> int:
    gate = check_reuse_gate()
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (LOCK_DIR / "gate_report.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )

    if not SALES_PATH.is_file():
        print(f"missing sales: {SALES_PATH}")
        print(f"wrote {LOCK_DIR / 'gate_report.json'} (gate_ok={gate['ok']})")
        return 2

    sales = pd.read_parquet(SALES_PATH)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    reusable, ineligible, required, all_jobs = build_inventory(gate=gate, sales=sales)
    reusable.to_csv(LOCK_DIR / "reusable_jobs.csv", index=False)
    ineligible.to_csv(LOCK_DIR / "ineligible_fill_jobs.csv", index=False)
    required.to_csv(LOCK_DIR / "required_jobs.csv", index=False)
    all_jobs.to_csv(LOCK_DIR / "all_jobs.csv", index=False)

    summary = {
        "gate_ok": bool(gate.get("ok")),
        "n_total": int(len(all_jobs)),
        "n_reusable": int(len(reusable)),
        "n_ineligible_fill": int(len(ineligible)),
        "n_required": int(len(required)),
        "expected_total": 1540,
    }
    (LOCK_DIR / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"gate_ok={gate.get('ok')}")
    for name, chk in (gate.get("checks") or {}).items():
        print(f"  check[{name}].ok={chk.get('ok')}")
    print(
        f"total={summary['n_total']} reusable={summary['n_reusable']} "
        f"ineligible_fill={summary['n_ineligible_fill']} "
        f"required={summary['n_required']}"
    )
    print(f"wrote {LOCK_DIR}")
    return 0 if gate.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
