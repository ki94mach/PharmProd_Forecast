"""Poll until arch-validation screening completes, then write the report."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXP_ID = "20260908T103000Z_archval"
BASE = ROOT / "src" / "data" / "ts_v3a" / "screening"
COMPLETE = BASE / EXP_ID / ".complete"
LOG = BASE / "validation_screen_run3.log"
INCOMPLETE = BASE / ".incomplete" / EXP_ID

from pkg.ts_v3a.analysis.report import write_validation_report
from pkg.ts_v3a.analysis.panel import PANEL_CSV


def main() -> int:
    print(f"waiting for {COMPLETE}", flush=True)
    while not COMPLETE.is_file():
        n_done = 0
        ckpt = INCOMPLETE / "checkpoint.json"
        if ckpt.is_file():
            try:
                import json

                n_done = len(json.loads(ckpt.read_text(encoding="utf-8")).get(
                    "completed_products", []
                ))
            except Exception:
                n_done = -1
        mtime = LOG.stat().st_mtime if LOG.exists() else 0
        print(
            f"not complete yet; completed_skus={n_done}; log_mtime={mtime}; "
            "sleeping 300s",
            flush=True,
        )
        time.sleep(300)
    print("complete; writing report", flush=True)
    path = write_validation_report(
        experiment_id=EXP_ID,
        base_dir=BASE,
        recigen_smoke_id="20260907T141655Z_a8650c2a",
        panel_csv=PANEL_CSV,
        origins=[140104, 140201, 140301, 140402],
    )
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
