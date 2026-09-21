#!/usr/bin/env python3
"""CLI for residual-XGBoost v1.1 OOF: validate / dry-run / execute / status / evaluate.

Hard-fails if HB or post-HB origins appear. Does not open HB outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.portfolio_ts_lock.constants import HB_ORIGINS, POST_HB_ORIGINS  # noqa: E402
from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS  # noqa: E402
from pkg.research.portfolio_residual_xgb_v1.evaluate_oof import (  # noqa: E402
    evaluate_oof_predictions,
)
from pkg.research.portfolio_residual_xgb_v1.execute_oof import (  # noqa: E402
    _load_residual_panel,
    run_oof_origin,
)


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_origins_safe(origins: list[int]) -> None:
    bad = sorted(set(int(o) for o in origins) & set(HB_ORIGINS | POST_HB_ORIGINS))
    if bad:
        raise SystemExit(f"HARD-FAIL: HB/post-HB origins in config: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--action",
        choices=["validate", "dry-run", "execute", "status", "evaluate"],
        required=True,
    )
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))
    origins = [int(o) for o in cfg["origins"]]
    _assert_origins_safe(origins)
    if cfg.get("HUMAN_XGB") not in (None, HUMAN_XGB_STATUS, "BLOCKED_NO_DEVELOPMENT_VINTAGES"):
        raise SystemExit("HUMAN_XGB must remain BLOCKED_NO_DEVELOPMENT_VINTAGES")

    run_id = str(cfg["run"]["run_id"])
    out_root = REPO / str(cfg["run"].get("output_dir", "src/data/results/portfolio_residual_xgb_v1"))
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "validate":
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": run_id,
                    "origins": origins,
                    "outer_workers": cfg.get("execution", {}).get("outer_workers", 1),
                    "HUMAN_XGB": HUMAN_XGB_STATUS,
                    "hb_outcomes_inspected": False,
                    "methods": cfg.get("methods"),
                },
                indent=2,
            )
        )
        return 0

    if args.action == "dry-run":
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "planned_origin_jobs": len(origins),
                    "planned_xgb_fits": len(origins),
                    "origins": origins,
                    "would_execute": False,
                },
                indent=2,
            )
        )
        return 0

    if args.action == "status":
        done = sorted(run_dir.glob("oof_predictions_origin_*.parquet"))
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "n_done": len(done),
                    "n_planned": len(origins),
                    "done_files": [p.name for p in done],
                },
                indent=2,
            )
        )
        return 0

    sales = REPO / cfg["sales"]["path"]
    v21 = REPO / cfg["sources"]["v21_export"]
    a0 = REPO / cfg["sources"]["a0_export"]

    if args.action == "execute":
        # Training path — intended for VPS only.
        residual, sales_df = _load_residual_panel(
            sales_path=sales, v21_export=v21, a0_export=a0
        )
        for o in origins:
            path = run_oof_origin(
                residual_panel=residual,
                sales=sales_df,
                evaluation_origin=int(o),
                out_dir=run_dir,
                resume=bool(cfg.get("execution", {}).get("resume", True)),
            )
            print(f"origin={o} wrote={path}")
        return 0

    if args.action == "evaluate":
        pred_paths = [
            run_dir / f"oof_predictions_origin_{o}.parquet" for o in origins
        ]
        missing = [p for p in pred_paths if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing prediction files: {[p.name for p in missing]}")
        gates = out_root / "residual_promotion_gates_v1.json"
        report = evaluate_oof_predictions(
            pred_paths,
            out_path=run_dir / "evaluation_report.json",
            gates_path=gates if gates.is_file() else None,
        )
        print(json.dumps({"promotion_gates_pass": report["promotion_gates_pass"]}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
