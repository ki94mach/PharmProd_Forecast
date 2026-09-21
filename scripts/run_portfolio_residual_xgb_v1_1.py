#!/usr/bin/env python3
"""CLI for residual-XGBoost v1.1 OOF: validate / dry-run / execute / status / evaluate / accept-smoke.

Hard-fails if HB or post-HB origins appear. Does not open HB outcomes.
Full execute requires a passing smoke acceptance artifact.
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
from pkg.portfolio_ts_lock.production_contract import file_sha256  # noqa: E402
from pkg.research.portfolio_residual_xgb_v1.constants import HUMAN_XGB_STATUS  # noqa: E402
from pkg.research.portfolio_residual_xgb_v1.evaluate_oof import (  # noqa: E402
    evaluate_oof_predictions,
)
from pkg.research.portfolio_residual_xgb_v1.execute_oof import (  # noqa: E402
    _load_residual_panel,
    run_oof_origin,
)
from pkg.research.portfolio_residual_xgb_v1.smoke_acceptance import (  # noqa: E402
    SMOKE_RUN_ID,
    assert_smoke_accepted,
    run_smoke_acceptance,
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
        choices=[
            "validate",
            "dry-run",
            "execute",
            "status",
            "evaluate",
            "accept-smoke",
        ],
        required=True,
    )
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))
    origins = [int(o) for o in cfg["origins"]]
    _assert_origins_safe(origins)
    if cfg.get("HUMAN_XGB") not in (None, HUMAN_XGB_STATUS, "BLOCKED_NO_DEVELOPMENT_VINTAGES"):
        raise SystemExit("HUMAN_XGB must remain BLOCKED_NO_DEVELOPMENT_VINTAGES")

    run_id = str(cfg["run"]["run_id"])
    out_root = REPO / str(
        cfg["run"].get("output_dir", "src/data/results/portfolio_residual_xgb_v1")
    )
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = (
        REPO
        / "src/data/results/product_universe/portfolio_ts_architecture_lock_v1"
    )
    is_smoke = run_id == SMOKE_RUN_ID or origins == [140107]
    is_full = "full" in run_id or len(origins) > 1

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
                    "requires_smoke_acceptance_before_full_execute": True,
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
                    "full_blocked_until_smoke_acceptance": bool(is_full),
                },
                indent=2,
            )
        )
        return 0

    if args.action == "accept-smoke":
        report = run_smoke_acceptance(
            repo_root=REPO,
            results_dir=out_root,
            lock_dir=lock_dir,
            require_predictions=True,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("passed") else 1

    if args.action == "status":
        done = sorted(run_dir.glob("oof_predictions_origin_*.parquet"))
        smoke_ok = (out_root / "smoke_acceptance_v1.json").is_file()
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "n_done": len(done),
                    "n_planned": len(origins),
                    "done_files": [p.name for p in done],
                    "smoke_acceptance_present": smoke_ok,
                },
                indent=2,
            )
        )
        return 0

    sales = REPO / cfg["sales"]["path"]
    v21 = REPO / cfg["sources"]["v21_export"]
    a0 = REPO / cfg["sources"]["a0_export"]

    if args.action == "execute":
        if is_full:
            assert_smoke_accepted(out_root)
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
        # After smoke execute, remind acceptance is required next
        if is_smoke:
            print(
                json.dumps(
                    {
                        "next": "accept-smoke",
                        "note": "Run accept-smoke before Full Execute",
                    }
                )
            )
        return 0

    if args.action == "evaluate":
        pred_paths = [
            run_dir / f"oof_predictions_origin_{o}.parquet" for o in origins
        ]
        missing = [p for p in pred_paths if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing prediction files: {[p.name for p in missing]}")
        gates = out_root / "residual_promotion_gates_v1.json"
        gates_sha_before = file_sha256(gates) if gates.is_file() else None

        if is_smoke:
            # Diagnostic only — do not treat as promotion decision; never mutate gates.
            report = evaluate_oof_predictions(
                pred_paths,
                out_path=run_dir / "smoke_evaluation_diagnostic.json",
                gates_path=None,
            )
            report["diagnostic_only"] = True
            report["affects_promotion_gates"] = False
            report["parameters_tuned"] = False
            # Rewrite without promotion_gates_pass as a lock decision
            report.pop("promotion_gates_pass", None)
            out = run_dir / "smoke_evaluation_diagnostic.json"
            out.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
            )
            if gates.is_file() and file_sha256(gates) != gates_sha_before:
                raise SystemExit("Smoke evaluate mutated promotion gates — FAIL")
            print(
                json.dumps(
                    {
                        "diagnostic_only": True,
                        "affects_promotion_gates": False,
                        "parameters_tuned": False,
                        "report": str(out.as_posix()),
                    },
                    indent=2,
                )
            )
            return 0

        if is_full:
            assert_smoke_accepted(out_root)
        report = evaluate_oof_predictions(
            pred_paths,
            out_path=run_dir / "evaluation_report.json",
            gates_path=gates if gates.is_file() else None,
        )
        if gates.is_file() and file_sha256(gates) != gates_sha_before:
            raise SystemExit("Evaluate mutated promotion gates — FAIL")
        print(json.dumps({"promotion_gates_pass": report["promotion_gates_pass"]}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
