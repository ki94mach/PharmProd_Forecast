"""CLI for V3A screening analysis / panel / Recigen diagnosis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pkg.ts_v3a.analysis.panel import (
    DEFAULT_VALIDATION_ORIGINS,
    PANEL_CSV,
    build_validation_panel,
    write_validation_panel,
)
from pkg.ts_v3a.analysis.recigen import diagnose_recigen_from_dir
from pkg.ts_v3a.analysis.report import write_validation_report
from pkg.ts_v3a.persistence import default_screening_root, load_screening_experiment
from pkg.ts_v3a.analysis.metrics import (
    architecture_summary,
    paired_architecture_comparisons,
    rebuild_ensemble,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pkg.ts_v3a.analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_panel = sub.add_parser("build-panel", help="Select validation products from frozen MVP sales")
    p_panel.add_argument("--target-n", type=int, default=20)
    p_panel.add_argument(
        "--origins",
        default=",".join(str(o) for o in DEFAULT_VALIDATION_ORIGINS),
    )

    p_smoke = sub.add_parser("smoke-metrics", help="Regenerate metrics from a completed experiment")
    p_smoke.add_argument("--experiment-id", required=True)
    p_smoke.add_argument("--base-dir", type=Path, default=None)
    p_smoke.add_argument("--output", type=Path, required=True)

    p_rec = sub.add_parser("recigen", help="Diagnose Recigen from smoke artifacts")
    p_rec.add_argument("--experiment-id", default="20260907T141655Z_a8650c2a")
    p_rec.add_argument("--base-dir", type=Path, default=None)
    p_rec.add_argument("--output", type=Path, required=True)

    p_rep = sub.add_parser("report", help="Full validation report for an experiment")
    p_rep.add_argument("--experiment-id", required=True)
    p_rep.add_argument("--base-dir", type=Path, default=None)
    p_rep.add_argument("--recigen-smoke-id", default="20260907T141655Z_a8650c2a")
    p_rep.add_argument("--panel-csv", type=Path, default=PANEL_CSV)

    args = p.parse_args(argv)
    base = args.base_dir if getattr(args, "base_dir", None) else default_screening_root()

    if args.cmd == "build-panel":
        origins = tuple(int(x.strip()) for x in str(args.origins).split(",") if x.strip())
        panel, cov = build_validation_panel(origins=origins, target_n=int(args.target_n))
        path = write_validation_panel(panel, origins=origins)
        cov_path = path.with_name(path.stem + "_origin_coverage.csv")
        cov.to_csv(cov_path, index=False)
        print(f"wrote {path} n={len(panel)}")
        print(f"wrote {cov_path}")
        print("products=" + ",".join(panel["product"].astype(str).tolist()))
        print("origins=" + ",".join(str(o) for o in origins))
        print(f"expected_folds={len(panel)*len(origins)*6*3}")
        return 0

    if args.cmd == "smoke-metrics":
        loaded = load_screening_experiment(args.experiment_id, base_dir=base)
        ens = rebuild_ensemble(loaded)
        summary = architecture_summary(loaded, ensemble=ens)
        paired = paired_architecture_comparisons(ens)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "architecture_summary.csv", index=False)
        paired.to_csv(out / "paired_comparisons.csv", index=False)
        ens.to_parquet(out / "ensemble_predictions.parquet", index=False)
        print(f"wrote {out}")
        print(summary.to_string(index=False))
        return 0

    if args.cmd == "recigen":
        exp_dir = Path(base) / args.experiment_id
        rec = diagnose_recigen_from_dir(exp_dir)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        for key in ("detail", "by_horizon", "fold_summary", "scaler"):
            df = rec.get(key)
            if hasattr(df, "to_csv") and df is not None and not df.empty:
                df.to_csv(out / f"recigen_{key}.csv", index=False)
        (out / "recigen_conclusion.txt").write_text(str(rec.get("conclusion", "")), encoding="utf-8")
        print(rec.get("conclusion", ""))
        return 0

    if args.cmd == "report":
        path = write_validation_report(
            experiment_id=args.experiment_id,
            base_dir=base,
            recigen_smoke_id=args.recigen_smoke_id,
            panel_csv=args.panel_csv,
        )
        print(f"wrote {path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
