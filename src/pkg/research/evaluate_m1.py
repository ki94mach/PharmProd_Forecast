"""CLI entry point for M1 Optuna Hyperparameter Optimization.

Usage:
    python -m pkg.research.evaluate_m1

Phases:
  1. F0 baseline reproduction gate (STOP if drift > 0.10 abs WMAPE).
  2. Optuna tuning on pre-PRIMARY origins (40 trials per anchor, SQLite-resumable).
  3. PRIMARY evaluation with frozen params (no early stopping, no retuning).
  4. Write all artifacts and docs/m1_optuna_tuning.md.

STOP after M1A / M1B. Do NOT start M1C or another family.

Exit codes:
    0 — success
    1 — environment / reproducibility problem (baseline gate fail)
    2 — assertion / logic error
"""
from __future__ import annotations

import sys
import traceback

from pkg.research.tuning.config import m1_output_dir
from pkg.research.tuning.evaluate_tuned import evaluate_tuned_models
from pkg.research.tuning.report import write_m1_report
from pkg.research.tuning.run_optuna import run_all_optuna_studies


def main() -> None:
    print("=" * 60)
    print("M1 Optuna Hyperparameter Optimization")
    print("M1A — TS + XGB residual")
    print("M1B — Human + XGB residual")
    print("No F3A–F3E features. No SQL. STOP after M1A/M1B.")
    print("=" * 60)

    out_dir = m1_output_dir()
    print(f"Output directory: {out_dir}\n")

    # Phase 1 + 2: Optuna studies (baseline gate is run inside evaluate_tuned)
    try:
        print("Phase 1+2: Optuna tuning on pre-PRIMARY origins …")
        optuna_results = run_all_optuna_studies(out_dir=out_dir)
    except EnvironmentError as exc:
        print(f"STOP — environment/reproducibility error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as exc:
        print(f"STOP — assertion error: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    except Exception as exc:
        print(f"ERROR in Optuna tuning: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    # Phase 3: PRIMARY evaluation
    try:
        print("\nPhase 3: PRIMARY evaluation with frozen params …")
        eval_result = evaluate_tuned_models(optuna_results, out_dir=out_dir)
    except EnvironmentError as exc:
        print(f"STOP — environment/reproducibility error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as exc:
        print(f"STOP — assertion error (key mismatch?): {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    except Exception as exc:
        print(f"ERROR in PRIMARY evaluation: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)

    # Phase 4: Report
    try:
        print("\nPhase 4: Writing report …")
        report_path = write_m1_report(
            eval_result, optuna_results, out_dir=out_dir
        )
        print(f"Report: {report_path}")
    except Exception as exc:
        print(f"WARNING: report write failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        # Non-fatal — artifacts were written

    # Summary
    verdicts = eval_result.get("verdicts", {}) if eval_result else {}
    print("\n" + "=" * 60)
    print("M1 COMPLETE")
    for anchor in ("ts", "human"):
        v = verdicts.get(anchor, "n/a")
        label = "M1A" if anchor == "ts" else "M1B"
        print(f"  {label} ({anchor.upper()}): {v}")
    print("=" * 60)
    print("STOP — do not start M1C or another feature family.")


if __name__ == "__main__":
    main()
