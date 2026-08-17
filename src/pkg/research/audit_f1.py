"""F1 feature audit orchestrator and CLI.

Run::

    python -m pkg.research.audit_f1
    python -m pkg.research.audit_f1 --section control
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from pkg.benchmark import load_benchmark
from pkg.research.audit.common import audit_output_dir
from pkg.research.audit.control import run_f0_control
from pkg.research.audit.decomposition import decompose_error_delta
from pkg.research.audit.encoding import analyze_missing_history_encoding
from pkg.research.audit.human_audit import analyze_human_granularity, analyze_human_sample_sizes
from pkg.research.audit.importance import analyze_xgb_usage
from pkg.research.audit.ratios import profile_ratio_features
from pkg.research.audit.redundancy import analyze_demand_redundancy
from pkg.research.audit.report import render_report

ALL_SECTIONS = (
    "control",
    "redundancy",
    "human",
    "encoding",
    "ratios",
    "decomposition",
    "importance",
    "report",
)


def run_audit(
    *,
    sections: Optional[Iterable[str]] = None,
    verify_checksums: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    """Run selected audit sections; return combined results dict."""
    out_dir = out_dir or audit_output_dir()
    selected = set(sections) if sections else set(ALL_SECTIONS)
    results: dict = {"out_dir": out_dir, "adapter_fix": "None"}

    ds = load_benchmark(verify_checksums=verify_checksums)

    if "control" in selected:
        print("=== Section 1: F0_CONTROL ===")
        results["control"] = run_f0_control(ds, out_dir=out_dir)
        if not results["control"]["passed"]:
            print(
                "WARNING: F0_CONTROL gate FAILED — investigate make_residual_model "
                "before interpreting F1 results."
            )

    if "redundancy" in selected:
        print("=== Section 2: Demand redundancy ===")
        results["redundancy"] = analyze_demand_redundancy(ds, out_dir=out_dir)

    if "human" in selected:
        print("=== Section 3-4: Human granularity & sample sizes ===")
        results["human_granularity"] = analyze_human_granularity(ds, out_dir=out_dir)
        results["human_samples"] = analyze_human_sample_sizes(ds, out_dir=out_dir)

    if "encoding" in selected:
        print("=== Section 5: Missing-history encoding ===")
        results["encoding"] = analyze_missing_history_encoding(ds, out_dir=out_dir)

    if "ratios" in selected:
        print("=== Section 6: Ratio instability ===")
        results["ratios"] = profile_ratio_features(ds, out_dir=out_dir)

    if "decomposition" in selected:
        print("=== Section 7: Error decomposition ===")
        results["decomposition"] = decompose_error_delta(ds, out_dir=out_dir)

    if "importance" in selected:
        print("=== Section 8: XGB feature usage ===")
        results["importance"] = analyze_xgb_usage(ds, out_dir=out_dir)

    if "report" in selected:
        print("=== Final report ===")
        report_path = render_report(results, out_dir=out_dir)
        results["report_path"] = report_path
        print(f"Wrote {report_path}")

    return results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="F1 feature audit on frozen benchmark v1")
    parser.add_argument(
        "--section",
        nargs="+",
        choices=list(ALL_SECTIONS),
        help="Run specific sections (default: all)",
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="CSV output directory (default: src/data/results/f1_audit)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        run_audit(
            sections=args.section,
            verify_checksums=args.verify_checksums,
            out_dir=args.out_dir,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run: python -m pkg.benchmark.freeze", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
