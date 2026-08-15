"""Offline verification of frozen benchmark v1 against expected WMAPEs.

Run::

    python -m pkg.benchmark.verify
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pkg.benchmark.config import (
    EXPECTED_ANALYSIS_A_PRIMARY,
    EXPECTED_ANALYSIS_B_PRIMARY,
)
from pkg.benchmark.dataset import load_benchmark, load_manifest
from pkg.benchmark.evaluate import backtest


def verify(
    root: Optional[Path] = None,
    *,
    wmape_tol: float = 0.05,
    check_checksums: bool = True,
    run_xgb: bool = True,
) -> dict:
    """Assert panels exist, checksums match, and frozen WMAPEs reproduce."""
    man = load_manifest()
    ds = load_benchmark(root, verify_checksums=check_checksums)
    report = {"ok": True, "checks": []}

    # Row counts
    expected_counts = man.get("row_counts", {})
    for key, panel in [
        ("ts_universe", ds.ts_universe),
        ("budget_universe", ds.budget_universe),
        ("matched_universe", ds.matched_universe),
    ]:
        actual = len(panel)
        exp = expected_counts.get(key)
        ok = exp is None or actual == exp
        report["checks"].append(
            {"name": f"rows_{key}", "expected": exp, "actual": actual, "ok": ok}
        )
        if not ok:
            report["ok"] = False

    b_models = ["ts", "human"]
    if run_xgb:
        b_models.extend(["ts_xgb", "human_xgb", "integrated"])

    expected_b = man.get(
        "expected_analysis_b_primary_wmape", EXPECTED_ANALYSIS_B_PRIMARY
    )
    for name in b_models:
        res = backtest(name, dataset=ds, universe="matched", eligibility="primary")
        actual = float(res.overall["wmape"].iloc[0])
        n = int(res.overall["n"].iloc[0])
        exp_w = expected_b.get(name if name != "human" else "human")
        # map names
        key = {
            "ts": "ts",
            "human": "human",
            "ts_xgb": "ts_xgb",
            "human_xgb": "human_xgb",
            "integrated": "integrated",
        }[name]
        exp_w = expected_b[key]
        exp_n = expected_b.get("n")
        ok_w = abs(actual - exp_w) <= wmape_tol
        ok_n = exp_n is None or n == exp_n
        report["checks"].append(
            {
                "name": f"analysis_b_{name}_wmape",
                "expected": exp_w,
                "actual": actual,
                "tol": wmape_tol,
                "ok": ok_w,
            }
        )
        report["checks"].append(
            {
                "name": f"analysis_b_{name}_n",
                "expected": exp_n,
                "actual": n,
                "ok": ok_n,
            }
        )
        if not (ok_w and ok_n):
            report["ok"] = False

    # Analysis A anchors + negative controls (budget universe)
    expected_a = man.get(
        "expected_analysis_a_primary_wmape", EXPECTED_ANALYSIS_A_PRIMARY
    )
    a_models = ["human"]
    if run_xgb:
        a_models.extend(
            [
                "bias_global",
                "bias_product",
                "bias_product_horizon",
                "af_ratio",
                "ridge",
                "human_xgb",
            ]
        )
    for name in a_models:
        res = backtest(name, dataset=ds, universe="budget", eligibility="primary")
        actual = float(res.overall["wmape"].iloc[0])
        n = int(res.overall["n"].iloc[0])
        key = name
        exp_w = expected_a[key]
        exp_n = expected_a.get("n")
        ok_w = abs(actual - exp_w) <= wmape_tol
        ok_n = exp_n is None or n == exp_n
        report["checks"].append(
            {
                "name": f"analysis_a_{name}_wmape",
                "expected": exp_w,
                "actual": actual,
                "tol": wmape_tol,
                "ok": ok_w,
            }
        )
        report["checks"].append(
            {
                "name": f"analysis_a_{name}_n",
                "expected": exp_n,
                "actual": n,
                "ok": ok_n,
            }
        )
        if not (ok_w and ok_n):
            report["ok"] = False

    return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify benchmark v1 offline")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--wmape-tol", type=float, default=0.05)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--anchors-only",
        action="store_true",
        help="Skip XGB/Ridge (fast smoke test of frozen anchors)",
    )
    args = parser.parse_args(argv)
    report = verify(
        root=args.root,
        wmape_tol=args.wmape_tol,
        check_checksums=not args.skip_checksums,
        run_xgb=not args.anchors_only,
    )
    for c in report["checks"]:
        status = "OK" if c["ok"] else "FAIL"
        print(f"[{status}] {c['name']}: expected={c.get('expected')} actual={c.get('actual')}")
    if report["ok"]:
        print("\nAll checks passed.")
        return 0
    print("\nSome checks FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
