"""F3E Step 2 CLI — Point-in-Time Peer Demand Feature Construction.

Usage:
    python -m pkg.research.prepare_f3e_step2

Exit codes:
    0  success — all assertions passed, artifacts written
    1  file not found (missing Step 1 freeze or benchmark files)
    2  PIT / exclusion / semantic assertion failed — STOP, do not run Step 3
"""
from __future__ import annotations

import sys


def main() -> int:
    from pathlib import Path

    try:
        import pandas as pd

        from pkg.benchmark import load_benchmark
        from pkg.benchmark.config import default_benchmark_root
        from pkg.research.f3e.config import f3e_feature_audit_dir, f3e_source_dir
        from pkg.research.f3e.config import (
            NORMALIZED_MONTHLY_SALES_PARQUET,
            PRODUCT_PEER_PROFILE_PARQUET,
        )
        from pkg.research.f3e.feature_audit import run_feature_audit
        from pkg.research.f3e.feature_report import write_feature_audit_report
        from pkg.research.f3e.features import build_f3e_features
        from pkg.research.f3e.prepare import (
            _file_fingerprint,
            assert_freeze_untouched,
            mvp_products,
        )
    except ImportError as exc:
        print(f"[F3E Step 2] ImportError: {exc}", file=sys.stderr)
        return 1

    print("[F3E Step 2] Starting — PIT Peer Demand Feature Construction")

    bench_root = default_benchmark_root()

    try:
        fingerprint_before = _file_fingerprint(bench_root)
    except Exception as exc:
        print(f"[F3E Step 2] ERROR fingerprinting benchmark: {exc}", file=sys.stderr)
        return 1

    # Load Step 1 frozen artifacts
    src = f3e_source_dir()
    norm_path = src / NORMALIZED_MONTHLY_SALES_PARQUET
    profile_path = src / PRODUCT_PEER_PROFILE_PARQUET

    for p in (norm_path, profile_path):
        if not p.exists():
            print(f"[F3E Step 2] ERROR: missing Step 1 artifact: {p}", file=sys.stderr)
            print("[F3E Step 2] Run `python -m pkg.research.prepare_f3e` first.", file=sys.stderr)
            return 1

    try:
        panel = pd.read_parquet(norm_path)
        profile = pd.read_parquet(profile_path)
    except Exception as exc:
        print(f"[F3E Step 2] ERROR loading parquets: {exc}", file=sys.stderr)
        return 1

    print(f"[F3E Step 2] Panel: {len(panel):,} rows, {panel['product'].nunique():,} products")

    # Load MVP list and benchmark PRIMARY rows
    try:
        mvp_list = mvp_products(bench_root)
        benchmark = load_benchmark()
    except FileNotFoundError as exc:
        print(f"[F3E Step 2] ERROR: {exc}", file=sys.stderr)
        return 1

    primary_rows = benchmark.matched_universe
    print(f"[F3E Step 2] PRIMARY rows: {len(primary_rows):,}, "
          f"products: {primary_rows['product'].nunique():,}")

    # Build features and run all assertions
    try:
        enriched = build_f3e_features(panel, profile, primary_rows)
    except AssertionError as exc:
        print(f"[F3E Step 2] STOP — assertion failed: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[F3E Step 2] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[F3E Step 2] Features built: {len(enriched):,} rows")

    # Run feature audit
    try:
        out_dir = f3e_feature_audit_dir()
        audit = run_feature_audit(enriched, panel, profile, mvp_list, out_dir=out_dir)
    except Exception as exc:
        print(f"[F3E Step 2] ERROR during feature audit: {exc}", file=sys.stderr)
        return 2

    # Write features parquet
    features_parquet = out_dir / "features.parquet"
    enriched.to_parquet(features_parquet, index=False)
    print(f"[F3E Step 2] Features written to: {features_parquet}")

    # Print coverage summary
    cov = audit.get("coverage_overall")
    if cov is not None and not cov.empty:
        for feat in [
            "log_generic_peer_dqtyunit_last_month",
            "log_generic_peer_dqtyunit_3m_mean",
            "log_cross_generic_field_consume_patients_last_month",
            "log_cross_generic_field_consume_patients_3m_mean",
        ]:
            avail_row = cov.loc[cov["metric"] == f"{feat}_available_rows"]
            pct_row = cov.loc[cov["metric"] == f"{feat}_coverage_pct"]
            if not avail_row.empty and not pct_row.empty:
                print(f"[F3E Step 2]   {feat}: "
                      f"{int(avail_row['value'].iloc[0]):,} rows "
                      f"({float(pct_row['value'].iloc[0]):.1f}%)")

    # Write report
    doc_path = write_feature_audit_report(audit)
    print(f"[F3E Step 2] Report written to: {doc_path}")

    # Assert benchmark freeze untouched
    try:
        assert_freeze_untouched(bench_root, fingerprint_before)
    except AssertionError as exc:
        print(f"[F3E Step 2] STOP — benchmark freeze modified: {exc}", file=sys.stderr)
        return 2

    print("[F3E Step 2] Complete. Review docs/f3e_peer_demand_feature_audit.md before Step 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
