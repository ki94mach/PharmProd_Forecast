"""Prepare / extract sales_full_portfolio_v2 (never overwrites v1).

Default mode is --prepare-only: writes a readiness manifest under
portfolio_forecast_reliability_v1 and does NOT write sales.parquet.

Full extract is gated on scoring-sales completeness through 140506 and
refuses to overwrite an existing v2 parquet or any v1 artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hashlib  # noqa: E402

from pkg.research.portfolio_forecast_reliability_v1.constants import (  # noqa: E402
    FAMILY_DIR_REL,
    PROSPECTIVE_MACHINE_TRAIN_END,
    PROSPECTIVE_SCORING_SALES_THROUGH,
    SALES_FULL_PORTFOLIO_REL,
    SALES_FULL_PORTFOLIO_V2_META_REL,
    SALES_FULL_PORTFOLIO_V2_REL,
    SALES_V2_PREPARE_MANIFEST_FILENAME,
    SALES_WAREHOUSE_CLOSURE_V2_REL,
)

V1_OUT_DIR = (
    ROOT / "src" / "data" / "results" / "product_universe" / "sales_full_portfolio_v1"
)
V1_SNAPSHOT = "sales_full_portfolio_v1"
OUT_DIR = (
    ROOT / "src" / "data" / "results" / "product_universe" / "sales_full_portfolio_v2"
)
SNAPSHOT_VERSION = "sales_full_portfolio_v2"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_prepare_manifest(*, repo_root: Path) -> dict[str, Any]:
    """Readiness / policy manifest only — no parquet written."""
    v1_parquet = repo_root / SALES_FULL_PORTFOLIO_REL
    v1_meta = v1_parquet.parent / "meta.json"
    v1_hashes = v1_parquet.parent / "product_history_hashes.csv"
    v2_parquet = repo_root / SALES_FULL_PORTFOLIO_V2_REL

    return {
        "status": "PREPARED_NOT_EXTRACTED",
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at_utc": _utc(),
        "parquet_written": False,
        "extract_allowed_after": PROSPECTIVE_SCORING_SALES_THROUGH,
        "extract_gate": (
            f"Refuse extract until scoring sales support completeness through "
            f"{PROSPECTIVE_SCORING_SALES_THROUGH}"
        ),
        "never_overwrite_v1": True,
        "v1": {
            "snapshot_version": V1_SNAPSHOT,
            "directory": V1_OUT_DIR.as_posix(),
            "sales_parquet": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "exists": v1_parquet.is_file(),
            "content_sha256": file_sha256(v1_parquet) if v1_parquet.is_file() else None,
            "meta_exists": v1_meta.is_file(),
            "history_hashes_exists": v1_hashes.is_file(),
            "role_for_140404_machine_backfill": (
                f"Frozen history through {PROSPECTIVE_MACHINE_TRAIN_END} "
                "(origin−2); never replace with v2 for training"
            ),
        },
        "v2_intended": {
            "directory": OUT_DIR.as_posix(),
            "sales_parquet": SALES_FULL_PORTFOLIO_V2_REL.as_posix(),
            "exists_already": v2_parquet.is_file(),
            "schema": {"columns": ["product", "date", "sales"]},
            "fields_to_record_at_freeze": [
                "content_sha256",
                "n_rows",
                "n_products_with_sales",
                "min_date",
                "max_date",
                "size_bytes",
                "products_sha256",
                "completeness_through_140506",
            ],
            "role": "scoring actuals only after 140506 closes",
            "meta_json": SALES_FULL_PORTFOLIO_V2_META_REL.as_posix(),
            "warehouse_closure_json": SALES_WAREHOUSE_CLOSURE_V2_REL.as_posix(),
            "warehouse_closure_required_fields": [
                "closed_through",
                "closed_at_utc",
                "validated",
                "source",
            ],
            "maturity_gate_note": (
                "v1 max(date) cannot prove warehouse closure. Maturity requires "
                "validated v2 parquet+meta and warehouse_closure.validated=true "
                f"with closed_through>={PROSPECTIVE_SCORING_SALES_THROUGH}."
            ),
            "historical_revision_audit_vs_v1": {
                "compare": "product_history_hashes.csv through_last_complete",
                "through_origin_minus_2": PROSPECTIVE_MACHINE_TRAIN_END,
                "must_match_v1_for_dates_le": PROSPECTIVE_MACHINE_TRAIN_END,
                "note": (
                    "When v2 is extracted, audit that history through 140402 "
                    "matches v1 hashes; divergences are recorded, never silent."
                ),
            },
        },
        "policy": {
            "machine_backfill_140404_sales": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "scoring_actuals_after_close": SALES_FULL_PORTFOLIO_V2_REL.as_posix(),
            "never_overwrite_v1": True,
            "force_forbidden_for_prepare": True,
            "v1_max_date_cannot_prove_closure": True,
        },
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }


def write_prepare_manifest(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    manifest = build_prepare_manifest(repo_root=repo_root)
    out_dir = repo_root / FAMILY_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SALES_V2_PREPARE_MANIFEST_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return {
        "ok": True,
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "parquet_written": False,
        "status": manifest["status"],
    }


def extract_v2(*, repo_root: Path) -> dict[str, Any]:
    """Full extract — gated; never overwrites v1; refuses existing v2 parquet."""
    repo_root = Path(repo_root).resolve()
    # Hard refuse touching v1
    v1_parquet = repo_root / SALES_FULL_PORTFOLIO_REL
    if not v1_parquet.is_file():
        raise FileNotFoundError(f"v1 sales required and must remain frozen: {v1_parquet}")

    # Gate: do not extract until 140506 can be supported — prepare-only is the
    # default path; full extract still refuses unless caller opts in carefully.
    # For safety in this prepare program, refuse full extract unless explicitly
    # allowed via environment that sales through 140506 is ready — still check
    # warehouse via extract_v1_impl into OUT_DIR only.
    raise RuntimeError(
        "Full sales_full_portfolio_v2 extract is not enabled in the prepare step. "
        f"Re-run after {PROSPECTIVE_SCORING_SALES_THROUGH} closes with an explicit "
        "post-maturity workflow. Use --prepare-only to write the readiness manifest."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        default=True,
        help="Write readiness manifest only (default).",
    )
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Attempt full v2 extract (refused until post-140506 workflow).",
    )
    args = ap.parse_args()
    if args.extract:
        try:
            extract_v2(repo_root=args.repo_root)
        except RuntimeError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            # Still write prepare manifest
            summary = write_prepare_manifest(repo_root=args.repo_root)
            print(json.dumps(summary, indent=2))
            return 1
    summary = write_prepare_manifest(repo_root=args.repo_root)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
