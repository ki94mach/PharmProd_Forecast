#!/usr/bin/env python
"""Build benchmarks/v2 budget_universe with Version→quarter→origin_from_quarter.

Does not overwrite v1. Does not join sales or compute outcome metrics.
Does not run machine backfills or freeze the reliability contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.benchmark.budget_origin_v2 import (  # noqa: E402
    build_budget_panel_v2,
    completeness_by_origin,
    file_sha256,
)
from pkg.benchmark.config import (  # noqa: E402
    EXCLUDED_ODD_COVERAGE_PRODUCTS,
    FORECAST_HORIZON_MONTHS,
    INCOMPLETE_SHAMSI_MONTHS,
    default_benchmark_root,
    manifest_path,
)
from pkg.db.query.product_budget import (  # noqa: E402
    load_line_budget_forecasts,
    version_to_qrt,
)
from pkg.portfolio_ts_lock.constants import HB_ORIGINS  # noqa: E402

V2_REL = Path("src/data/benchmarks/v2")
COVERAGE_REL = Path(
    "src/data/results/portfolio_forecast_reliability_v1/hb_origin_v2_coverage_audit.json"
)
MODULE_PATH = ROOT / "src/pkg/benchmark/budget_origin_v2.py"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _product_filter(repo: Path) -> set[str] | None:
    """Unused hook; full coverage uses exclude-list in ``_load_raw``."""
    return None


def _incompleteness_summary(panel: pd.DataFrame) -> dict[str, Any]:
    """Per-origin: how many products lack exact horizons {1..15}."""
    comp = completeness_by_origin(panel)
    by_origin_summary: dict[str, Any] = {}
    for origin, block in (comp.get("by_origin") or {}).items():
        n_any = int(block.get("n_products_with_any_row") or 0)
        n_complete = int(block.get("n_complete_1_15") or 0)
        n_incomplete = int(block.get("n_incomplete") or 0)
        by_origin_summary[str(origin)] = {
            "n_products_with_any_row": n_any,
            "n_products_complete_1_15": n_complete,
            "n_products_incomplete": n_incomplete,
            "pct_incomplete": round(100.0 * n_incomplete / n_any, 2) if n_any else None,
            "incomplete_products": [
                {
                    "product": g["product"],
                    "missing": g["missing"],
                    "extra": g["extra"],
                    "horizons_present": g["horizons_present"],
                }
                for g in (block.get("incomplete_groups") or [])
            ],
        }
    return {
        "n_distinct_products_panel": int(panel["product"].nunique()) if len(panel) else 0,
        "n_complete_groups_total": int(comp.get("n_complete_groups") or 0),
        "n_incomplete_groups_total": int(comp.get("n_incomplete_groups") or 0),
        "by_origin": by_origin_summary,
    }


def _load_raw(repo: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all Line Budget SKUs (~full portfolio). Do not require sales.

    MVP (~55–60) is TARGET_GENERIC_EN only. Full portfolio is unrestricted
    Line Budget. Sales joins belong in outcome evaluation, not HB panel build.
    """
    del repo  # reserved for future path hooks; HB load is SQL-only
    raw = load_line_budget_forecasts(
        earliest_edition_only=True,
        restrict_to_target_generics=False,
    )
    src_meta: dict[str, Any] = {
        "loader": "pkg.db.query.product_budget.load_line_budget_forecasts",
        "earliest_edition_only": True,
        "restrict_to_target_generics": False,
        "n_rows_loaded": int(len(raw)),
        "n_distinct_products_loaded_raw": int(raw["product"].nunique())
        if len(raw)
        else 0,
        "sql_generic_filter": None,
        "sales_joined": False,
        "product_filter": "all_line_budget_minus_EXCLUDED_ODD_COVERAGE",
        "note": (
            "HB panel is forecast-only. No sales intersect. "
            "MVP TARGET_GENERIC_EN (~55–60) is not applied; expect ~full portfolio."
        ),
    }
    before = len(raw)
    raw = raw.loc[
        ~raw["product"].astype(str).isin(EXCLUDED_ODD_COVERAGE_PRODUCTS)
    ].copy()
    src_meta["n_rows_after_product_filter"] = int(len(raw))
    src_meta["n_rows_dropped_odd_coverage"] = int(before - len(raw))
    src_meta["n_distinct_products_loaded"] = (
        int(raw["product"].nunique()) if len(raw) else 0
    )
    if "date" in raw.columns and INCOMPLETE_SHAMSI_MONTHS:
        before = len(raw)
        raw = raw.loc[~raw["date"].astype(int).isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
        src_meta["n_rows_dropped_incomplete_shamsi_months"] = int(before - len(raw))
    return raw, src_meta


def _coverage_audit(
    *,
    repo: Path,
    panel: pd.DataFrame,
) -> dict[str, Any]:
    """Identifier-only coverage vs A0/V2.1; never open outcome columns."""
    hb_outcomes_inspected = False
    id_cols_hb = [
        "product",
        "source_budget_quarter",
        "budget_origin",
        "target_date",
        "horizon",
    ]
    origins = sorted(int(x) for x in panel["budget_origin"].unique()) if len(panel) else []

    # Remap check: rows that would have been origin 140312 under min(date) with qrt=1404Q1
    q1404q1 = panel.loc[panel["source_budget_quarter"].astype(str).str.upper() == "1404Q1"]
    targets_140312_under_1404q1 = q1404q1.loc[
        q1404q1["target_date"].astype(int) == 140312
    ]
    # After correct assignment those targets are pre_origin and should NOT be in panel;
    # check whether 1404Q1 keeps originate at 140401
    origin_for_1404q1 = sorted(
        int(x) for x in q1404q1["budget_origin"].unique()
    ) if len(q1404q1) else []

    likely_protected = [140304, 140307, 140310, 140401]
    likely_protected_present = all(o in origins for o in likely_protected)

    # Machine panels from existing reliability coverage audit paths if present
    cov_path = (
        repo
        / "src/data/results/portfolio_residual_xgb_v1/hb_reliability_coverage_audit_v1.json"
    )
    machine: dict[str, Any] = {"a0": {}, "v21": {}}
    matched: dict[str, Any] = {}
    if cov_path.is_file():
        prev = json.loads(cov_path.read_text(encoding="utf-8"))
        a0_path = Path(prev["a0_panel"]["relative_path"])
        v21_path = Path(prev["v21_panel"]["relative_path"])
        for label, path in (("a0", a0_path), ("v21", v21_path)):
            if not path.is_file():
                machine[label] = {"exists": False, "path": str(path)}
                continue
            mdf = pd.read_parquet(
                path, columns=["product", "forecast_origin", "horizon", "target_date"]
            )
            mdf["forecast_origin"] = pd.to_numeric(
                mdf["forecast_origin"], errors="coerce"
            ).astype("Int64")
            by_o: dict[str, Any] = {}
            for o in origins:
                g = mdf.loc[mdf["forecast_origin"] == o]
                by_o[str(o)] = {
                    "origin_exists": bool(len(g) > 0),
                    "n_products_with_any_row": int(g["product"].nunique())
                    if len(g)
                    else 0,
                }
            machine[label] = {
                "exists": True,
                "path": str(path),
                "columns_read": [
                    "product",
                    "forecast_origin",
                    "horizon",
                    "target_date",
                ],
                "by_origin": by_o,
            }

        # Matched product×origin complete groups (ID only)
        for o in origins:
            hb_o = panel.loc[panel["budget_origin"].astype(int) == o]
            hb_complete_products = set()
            for product, pg in hb_o.groupby("product", sort=False):
                hs = set(int(h) for h in pg["horizon"].astype(int))
                if hs == set(range(1, FORECAST_HORIZON_MONTHS + 1)):
                    hb_complete_products.add(str(product))
            entry: dict[str, Any] = {
                "hb_complete_products": len(hb_complete_products),
            }
            for label in ("a0", "v21"):
                path = Path(
                    prev["a0_panel"]["relative_path"]
                    if label == "a0"
                    else prev["v21_panel"]["relative_path"]
                )
                if not path.is_file():
                    entry[f"{label}_matched_complete"] = None
                    continue
                mdf = pd.read_parquet(
                    path, columns=["product", "forecast_origin", "horizon"]
                )
                mdf = mdf.loc[
                    pd.to_numeric(mdf["forecast_origin"], errors="coerce") == o
                ]
                m_complete = set()
                for product, pg in mdf.groupby("product", sort=False):
                    hs = set(
                        int(h)
                        for h in pd.to_numeric(pg["horizon"], errors="coerce").dropna()
                    )
                    if hs == set(range(1, FORECAST_HORIZON_MONTHS + 1)):
                        m_complete.add(str(product))
                entry[f"{label}_matched_complete"] = len(
                    hb_complete_products & m_complete
                )
            matched[str(o)] = entry

    machine_missing_140401 = False
    if 140401 in origins:
        for label in ("a0", "v21"):
            block = machine.get(label) or {}
            by_o = block.get("by_origin") or {}
            info = by_o.get("140401") or {}
            if block.get("exists") and not info.get("origin_exists"):
                machine_missing_140401 = True

    return {
        "hb_outcomes_inspected": hb_outcomes_inspected,
        "columns_read_hb": id_cols_hb,
        "forecast_or_outcome_columns_read": [],
        "corrected_distinct_hb_origins": origins,
        "completeness": completeness_by_origin(panel),
        "incompleteness_by_origin": _incompleteness_summary(panel),
        "qrt_1404Q1_budget_origins": origin_for_1404q1,
        "target_140312_under_1404Q1_in_kept_panel": int(len(targets_140312_under_1404q1)),
        "note_140312_to_140401": (
            "Rows with source_budget_quarter=1404Q1 receive budget_origin=140401. "
            "Targets at 140312 under 1404Q1 are horizon 0 and must be excluded "
            "(not kept as origin 140312)."
        ),
        "likely_protected_set": likely_protected,
        "likely_protected_set_all_present_in_hb_v2": likely_protected_present,
        "old_HB_ORIGINS_constant": sorted(HB_ORIGINS),
        "machine_panel_coverage_id_only": machine,
        "matched_complete_product_origins": matched,
        "machine_forecasts_required": {
            "origin_140401_required": bool(140401 in origins and machine_missing_140401),
            "do_not_rename_existing_machine_forecasts": True,
            "production_cutoff_if_generated": {
                "train_through": 140311,
                "bridge_month": 140312,
                "h1": 140401,
            },
            "backfill_executed": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    args = ap.parse_args()
    repo = args.repo_root.resolve()

    out_dir = repo / V2_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to touch v1
    v1 = default_benchmark_root()
    assert v1.resolve() != out_dir.resolve()

    raw, src_meta = _load_raw(repo)
    panel, excluded, stats = build_budget_panel_v2(raw)

    bud_path = out_dir / "budget_universe.parquet"
    excl_path = out_dir / "excluded_rows_audit.parquet"
    meta_path = out_dir / "budget_universe_metadata.json"

    panel_sorted = panel.sort_values(
        ["budget_origin", "product", "horizon", "target_date"]
    ).reset_index(drop=True)
    panel_sorted.to_parquet(bud_path, index=False)
    excluded.to_parquet(excl_path, index=False)

    v1_bud = v1 / "budget_universe.parquet"
    v1_hash = file_sha256(v1_bud) if v1_bud.is_file() else None
    manifest_v1 = None
    if manifest_path().is_file():
        man = json.loads(manifest_path().read_text(encoding="utf-8"))
        manifest_v1 = (
            man.get("file_hashes", {}) or man.get("hashes", {})
        ).get("budget_universe.parquet")

    coverage = _coverage_audit(repo=repo, panel=panel_sorted)

    metadata: dict[str, Any] = {
        "benchmark_version": "v2",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "source": {
            **src_meta,
            "version_semantics": (
                "SQL Version YYYYQQEE → version_to_qrt → source_budget_quarter; "
                "budget_origin = origin_from_quarter(quarter). "
                "Do not infer quarter from target_date."
            ),
            "version_to_qrt_example": {
                "14040101": version_to_qrt(14040101),
                "14030401": version_to_qrt(14030401),
            },
        },
        "previous_benchmark": {
            "path": (Path("src/data/benchmarks/v1/budget_universe.parquet")).as_posix(),
            "sha256_file": v1_hash,
            "sha256_manifest": manifest_v1,
        },
        "quarter_to_origin_mapping": stats["quarter_to_origin_mapping"],
        "origin_formula": "budget_origin = origin_from_quarter(version_to_qrt(Version))",
        "horizon_formula": (
            "horizon = 12*(target_year-origin_year)+(target_month-origin_month)+1 "
            f"; keep 1..{FORECAST_HORIZON_MONTHS}"
        ),
        "row_counts": {
            "before_filter": stats["n_rows_before_filter"],
            "after_aggregate": stats["n_rows_after_aggregate"],
            "after_horizon_filter": stats["n_rows_kept_1_15"],
            "excluded_total": stats["n_rows_excluded"],
            "excluded_pre_origin": stats["n_excluded_pre_origin"],
            "excluded_beyond_h15": stats["n_excluded_beyond_h15"],
            "excluded_non_finite_date": stats["n_excluded_non_finite_date"],
        },
        "complete_and_incomplete_groups_by_origin": stats["completeness"],
        "incompleteness_by_origin": _incompleteness_summary(panel_sorted),
        "distinct_budget_origins": stats["distinct_budget_origins"],
        "paths": {
            "budget_universe": (V2_REL / "budget_universe.parquet").as_posix(),
            "excluded_rows_audit": (V2_REL / "excluded_rows_audit.parquet").as_posix(),
            "metadata": (V2_REL / "budget_universe_metadata.json").as_posix(),
        },
        "budget_universe_sha256": file_sha256(bud_path),
        "excluded_rows_audit_sha256": file_sha256(excl_path),
        "transform_code_sha256": file_sha256(MODULE_PATH),
        "git_commit": _git_commit(repo),
        "v1_not_modified": True,
        "sales_joined": False,
        "metrics_computed": False,
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cov_out = repo / COVERAGE_REL
    cov_out.parent.mkdir(parents=True, exist_ok=True)
    coverage_payload = {
        "audit_version": "hb_origin_v2_coverage_audit",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "budget_universe_v2_sha256": metadata["budget_universe_sha256"],
        "metadata_path": (V2_REL / "budget_universe_metadata.json").as_posix(),
        **coverage,
    }
    cov_out.write_text(
        json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "ok": True,
        "hb_outcomes_inspected": False,
        "budget_universe": bud_path.as_posix(),
        "product_filter": src_meta.get("product_filter"),
        "n_distinct_products": metadata["incompleteness_by_origin"][
            "n_distinct_products_panel"
        ],
        "distinct_origins": stats["distinct_budget_origins"],
        "incompleteness_by_origin": {
            o: {
                "n_products": b["n_products_with_any_row"],
                "n_complete": b["n_products_complete_1_15"],
                "n_incomplete": b["n_products_incomplete"],
            }
            for o, b in metadata["incompleteness_by_origin"]["by_origin"].items()
        },
        "likely_protected_set_present": coverage[
            "likely_protected_set_all_present_in_hb_v2"
        ],
        "machine_140401_required": coverage["machine_forecasts_required"][
            "origin_140401_required"
        ],
        "coverage_audit": cov_out.as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
