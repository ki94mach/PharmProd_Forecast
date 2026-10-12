"""Identifier-only trace of missing 140505 / h14 for origin 140404.

Traces Versioned Line Budget → filters → v2 panel without reading HB quantities.
Does not mutate frozen benchmarks, sales, or locks.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.benchmark.config import EXCLUDED_ODD_COVERAGE_PRODUCTS, INCOMPLETE_SHAMSI_MONTHS
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    BUDGET_UNIVERSE_V2_REL,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    IDENTIFIER_TRACE_FILENAME,
    PROSPECTIVE_HB_ORIGIN_140404,
    PROSPECTIVE_HB_QUARTER_140404,
    REQUIRED_HORIZONS,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_origin_mapping_audit import (
    _load_hb_id,
)

# Horizon 14 at origin 140404 → target 140505 (incomplete Shamsi month).
TARGET_MONTH_H14 = 140505
HORIZON_H14 = 14


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_target_months_for_origin(origin: int) -> dict[str, int]:
    """Map horizon → target_date for a full 1..15 window."""
    return {str(h): int(shamsi_add_months(int(origin), h - 1)) for h in range(1, 16)}


def _per_sku_horizon_completeness(panel_o: pd.DataFrame) -> dict[str, Any]:
    missing_horizon_counts: Counter[int] = Counter()
    pattern_counts: Counter[str] = Counter()
    n_complete = 0
    n_skus = 0
    for _product, g in panel_o.groupby(panel_o["product"].astype(str), sort=False):
        n_skus += 1
        hs = {int(h) for h in pd.to_numeric(g["horizon"], errors="coerce").dropna()}
        missing = sorted(REQUIRED_HORIZONS - hs)
        if not missing and hs == set(REQUIRED_HORIZONS):
            n_complete += 1
            pattern_counts["complete_1_15"] += 1
        else:
            for m in missing:
                missing_horizon_counts[m] += 1
            key = "missing_" + "_".join(str(x) for x in missing) if missing else "extra_only"
            pattern_counts[key] += 1
    return {
        "n_skus_with_any_row": n_skus,
        "n_skus_complete_1_15": n_complete,
        "n_skus_incomplete": n_skus - n_complete,
        "n_skus_missing_each_horizon": {
            str(h): int(missing_horizon_counts.get(h, 0)) for h in range(1, 16)
        },
        "missing_pattern_counts_top": dict(pattern_counts.most_common(20)),
    }


def _counts_by_target_month(panel_o: pd.DataFrame) -> dict[str, int]:
    if panel_o.empty or "target_date" not in panel_o.columns:
        return {}
    s = (
        pd.to_numeric(panel_o["target_date"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {str(int(k)): int(v) for k, v in s.items()}


def _counts_by_horizon(panel_o: pd.DataFrame) -> dict[str, int]:
    if panel_o.empty or "horizon" not in panel_o.columns:
        return {}
    s = (
        pd.to_numeric(panel_o["horizon"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {str(int(k)): int(v) for k, v in s.items()}


def trace_raw_through_filters(*, load_sql: bool = True) -> dict[str, Any]:
    """Trace 1404Q2 / 140505 identifiers through build filters (qty dropped immediately)."""
    stages: list[dict[str, Any]] = []
    expected = expected_target_months_for_origin(PROSPECTIVE_HB_ORIGIN_140404)

    stages.append(
        {
            "stage": "0_policy_calendar",
            "origin": PROSPECTIVE_HB_ORIGIN_140404,
            "quarter": PROSPECTIVE_HB_QUARTER_140404,
            "expected_horizon_to_target": expected,
            "h14_target": expected[str(HORIZON_H14)],
            "INCOMPLETE_SHAMSI_MONTHS": sorted(INCOMPLETE_SHAMSI_MONTHS),
            "h14_target_in_INCOMPLETE_SHAMSI_MONTHS": (
                TARGET_MONTH_H14 in INCOMPLETE_SHAMSI_MONTHS
            ),
        }
    )

    sql_trace: dict[str, Any] = {
        "attempted": bool(load_sql),
        "ok": False,
        "error": None,
    }
    if load_sql:
        try:
            from pkg.db.query.product_budget import load_line_budget_forecasts

            raw = load_line_budget_forecasts(
                earliest_edition_only=True,
                restrict_to_target_generics=False,
            )
            # Drop quantity immediately — never use for scoring.
            id_raw = raw.drop(columns=["forecast"], errors="ignore").copy()
            assert "forecast" not in id_raw.columns
            assert not any(c in id_raw.columns for c in FORBIDDEN_OUTCOME_COLS)
            sql_trace["ok"] = True
            sql_trace["n_rows_loaded_id_only"] = int(len(id_raw))
            sql_trace["columns_kept"] = list(id_raw.columns)

            stages.append(
                {
                    "stage": "1_sql_load_line_budget_all",
                    "loader": "load_line_budget_forecasts",
                    "earliest_edition_only": True,
                    "restrict_to_target_generics": False,
                    "n_rows": int(len(id_raw)),
                    "n_products": int(id_raw["product"].nunique()) if len(id_raw) else 0,
                    "forecast_qty_read": False,
                    "forecast_qty_dropped_immediately": True,
                }
            )

            q = id_raw.loc[
                id_raw["qrt"].astype(str).str.upper() == PROSPECTIVE_HB_QUARTER_140404
            ].copy()
            dates_q = sorted(
                int(x)
                for x in pd.to_numeric(q["date"], errors="coerce").dropna().unique()
            )
            n_505 = int((pd.to_numeric(q["date"], errors="coerce") == TARGET_MONTH_H14).sum())
            n_505_skus = int(
                q.loc[
                    pd.to_numeric(q["date"], errors="coerce") == TARGET_MONTH_H14,
                    "product",
                ]
                .astype(str)
                .nunique()
            )
            stages.append(
                {
                    "stage": "2_filter_quarter_1404Q2",
                    "n_rows": int(len(q)),
                    "n_products": int(q["product"].nunique()) if len(q) else 0,
                    "distinct_target_dates": dates_q,
                    "n_rows_target_140505": n_505,
                    "n_skus_target_140505": n_505_skus,
                }
            )

            before = len(q)
            q2 = q.loc[
                ~q["product"].astype(str).isin(EXCLUDED_ODD_COVERAGE_PRODUCTS)
            ].copy()
            n_505_b = int(
                (pd.to_numeric(q2["date"], errors="coerce") == TARGET_MONTH_H14).sum()
            )
            n_505_skus_b = int(
                q2.loc[
                    pd.to_numeric(q2["date"], errors="coerce") == TARGET_MONTH_H14,
                    "product",
                ]
                .astype(str)
                .nunique()
            )
            stages.append(
                {
                    "stage": "3_drop_EXCLUDED_ODD_COVERAGE_PRODUCTS",
                    "n_rows_before": before,
                    "n_rows_after": int(len(q2)),
                    "n_rows_dropped": int(before - len(q2)),
                    "n_rows_target_140505": n_505_b,
                    "n_skus_target_140505": n_505_skus_b,
                }
            )

            before = len(q2)
            q3 = q2.loc[
                ~pd.to_numeric(q2["date"], errors="coerce")
                .astype("Int64")
                .isin(INCOMPLETE_SHAMSI_MONTHS)
            ].copy()
            n_505_c = int(
                (pd.to_numeric(q3["date"], errors="coerce") == TARGET_MONTH_H14).sum()
            )
            stages.append(
                {
                    "stage": "4_drop_INCOMPLETE_SHAMSI_MONTHS",
                    "incomplete_months": sorted(INCOMPLETE_SHAMSI_MONTHS),
                    "n_rows_before": before,
                    "n_rows_after": int(len(q3)),
                    "n_rows_dropped": int(before - len(q3)),
                    "n_rows_target_140505_after": n_505_c,
                    "root_cause": (
                        f"All target_date={TARGET_MONTH_H14} (h{HORIZON_H14} for "
                        f"origin {PROSPECTIVE_HB_ORIGIN_140404}) rows are removed "
                        "because 140505 ∈ INCOMPLETE_SHAMSI_MONTHS in "
                        "scripts/build_budget_universe_v2._load_raw."
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 — audit must still write offline
            sql_trace["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "sql_trace": sql_trace,
        "pipeline_stages": stages,
        "primary_root_cause": (
            "INCOMPLETE_SHAMSI_MONTHS={140505} drops every product×target 140505 "
            "row before Version→origin assignment / horizon keep-1..15. "
            f"For origin {PROSPECTIVE_HB_ORIGIN_140404}, that target is exactly "
            f"horizon {HORIZON_H14}, so no SKU can be full-horizon complete."
        ),
    }


def build_hb_140404_identifier_trace(
    *,
    repo_root: Path,
    load_sql: bool = True,
) -> dict[str, Any]:
    """Combine frozen v2 panel counts with optional live SQL filter trace."""
    repo_root = Path(repo_root).resolve()
    hb_path = repo_root / BUDGET_UNIVERSE_V2_REL
    expected = expected_target_months_for_origin(PROSPECTIVE_HB_ORIGIN_140404)

    panel_block: dict[str, Any] = {"exists": hb_path.is_file()}
    if hb_path.is_file():
        hb = _load_hb_id(hb_path)
        o = hb.loc[
            pd.to_numeric(hb["budget_origin"], errors="coerce")
            == PROSPECTIVE_HB_ORIGIN_140404
        ].copy()
        panel_block.update(
            {
                "relative_path": BUDGET_UNIVERSE_V2_REL.as_posix(),
                "sha256": file_sha256(hb_path),
                "columns_read": list(hb.attrs.get("columns_read") or []),
                "outcome_columns_present_but_not_read": list(
                    hb.attrs.get("forbidden_present_but_not_read") or []
                ),
                "n_rows_origin_140404": int(len(o)),
                "counts_by_target_month": _counts_by_target_month(o),
                "counts_by_horizon": _counts_by_horizon(o),
                "per_sku_horizon_completeness": _per_sku_horizon_completeness(o),
                "target_140505_present_in_panel": bool(
                    (
                        pd.to_numeric(o["target_date"], errors="coerce")
                        == TARGET_MONTH_H14
                    ).any()
                )
                if len(o)
                else False,
                "horizon_14_present_in_panel": bool(
                    (pd.to_numeric(o["horizon"], errors="coerce") == HORIZON_H14).any()
                )
                if len(o)
                else False,
            }
        )

    filter_trace = trace_raw_through_filters(load_sql=load_sql)

    return {
        "audit_version": "hb_140404_identifier_trace_v1",
        "family": FAMILY,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
        "prospective_origin": PROSPECTIVE_HB_ORIGIN_140404,
        "prospective_quarter": PROSPECTIVE_HB_QUARTER_140404,
        "expected_horizon_to_target": expected,
        "diagnosis": {
            "missing_target_month": TARGET_MONTH_H14,
            "missing_horizon": HORIZON_H14,
            "mechanism": "INCOMPLETE_SHAMSI_MONTHS_filter_in_v2_build",
            "config_constant": "pkg.benchmark.config.INCOMPLETE_SHAMSI_MONTHS",
            "applied_in": "scripts/build_budget_universe_v2._load_raw",
            "note": (
                "This is an identifier drop of an incomplete Shamsi month, not a "
                "silent horizon renumber. Horizons are never renumbered; h14 is absent."
            ),
        },
        "frozen_v2_panel": panel_block,
        "raw_filter_trace": filter_trace,
        "frozen_locks_mutated": False,
        "final_contract_written": False,
    }


def write_hb_140404_identifier_trace(
    *,
    repo_root: Path,
    load_sql: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    audit = build_hb_140404_identifier_trace(repo_root=repo_root, load_sql=load_sql)
    out_dir = repo_root / FAMILY_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / IDENTIFIER_TRACE_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return {
        "ok": True,
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "hb_outcomes_inspected": False,
        "target_140505_present_in_panel": (audit.get("frozen_v2_panel") or {}).get(
            "target_140505_present_in_panel"
        ),
        "n_skus_complete_1_15": (
            (audit.get("frozen_v2_panel") or {})
            .get("per_sku_horizon_completeness")
            or {}
        ).get("n_skus_complete_1_15"),
        "sql_trace_ok": ((audit.get("raw_filter_trace") or {}).get("sql_trace") or {}).get(
            "ok"
        ),
    }
