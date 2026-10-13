"""Versioned HB reliability extract (outside frozen benchmarks/).

Identifier-first path: SQL never selects BudgetQty. Incomplete Shamsi months are
**not** dropped — incomplete actual sales must never delete stored HB forecasts.

Does not mutate ``src/data/benchmarks/``, TS/residual locks, or write the final
reliability contract. Does not open HB quantities for scoring.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from pkg.benchmark.budget_origin_v2 import (
    assert_no_duplicate_poh,
    assert_version_quarter_consistent,
    assign_budget_origin_from_version,
    completeness_by_origin,
    filter_horizons_1_15,
)
from pkg.benchmark.calendar import origin_from_quarter, shamsi_add_months
from pkg.benchmark.config import EXCLUDED_ODD_COVERAGE_PRODUCTS
from pkg.db.query.product_budget import (
    load_line_budget_identifiers,
    version_to_qrt,
)
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    ELIGIBLE_UNIVERSE_REL,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    GROUP_SPECIFIC_ONLY_ORIGIN,
    HB_FORECAST_COVERAGE_AUDIT_FILENAME,
    HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME,
    HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME,
    HB_RELIABILITY_COVERAGE_ORIGINS,
    HB_RELIABILITY_EXTRACT_DIR_REL,
    HB_RELIABILITY_EXTRACT_META_REL,
    HB_RELIABILITY_IDENTIFIERS_PANEL_REL,
    HB_SCORING_PATH_PREPARE_FILENAME,
    MACHINE_140404_A0_CONFIG_REL,
    MACHINE_140404_A0_EXPORT_REL,
    MACHINE_140404_A0_RUN_ID,
    MACHINE_140404_A0_SEEDS,
    MACHINE_140404_TRAINING_SALES_N_ROWS,
    MACHINE_140404_TRAINING_SALES_REL,
    MACHINE_140404_TRAINING_SALES_SHA256,
    MACHINE_140404_V21_CONFIG_REL,
    MACHINE_140404_V21_EXPORT_REL,
    MACHINE_140404_V21_RUN_ID,
    PROPOSED_FINAL_FULL_ORIGINS,
    PROSPECTIVE_HB_ORIGIN_140404,
    PROSPECTIVE_MACHINE_TRAIN_END,
    PROSPECTIVE_SCORING_SALES_THROUGH,
    PROTECTED_HB_ORIGINS,
    REQUIRED_HORIZONS,
    SALES_FULL_PORTFOLIO_REL,
    SALES_FULL_PORTFOLIO_V2_REL,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
    evaluate_scoring_sales_maturity_gate,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def build_identifier_panel(
    raw_ids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Version→quarter→origin, exact calendar horizons; no INCOMPLETE_SHAMSI drop.

    ``raw_ids`` must already exclude odd-coverage products if desired; this
    function never filters incomplete Shamsi months.
    """
    if raw_ids.empty:
        empty = pd.DataFrame(
            columns=[
                "product",
                "generic",
                "source_budget_quarter",
                "budget_version",
                "budget_origin",
                "target_date",
                "horizon",
                "edition",
            ]
        )
        return empty, empty.copy(), {"n_rows_in": 0}

    work = raw_ids.copy()
    if "forecast" in work.columns or "budget_forecast" in work.columns:
        raise ValueError(
            "Identifier panel must not carry forecast quantities; "
            "use load_line_budget_identifiers() (BudgetQty never selected)."
        )
    # Placeholder forecast for shared assign/aggregate helpers — never retained.
    work["forecast"] = 0.0
    assert_version_quarter_consistent(work)
    # Deterministic collapse on identity keys (no qty aggregation).
    keys = ["product", "qrt", "version", "date"]
    agg_cols: dict[str, str] = {"forecast": "first"}
    if "generic" in work.columns:
        agg_cols["generic"] = "first"
    if "edition" in work.columns:
        agg_cols["edition"] = "first"
    agg = work.groupby(keys, as_index=False).agg(agg_cols)
    assigned = assign_budget_origin_from_version(agg)
    kept, excluded = filter_horizons_1_15(assigned)
    assert_no_duplicate_poh(kept)

    panel = kept.copy()
    if "budget_version" not in panel.columns and "version" in panel.columns:
        panel["budget_version"] = panel["version"]
    if "source_budget_quarter" not in panel.columns and "qrt" in panel.columns:
        panel["source_budget_quarter"] = panel["qrt"]
    if "edition" not in panel.columns and "budget_version" in panel.columns:
        panel["edition"] = panel["budget_version"].astype(int) % 100
    keep_cols = [
        "product",
        "generic",
        "source_budget_quarter",
        "qrt",
        "budget_version",
        "budget_origin",
        "target_date",
        "date",
        "horizon",
        "edition",
    ]
    panel = panel[[c for c in keep_cols if c in panel.columns]].copy()
    # Never retain placeholder forecast
    assert "forecast" not in panel.columns
    assert "budget_forecast" not in panel.columns
    assert not any(c in panel.columns for c in FORBIDDEN_OUTCOME_COLS)

    stats = {
        "n_rows_in": int(len(raw_ids)),
        "n_rows_after_identity_collapse": int(len(agg)),
        "n_rows_kept_1_15": int(len(panel)),
        "n_rows_excluded_horizon": int(len(excluded)),
        "incomplete_shamsi_months_filter_applied": False,
        "columns_retained": list(panel.columns),
    }
    return panel, excluded, stats


def provenance_audit(raw_before_earliest: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    """Earliest-edition provenance; publication timing remains unverified."""
    edition_gt1 = 0
    edition_gt1_sample: list[dict[str, Any]] = []
    if len(panel) and "budget_version" in panel.columns:
        editions = pd.to_numeric(panel["budget_version"], errors="coerce").fillna(0).astype(
            int
        ) % 100
        mask = editions > 1
        edition_gt1 = int(mask.to_numpy().sum())
        for _, row in panel.loc[mask].head(30).iterrows():
            edition_gt1_sample.append(
                {
                    "product": str(row["product"]),
                    "budget_origin": int(row["budget_origin"]),
                    "budget_version": int(row["budget_version"]),
                    "edition": int(row["budget_version"]) % 100,
                    "note": (
                        "edition>01 is recorded for provenance only; it does NOT "
                        "prove the forecast was issued after the stated origin"
                    ),
                }
            )

    origin_mismatch = 0
    if len(panel) and "source_budget_quarter" in panel.columns:
        q = panel["source_budget_quarter"].astype(str)
        derived = q.map(lambda x: int(origin_from_quarter(x)))
        origin_mismatch = int(
            (
                derived.to_numpy()
                != pd.to_numeric(panel["budget_origin"], errors="coerce").to_numpy()
            ).sum()
        )

    return {
        "publication_timing_status": "PUBLICATION_TIMING_UNVERIFIED",
        "publication_timestamp_columns_in_sql": [],
        "publication_timestamps_available": False,
        "earliest_edition_only": True,
        "earliest_edition_is_useful_provenance": True,
        "earliest_edition_proves_availability_at_origin": False,
        "edition_gt_01_proves_late_forecast": False,
        "n_panel_rows_earliest_edition_gt_01": edition_gt1,
        "earliest_edition_gt_01_sample": edition_gt1_sample,
        "n_origin_quarter_mismatches": origin_mismatch,
        "note": (
            "No publication timestamps are available from the identifier SQL. "
            "Keeping the earliest Version/edition is useful provenance, but it "
            "does not prove HB availability at the forecast origin. edition>01 "
            "alone does not prove a late forecast. Status remains "
            "PUBLICATION_TIMING_UNVERIFIED unless another source establishes timing."
        ),
    }


def per_origin_completeness(panel: pd.DataFrame, origins: list[int]) -> dict[str, Any]:
    """Exact {1..15} completeness; incompletes explicit (never fill/renumber)."""
    out: dict[str, Any] = {}
    for origin in origins:
        g = panel.loc[
            pd.to_numeric(panel["budget_origin"], errors="coerce") == int(origin)
        ]
        by_target: dict[str, int] = {}
        if len(g):
            vc = (
                pd.to_numeric(g["target_date"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )
            by_target = {str(int(k)): int(v) for k, v in vc.items()}
        miss_h: Counter[int] = Counter()
        n_complete = 0
        n_skus = int(g["product"].nunique()) if len(g) else 0
        incomplete_sample: list[dict[str, Any]] = []
        for product, pg in g.groupby(g["product"].astype(str), sort=False):
            hs = {int(h) for h in pd.to_numeric(pg["horizon"], errors="coerce").dropna()}
            missing = sorted(REQUIRED_HORIZONS - hs)
            if missing:
                for m in missing:
                    miss_h[m] += 1
                if len(incomplete_sample) < 25:
                    incomplete_sample.append(
                        {
                            "product": str(product),
                            "horizons_present": sorted(hs),
                            "missing": missing,
                        }
                    )
            elif hs == set(REQUIRED_HORIZONS):
                n_complete += 1
        out[str(origin)] = {
            "n_skus_with_any_row": n_skus,
            "n_skus_complete_1_15": n_complete,
            "n_skus_incomplete": n_skus - n_complete,
            "n_skus_missing_each_horizon": {
                str(h): int(miss_h.get(h, 0)) for h in range(1, 16)
            },
            "counts_by_target_month": by_target,
            "target_140505_present": "140505" in by_target,
            "incomplete_groups_sample": incomplete_sample,
            "never_fill_or_renumber": True,
        }
    return out


def evaluate_hb_forecast_coverage_gate(
    *,
    panel: pd.DataFrame,
    origin: int = PROSPECTIVE_HB_ORIGIN_140404,
) -> dict[str, Any]:
    """Gate 1: original HB identifiers + complete SKU×origin groups for 140404."""
    g = panel.loc[
        pd.to_numeric(panel["budget_origin"], errors="coerce") == int(origin)
    ]
    has_140505 = bool(
        (pd.to_numeric(g["target_date"], errors="coerce") == 140505).any()
    ) if len(g) else False
    has_h14 = bool(
        (pd.to_numeric(g["horizon"], errors="coerce") == 14).any()
    ) if len(g) else False
    n_complete = 0
    for _p, pg in g.groupby(g["product"].astype(str), sort=False):
        hs = {int(h) for h in pd.to_numeric(pg["horizon"], errors="coerce").dropna()}
        if hs == set(REQUIRED_HORIZONS):
            n_complete += 1
    n_skus = int(g["product"].nunique()) if len(g) else 0
    passed = has_140505 and has_h14 and n_complete > 0
    return {
        "gate": "hb_forecast_coverage",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "origin": origin,
        "n_skus_with_any_row": n_skus,
        "n_skus_complete_1_15": n_complete,
        "target_140505_restored": has_140505,
        "horizon_14_present": has_h14,
        "incomplete_shamsi_filter_absent": True,
    }


def evaluate_actuals_mature_gate(*, repo_root: Path) -> dict[str, Any]:
    """Gate 2: validated sales_full_portfolio_v2 + warehouse closure through 140506."""
    detail = evaluate_scoring_sales_maturity_gate(repo_root=repo_root)
    passed = bool(detail.get("passed"))
    return {
        "gate": "actuals_mature",
        "status": "PASS" if passed else "PENDING",
        "passed": passed,
        "required_closed_through": PROSPECTIVE_SCORING_SALES_THROUGH,
        "detail": detail,
    }


def expected_target_dates_for_origin(origin: int) -> dict[int, int]:
    """horizon → exact Shamsi target_date for origin (never renumber)."""
    return {h: int(shamsi_add_months(int(origin), h - 1)) for h in range(1, 16)}


def _check_pinned_training_contract(*, repo_root: Path, config_rel: Path) -> dict[str, Any]:
    """Assert 140404 job config pins frozen v1 sales + train-through 140402."""
    import yaml

    path = repo_root / config_rel
    out: dict[str, Any] = {
        "config_relative": config_rel.as_posix(),
        "exists": path.is_file(),
        "passed": False,
    }
    if not path.is_file():
        out["fail_reasons"] = ["config missing"]
        return out
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sales = cfg.get("sales") or {}
    expected = sales.get("expected") or {}
    reasons: list[str] = []
    path_ok = str(sales.get("path") or "").replace("\\", "/") == (
        MACHINE_140404_TRAINING_SALES_REL.as_posix()
    )
    if not path_ok:
        reasons.append(
            f"sales.path={sales.get('path')!r} != {MACHINE_140404_TRAINING_SALES_REL.as_posix()}"
        )
    sha = str(expected.get("content_sha256") or "").lower()
    if sha != MACHINE_140404_TRAINING_SALES_SHA256.lower():
        reasons.append("sales.expected.content_sha256 mismatch vs pinned v1 contract")
    n_rows = expected.get("n_rows")
    if n_rows is not None and int(n_rows) != MACHINE_140404_TRAINING_SALES_N_ROWS:
        reasons.append(
            f"sales.expected.n_rows={n_rows} != {MACHINE_140404_TRAINING_SALES_N_ROWS}"
        )
    prep = cfg.get("reliability_prepare") or {}
    train_end = prep.get("machine_train_end")
    if train_end is not None and int(train_end) != PROSPECTIVE_MACHINE_TRAIN_END:
        reasons.append(
            f"machine_train_end={train_end} != {PROSPECTIVE_MACHINE_TRAIN_END}"
        )
    origins = [int(x) for x in (cfg.get("origins") or [])]
    if origins != [PROSPECTIVE_HB_ORIGIN_140404]:
        reasons.append(f"origins={origins} (require [{PROSPECTIVE_HB_ORIGIN_140404}])")
    out["passed"] = len(reasons) == 0
    out["fail_reasons"] = reasons
    out["pinned"] = {
        "sales_path": MACHINE_140404_TRAINING_SALES_REL.as_posix(),
        "content_sha256": MACHINE_140404_TRAINING_SALES_SHA256,
        "n_rows": MACHINE_140404_TRAINING_SALES_N_ROWS,
        "machine_train_end": PROSPECTIVE_MACHINE_TRAIN_END,
    }
    return out


def _status_histogram_from_run(run_dir: Path, *, origin: int) -> dict[str, Any]:
    """Read job statuses from state.sqlite (no forecast qty)."""
    import sqlite3

    db = run_dir / "state.sqlite"
    if not db.is_file():
        return {"exists": False, "by_status": {}, "n_jobs_origin": 0}
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            """
            SELECT status, product_id, selected_model, COUNT(*) AS n
            FROM jobs
            WHERE forecast_origin = ?
            GROUP BY status, product_id, selected_model
            """,
            (int(origin),),
        ).fetchall()
        by_status: Counter[str] = Counter()
        n_jobs = 0
        for status, _pid, _model, n in rows:
            by_status[str(status)] += int(n)
            n_jobs += int(n)
    finally:
        conn.close()
    return {
        "exists": True,
        "by_status": dict(by_status),
        "n_jobs_origin": n_jobs,
    }


def _validate_panel_targets_and_seeds(
    panel: pd.DataFrame,
    *,
    origin: int,
    require_a0_seeds: bool,
) -> dict[str, Any]:
    """Identifier checks on export panel: exact targets; A0 seeds; no qty used."""
    expected = expected_target_dates_for_origin(origin)
    g = panel.loc[pd.to_numeric(panel["forecast_origin"], errors="coerce") == origin]
    # Never require forecast column for gate logic
    id_cols = [c for c in ("product", "forecast_origin", "horizon", "target_date", "seed", "job_status", "selected_model", "engine") if c in g.columns]
    work = g[id_cols].copy() if len(g) else g

    target_mismatches = 0
    if len(work) and "horizon" in work.columns and "target_date" in work.columns:
        for h, td in zip(
            pd.to_numeric(work["horizon"], errors="coerce"),
            pd.to_numeric(work["target_date"], errors="coerce"),
            strict=False,
        ):
            if h != h or td != td:
                continue
            hi, ti = int(h), int(td)
            if hi in expected and ti != expected[hi]:
                target_mismatches += 1

    n_skus = int(work["product"].nunique()) if len(work) and "product" in work.columns else 0
    n_complete_native = 0
    n_fallback = 0
    n_unavailable = 0
    seed_coverage: dict[str, Any] = {"required_seeds": list(MACHINE_140404_A0_SEEDS)}

    # Classification rule (frozen-policy aligned):
    # - SUCCESS rows are native engine candidates (incl. naive / seasonal_naive).
    # - INELIGIBLE is scientific ineligibility / explicit fallback provenance
    #   (typically naive_last_value). Never treat SUCCESS model names as fallback.
    if require_a0_seeds and len(work) and "seed" in work.columns:
        complete_products = 0
        unavailable_products = 0
        for _product, pg in work.groupby(work["product"].astype(str), sort=False):
            seeds_ok = True
            for seed in MACHINE_140404_A0_SEEDS:
                sg = pg.loc[pd.to_numeric(pg["seed"], errors="coerce") == seed]
                statuses = (
                    set(sg["job_status"].astype(str).str.upper())
                    if "job_status" in sg.columns and len(sg)
                    else set()
                )
                if statuses and statuses != {"SUCCESS"}:
                    seeds_ok = False
                hs = {
                    int(h)
                    for h in pd.to_numeric(sg["horizon"], errors="coerce").dropna()
                }
                if hs != set(REQUIRED_HORIZONS):
                    seeds_ok = False
            if seeds_ok:
                complete_products += 1
            else:
                unavailable_products += 1
        n_complete_native = complete_products
        n_fallback = 0  # INELIGIBLE A0 jobs are not in SUCCESS export panel
        n_unavailable = unavailable_products
        seed_coverage["n_products_all_three_seeds_complete_native"] = complete_products
        seed_coverage["n_products_legitimate_fallback"] = 0
        seed_coverage["n_products_unavailable"] = unavailable_products
        seed_coverage["classification_rule"] = (
            "SUCCESS panel rows = native candidates; "
            "INELIGIBLE reconciled from state.sqlite separately"
        )
    elif len(work):
        for _product, pg in work.groupby(work["product"].astype(str), sort=False):
            hs = {
                int(h)
                for h in pd.to_numeric(pg["horizon"], errors="coerce").dropna()
            }
            statuses = (
                set(pg["job_status"].astype(str).str.upper())
                if "job_status" in pg.columns
                else {"SUCCESS"}
            )
            # Export panel is SUCCESS-only; treat complete SUCCESS as native.
            if hs == set(REQUIRED_HORIZONS) and (
                not statuses or statuses == {"SUCCESS"}
            ):
                n_complete_native += 1
            else:
                n_unavailable += 1

    return {
        "n_skus_with_origin_rows": n_skus,
        "n_skus_complete_native_1_15": n_complete_native,
        "n_skus_legitimate_fallback": n_fallback,
        "n_skus_unavailable": n_unavailable,
        "n_target_date_mismatches": target_mismatches,
        "exact_target_dates_ok": target_mismatches == 0 and n_skus > 0,
        "seed_coverage": seed_coverage if require_a0_seeds else None,
        "classification_rule": (
            "SUCCESS (any selected_model incl. naive/seasonal_naive) = native; "
            "INELIGIBLE = scientific fallback (state.sqlite); never silent drop"
        ),
        "columns_read": id_cols,
        "forecast_qty_read": False,
    }


def evaluate_machine_ready_gate(*, repo_root: Path) -> dict[str, Any]:
    """Gate 3: discover portfolio_*_backfill_140404 outputs (not frozen v2 exports)."""
    repo_root = Path(repo_root).resolve()
    origin = PROSPECTIVE_HB_ORIGIN_140404
    training = {
        "v21": _check_pinned_training_contract(
            repo_root=repo_root, config_rel=MACHINE_140404_V21_CONFIG_REL
        ),
        "a0": _check_pinned_training_contract(
            repo_root=repo_root, config_rel=MACHINE_140404_A0_CONFIG_REL
        ),
    }

    engines: dict[str, Any] = {}
    specs = (
        (
            "v21",
            MACHINE_140404_V21_RUN_ID,
            MACHINE_140404_V21_EXPORT_REL,
            False,
        ),
        (
            "a0",
            MACHINE_140404_A0_RUN_ID,
            MACHINE_140404_A0_EXPORT_REL,
            True,
        ),
    )
    all_ok = True
    for label, run_id, export_rel, require_seeds in specs:
        run_dir = repo_root / "src" / "data" / "forecast_jobs" / run_id
        export_path = repo_root / export_rel
        status_hist = _status_histogram_from_run(run_dir, origin=origin)
        block: dict[str, Any] = {
            "run_id": run_id,
            "run_dir": f"src/data/forecast_jobs/{run_id}",
            "run_dir_exists": run_dir.is_dir(),
            "export_relative": export_rel.as_posix(),
            "export_exists": export_path.is_file(),
            "not_frozen_v2_export": True,
            "job_status_histogram": status_hist,
            "training_contract": training[label],
            "required_a0_seeds": list(MACHINE_140404_A0_SEEDS) if require_seeds else [],
        }
        panel_check: dict[str, Any] | None = None
        if export_path.is_file():
            want = [
                "product",
                "forecast_origin",
                "horizon",
                "target_date",
                "seed",
                "job_status",
                "selected_model",
                "engine",
            ]
            import pyarrow.parquet as pq

            schema_cols = [f.name for f in pq.read_schema(export_path)]
            read_cols = [
                c
                for c in want
                if c in schema_cols and c not in FORBIDDEN_OUTCOME_COLS
            ]
            df = pd.read_parquet(export_path, columns=read_cols)
            assert "forecast" not in df.columns
            assert not any(c in df.columns for c in FORBIDDEN_OUTCOME_COLS)
            panel_check = _validate_panel_targets_and_seeds(
                df, origin=origin, require_a0_seeds=require_seeds
            )
            block["export_sha256"] = file_sha256(export_path)
            block["panel_checks"] = panel_check
        else:
            block["panel_checks"] = None
            block["pending_reason"] = (
                "140404 export panel missing; run VPS backfill then "
                f"export_portfolio_backfill_panel --run-id {run_id}"
            )

        # Pass criteria: training pin OK, export exists, exact targets, native complete>0,
        # A0 has all three seeds where required; fallbacks reported separately (not fail alone).
        passed_engine = bool(training[label].get("passed")) and export_path.is_file()
        if panel_check is not None:
            n_native = int(panel_check.get("n_skus_complete_native_1_15") or 0)
            n_fb = int(panel_check.get("n_skus_legitimate_fallback") or 0)
            n_unavail = int(panel_check.get("n_skus_unavailable") or 0)
            n_skus = int(panel_check.get("n_skus_with_origin_rows") or 0)
            # Every SUCCESS panel product must be complete native or recorded fallback.
            passed_engine = (
                passed_engine
                and bool(panel_check.get("exact_target_dates_ok"))
                and n_skus > 0
                and n_unavail == 0
                and (n_native + n_fb) == n_skus
            )
            if require_seeds:
                sc = panel_check.get("seed_coverage") or {}
                n_seed_un = int(sc.get("n_products_unavailable") or 0)
                passed_engine = passed_engine and n_seed_un == 0
        else:
            passed_engine = False

        # Valid SUCCESS presence when run dir exists
        if status_hist.get("exists"):
            n_success = int((status_hist.get("by_status") or {}).get("SUCCESS", 0))
            n_ineligible = int((status_hist.get("by_status") or {}).get("INELIGIBLE", 0))
            block["n_SUCCESS"] = n_success
            block["n_INELIGIBLE_fallback"] = n_ineligible
            block["n_FAILED"] = int((status_hist.get("by_status") or {}).get("FAILED", 0))
            block["n_PENDING"] = int((status_hist.get("by_status") or {}).get("PENDING", 0))
            if n_success == 0 and (panel_check is None or int(panel_check.get("n_skus_complete_native_1_15") or 0) == 0):
                passed_engine = False

        block["passed"] = bool(passed_engine)
        if not passed_engine:
            all_ok = False
        engines[label] = block

    return {
        "gate": "machine_ready",
        "status": "PASS" if all_ok else "PENDING",
        "passed": all_ok,
        "origin": origin,
        "discovers": {
            "run_ids": [MACHINE_140404_V21_RUN_ID, MACHINE_140404_A0_RUN_ID],
            "export_panels": [
                MACHINE_140404_V21_EXPORT_REL.as_posix(),
                MACHINE_140404_A0_EXPORT_REL.as_posix(),
            ],
            "not": [
                "src/data/results/product_universe/portfolio_v21_backfill_v2_export/",
                "src/data/results/product_universe/portfolio_a0_backfill_v2_export/",
            ],
        },
        "engines": engines,
        "expected_target_dates": {
            str(k): v for k, v in expected_target_dates_for_origin(origin).items()
        },
        "note": (
            "machine_ready discovers portfolio_*_backfill_140404 run dirs and exports. "
            "Requires pinned v1 training sales contract, SUCCESS statuses, exact "
            "calendar target dates, and A0 seeds 41/42/43. Legitimate naive/"
            "INELIGIBLE fallbacks and unavailable SKUs are reported separately."
        ),
    }


def build_scoring_path_prepare(*, repo_root: Path) -> dict[str, Any]:
    """Prepare notes: one pinned v2 actuals snapshot across origins; v1 for train."""
    return {
        "artifact": HB_SCORING_PATH_PREPARE_FILENAME,
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "scoring_actuals": {
            "pinned_snapshot": SALES_FULL_PORTFOLIO_V2_REL.as_posix(),
            "applies_to_all_reliability_origins": True,
            "origins": sorted(HB_RELIABILITY_COVERAGE_ORIGINS),
            "status": "PREPARED_AWAITING_V2_EXTRACT",
            "historical_sales_revisions": "report_separately_vs_v1_hashes",
        },
        "machine_training_history": {
            "source": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "retain_frozen_v1": True,
            "never_overwrite_v1": True,
            "train_through_for_140404": int(
                shamsi_add_months(PROSPECTIVE_HB_ORIGIN_140404, -2)
            ),
        },
        "hb_forecast_source": {
            "extract": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
            "outside_frozen_benchmarks": True,
            "incomplete_shamsi_months_do_not_drop_forecasts": True,
        },
        "eligible_universe": ELIGIBLE_UNIVERSE_REL.as_posix(),
    }


QUANTITY_RETRIEVAL_KEY_COLS: tuple[str, ...] = (
    "product",
    "budget_version",
    "edition",
    "source_budget_quarter",
    "budget_origin",
    "target_date",
    "horizon",
)


def assert_exact_quantity_key_agreement(
    identifiers: pd.DataFrame,
    quantities: pd.DataFrame,
    *,
    key_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assert quantity rows match audited identifier keys exactly.

    Refuses silent edition/cohort drift: every quantity key must appear in the
    identifiers panel, and the matched set must equal the identifier key set
    for the requested cohort (no extras, no missing keys after join).
    """
    keys = list(key_cols) if key_cols is not None else list(QUANTITY_RETRIEVAL_KEY_COLS)
    missing_id = [c for c in keys if c not in identifiers.columns]
    missing_qty = [c for c in keys if c not in quantities.columns]
    if missing_id or missing_qty:
        raise ValueError(
            "quantity key agreement requires columns "
            f"identifiers_missing={missing_id} quantities_missing={missing_qty}"
        )
    id_keys = identifiers[keys].drop_duplicates()
    qty_keys = quantities[keys].drop_duplicates()
    merged = id_keys.merge(qty_keys, on=keys, how="outer", indicator=True)
    only_id = int((merged["_merge"] == "left_only").sum())
    only_qty = int((merged["_merge"] == "right_only").sum())
    both = int((merged["_merge"] == "both").sum())
    if only_qty > 0:
        raise AssertionError(
            "quantity retrieval introduced keys absent from identifiers_panel "
            f"(n_extra={only_qty}); refuse silent edition/cohort change"
        )
    if only_id > 0:
        raise AssertionError(
            "quantity retrieval missing audited identifier keys "
            f"(n_missing={only_id}); refuse cohort shrinkage"
        )
    return {
        "assert_exact_key_agreement": True,
        "n_keys_identifiers": int(len(id_keys)),
        "n_keys_quantities": int(len(qty_keys)),
        "n_keys_matched": both,
        "n_keys_only_identifiers": only_id,
        "n_keys_only_quantities": only_qty,
        "passed": only_id == 0 and only_qty == 0,
    }


def build_quantity_retrieval_prepare(*, repo_root: Path) -> dict[str, Any]:
    """Pin later HB quantity retrieval to audited identifier keys.

    ``identifiers_panel.parquet`` establishes coverage only. A later quantity
    query must join on exact product / Version / edition / origin / target keys
    so restoring BudgetQty cannot silently introduce another edition or change
    the cohort. This step does **not** fetch quantities.
    """
    repo_root = Path(repo_root).resolve()
    panel_path = repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
    key_cols = list(QUANTITY_RETRIEVAL_KEY_COLS)
    key_stats: dict[str, Any] = {"panel_exists": panel_path.is_file()}
    if panel_path.is_file():
        df = pd.read_parquet(panel_path)
        assert not any(c in df.columns for c in FORBIDDEN_OUTCOME_COLS)
        present = [c for c in key_cols if c in df.columns]
        # Self-check: keys are unique on the audited panel (no duplicate edition
        # for the same product×origin×target).
        key_dupes = 0
        if all(c in df.columns for c in key_cols):
            key_dupes = int(df.duplicated(subset=key_cols).sum())
        key_stats.update(
            {
                "n_rows": int(len(df)),
                "n_products": int(df["product"].nunique()) if len(df) else 0,
                "key_columns_present": present,
                "key_columns_missing": [c for c in key_cols if c not in df.columns],
                "n_duplicate_key_rows": key_dupes,
                "content_sha256": file_sha256(panel_path),
                "assert_exact_key_agreement": True,
                "assert_helper": (
                    "pkg.research.portfolio_forecast_reliability_v1."
                    "hb_reliability_extract.assert_exact_quantity_key_agreement"
                ),
            }
        )
    return {
        "artifact": HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME,
        "created_at": _utc(),
        "status": "PREPARED_NOT_EXECUTED",
        "hb_outcomes_inspected": False,
        "hb_quantities_fetched": False,
        "final_contract_written": False,
        "identifiers_panel": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
        "identifiers_establish_coverage_only": True,
        "identifiers_cannot_provide_hb_values": True,
        "quantity_query": {
            "loader_planned": "pkg.db.query.product_budget.load_line_budget_forecasts",
            "restrict_to_target_generics": False,
            "earliest_edition_only": False,
            "join_keys_required": key_cols,
            "join_rule": (
                "INNER JOIN fetched (product, version, date→target_date) to "
                "identifiers_panel on product + budget_version + edition + "
                "budget_origin + target_date + horizon. Refuse rows whose "
                "Version/edition is not in the audited key set. Call "
                "assert_exact_quantity_key_agreement before accepting quantities."
            ),
            "assert_exact_key_agreement": True,
            "refuse_if_extra_edition_or_cohort_change": True,
            "budget_qty_column": "forecast",
            "not_run_in_this_step": True,
        },
        "key_stats": key_stats,
        "note": (
            "Later quantity retrieval must pin to these audited keys. Restoring "
            "HB values must not silently introduce another edition or change the cohort."
        ),
    }


def run_identifier_extract_and_audits(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load identifier-only SQL, write extract + coverage/candidate manifests."""
    repo_root = Path(repo_root).resolve()
    columns_fetched_from_sql = [
        "product",
        "generic",
        "fk_date_id",
        "version",
    ]
    raw = load_line_budget_identifiers(earliest_edition_only=True)
    assert "forecast" not in raw.columns
    assert "budget_forecast" not in raw.columns

    n_before_odd = len(raw)
    raw = raw.loc[
        ~raw["product"].astype(str).isin(EXCLUDED_ODD_COVERAGE_PRODUCTS)
    ].copy()
    n_dropped_odd = n_before_odd - len(raw)

    # Restore 140505 presence check pre-panel
    n_140505_ids = int(
        (pd.to_numeric(raw["date"], errors="coerce") == 140505).sum()
    )
    n_140505_skus = int(
        raw.loc[pd.to_numeric(raw["date"], errors="coerce") == 140505, "product"]
        .astype(str)
        .nunique()
    )

    panel, excluded, build_stats = build_identifier_panel(raw)
    prov = provenance_audit(raw, panel)
    origins = sorted(HB_RELIABILITY_COVERAGE_ORIGINS)
    completeness = per_origin_completeness(panel, origins)

    # Write extract (identifiers only)
    out_dir = repo_root / HB_RELIABILITY_EXTRACT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
    panel.to_parquet(panel_path, index=False)

    meta = {
        "extract_version": "hb_reliability_extract_v1",
        "created_at": _utc(),
        "family": FAMILY,
        "mode": "identifiers_only",
        "outside_frozen_benchmarks": True,
        "frozen_benchmarks_mutated": False,
        "incomplete_shamsi_months_filter_applied": False,
        "sql": {
            "loader": "pkg.db.query.product_budget.load_line_budget_identifiers",
            "columns_fetched_from_sql": columns_fetched_from_sql,
            "budget_qty_selected": False,
            "identifier_only_query": True,
            "note": (
                "Dropping forecast after a qty-bearing load is NOT identifier-only; "
                "this extract never selects BudgetQty."
            ),
        },
        "columns_retained": list(panel.columns),
        "product_exclusions": {
            "EXCLUDED_ODD_COVERAGE_PRODUCTS": sorted(EXCLUDED_ODD_COVERAGE_PRODUCTS),
            "n_rows_dropped": int(n_dropped_odd),
        },
        "n_rows_target_140505_after_exclusions": n_140505_ids,
        "n_skus_target_140505_after_exclusions": n_140505_skus,
        "build_stats": build_stats,
        "panel": {
            "relative_path": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
            "n_rows": int(len(panel)),
            "n_products": int(panel["product"].nunique()) if len(panel) else 0,
            "content_sha256": file_sha256(panel_path),
        },
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }
    _atomic_json(repo_root / HB_RELIABILITY_EXTRACT_META_REL, meta)

    gate_hb = evaluate_hb_forecast_coverage_gate(panel=panel)
    gate_act = evaluate_actuals_mature_gate(repo_root=repo_root)
    gate_mach = evaluate_machine_ready_gate(repo_root=repo_root)

    coverage_audit = {
        "audit_version": "hb_reliability_forecast_coverage_audit_v1",
        "created_at": _utc(),
        "family": FAMILY,
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
        "columns_fetched_from_sql": columns_fetched_from_sql,
        "columns_retained": list(panel.columns),
        "identifier_only_query": True,
        "extract": {
            "relative_path": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
            "meta": HB_RELIABILITY_EXTRACT_META_REL.as_posix(),
            "sha256": meta["panel"]["content_sha256"],
        },
        "policy": {
            "incomplete_shamsi_months_filter_applied": False,
            "incomplete_actuals_never_delete_hb_forecasts": True,
            "never_fill_or_renumber_horizons": True,
            "product_exclusions_preserved": sorted(EXCLUDED_ODD_COVERAGE_PRODUCTS),
        },
        "n_rows_target_140505_restored": n_140505_ids,
        "n_skus_target_140505_restored": n_140505_skus,
        "reliability_origins": origins,
        "per_origin_completeness": completeness,
        "provenance": prov,
        "gates": {
            "hb_forecast_coverage": gate_hb,
            "actuals_mature": {
                "status": gate_act["status"],
                "passed": gate_act["passed"],
            },
            "machine_ready": {
                "status": gate_mach["status"],
                "passed": gate_mach["passed"],
                "discovers": gate_mach.get("discovers"),
            },
        },
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
        "frozen_locks_mutated": False,
    }
    cov_path = repo_root / FAMILY_DIR_REL / HB_FORECAST_COVERAGE_AUDIT_FILENAME
    _atomic_json(cov_path, coverage_audit)

    # Candidate manifest (additive; separate from earlier candidate_origin_manifest_v1)
    all_hb = bool(gate_hb["passed"])
    manifest = {
        "manifest_version": "hb_reliability_candidate_manifest_v1",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "gates": {
            "hb_forecast_coverage": gate_hb,
            "actuals_mature": gate_act,
            "machine_ready": gate_mach,
        },
        "140404_hb_forecast_coverage": gate_hb["status"],
        "excluded_from_contract_freeze": not (
            gate_hb["passed"] and gate_act["passed"] and gate_mach["passed"]
        ),
        "stop_before_final_contract": True,
        "proposed_final_origins_if_all_gates_pass": list(PROPOSED_FINAL_FULL_ORIGINS),
        "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "note": (
            "HB forecast coverage is independent of actuals maturity. "
            "actuals_mature and machine_ready remain PENDING until v2 sales "
            "closure and 140404 machine exports exist."
        ),
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
    }
    man_path = repo_root / FAMILY_DIR_REL / HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME
    _atomic_json(man_path, manifest)

    scoring = build_scoring_path_prepare(repo_root=repo_root)
    score_path = repo_root / FAMILY_DIR_REL / HB_SCORING_PATH_PREPARE_FILENAME
    _atomic_json(score_path, scoring)

    qty_prep = build_quantity_retrieval_prepare(repo_root=repo_root)
    qty_path = repo_root / FAMILY_DIR_REL / HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME
    _atomic_json(qty_path, qty_prep)

    return {
        "ok": True,
        "extract_dir": out_dir.as_posix(),
        "panel_path": panel_path.as_posix(),
        "panel_sha256": meta["panel"]["content_sha256"],
        "coverage_audit_path": cov_path.as_posix(),
        "candidate_manifest_path": man_path.as_posix(),
        "scoring_path_prepare": score_path.as_posix(),
        "quantity_retrieval_prepare": qty_path.as_posix(),
        "n_skus_target_140505_restored": n_140505_skus,
        "gates": {
            "hb_forecast_coverage": gate_hb["status"],
            "actuals_mature": gate_act["status"],
            "machine_ready": gate_mach["status"],
        },
        "publication_timing_status": prov.get("publication_timing_status"),
        "n_skus_complete_140404": completeness.get("140404", {}).get(
            "n_skus_complete_1_15"
        ),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }


def refresh_gates_from_existing_panel(*, repo_root: Path) -> dict[str, Any]:
    """Recompute gates/provenance/prepare artifacts from the pinned panel (no SQL)."""
    repo_root = Path(repo_root).resolve()
    panel_path = repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path)
    assert not any(c in panel.columns for c in FORBIDDEN_OUTCOME_COLS)
    meta_path = repo_root / HB_RELIABILITY_EXTRACT_META_REL
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else {}
    )
    columns_fetched = list(
        (meta.get("sql") or {}).get("columns_fetched_from_sql")
        or ["product", "generic", "fk_date_id", "version"]
    )
    n_140505_ids = int(
        (pd.to_numeric(panel["target_date"], errors="coerce") == 140505).sum()
    ) if len(panel) else 0
    n_140505_skus = int(
        panel.loc[
            pd.to_numeric(panel["target_date"], errors="coerce") == 140505, "product"
        ]
        .astype(str)
        .nunique()
    ) if len(panel) else 0

    prov = provenance_audit(panel, panel)
    origins = sorted(HB_RELIABILITY_COVERAGE_ORIGINS)
    completeness = per_origin_completeness(panel, origins)
    gate_hb = evaluate_hb_forecast_coverage_gate(panel=panel)
    gate_act = evaluate_actuals_mature_gate(repo_root=repo_root)
    gate_mach = evaluate_machine_ready_gate(repo_root=repo_root)

    coverage_audit = {
        "audit_version": "hb_reliability_forecast_coverage_audit_v1",
        "created_at": _utc(),
        "family": FAMILY,
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read": [],
        "columns_fetched_from_sql": columns_fetched,
        "columns_retained": list(panel.columns),
        "identifier_only_query": True,
        "extract": {
            "relative_path": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
            "meta": HB_RELIABILITY_EXTRACT_META_REL.as_posix(),
            "sha256": file_sha256(panel_path),
        },
        "policy": {
            "incomplete_shamsi_months_filter_applied": False,
            "incomplete_actuals_never_delete_hb_forecasts": True,
            "never_fill_or_renumber_horizons": True,
            "product_exclusions_preserved": sorted(EXCLUDED_ODD_COVERAGE_PRODUCTS),
        },
        "n_rows_target_140505_restored": n_140505_ids,
        "n_skus_target_140505_restored": n_140505_skus,
        "reliability_origins": origins,
        "per_origin_completeness": completeness,
        "provenance": prov,
        "gates": {
            "hb_forecast_coverage": gate_hb,
            "actuals_mature": {
                "status": gate_act["status"],
                "passed": gate_act["passed"],
            },
            "machine_ready": {
                "status": gate_mach["status"],
                "passed": gate_mach["passed"],
                "discovers": gate_mach.get("discovers"),
            },
        },
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
        "frozen_locks_mutated": False,
        "refreshed_from_existing_panel": True,
        "sql_requery": False,
    }
    cov_path = repo_root / FAMILY_DIR_REL / HB_FORECAST_COVERAGE_AUDIT_FILENAME
    _atomic_json(cov_path, coverage_audit)

    manifest = {
        "manifest_version": "hb_reliability_candidate_manifest_v1",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "gates": {
            "hb_forecast_coverage": gate_hb,
            "actuals_mature": gate_act,
            "machine_ready": gate_mach,
        },
        "140404_hb_forecast_coverage": gate_hb["status"],
        "excluded_from_contract_freeze": not (
            gate_hb["passed"] and gate_act["passed"] and gate_mach["passed"]
        ),
        "stop_before_final_contract": True,
        "proposed_final_origins_if_all_gates_pass": list(PROPOSED_FINAL_FULL_ORIGINS),
        "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "note": (
            "HB forecast coverage is independent of actuals maturity. "
            "actuals_mature and machine_ready remain PENDING until v2 sales "
            "closure and 140404 machine exports exist."
        ),
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
        "refreshed_from_existing_panel": True,
    }
    man_path = repo_root / FAMILY_DIR_REL / HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME
    _atomic_json(man_path, manifest)

    scoring = build_scoring_path_prepare(repo_root=repo_root)
    score_path = repo_root / FAMILY_DIR_REL / HB_SCORING_PATH_PREPARE_FILENAME
    _atomic_json(score_path, scoring)

    qty_prep = build_quantity_retrieval_prepare(repo_root=repo_root)
    qty_path = repo_root / FAMILY_DIR_REL / HB_QUANTITY_RETRIEVAL_PREPARE_FILENAME
    _atomic_json(qty_path, qty_prep)

    return {
        "ok": True,
        "refreshed_from_existing_panel": True,
        "coverage_audit_path": cov_path.as_posix(),
        "candidate_manifest_path": man_path.as_posix(),
        "scoring_path_prepare": score_path.as_posix(),
        "quantity_retrieval_prepare": qty_path.as_posix(),
        "gates": {
            "hb_forecast_coverage": gate_hb["status"],
            "actuals_mature": gate_act["status"],
            "machine_ready": gate_mach["status"],
        },
        "publication_timing_status": prov.get("publication_timing_status"),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }
