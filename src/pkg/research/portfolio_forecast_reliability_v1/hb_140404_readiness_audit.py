"""140404 machine backfill readiness + reconciliation (no HB outcome scoring).

Validates V2.1/A0 portfolio_*_backfill_140404 exports against the frozen TS
production contract pins, reconciles native/fallback/INELIGIBLE/FAILED/
NO_MACHINE_FORECAST without silent drops, and rebuilds identifier-only
machine–HB coverage for reliability origins (140401 remains group-specific).

Does not mutate locks, frozen v2 exports, or write the final reliability contract.
Does not fetch HB BudgetQty.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from pkg.benchmark.calendar import shamsi_add_months
from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    A0_PANEL_REL,
    FAMILY,
    FAMILY_DIR_REL,
    FORBIDDEN_OUTCOME_COLS,
    GROUP_SPECIFIC_ONLY_ORIGIN,
    HB_140404_MACHINE_RECONCILIATION_FILENAME,
    HB_140404_READINESS_AUDIT_FILENAME,
    HB_FORECAST_COVERAGE_AUDIT_FILENAME,
    HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME,
    HB_RELIABILITY_COVERAGE_ORIGINS,
    HB_RELIABILITY_IDENTIFIERS_PANEL_REL,
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
    V21_PANEL_REL,
    VALIDITY_GROUPS,
)
from pkg.research.portfolio_forecast_reliability_v1.deployable_policy import (
    reconcile_hb_complete_deployable,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_140404_maturity_audit import (
    evaluate_scoring_sales_maturity_gate,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_origin_mapping_audit import (
    _branch_breakdown,
    _group_coverage_block,
    _load_machine_id,
    _sku_origin_histogram,
    complete_pairs_full,
    group_complete_pairs,
)
from pkg.research.portfolio_forecast_reliability_v1.hb_reliability_extract import (
    evaluate_actuals_mature_gate,
    evaluate_hb_forecast_coverage_gate,
    evaluate_machine_ready_gate,
    expected_target_dates_for_origin,
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


def _check_config_training_contract(
    *, repo_root: Path, config_rel: Path
) -> dict[str, Any]:
    path = repo_root / config_rel
    out: dict[str, Any] = {
        "config_relative": config_rel.as_posix(),
        "exists": path.is_file(),
        "passed": False,
    }
    if not path.is_file():
        out["fail_reasons"] = ["config missing"]
        return out
    out["config_sha256"] = file_sha256(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sales = cfg.get("sales") or {}
    expected = sales.get("expected") or {}
    reasons: list[str] = []
    path_ok = str(sales.get("path") or "").replace("\\", "/") == (
        MACHINE_140404_TRAINING_SALES_REL.as_posix()
    )
    if not path_ok:
        reasons.append(f"sales.path mismatch: {sales.get('path')!r}")
    sha = str(expected.get("content_sha256") or "").lower()
    if sha != MACHINE_140404_TRAINING_SALES_SHA256.lower():
        reasons.append("sales.expected.content_sha256 mismatch vs pinned v1")
    n_rows = expected.get("n_rows")
    if n_rows is not None and int(n_rows) != MACHINE_140404_TRAINING_SALES_N_ROWS:
        reasons.append(f"sales.expected.n_rows={n_rows}")
    prep = cfg.get("reliability_prepare") or {}
    train_end = prep.get("machine_train_end")
    if train_end is not None and int(train_end) != PROSPECTIVE_MACHINE_TRAIN_END:
        reasons.append(f"machine_train_end={train_end}")
    origins = [int(x) for x in (cfg.get("origins") or [])]
    if origins != [PROSPECTIVE_HB_ORIGIN_140404]:
        reasons.append(f"origins={origins}")
    # Bridge / cutoff: production contract uses train_through origin-2.
    cutoff = cfg.get("cutoff_policy") or {}
    out["cutoff_policy"] = cutoff
    out["bridge_handling"] = {
        "production_contract": {
            "train_through": "origin-2",
            "bridge_month": "origin-1",
            "bridge_excluded_from_training": True,
            "discard_bridge_step": True,
        },
        "config_cutoff_policy_name": cutoff.get("name"),
        "machine_train_end": PROSPECTIVE_MACHINE_TRAIN_END,
        "origin_minus_2": int(
            shamsi_add_months(PROSPECTIVE_HB_ORIGIN_140404, -2)
        ),
        "aligned": int(shamsi_add_months(PROSPECTIVE_HB_ORIGIN_140404, -2))
        == PROSPECTIVE_MACHINE_TRAIN_END,
    }
    if not out["bridge_handling"]["aligned"]:
        reasons.append("origin-2 train end misaligned")
    out["fail_reasons"] = reasons
    out["passed"] = len(reasons) == 0
    out["pinned"] = {
        "sales_path": MACHINE_140404_TRAINING_SALES_REL.as_posix(),
        "content_sha256": MACHINE_140404_TRAINING_SALES_SHA256,
        "n_rows": MACHINE_140404_TRAINING_SALES_N_ROWS,
        "machine_train_end": PROSPECTIVE_MACHINE_TRAIN_END,
    }
    return out


def _job_status_reconciliation(run_dir: Path, *, origin: int) -> dict[str, Any]:
    db = run_dir / "state.sqlite"
    if not db.is_file():
        return {"exists": False, "by_status": {}, "n_jobs": 0}
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            """
            SELECT status, COALESCE(selected_model, ''), COUNT(*)
            FROM jobs
            WHERE forecast_origin = ?
            GROUP BY status, COALESCE(selected_model, '')
            """,
            (int(origin),),
        ).fetchall()
        n_products = conn.execute(
            """
            SELECT COUNT(DISTINCT product_id)
            FROM jobs WHERE forecast_origin = ?
            """,
            (int(origin),),
        ).fetchone()[0]
    finally:
        conn.close()
    by_status: Counter[str] = Counter()
    by_status_model: list[dict[str, Any]] = []
    for status, model, n in rows:
        by_status[str(status)] += int(n)
        by_status_model.append(
            {"status": str(status), "selected_model": str(model), "n": int(n)}
        )
    by_status_model.sort(key=lambda r: (-r["n"], r["status"], r["selected_model"]))
    n_success = int(by_status.get("SUCCESS", 0))
    n_inel = int(by_status.get("INELIGIBLE", 0))
    n_failed = int(by_status.get("FAILED", 0))
    n_pending = int(by_status.get("PENDING", 0))
    n_running = int(by_status.get("RUNNING", 0))
    n_skipped = int(by_status.get("SKIPPED", 0))
    n_jobs = int(sum(by_status.values()))
    accounted = n_success + n_inel + n_failed + n_pending + n_running + n_skipped
    return {
        "exists": True,
        "run_dir": run_dir.as_posix(),
        "n_distinct_product_ids": int(n_products),
        "n_jobs": n_jobs,
        "by_status": dict(by_status),
        "by_status_model_top": by_status_model[:40],
        "n_SUCCESS": n_success,
        "n_INELIGIBLE_scientific_or_fallback": n_inel,
        "n_FAILED_runtime": n_failed,
        "n_PENDING": n_pending,
        "n_RUNNING": n_running,
        "n_SKIPPED": n_skipped,
        "identity": {
            "equation": (
                "SUCCESS + INELIGIBLE + FAILED + PENDING + RUNNING + SKIPPED = n_jobs"
            ),
            "left": accounted,
            "right": n_jobs,
            "holds": accounted == n_jobs,
        },
        "never_silent_drop": True,
        "complete": n_pending == 0 and n_running == 0 and n_failed == 0,
    }


def _validate_machine_panel(
    panel_path: Path,
    *,
    origin: int,
    require_a0_seeds: bool,
    engine_label: str,
) -> dict[str, Any]:
    """Validate export panel vs production contract forecast constraints.

    Reads machine ``forecast`` for finite/non-negative checks only — not HB qty.
    """
    expected = expected_target_dates_for_origin(origin)
    df = pd.read_parquet(panel_path)
    out: dict[str, Any] = {
        "export_relative": None,
        "export_sha256": file_sha256(panel_path),
        "n_rows": int(len(df)),
        "n_products": int(df["product"].nunique()) if len(df) else 0,
        "columns": list(df.columns),
        "engine_label": engine_label,
    }
    g = df.loc[pd.to_numeric(df["forecast_origin"], errors="coerce") == origin]
    # Exact targets
    mismatches = 0
    if len(g):
        for h, td in zip(
            pd.to_numeric(g["horizon"], errors="coerce"),
            pd.to_numeric(g["target_date"], errors="coerce"),
            strict=False,
        ):
            if h != h or td != td:
                continue
            hi, ti = int(h), int(td)
            if hi in expected and ti != expected[hi]:
                mismatches += 1
    horizons = sorted(
        {int(h) for h in pd.to_numeric(g["horizon"], errors="coerce").dropna()}
    ) if len(g) else []
    targets = sorted(
        {int(t) for t in pd.to_numeric(g["target_date"], errors="coerce").dropna()}
    ) if len(g) else []
    out["exact_target_dates_ok"] = mismatches == 0 and len(g) > 0
    out["n_target_date_mismatches"] = mismatches
    out["horizons_observed"] = horizons
    out["horizons_exact_1_15"] = horizons == list(range(1, 16))
    out["targets_observed"] = targets
    out["targets_expected"] = [expected[h] for h in range(1, 16)]
    out["targets_exact_140404_140506"] = targets == out["targets_expected"]

    # Forecast constraints (machine panel only)
    if "forecast" in g.columns and len(g):
        f_arr = pd.to_numeric(g["forecast"], errors="coerce").to_numpy(dtype=float)
        n_nonfinite = int((~np.isfinite(f_arr)).sum())
        n_neg = int((np.isfinite(f_arr) & (f_arr < 0)).sum())
        out["forecast_constraints"] = {
            "finite": n_nonfinite == 0,
            "n_nonfinite": n_nonfinite,
            "non_negative": n_neg == 0,
            "n_negative": n_neg,
            "min": float(np.nanmin(f_arr)) if len(f_arr) else None,
            "max": float(np.nanmax(f_arr)) if len(f_arr) else None,
            "production_contract_requires_non_negative_final": True,
            "hb_quantities_read": False,
            "note": (
                "Machine-panel forecast check only; HB BudgetQty not read. "
                "Negative A0 values must be clipped before any final scored use."
            ),
        }
    else:
        out["forecast_constraints"] = {
            "finite": False,
            "non_negative": False,
            "error": "forecast column missing",
        }

    # Native completeness / seeds
    n_complete_native = 0
    n_fallback = 0
    n_unavailable = 0
    seed_block: dict[str, Any] | None = None
    if require_a0_seeds and "seed" in g.columns and len(g):
        complete_products = 0
        fallback_products = 0
        unavailable_products = 0
        for _product, pg in g.groupby(g["product"].astype(str), sort=False):
            seeds_ok = True
            is_fallback = False
            for seed in MACHINE_140404_A0_SEEDS:
                sg = pg.loc[pd.to_numeric(pg["seed"], errors="coerce") == seed]
                if "selected_model" in sg.columns and len(sg):
                    models = set(sg["selected_model"].astype(str).str.lower())
                    if any("naive" in m for m in models):
                        is_fallback = True
                hs = {
                    int(h)
                    for h in pd.to_numeric(sg["horizon"], errors="coerce").dropna()
                }
                if hs != set(REQUIRED_HORIZONS):
                    seeds_ok = False
            if seeds_ok and not is_fallback:
                complete_products += 1
            elif is_fallback:
                fallback_products += 1
            else:
                unavailable_products += 1
        n_complete_native = complete_products
        n_fallback = fallback_products
        n_unavailable = unavailable_products
        seed_block = {
            "required_seeds": list(MACHINE_140404_A0_SEEDS),
            "seeds_observed": sorted(
                {
                    int(s)
                    for s in pd.to_numeric(g["seed"], errors="coerce").dropna()
                }
            ),
            "all_three_seeds_present": set(MACHINE_140404_A0_SEEDS).issubset(
                {
                    int(s)
                    for s in pd.to_numeric(g["seed"], errors="coerce").dropna()
                }
            ),
            "n_products_all_three_seeds_complete_native": complete_products,
            "n_products_legitimate_fallback": fallback_products,
            "n_products_unavailable": unavailable_products,
        }
    elif len(g):
        for _product, pg in g.groupby(g["product"].astype(str), sort=False):
            hs = {
                int(h)
                for h in pd.to_numeric(pg["horizon"], errors="coerce").dropna()
            }
            models = (
                set(pg["selected_model"].astype(str).str.lower())
                if "selected_model" in pg.columns
                else set()
            )
            is_fallback = any("naive" in m for m in models)
            if hs == set(REQUIRED_HORIZONS) and not is_fallback:
                n_complete_native += 1
            elif hs == set(REQUIRED_HORIZONS) and is_fallback:
                n_fallback += 1
            else:
                n_unavailable += 1

    out["n_skus_complete_native_1_15"] = n_complete_native
    out["n_skus_legitimate_fallback"] = n_fallback
    out["n_skus_unavailable_in_panel"] = n_unavailable
    out["seed_coverage"] = seed_block
    # Horizon presence across panel (catches systemic missing h14 etc.)
    h_counts = (
        pd.to_numeric(g["horizon"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    out["n_rows_by_horizon"] = {str(int(k)): int(v) for k, v in h_counts.items()}
    missing_h = sorted(set(REQUIRED_HORIZONS) - set(h_counts.index.tolist()))
    out["horizons_missing_from_panel"] = missing_h
    # Per-product incomplete when any required horizon absent for the product
    # (A0: after collapsing seeds — already captured in n_unavailable).
    n_panel_products = int(g["product"].nunique()) if len(g) else 0
    n_accounted = n_complete_native + n_fallback + n_unavailable
    out["panel_product_identity"] = {
        "n_products_in_panel": n_panel_products,
        "n_complete_native": n_complete_native,
        "n_legitimate_fallback": n_fallback,
        "n_incomplete_or_unavailable": n_unavailable,
        "equation": "native + fallback + unavailable = n_products_in_panel",
        "holds": n_accounted == n_panel_products,
    }
    fc = out["forecast_constraints"]
    # Strict: every SUCCESS panel product must be complete native or recorded fallback.
    completeness_ok = (
        n_panel_products > 0
        and n_unavailable == 0
        and (n_complete_native + n_fallback) == n_panel_products
    )
    out["passed"] = bool(
        out["exact_target_dates_ok"]
        and out["horizons_exact_1_15"]
        and out["targets_exact_140404_140506"]
        and fc.get("finite")
        and fc.get("non_negative")
        and completeness_ok
        and (
            (not require_a0_seeds)
            or (
                seed_block
                and seed_block.get("all_three_seeds_present")
                and int(
                    seed_block.get("n_products_all_three_seeds_complete_native") or 0
                )
                + int(seed_block.get("n_products_legitimate_fallback") or 0)
                == n_panel_products
            )
        )
    )
    return out


def build_machine_backfill_reconciliation(*, repo_root: Path) -> dict[str, Any]:
    """Reconcile 140404 V2.1/A0 job statuses + panel contract checks."""
    repo_root = Path(repo_root).resolve()
    origin = PROSPECTIVE_HB_ORIGIN_140404
    engines: dict[str, Any] = {}
    specs = (
        (
            "v21",
            MACHINE_140404_V21_RUN_ID,
            MACHINE_140404_V21_EXPORT_REL,
            MACHINE_140404_V21_CONFIG_REL,
            False,
            "v2.1",
        ),
        (
            "a0",
            MACHINE_140404_A0_RUN_ID,
            MACHINE_140404_A0_EXPORT_REL,
            MACHINE_140404_A0_CONFIG_REL,
            True,
            "a0",
        ),
    )
    for label, run_id, export_rel, config_rel, require_seeds, eng in specs:
        run_dir = repo_root / "src" / "data" / "forecast_jobs" / run_id
        export_path = repo_root / export_rel
        training = _check_config_training_contract(
            repo_root=repo_root, config_rel=config_rel
        )
        jobs = _job_status_reconciliation(run_dir, origin=origin)
        panel = None
        if export_path.is_file():
            panel = _validate_machine_panel(
                export_path,
                origin=origin,
                require_a0_seeds=require_seeds,
                engine_label=eng,
            )
            panel["export_relative"] = export_rel.as_posix()
            # export_summary if present
            summ = export_path.parent / "export_summary.json"
            if summ.is_file():
                panel["export_summary"] = json.loads(
                    summ.read_text(encoding="utf-8")
                )
                panel["export_summary_sha256"] = file_sha256(summ)
        engines[label] = {
            "run_id": run_id,
            "not_frozen_v2_export": True,
            "source_exports_preserved": True,
            "training_contract": training,
            "job_status_reconciliation": jobs,
            "panel_validation": panel,
            "passed": bool(
                training.get("passed")
                and jobs.get("complete")
                and panel is not None
                and panel.get("passed")
            ),
        }

    # Universe identity: V2.1 SUCCESS+INELIGIBLE should cover eligible cohort
    v21_jobs = engines["v21"]["job_status_reconciliation"]
    a0_jobs = engines["a0"]["job_status_reconciliation"]
    blockers: list[str] = []
    for label, block in engines.items():
        if not block["training_contract"].get("passed"):
            blockers.append(f"{label}: training contract failed")
        if not (block["job_status_reconciliation"] or {}).get("complete"):
            blockers.append(f"{label}: jobs not complete (pending/running/failed)")
        pv = block.get("panel_validation") or {}
        if not pv:
            blockers.append(f"{label}: export panel missing")
        else:
            if not pv.get("exact_target_dates_ok"):
                blockers.append(f"{label}: exact target dates failed")
            fc = pv.get("forecast_constraints") or {}
            if not fc.get("finite"):
                blockers.append(f"{label}: non-finite forecasts")
            if not fc.get("non_negative"):
                blockers.append(
                    f"{label}: negative forecasts "
                    f"(n={fc.get('n_negative')}; production requires non-negative final)"
                )
            n_unavail = int(pv.get("n_skus_unavailable_in_panel") or 0)
            n_native = int(pv.get("n_skus_complete_native_1_15") or 0)
            n_fb = int(pv.get("n_skus_legitimate_fallback") or 0)
            n_prod = int(
                (pv.get("panel_product_identity") or {}).get("n_products_in_panel")
                or 0
            )
            if n_prod <= 0:
                blockers.append(f"{label}: empty export panel at origin 140404")
            elif n_unavail > 0:
                missing_h = pv.get("horizons_missing_from_panel") or []
                by_h = pv.get("n_rows_by_horizon") or {}
                blockers.append(
                    f"{label}: {n_unavail}/{n_prod} SUCCESS products lack complete "
                    f"horizons 1–15 (native={n_native}, fallback={n_fb}); "
                    f"rows_by_horizon={by_h}"
                    + (f"; panel_missing_horizons={missing_h}" if missing_h else "")
                )
            sc = pv.get("seed_coverage")
            if label == "a0" and sc and not sc.get("all_three_seeds_present"):
                blockers.append("a0: seeds 41/42/43 incomplete")

    return {
        "artifact": HB_140404_MACHINE_RECONCILIATION_FILENAME,
        "created_at": _utc(),
        "family": FAMILY,
        "origin": origin,
        "hb_outcomes_inspected": False,
        "hb_quantities_fetched": False,
        "final_contract_written": False,
        "frozen_locks_mutated": False,
        "frozen_v2_exports_mutated": False,
        "engines": engines,
        "universe_notes": {
            "v21_SUCCESS_plus_INELIGIBLE": int(v21_jobs.get("n_SUCCESS") or 0)
            + int(v21_jobs.get("n_INELIGIBLE_scientific_or_fallback") or 0),
            "v21_n_product_ids": v21_jobs.get("n_distinct_product_ids"),
            "a0_SUCCESS_jobs": a0_jobs.get("n_SUCCESS"),
            "a0_INELIGIBLE_jobs": a0_jobs.get(
                "n_INELIGIBLE_scientific_or_fallback"
            ),
            "a0_n_product_ids_across_seeds": a0_jobs.get("n_distinct_product_ids"),
            "INELIGIBLE_means_scientific_ineligibility_or_naive_fallback": True,
            "FAILED_means_runtime_failure": True,
            "never_silent_drop": True,
        },
        "blockers": blockers,
        "passed": len(blockers) == 0,
    }


def _load_hb_identifiers(repo_root: Path) -> pd.DataFrame:
    path = repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    assert not any(c in df.columns for c in FORBIDDEN_OUTCOME_COLS)
    # Keep reliability coverage origins only for population calcs
    origins = sorted(HB_RELIABILITY_COVERAGE_ORIGINS)
    work = df.loc[
        pd.to_numeric(df["budget_origin"], errors="coerce").isin(origins)
    ].copy()
    work = work.rename(columns={"budget_origin": "forecast_origin"})
    return work


def _machine_pairs_for_coverage(
    *, repo_root: Path
) -> tuple[dict[str, set[tuple[str, int]]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Compose machine completeness: frozen v2 for protected + 140404 exports."""
    meta: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    pairs: dict[str, set[tuple[str, int]]] = {}

    # Protected-origin panels: frozen v2 exports (unchanged)
    for label, rel in (("a0_v2", A0_PANEL_REL), ("v21_v2", V21_PANEL_REL)):
        path = repo_root / rel
        meta[label] = {
            "relative_path": rel.as_posix(),
            "exists": path.is_file(),
            "role": "frozen_v2_for_protected_origins",
            "sha256": file_sha256(path) if path.is_file() else None,
        }
        if path.is_file():
            frames[label] = _load_machine_id(path)
        else:
            frames[label] = pd.DataFrame(columns=list(("product", "forecast_origin", "horizon", "target_date")))

    # 140404 panels
    for label, rel in (
        ("a0_140404", MACHINE_140404_A0_EXPORT_REL),
        ("v21_140404", MACHINE_140404_V21_EXPORT_REL),
    ):
        path = repo_root / rel
        meta[label] = {
            "relative_path": rel.as_posix(),
            "exists": path.is_file(),
            "role": "prospective_140404_only",
            "sha256": file_sha256(path) if path.is_file() else None,
        }
        if path.is_file():
            frames[label] = _load_machine_id(path)
        else:
            frames[label] = pd.DataFrame(
                columns=list(("product", "forecast_origin", "horizon", "target_date"))
            )

    def _filter_origins(df: pd.DataFrame, origins: list[int]) -> pd.DataFrame:
        if df.empty or "forecast_origin" not in df.columns:
            return df
        return df.loc[
            pd.to_numeric(df["forecast_origin"], errors="coerce").isin(origins)
        ].copy()

    a0 = pd.concat(
        [
            _filter_origins(frames["a0_v2"], sorted(PROTECTED_HB_ORIGINS)),
            _filter_origins(frames["a0_140404"], [PROSPECTIVE_HB_ORIGIN_140404]),
        ],
        ignore_index=True,
    )
    v21 = pd.concat(
        [
            _filter_origins(frames["v21_v2"], sorted(PROTECTED_HB_ORIGINS)),
            _filter_origins(frames["v21_140404"], [PROSPECTIVE_HB_ORIGIN_140404]),
        ],
        ignore_index=True,
    )

    # A0 @ 140404: require all three seeds × horizons 1..15
    a0_complete: set[tuple[str, int]] = set()
    a0_140404_path = repo_root / MACHINE_140404_A0_EXPORT_REL
    if a0_140404_path.is_file():
        cols = [f.name for f in pq.read_schema(a0_140404_path)]
        want = [
            "product",
            "forecast_origin",
            "horizon",
            "target_date",
            "seed",
            "selected_model",
        ]
        a0_full = pd.read_parquet(
            a0_140404_path, columns=[c for c in want if c in cols]
        )
        for product, pg in a0_full.groupby(a0_full["product"].astype(str), sort=False):
            ok = True
            for seed in MACHINE_140404_A0_SEEDS:
                sg = pg.loc[pd.to_numeric(pg["seed"], errors="coerce") == seed]
                hs = {
                    int(h)
                    for h in pd.to_numeric(sg["horizon"], errors="coerce").dropna()
                }
                if hs != set(REQUIRED_HORIZONS):
                    ok = False
                    break
            if ok:
                a0_complete.add((str(product), PROSPECTIVE_HB_ORIGIN_140404))

    a0_v2_pairs, _ = complete_pairs_full(
        _filter_origins(frames["a0_v2"], sorted(PROTECTED_HB_ORIGINS)),
        "forecast_origin",
    )
    a0_complete |= a0_v2_pairs
    v21_complete, _ = complete_pairs_full(v21, "forecast_origin")
    pairs["a0"] = a0_complete
    pairs["v21"] = v21_complete
    frames["a0"] = a0
    frames["v21"] = v21
    meta["composition"] = {
        "protected_origins_from": "portfolio_*_backfill_v2_export",
        "origin_140404_from": "portfolio_*_backfill_140404_export",
        "frozen_v2_exports_mutated": False,
    }
    return pairs, frames, meta


def build_identifier_machine_hb_coverage(*, repo_root: Path) -> dict[str, Any]:
    """Identifier-only machine–HB coverage for all reliability origins."""
    repo_root = Path(repo_root).resolve()
    hb = _load_hb_identifiers(repo_root)
    origins = sorted(HB_RELIABILITY_COVERAGE_ORIGINS)
    human_pairs, human_incomplete = complete_pairs_full(hb, "forecast_origin")
    # Restrict human complete to reliability origins
    human_pairs = {(p, o) for p, o in human_pairs if o in origins}

    machine_pairs, machine_frames, machine_meta = _machine_pairs_for_coverage(
        repo_root=repo_root
    )
    a0_p = machine_pairs["a0"]
    v21_p = machine_pairs["v21"]

    sales_path = repo_root / SALES_FULL_PORTFOLIO_REL
    if not sales_path.is_file():
        raise FileNotFoundError(sales_path)
    sales = pd.read_parquet(sales_path, columns=["product", "date", "sales"])
    deployable = reconcile_hb_complete_deployable(
        human_complete=human_pairs,
        a0_complete=a0_p,
        v21_complete=v21_p,
        sales=sales,
    )
    branches = _branch_breakdown(human_pairs, a0_p, v21_p, origins)

    # Full-horizon portfolio population (exclude 140401 from proposed final set)
    full_origins = [o for o in origins if o != GROUP_SPECIFIC_ONLY_ORIGIN]
    human_full = {(p, o) for p, o in human_pairs if o in full_origins}
    deployable_full = reconcile_hb_complete_deployable(
        human_complete=human_full,
        a0_complete=a0_p,
        v21_complete=v21_p,
        sales=sales,
    )

    group_coverage: dict[str, Any] = {}
    for gname, ghs in VALIDITY_GROUPS:
        block = _group_coverage_block(
            group_name=gname,
            horizons=ghs,
            hb_df=hb,
            a0_df=machine_frames["a0"],
            v21_df=machine_frames["v21"],
            origins=origins,
        )
        hb_g, _ = group_complete_pairs(hb, horizons=ghs, origin_col="forecast_origin")
        block["origin_140401_group_specific_only"] = True
        block["n_140401_pairs_in_group"] = sum(
            1 for _p, o in hb_g if o == GROUP_SPECIFIC_ONLY_ORIGIN
        )
        group_coverage[gname] = block

    per_origin = {}
    for o in origins:
        h_o = {(p, oo) for p, oo in human_pairs if oo == o}
        per_origin[str(o)] = {
            "n_hb_complete_1_15": len(h_o),
            "n_native_any": len(h_o & (a0_p | v21_p)),
            "n_both": len(h_o & a0_p & v21_p),
            "n_v21_only": len(h_o & (v21_p - a0_p)),
            "n_a0_only": len(h_o & (a0_p - v21_p)),
            "n_neither": len(h_o - (a0_p | v21_p)),
            "group_specific_only": o == GROUP_SPECIFIC_ONLY_ORIGIN,
            "in_proposed_final_full_origins": o in PROPOSED_FINAL_FULL_ORIGINS,
        }

    return {
        "hb_identifiers_panel": HB_RELIABILITY_IDENTIFIERS_PANEL_REL.as_posix(),
        "hb_identifiers_sha256": file_sha256(
            repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
        ),
        "hb_outcomes_inspected": False,
        "forecast_or_outcome_columns_read_from_hb": [],
        "reliability_origins": origins,
        "machine_sources": machine_meta,
        "n_hb_complete_product_origins": len(human_pairs),
        "n_hb_incomplete_groups_recorded": len(human_incomplete),
        "incomplete_groups_sample": human_incomplete[:40],
        "never_silent_drop": True,
        "never_renumber_horizons": True,
        "deployable_reconciliation_all_reliability_origins": deployable,
        "deployable_reconciliation_full_horizon_origins_excluding_140401": deployable_full,
        "branch_breakdown": branches,
        "sku_origin_depth_matched_native": _sku_origin_histogram(
            human_pairs & (a0_p | v21_p)
        ),
        "per_origin": per_origin,
        "validity_group_coverage": group_coverage,
        "group_specific_only_origin": GROUP_SPECIFIC_ONLY_ORIGIN,
        "proposed_final_full_origins": list(PROPOSED_FINAL_FULL_ORIGINS),
    }


def build_readiness_audit(*, repo_root: Path) -> dict[str, Any]:
    """Full readiness audit: machine reconcile + coverage + sales gates."""
    repo_root = Path(repo_root).resolve()
    recon = build_machine_backfill_reconciliation(repo_root=repo_root)
    coverage = build_identifier_machine_hb_coverage(repo_root=repo_root)

    # Load HB panel for forecast coverage gate
    hb_path = repo_root / HB_RELIABILITY_IDENTIFIERS_PANEL_REL
    hb_panel = pd.read_parquet(hb_path)
    gate_hb = evaluate_hb_forecast_coverage_gate(panel=hb_panel)
    gate_act = evaluate_actuals_mature_gate(repo_root=repo_root)
    gate_mach = evaluate_machine_ready_gate(repo_root=repo_root)
    sales_detail = evaluate_scoring_sales_maturity_gate(repo_root=repo_root)

    # machine_ready may PASS on id completeness even if non-neg fails —
    # readiness uses the stricter reconciliation blockers.
    freeze_blockers: list[str] = []
    freeze_blockers.extend(recon.get("blockers") or [])
    if not gate_hb.get("passed"):
        freeze_blockers.append("hb_forecast_coverage gate failed")
    if not gate_act.get("passed"):
        freeze_blockers.append(
            "actuals_mature PENDING: validated sales_full_portfolio_v2 + "
            f"warehouse_closure through {PROSPECTIVE_SCORING_SALES_THROUGH} required"
        )
    if not gate_mach.get("passed"):
        freeze_blockers.append("machine_ready gate not PASS")
    if not (coverage.get("deployable_reconciliation_all_reliability_origins") or {}).get(
        "identity", {}
    ).get("holds"):
        freeze_blockers.append("deployable identity broken")
    # Explicit: no final freeze while publication timing unverified is OK for
    # machine readiness, but sales v2 is required for scoring maturity.
    ready = len(freeze_blockers) == 0

    return {
        "audit_version": "hb_140404_readiness_audit_v1",
        "created_at": _utc(),
        "family": FAMILY,
        "hb_outcomes_inspected": False,
        "hb_quantities_fetched": False,
        "final_contract_written": False,
        "frozen_locks_mutated": False,
        "frozen_v2_exports_mutated": False,
        "source_exports_preserved": True,
        "prospective_origin": PROSPECTIVE_HB_ORIGIN_140404,
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
        "machine_backfill_reconciliation": {
            "passed": recon.get("passed"),
            "blockers": recon.get("blockers"),
            "engines_passed": {
                k: v.get("passed") for k, v in (recon.get("engines") or {}).items()
            },
            "artifact": HB_140404_MACHINE_RECONCILIATION_FILENAME,
        },
        "gates": {
            "hb_forecast_coverage": gate_hb,
            "actuals_mature": gate_act,
            "machine_ready": {
                "status": gate_mach.get("status"),
                "passed": gate_mach.get("passed"),
                "discovers": gate_mach.get("discovers"),
            },
            "scoring_sales_v2_closure": sales_detail,
        },
        "identifier_machine_hb_coverage": {
            "n_hb_complete_product_origins": coverage.get(
                "n_hb_complete_product_origins"
            ),
            "deployable_identity_holds": (
                coverage.get("deployable_reconciliation_all_reliability_origins") or {}
            )
            .get("identity", {})
            .get("holds"),
            "deployable_coverage_pct": (
                coverage.get("deployable_reconciliation_all_reliability_origins") or {}
            ).get("deployable_coverage_pct"),
            "identity": (
                coverage.get("deployable_reconciliation_all_reliability_origins") or {}
            ).get("identity"),
            "per_origin": coverage.get("per_origin"),
            "group_specific_only_origin": GROUP_SPECIFIC_ONLY_ORIGIN,
            "validity_groups": {
                g: {
                    "n_skus_mnih_eligible_ge3_origins": (
                        coverage["validity_group_coverage"][g].get(
                            "n_skus_mnih_eligible_ge3_origins"
                        )
                    ),
                    "n_140401_pairs_in_group": coverage["validity_group_coverage"][g].get(
                        "n_140401_pairs_in_group"
                    ),
                }
                for g, _hs in VALIDITY_GROUPS
            },
        },
        "contract_freeze_readiness": {
            "ready_to_freeze": ready,
            "blockers": freeze_blockers,
            "stop_before_final_contract": True,
            "proposed_final_origins_if_ready": list(PROPOSED_FINAL_FULL_ORIGINS)
            if ready
            else None,
            "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        },
        "coverage_detail_embedded": coverage,
    }


def write_140404_readiness_artifacts(*, repo_root: Path) -> dict[str, Any]:
    """Write reconciliation + readiness audits; refresh candidate manifest."""
    repo_root = Path(repo_root).resolve()
    recon = build_machine_backfill_reconciliation(repo_root=repo_root)
    readiness = build_readiness_audit(repo_root=repo_root)

    out_dir = repo_root / FAMILY_DIR_REL
    recon_path = out_dir / HB_140404_MACHINE_RECONCILIATION_FILENAME
    ready_path = out_dir / HB_140404_READINESS_AUDIT_FILENAME
    _atomic_json(recon_path, recon)
    _atomic_json(ready_path, readiness)

    # Refresh candidate manifest (additive; no final contract)
    gate_hb = readiness["gates"]["hb_forecast_coverage"]
    gate_act = readiness["gates"]["actuals_mature"]
    gate_mach = readiness["gates"]["machine_ready"]
    freeze = readiness["contract_freeze_readiness"]
    manifest = {
        "manifest_version": "hb_reliability_candidate_manifest_v1",
        "created_at": _utc(),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "gates": {
            "hb_forecast_coverage": {
                "status": gate_hb.get("status"),
                "passed": gate_hb.get("passed"),
            },
            "actuals_mature": {
                "status": gate_act.get("status"),
                "passed": gate_act.get("passed"),
            },
            "machine_ready": gate_mach,
            "machine_backfill_reconciliation": {
                "passed": recon.get("passed"),
                "blockers": recon.get("blockers"),
            },
        },
        "140404_hb_forecast_coverage": gate_hb.get("status"),
        "excluded_from_contract_freeze": not freeze.get("ready_to_freeze"),
        "stop_before_final_contract": True,
        "ready_to_freeze": freeze.get("ready_to_freeze"),
        "freeze_blockers": freeze.get("blockers"),
        "proposed_final_origins_if_all_gates_pass": list(PROPOSED_FINAL_FULL_ORIGINS),
        "group_specific_evidence_only": [GROUP_SPECIFIC_ONLY_ORIGIN],
        "artifacts": {
            "machine_reconciliation": (
                FAMILY_DIR_REL / HB_140404_MACHINE_RECONCILIATION_FILENAME
            ).as_posix(),
            "readiness_audit": (
                FAMILY_DIR_REL / HB_140404_READINESS_AUDIT_FILENAME
            ).as_posix(),
            "forecast_coverage_audit": (
                FAMILY_DIR_REL / HB_FORECAST_COVERAGE_AUDIT_FILENAME
            ).as_posix(),
        },
        "note": (
            "Additive readiness after 140404 V2.1/A0 backfill. "
            "No HB quantities, outcome scoring, or final contract write."
        ),
        "current_PROTECTED_HB_ORIGINS_unchanged": sorted(PROTECTED_HB_ORIGINS),
    }
    man_path = out_dir / HB_RELIABILITY_CANDIDATE_MANIFEST_FILENAME
    _atomic_json(man_path, manifest)

    return {
        "ok": True,
        "reconciliation_path": recon_path.as_posix(),
        "reconciliation_sha256": file_sha256(recon_path),
        "readiness_path": ready_path.as_posix(),
        "readiness_sha256": file_sha256(ready_path),
        "candidate_manifest_path": man_path.as_posix(),
        "ready_to_freeze": freeze.get("ready_to_freeze"),
        "freeze_blockers": freeze.get("blockers"),
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
    }
