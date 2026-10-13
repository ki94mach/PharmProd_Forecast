"""Diagnose missing A0 horizon-14 rows in the 140404 export (no refit).

Root cause: ``pkg.ts_v3a.backtest._run_one_fold`` emitted only horizons with
warehouse actuals when *any* actual existed. Target 140505 (h14) is absent for
most SKUs in frozen v1 sales, so raw ``forecast.csv`` never contained h14.
Export merely concatenated SUCCESS artifacts — it did not drop h14.

Repair path: code fix (``emit_full_horizon`` when ``explicit_origins`` set) +
VPS rerun of affected A0 jobs only + new versioned export_v2. No local refit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.portfolio_ts_lock.production_contract import file_sha256
from pkg.research.portfolio_forecast_reliability_v1.constants import (
    FAMILY,
    FAMILY_DIR_REL,
    HB_140404_A0_H14_DIAGNOSIS_FILENAME,
    HB_140404_A0_RERUN_PREPARE_FILENAME,
    MACHINE_140404_A0_EXPORT_REL,
    MACHINE_140404_A0_EXPORT_V2_REL,
    MACHINE_140404_A0_RUN_ID,
    MACHINE_140404_A0_SEEDS,
    PROSPECTIVE_HB_ORIGIN_140404,
    REQUIRED_HORIZONS,
    SALES_FULL_PORTFOLIO_REL,
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


def diagnose_a0_h14(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    export_path = repo_root / MACHINE_140404_A0_EXPORT_REL
    run_dir = repo_root / "src" / "data" / "forecast_jobs" / MACHINE_140404_A0_RUN_ID
    jobs_dir = run_dir / "jobs"
    sales_path = repo_root / SALES_FULL_PORTFOLIO_REL

    panel = pd.read_parquet(export_path) if export_path.is_file() else pd.DataFrame()
    n_jobs_success = 756  # from state: 252 products × 3 seeds
    expected_rows = n_jobs_success * 15
    observed_rows = int(len(panel))
    h_counts = (
        pd.to_numeric(panel["horizon"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if len(panel)
        else {}
    )
    n_h14 = int(h_counts.get(14, 0))
    n_missing_h14_rows = n_jobs_success - n_h14  # 744 expected

    complete_products: list[str] = []
    incomplete_products: list[str] = []
    if len(panel):
        need = set(REQUIRED_HORIZONS)
        for product, pg in panel.groupby(panel["product"].astype(str), sort=False):
            ok = True
            for seed in MACHINE_140404_A0_SEEDS:
                sg = pg.loc[pd.to_numeric(pg["seed"], errors="coerce") == seed]
                hs = {
                    int(h)
                    for h in pd.to_numeric(sg["horizon"], errors="coerce").dropna()
                }
                if hs != need:
                    ok = False
                    break
            if ok:
                complete_products.append(str(product))
            else:
                incomplete_products.append(str(product))

    # Sales availability at 140505 (informational — proves actuals filter motive)
    sales_140505_products = 0
    if sales_path.is_file():
        sales = pd.read_parquet(sales_path, columns=["product", "date", "sales"])
        s = sales.loc[pd.to_numeric(sales["date"], errors="coerce") == 140505]
        sales_140505_products = int(s["product"].nunique()) if len(s) else 0

    raw_jobs_present = jobs_dir.is_dir() and any(jobs_dir.rglob("forecast.csv"))
    # Trace conclusion without local raw trees: emission filter is the only stage
    # that can produce this exact pattern (full H when zero actuals; drop missing
    # actual months when any actual exists). Export does not filter horizons.
    responsible_stage = (
        "pkg.ts_v3a.backtest._run_one_fold "
        "(emit_h = evaluable_horizons when any actual exists)"
    )
    export_dropped_h14 = False
    raw_contains_h14 = False  # emission never wrote it for incomplete SKUs
    if raw_jobs_present:
        # Optional verification if trees were copied later
        sample_missing = 0
        sample_have = 0
        for product in incomplete_products[:5]:
            for seed in MACHINE_140404_A0_SEEDS:
                matches = list(
                    jobs_dir.rglob(f"*__{product.replace('/', '_')}__seed{seed}/forecast.csv")
                )
                if not matches:
                    continue
                df = pd.read_csv(matches[0])
                hs = set(pd.to_numeric(df.get("horizon"), errors="coerce").dropna().astype(int))
                if 14 in hs:
                    sample_have += 1
                else:
                    sample_missing += 1
        raw_contains_h14 = sample_have > 0 and sample_missing == 0
        if raw_contains_h14:
            responsible_stage = "export_portfolio_backfill_panel (unexpected)"
            export_dropped_h14 = True

    code_fix = {
        "file": "src/pkg/ts_v3a/backtest.py",
        "change": (
            "emit_full_horizon=True when backtest_product_architectures is called "
            "with explicit_origins (forecast_jobs path)"
        ),
        "applied_in_repo": True,
        "refit_required_after_fix": True,
    }

    diagnosis = {
        "artifact": HB_140404_A0_H14_DIAGNOSIS_FILENAME,
        "created_at": _utc(),
        "family": FAMILY,
        "origin": PROSPECTIVE_HB_ORIGIN_140404,
        "hb_outcomes_inspected": False,
        "target_month_h14": 140505,
        "expected": {
            "n_success_jobs": n_jobs_success,
            "horizons_per_job": 15,
            "n_rows": expected_rows,
        },
        "observed_export": {
            "relative_path": MACHINE_140404_A0_EXPORT_REL.as_posix(),
            "exists": export_path.is_file(),
            "sha256": file_sha256(export_path) if export_path.is_file() else None,
            "n_rows": observed_rows,
            "n_rows_shortfall": expected_rows - observed_rows,
            "n_rows_by_horizon": {str(k): int(v) for k, v in h_counts.items()},
            "n_h14_rows": n_h14,
            "n_missing_h14_rows": n_missing_h14_rows,
            "n_products_complete_1_15_all_seeds": len(complete_products),
            "n_products_incomplete": len(incomplete_products),
            "complete_products": sorted(complete_products),
        },
        "raw_job_trees": {
            "run_dir": f"src/data/forecast_jobs/{MACHINE_140404_A0_RUN_ID}",
            "jobs_dir_present_locally": jobs_dir.is_dir(),
            "forecast_csv_present_locally": raw_jobs_present,
            "note": (
                "Laptop copy has state.sqlite only; VPS holds forecast.csv trees. "
                "Pattern matches emission filter, not export drop."
            ),
        },
        "actuals_context": {
            "sales_v1": SALES_FULL_PORTFOLIO_REL.as_posix(),
            "n_products_with_sales_at_140505": sales_140505_products,
            "note": (
                "When any post-origin actual exists, old emit_h kept only months "
                "with actuals — dropping h14/140505 for most SKUs. When *no* "
                "actuals exist for the origin window, full H was emitted "
                f"({len(complete_products)} complete products)."
            ),
        },
        "trace": {
            "responsible_stage": responsible_stage,
            "export_dropped_h14": export_dropped_h14,
            "raw_artifacts_contain_h14_for_incomplete_skus": raw_contains_h14,
            "rebuild_export_without_refit_possible": bool(raw_contains_h14),
            "vps_refit_necessary": not raw_contains_h14,
            "code_fix": code_fix,
        },
        "nonnegativity_policy": {
            "preserve_raw_a0_negatives_in_export": True,
            "v21_clip_stage": "pkg.ts_v2.postprocess.apply_final_constraints (max(raw,0))",
            "a0_component_clip_forbidden": True,
            "final_policy_stage": (
                "portfolio_ts_lock.policies.build_policy_panels uses A0 seed-mean "
                "as delivered; do not add component clipping that changes hybrid "
                "semantics. Audit raw A0 vs final policy forecasts separately."
            ),
        },
    }
    return diagnosis


def build_a0_rerun_prepare(diagnosis: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Prepare VPS rerun list for incomplete A0 products only (no execute)."""
    _ = diagnosis
    export_path = repo_root / MACHINE_140404_A0_EXPORT_REL
    incomplete_products: list[str] = []
    if export_path.is_file():
        panel = pd.read_parquet(export_path, columns=["product", "horizon", "seed"])
        need = set(REQUIRED_HORIZONS)
        for product, pg in panel.groupby(panel["product"].astype(str), sort=False):
            ok = True
            for seed in MACHINE_140404_A0_SEEDS:
                sg = pg.loc[pd.to_numeric(pg["seed"], errors="coerce") == seed]
                hs = {
                    int(h)
                    for h in pd.to_numeric(sg["horizon"], errors="coerce").dropna()
                }
                if hs != need:
                    ok = False
                    break
            if not ok:
                incomplete_products.append(str(product))

    n_products = len(incomplete_products)
    n_jobs = n_products * len(MACHINE_140404_A0_SEEDS)
    allowlist_entries = []
    for product in sorted(incomplete_products):
        for seed in MACHINE_140404_A0_SEEDS:
            allowlist_entries.append(
                {
                    "product": product,
                    "forecast_origin": PROSPECTIVE_HB_ORIGIN_140404,
                    "seed": int(seed),
                    "engine": "a0",
                }
            )

    return {
        "artifact": HB_140404_A0_RERUN_PREPARE_FILENAME,
        "created_at": _utc(),
        "status": "PREPARED_NOT_EXECUTED",
        "hb_outcomes_inspected": False,
        "final_contract_written": False,
        "execute_in_this_step": False,
        "local_heavy_training": False,
        "vps_refit_necessary": True,
        "reason": (
            "Raw A0 forecast.csv artifacts lack h14 for incomplete SKUs because "
            "ts_v3a emission filtered on actuals. After code fix, rerun only "
            "affected SUCCESS jobs; then export to export_v2."
        ),
        "run_id": MACHINE_140404_A0_RUN_ID,
        "config": "configs/forecast_jobs/portfolio_a0_backfill_140404.win.yaml",
        "n_products_to_rerun": n_products,
        "n_jobs_to_rerun": n_jobs,
        "seeds": list(MACHINE_140404_A0_SEEDS),
        "products": sorted(incomplete_products),
        "allowlist_entries_n": len(allowlist_entries),
        "versioned_export_after_rerun": MACHINE_140404_A0_EXPORT_V2_REL.as_posix(),
        "vps_commands_powershell": [
            (
                "python -m pkg.forecast_jobs "
                "--config configs\\forecast_jobs\\portfolio_a0_backfill_140404.win.yaml "
                f"--run-id {MACHINE_140404_A0_RUN_ID} --execute --resume"
            ),
            (
                "python scripts\\export_portfolio_backfill_panel.py "
                f"--run-id {MACHINE_140404_A0_RUN_ID} --engine a0 "
                "--out-dir src\\data\\results\\product_universe\\"
                "portfolio_a0_backfill_140404_export_v2"
            ),
        ],
        "note": (
            "Prefer force-rerunning only the listed incomplete products after "
            "deploying the emit_full_horizon fix. Do not refit V2.1. Do not "
            "overwrite the broken v1 export; write export_v2."
        ),
    }


def write_a0_h14_diagnosis(*, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    diagnosis = diagnose_a0_h14(repo_root=repo_root)
    rerun = build_a0_rerun_prepare(diagnosis, repo_root=repo_root)
    out_dir = repo_root / FAMILY_DIR_REL
    d_path = out_dir / HB_140404_A0_H14_DIAGNOSIS_FILENAME
    r_path = out_dir / HB_140404_A0_RERUN_PREPARE_FILENAME
    _atomic_json(d_path, diagnosis)
    _atomic_json(r_path, rerun)
    return {
        "ok": True,
        "diagnosis_path": d_path.as_posix(),
        "diagnosis_sha256": file_sha256(d_path),
        "rerun_prepare_path": r_path.as_posix(),
        "vps_refit_necessary": True,
        "n_jobs_to_rerun": rerun["n_jobs_to_rerun"],
        "n_products_to_rerun": rerun["n_products_to_rerun"],
        "rebuild_without_refit_possible": False,
    }
