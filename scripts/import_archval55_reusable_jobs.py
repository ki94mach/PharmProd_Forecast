#!/usr/bin/env python3
"""Import reusable Archval25 forecasts into the Archval55 run tree.

After the reuse gate passes:
  - SUCCESS jobs → copy forecasts from archval25 panel into new job dirs + mark SUCCESS
  - Expected-ineligible → import naive_last_value from fallback_v1 panel (or rebuild)
  - Leaves required jobs PENDING for VPS execute

Does not mutate archval25_* trees. Writes a partial merged panel under
``archval55_export/`` for inspection (full panel after VPS completes).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
_SRC = _REPO / "src"
for p in (_SCRIPTS, str(_SRC)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from archval55_reuse import (  # noqa: E402
    ARCHVAL25_EXPORT,
    ARCHVAL25_FALLBACK_EXPORT,
    FALLBACK_POLICY,
    FALLBACK_VERSION,
    LOCK_DIR,
    RESULTS,
    build_inventory,
    check_reuse_gate,
)
from pkg.benchmark.backfill_runner.state import resolve_git_commit  # noqa: E402
from pkg.forecast_jobs.config import load_job_config  # noqa: E402
from pkg.forecast_jobs.cutoff import apply_cutoff_policy  # noqa: E402
from pkg.forecast_jobs.engines import (  # noqa: E402
    FALLBACK_MODEL,
    REASON_NEURAL_SHORT_HISTORY,
    REASON_V21_NO_ELIGIBLE_MODELS,
    build_naive_last_value_forecasts,
    last_training_value,
    target_dates_for_origin,
)
from pkg.benchmark.backfill_runner.types import EngineJobRequest  # noqa: E402
from pkg.forecast_jobs.plan import PlannedJob, encode_product_key  # noqa: E402
from pkg.forecast_jobs.runner import prepare_run  # noqa: E402
from pkg.forecast_jobs.store import (  # noqa: E402
    atomic_write_json,
    job_artifact_dir,
    run_dir,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_panel(path: Path) -> pd.DataFrame:
    pq = path / "forecasts_panel.parquet"
    csv = path / "forecasts_panel.csv"
    if pq.is_file():
        df = pd.read_parquet(pq)
    elif csv.is_file():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"no forecasts_panel under {path}")
    df = df.copy()
    df["product"] = df["product"].astype(str)
    df["engine"] = df["engine"].astype(str).str.lower()
    for c in ("forecast_origin", "target_date", "horizon"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


def _slice_job_forecasts(
    panel: pd.DataFrame,
    *,
    product: str,
    origin: int,
    engine: str,
    seed: Optional[int],
) -> pd.DataFrame:
    m = (
        (panel["product"] == str(product))
        & (panel["forecast_origin"] == int(origin))
        & (panel["engine"] == str(engine).lower())
    )
    if seed is None:
        if "seed" in panel.columns:
            m = m & (panel["seed"].isna() | (panel["seed"] == 0))
            # V2.1 rows often have NaN seed; also accept any if only one group
            sub = panel.loc[m]
            if sub.empty:
                sub = panel.loc[
                    (panel["product"] == str(product))
                    & (panel["forecast_origin"] == int(origin))
                    & (panel["engine"] == str(engine).lower())
                ]
        else:
            sub = panel.loc[m]
    else:
        sub = panel.loc[m & (panel["seed"] == int(seed))]
    return sub.copy()


def _planned_for_key(plan_jobs: list[PlannedJob], row: dict[str, Any]) -> Optional[PlannedJob]:
    seed = row.get("seed")
    seed_i = None if seed in ("", None) or (isinstance(seed, float) and pd.isna(seed)) else int(seed)
    for job in plan_jobs:
        if (
            str(job.product) == str(row["product"])
            and int(job.origin) == int(row["forecast_origin"])
            and str(job.architecture).lower() == str(row["engine"]).lower()
            and job.seed == seed_i
        ):
            return job
    return None


def _write_imported_job(
    *,
    root: Path,
    job: PlannedJob,
    forecasts: pd.DataFrame,
    status: str,
    selected_model: Optional[str],
    error_type: Optional[str],
    error_message: Optional[str],
    extras: dict[str, Any],
    dry_run: bool,
) -> Path:
    out_dir = job_artifact_dir(root, job.identity)
    if dry_run:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job.identity.job_id,
        "run_id": job.identity.experiment_id,
        "architecture": job.architecture,
        "config_hash": job.identity.config_hash,
        "quarter": job.quarter,
        "forecast_origin": job.origin,
        "product": job.product,
        "seed": job.seed,
        "success": status in {"SUCCESS", "INELIGIBLE"},
        "ineligible": status == "INELIGIBLE",
        "selected_model": selected_model,
        "error_type": error_type,
        "error_message": error_message,
        "imported_from": "archval25",
        "extras": extras,
    }
    atomic_write_json(out_dir / "job.json", payload)
    forecasts.to_csv(out_dir / "forecast.csv", index=False)
    if status == "INELIGIBLE":
        atomic_write_json(
            out_dir / "fallback.json",
            {
                "ineligible": True,
                "error_type": error_type,
                "error_message": error_message,
                "selected_model": selected_model,
                "extras": extras,
            },
        )
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1.win.yaml",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume run dir already prepared (state.sqlite exists)",
    )
    args = p.parse_args()

    gate = check_reuse_gate()
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (LOCK_DIR / "gate_report.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    if not gate.get("ok"):
        print("reuse gate FAILED — refusing import (would refit all 1540)")
        for name, chk in (gate.get("checks") or {}).items():
            print(f"  {name}: ok={chk.get('ok')} detail={chk}")
        return 2

    config = load_job_config(args.config)
    if args.skip_prepare:
        root = run_dir(config.output_dir, config.run_id)
        from pkg.forecast_jobs.snapshot import validate_sales_config
        from pkg.forecast_jobs.runner import scientific_config_hash
        from pkg.forecast_jobs.plan import build_job_plan
        from pkg.benchmark.backfill_runner.state import JobStateStore

        snapshot = validate_sales_config(config.sales)
        cfg_hash = scientific_config_hash(config, snapshot.content_sha256)
        state = JobStateStore(root)
        plan = build_job_plan(
            config, config_hash=cfg_hash, state=state, ensure_jobs=True
        )
        sales_df = pd.read_parquet(config.sales.path)
    else:
        root, cfg_hash, plan, sales_df, _snap = prepare_run(
            config, dry_run=False
        )
        from pkg.benchmark.backfill_runner.state import JobStateStore

        state = JobStateStore(root)

    sales = sales_df.copy()
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    reusable, ineligible, required, all_jobs = build_inventory(gate=gate, sales=sales)
    reusable.to_csv(LOCK_DIR / "reusable_jobs.csv", index=False)
    ineligible.to_csv(LOCK_DIR / "ineligible_fill_jobs.csv", index=False)
    required.to_csv(LOCK_DIR / "required_jobs.csv", index=False)

    panel25 = _load_panel(ARCHVAL25_EXPORT)
    fb_panel = None
    if ARCHVAL25_FALLBACK_EXPORT.is_dir():
        try:
            fb_panel = _load_panel(ARCHVAL25_FALLBACK_EXPORT)
        except FileNotFoundError:
            fb_panel = None

    git_commit = resolve_git_commit()
    now = _utc_stamp()
    imported_success = 0
    imported_ineligible = 0
    skipped = 0
    errors: list[str] = []
    imported_frames: list[pd.DataFrame] = []

    # Map plan jobs for identity under archval55 config_hash
    plan_by_key = {
        (
            j.product,
            j.origin,
            str(j.architecture).lower(),
            j.seed,
        ): j
        for j in plan.jobs
    }
    # Also include already-completed (not in plan.jobs when resume skips)
    from pkg.forecast_jobs.plan import build_job_plan as _bjp

    full_plan = _bjp(config, config_hash=cfg_hash, state=None, ensure_jobs=False)
    for j in full_plan.jobs:
        plan_by_key[(j.product, j.origin, str(j.architecture).lower(), j.seed)] = j

    def _seed_key(seed_val: Any) -> Optional[int]:
        if seed_val in ("", None):
            return None
        if isinstance(seed_val, float) and pd.isna(seed_val):
            return None
        return int(seed_val)

    for _, row in reusable.iterrows():
        seed = _seed_key(row["seed"])
        job = plan_by_key.get(
            (str(row["product"]), int(row["forecast_origin"]), str(row["engine"]), seed)
        )
        if job is None:
            errors.append(f"no planned job for reusable {dict(row)}")
            skipped += 1
            continue
        state.ensure_job(job.identity, git_commit=git_commit)
        fc = _slice_job_forecasts(
            panel25,
            product=str(row["product"]),
            origin=int(row["forecast_origin"]),
            engine=str(row["engine"]),
            seed=seed,
        )
        if fc.empty:
            errors.append(f"empty SUCCESS slice {job.identity.slug}")
            skipped += 1
            continue
        model = (
            str(fc["model"].iloc[0])
            if "model" in fc.columns and pd.notna(fc["model"].iloc[0])
            else (row.get("source_selected_model") or "imported")
        )
        out_dir = _write_imported_job(
            root=root,
            job=job,
            forecasts=fc,
            status="SUCCESS",
            selected_model=model,
            error_type=None,
            error_message=None,
            extras={"import_source": "archval25_export", "bucket": "reusable"},
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            state.mark_success(
                job.identity,
                output_path=str(out_dir),
                selected_model=model,
                started_at=now,
                finished_at=now,
                runtime_seconds=0.0,
                git_commit=git_commit,
            )
        imported_success += 1
        imported_frames.append(fc.assign(import_bucket="reusable"))

    for _, row in ineligible.iterrows():
        seed = _seed_key(row["seed"])
        engine = str(row["engine"]).lower()
        product = str(row["product"])
        origin = int(row["forecast_origin"])
        job = plan_by_key.get((product, origin, engine, seed))
        if job is None:
            errors.append(f"no planned job for ineligible {product} {origin} {engine}")
            skipped += 1
            continue
        state.ensure_job(job.identity, git_commit=git_commit)

        fc = pd.DataFrame()
        if fb_panel is not None:
            fc = _slice_job_forecasts(
                fb_panel, product=product, origin=origin, engine=engine, seed=seed
            )
            if "ineligible" in fc.columns:
                fc = fc.loc[fc["ineligible"].fillna(False).astype(bool) | (fc["model"] == FALLBACK_MODEL)]
            elif "model" in fc.columns:
                fc = fc.loc[fc["model"].astype(str) == FALLBACK_MODEL]

        if fc.empty:
            # Rebuild naive_last_value from sales (same semantics as fallback_v1).
            cut = apply_cutoff_policy(sales, origin, policy_name="production")
            product_sales = cut.loc[cut["product"].astype(str) == product].copy()
            last = last_training_value(product_sales, product)
            if last is None:
                errors.append(f"fallback unavailable {product}@{origin}")
                skipped += 1
                continue
            request = EngineJobRequest(
                engine=engine,
                product=product,
                quarter=job.quarter,
                forecast_origin=origin,
                horizon=int(config.horizon),
                target_dates=target_dates_for_origin(origin, int(config.horizon)),
                training_sales=product_sales,
                meta={
                    "fallback_policy": FALLBACK_POLICY,
                    "fallback_version": FALLBACK_VERSION,
                },
            )
            fc = build_naive_last_value_forecasts(
                request,
                engine=engine,
                last_value=float(last),
                seed=seed,
                architecture=engine if engine in {"a0", "a1"} else None,
            )
            reason = (
                REASON_V21_NO_ELIGIBLE_MODELS
                if engine == "v2.1"
                else REASON_NEURAL_SHORT_HISTORY
            )
            detail = str(row.get("source_error_message") or reason)
            extras = {
                "import_source": "rebuilt_naive_last_value",
                "fallback_reason": f"{reason}|policy={FALLBACK_POLICY}|version={FALLBACK_VERSION}",
                "fallback_last_value": float(last),
                "ineligibility_reason": reason,
            }
        else:
            reason = REASON_V21_NO_ELIGIBLE_MODELS if engine == "v2.1" else REASON_NEURAL_SHORT_HISTORY
            detail = str(row.get("source_error_message") or reason)
            extras = {
                "import_source": "archval25_export_fallback_v1",
                "fallback_reason": f"{reason}|policy={FALLBACK_POLICY}|version={FALLBACK_VERSION}",
                "ineligibility_reason": reason,
            }

        out_dir = _write_imported_job(
            root=root,
            job=job,
            forecasts=fc,
            status="INELIGIBLE",
            selected_model=FALLBACK_MODEL,
            error_type="ExpectedIneligible",
            error_message=detail,
            extras=extras,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            state.mark_ineligible(
                job.identity,
                output_path=str(out_dir),
                selected_model=FALLBACK_MODEL,
                error_type="ExpectedIneligible",
                error_message=detail,
                started_at=now,
                finished_at=now,
                runtime_seconds=0.0,
                git_commit=git_commit,
            )
        imported_ineligible += 1
        imported_frames.append(fc.assign(import_bucket="ineligible_fill"))

    export_out = RESULTS / "archval55_export"
    summary = {
        "dry_run": bool(args.dry_run),
        "run_id": config.run_id,
        "run_dir": str(root.as_posix()),
        "config_hash": cfg_hash,
        "gate_ok": True,
        "imported_success": imported_success,
        "imported_ineligible": imported_ineligible,
        "required_remaining": int(len(required)),
        "skipped": skipped,
        "errors": errors[:50],
        "n_errors": len(errors),
    }
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (LOCK_DIR / "import_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if not args.dry_run and imported_frames:
        export_out.mkdir(parents=True, exist_ok=True)
        merged = pd.concat(imported_frames, ignore_index=True)
        merged.to_parquet(export_out / "forecasts_panel_imported.parquet", index=False)
        merged.to_csv(export_out / "forecasts_panel_imported.csv", index=False)
        # Copy gate + inventory pointers; do not overwrite a full panel if present.
        shutil.copy2(LOCK_DIR / "gate_report.json", export_out / "gate_report.json")
        (export_out / "README.md").write_text(
            "# Archval55 export (partial until VPS completes)\n\n"
            f"- Imported SUCCESS: {imported_success}\n"
            f"- Imported INELIGIBLE fills: {imported_ineligible}\n"
            f"- Required remaining: {len(required)}\n"
            "- Full `forecasts_panel.parquet` is written after the VPS run finishes.\n"
            "- Do not confuse with `archval25_export` (frozen).\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print(f"required_jobs={len(required)} (execute on VPS with --resume)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
