"""Audit imported SUCCESS jobs with effective prepare_monthly_series hashes.

Compares inventory/sparse hashes to the zero-filled training series through
origin-2. Demotes mismatches from SUCCESS to PENDING in v2 full run trees.
Does not execute forecasting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_SRC = REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.backfill_runner.state import (  # noqa: E402
    JobIdentity,
    JobStateStore,
    resolve_git_commit,
)
from pkg.benchmark.calendar import shamsi_add_months  # noqa: E402
from pkg.forecast_jobs.plan import decode_product_key  # noqa: E402
from pkg.ts_v2.data import prepare_monthly_series  # noqa: E402

SALES = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "sales_full_portfolio_v1"
    / "sales.parquet"
)
OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "product_universe"
    / "portfolio_backfill_prep_v2"
)
FORECAST_JOBS = REPO / "src" / "data" / "forecast_jobs"
PRIOR_AUDIT = OUT / "reuse_hash_audit.csv"
RUNS = (
    ("portfolio_v21_backfill_v2", "v2.1", "configs/forecast_jobs/portfolio_v21_backfill_v2.allowlist.json"),
    ("portfolio_a0_backfill_v2", "a0", "configs/forecast_jobs/portfolio_a0_backfill_v2.allowlist.json"),
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def series_history_sha256(dates, values) -> str:
    h = hashlib.sha256()
    for d, v in zip(dates, values):
        if v != v:
            token = f"{int(d)}|\n"
        else:
            token = f"{int(d)}|{float(v):.10g}\n"
        h.update(token.encode("utf-8"))
    return h.hexdigest()


def sparse_history_hash(sales: pd.DataFrame, product: str, origin: int) -> str:
    lc = shamsi_add_months(int(origin), -2)
    g = sales.loc[
        (sales["product"] == product)
        & (pd.to_numeric(sales["date"], errors="coerce") <= lc)
    ].sort_values("date")
    return series_history_sha256(
        [int(x) for x in g["date"].tolist()],
        [float(x) if pd.notna(x) else float("nan") for x in g["sales"].tolist()],
    )


def effective_history_hash(sales: pd.DataFrame, product: str, origin: int) -> tuple[str, int, int]:
    """Hash of prepare_monthly_series values through training_end (= origin-2)."""
    prepared = prepare_monthly_series(sales, product, int(origin))
    series = prepared.values
    training_end = shamsi_add_months(int(origin), -2)
    if series.empty:
        return "", training_end, 0
    max_date = int(series.index.max())
    if max_date != int(training_end):
        raise RuntimeError(
            f"effective series max {max_date} != training_end {training_end} "
            f"for {product}@{origin}"
        )
    h = series_history_sha256(
        [int(x) for x in series.index.tolist()],
        [float(x) if pd.notna(x) else float("nan") for x in series.tolist()],
    )
    return h, training_end, int(len(series))


def _load_prior_inventory_hashes() -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    if not PRIOR_AUDIT.is_file():
        return out
    df = pd.read_csv(PRIOR_AUDIT)
    for r in df.itertuples(index=False):
        seed_raw = getattr(r, "seed", "")
        seed = (
            None
            if seed_raw == "" or (isinstance(seed_raw, float) and pd.isna(seed_raw))
            else int(seed_raw)
        )
        key = (
            str(r.product),
            int(r.forecast_origin),
            str(r.engine).lower(),
            seed,
        )
        out[key] = str(getattr(r, "inventory_hash", "") or "")
    return out


def _list_success_jobs(run_dir: Path) -> list[sqlite3.Row]:
    db = run_dir / "state.sqlite"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT job_id, experiment_id, engine_version, config_hash, quarter,
               forecast_origin, product_id, status, output_path
        FROM jobs WHERE status = 'SUCCESS'
        """
    ).fetchall()
    conn.close()
    return list(rows)


def _append_allowlist(path: Path, jobs: list[dict[str, Any]]) -> int:
    """Ensure demoted keys are on the missing allowlist for a later full run."""
    if not jobs:
        return 0
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"jobs": []}
    existing = payload.get("jobs") or []
    seen: set[tuple] = set()
    for j in existing:
        seed = j.get("seed")
        seed_k = None if seed is None or seed == "" else int(seed)
        seen.add((str(j["product"]), int(j["forecast_origin"]), seed_k))
    added = 0
    for j in jobs:
        seed = j.get("seed")
        seed_k = None if seed is None or seed == "" else int(seed)
        key = (str(j["product"]), int(j["forecast_origin"]), seed_k)
        if key in seen:
            continue
        existing.append(
            {
                "product": str(j["product"]),
                "forecast_origin": int(j["forecast_origin"]),
                "seed": seed_k,
            }
        )
        seen.add(key)
        added += 1
    payload["jobs"] = existing
    payload["n_jobs"] = len(existing)
    payload["note"] = (
        payload.get("note")
        or "Missing allowlist; includes effective-hash demotions from reuse audit v2."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return added


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demote",
        action="store_true",
        help="Demote reuse_ok_effective=false SUCCESS jobs to PENDING",
    )
    p.add_argument("--dry-run-demote", action="store_true")
    args = p.parse_args()

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str).str.strip()
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")
    sales = sales.loc[sales["date"] <= 140504].copy()

    inv_map = _load_prior_inventory_hashes()
    cache: dict[tuple[str, int], tuple[str, int, int]] = {}
    sparse_cache: dict[tuple[str, int], str] = {}
    rows: list[dict[str, Any]] = []
    git = resolve_git_commit()

    demoted = 0
    allowlist_adds: dict[str, list[dict[str, Any]]] = {}
    for run_id, engine_filter, allowlist_rel in RUNS:
        run_dir = FORECAST_JOBS / run_id
        if not (run_dir / "state.sqlite").is_file():
            raise SystemExit(f"missing run tree: {run_dir}")
        state = JobStateStore(run_dir)
        demote_jobs: list[dict[str, Any]] = []
        for row in _list_success_jobs(run_dir):
            product, seed = decode_product_key(str(row["product_id"]))
            origin = int(row["forecast_origin"])
            eng = str(row["engine_version"]).lower()
            if eng != engine_filter:
                continue
            ck = (str(product), origin)
            if ck not in cache:
                cache[ck] = effective_history_hash(sales, str(product), origin)
            if ck not in sparse_cache:
                sparse_cache[ck] = sparse_history_hash(sales, str(product), origin)
            eff_h, training_end, n_obs = cache[ck]
            sparse_h = sparse_cache[ck]
            inv_h = inv_map.get((str(product), origin, eng, seed), "")
            # Effective reuse: sparse-keyed inventory must equal prepared series hash
            reuse_ok = bool(eff_h) and sparse_h == eff_h
            sparse_eq_effective = sparse_h == eff_h
            rec = {
                "run_id": run_id,
                "job_id": str(row["job_id"]),
                "product": str(product),
                "forecast_origin": origin,
                "engine": eng,
                "seed": "" if seed is None else int(seed),
                "training_end": training_end,
                "n_effective_months": n_obs,
                "inventory_hash": inv_h,
                "sparse_hash": sparse_h,
                "effective_hash": eff_h,
                "sparse_eq_effective": sparse_eq_effective,
                "reuse_ok_effective": reuse_ok,
                "output_path": row["output_path"],
            }
            rows.append(rec)

            if (args.demote or args.dry_run_demote) and not reuse_ok:
                demoted += 1
                demote_jobs.append(
                    {
                        "product": str(product),
                        "forecast_origin": origin,
                        "seed": None if seed is None else int(seed),
                    }
                )
                if args.demote and not args.dry_run_demote:
                    identity = JobIdentity(
                        experiment_id=str(row["experiment_id"]),
                        engine_version=str(row["engine_version"]),
                        config_hash=str(row["config_hash"]),
                        quarter=str(row["quarter"]),
                        forecast_origin=origin,
                        product_id=str(row["product_id"]),
                    )
                    state.reset_for_force(identity, git_commit=git)
        allowlist_adds[allowlist_rel] = demote_jobs

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "reuse_effective_hash_audit_v2.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")

    by_engine: dict[str, Any] = {}
    for eng, g in frame.groupby("engine"):
        by_engine[str(eng)] = {
            "n_success_audited": int(len(g)),
            "n_ok": int(g["reuse_ok_effective"].sum()),
            "n_mismatch": int((~g["reuse_ok_effective"]).sum()),
            "n_sparse_ne_effective": int((~g["sparse_eq_effective"]).sum()),
        }

    allowlist_added = 0
    if args.demote and not args.dry_run_demote:
        for rel, jobs in allowlist_adds.items():
            allowlist_added += _append_allowlist(REPO / rel, jobs)

    # Post-demote status counts
    status_after: dict[str, Any] = {}
    for run_id, _eng, _al in RUNS:
        store = JobStateStore(FORECAST_JOBS / run_id)
        status_after[run_id] = store.status_counts(run_id)

    summary = {
        "extracted_at_utc": _utc(),
        "n_total": int(len(frame)),
        "n_ok": int(frame["reuse_ok_effective"].sum()) if len(frame) else 0,
        "n_mismatch": int((~frame["reuse_ok_effective"]).sum()) if len(frame) else 0,
        "n_sparse_ne_effective": int((~frame["sparse_eq_effective"]).sum())
        if len(frame)
        else 0,
        "by_engine": by_engine,
        "demoted": demoted if (args.demote or args.dry_run_demote) else 0,
        "demote_applied": bool(args.demote and not args.dry_run_demote),
        "allowlist_jobs_added": allowlist_added,
        "status_after": status_after,
        "csv": str(csv_path),
    }
    (OUT / "reuse_effective_hash_audit_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())