#!/usr/bin/env python3
"""Prepare archval55_a0_empty_fix_v1 so only the 90 allow-list jobs run.

Marks non-allow-list Cartesian A0 jobs as SUCCESS (allowlist_skip) and reclaims
stale RUNNING rows. Then ``python -m pkg.forecast_jobs --execute --resume`` only
fits the remaining allow-list PENDING jobs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pkg.benchmark.backfill_runner.state import JobIdentity, JobStateStore, resolve_git_commit
from pkg.benchmark.calendar import quarter_from_origin
from pkg.forecast_jobs.config import load_job_config
from pkg.forecast_jobs.plan import build_job_plan, encode_product_key
from pkg.forecast_jobs.runner import prepare_run

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "forecast_jobs" / "archval55_a0_empty_fix_v1.win.yaml"
ALLOWLIST = REPO / "configs" / "forecast_jobs" / "archval55_a0_empty_fix_v1.allowlist.json"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    args = p.parse_args()

    payload = json.loads(Path(args.allowlist).read_text(encoding="utf-8"))
    allow = {
        (str(j["product"]), int(j["forecast_origin"]), int(j["seed"]))
        for j in (payload.get("jobs") or [])
    }
    if not allow:
        raise SystemExit("empty allow-list")

    config = load_job_config(args.config)
    root, cfg_hash, plan, _sales, _snap = prepare_run(config, dry_run=False)
    state = JobStateStore(root)
    state.reclaim_stale_running(config.run_id)

    # Re-expand plan after reclaim.
    git = resolve_git_commit()
    plan = build_job_plan(
        config,
        config_hash=cfg_hash,
        state=state,
        git_commit=git,
        ensure_jobs=True,
    )

    now = _utc()
    skipped = 0
    keep = 0
    already = 0
    for job in plan.jobs:
        key = (str(job.product), int(job.origin), int(job.seed or -1))
        if key in allow:
            keep += 1
            continue
        rec = state.ensure_job(job.identity, git_commit=git)
        if rec.status == "SUCCESS":
            already += 1
            continue
        state.mark_success(
            job.identity,
            output_path="",
            selected_model="allowlist_skip",
            started_at=now,
            finished_at=now,
            runtime_seconds=0.0,
            git_commit=git,
        )
        skipped += 1

    # Ensure every allow-list key exists as a job row.
    for product, origin, seed in sorted(allow):
        identity = JobIdentity(
            experiment_id=config.run_id,
            engine_version="a0",
            config_hash=cfg_hash,
            quarter=quarter_from_origin(int(origin)),
            forecast_origin=int(origin),
            product_id=encode_product_key(product, seed),
        )
        state.ensure_job(identity, git_commit=git)

    counts = state.status_counts(config.run_id)
    print(
        json.dumps(
            {
                "run_root": str(root),
                "allow_list": len(allow),
                "marked_allowlist_skip": skipped,
                "already_success_non_allow": already,
                "allowlist_in_remaining_plan": keep,
                "status_counts": counts,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
