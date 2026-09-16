"""Prepare portfolio v2 allowlist: mark non-allowlist jobs SKIPPED (not SUCCESS)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pkg.benchmark.backfill_runner.state import JobStateStore, resolve_git_commit
from pkg.forecast_jobs.config import load_job_config
from pkg.forecast_jobs.plan import build_job_plan
from pkg.forecast_jobs.runner import prepare_run


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, required=True)
    args = p.parse_args()

    payload = json.loads(Path(args.allowlist).read_text(encoding="utf-8"))
    allow: set[tuple] = set()
    for j in payload.get("jobs") or []:
        product = str(j["product"])
        origin = int(j["forecast_origin"])
        seed = j.get("seed")
        if seed is None or seed == "":
            allow.add((product, origin, None))
        else:
            allow.add((product, origin, int(seed)))
    if not allow:
        raise SystemExit("empty allow-list")

    config = load_job_config(args.config)
    root, cfg_hash, _plan, _sales, _snap = prepare_run(config, dry_run=False)
    state = JobStateStore(root)
    state.reclaim_stale_running(config.run_id)
    git = resolve_git_commit()
    plan = build_job_plan(
        config, config_hash=cfg_hash, state=state, git_commit=git, ensure_jobs=True
    )

    now = _utc()
    skipped = 0
    keep = 0
    already_terminal = 0
    for job in plan.jobs:
        key = (str(job.product), int(job.origin), job.seed)
        key_alt = (str(job.product), int(job.origin), None)
        if key in allow or key_alt in allow:
            keep += 1
            continue
        rec = state.ensure_job(job.identity, git_commit=git)
        if rec.status in {"SUCCESS", "INELIGIBLE", "SKIPPED"}:
            already_terminal += 1
            continue
        state.mark_skipped(
            job.identity,
            reason_code="not_in_allowlist",
            reason_detail=None,
            started_at=now,
            finished_at=now,
            git_commit=git,
        )
        skipped += 1

    counts = state.status_counts(config.run_id)
    summary = {
        "run_id": config.run_id,
        "allowlist_n": len(allow),
        "keep_pending_candidates": keep,
        "marked_skipped": skipped,
        "already_terminal": already_terminal,
        "status_counts": counts,
    }
    print(json.dumps(summary, indent=2))
    out = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "results"
        / "product_universe"
        / "portfolio_backfill_prep_v2"
        / f"allowlist_prepare_{config.run_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
