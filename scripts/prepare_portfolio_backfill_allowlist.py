"""Mark non-allowlist Cartesian jobs as allowlist_skip SUCCESS (no execute)."""
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
    root, cfg_hash, plan, _sales, _snap = prepare_run(config, dry_run=False)
    state = JobStateStore(root)
    state.reclaim_stale_running(config.run_id)
    git = resolve_git_commit()
    plan = build_job_plan(
        config, config_hash=cfg_hash, state=state, git_commit=git, ensure_jobs=True
    )

    now = _utc()
    skipped = 0
    keep = 0
    already = 0
    for job in plan.jobs:
        key = (str(job.product), int(job.origin), job.seed)
        # Also match seed None vs missing for v2.1
        key_alt = (str(job.product), int(job.origin), None)
        if key in allow or key_alt in allow:
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

    # Count PENDING after skip
    counts = state.status_counts(config.run_id)
    summary = {
        "run_id": config.run_id,
        "allowlist_n": len(allow),
        "keep_pending_candidates": keep,
        "marked_allowlist_skip": skipped,
        "already_success": already,
        "status_counts": counts,
    }
    print(json.dumps(summary, indent=2))
    out = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "data"
        / "results"
        / "product_universe"
        / "portfolio_backfill_prep_v1"
        / f"allowlist_prepare_{config.run_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
