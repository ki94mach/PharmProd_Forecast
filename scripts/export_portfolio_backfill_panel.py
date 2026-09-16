"""Export portfolio forecast_jobs panel (SUCCESS + valid artifacts only).

Ignores SKIPPED / PENDING / FAILED. Optionally excludes INELIGIBLE from the
native panel. Dedupes on product × origin × engine × seed. Writes seed-level
rows and an A0 seed-mean panel.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.forecast_jobs.plan import decode_product_key

REPO = Path(__file__).resolve().parents[1]
FORECAST_JOBS = REPO / "src" / "data" / "forecast_jobs"


def _load_statuses(run_dir: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    db = run_dir / "state.sqlite"
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not db.is_file():
        return out
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(
        """
        SELECT engine_version, forecast_origin, product_id, status,
               error_type, error_message, selected_model, output_path
        FROM jobs
        """
    ):
        key = (
            str(row["engine_version"]).lower(),
            int(row["forecast_origin"]),
            str(row["product_id"]),
        )
        out[key] = {
            "job_status": str(row["status"]),
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "selected_model": row["selected_model"],
            "output_path": row["output_path"],
        }
    conn.close()
    return out


def assemble_success_panel(run_dir: Path, *, engine: Optional[str] = None) -> pd.DataFrame:
    """Collect SUCCESS jobs with non-empty forecast.csv; dedupe logical keys."""
    run_dir = Path(run_dir)
    statuses = _load_statuses(run_dir)
    jobs_root = run_dir / "jobs"
    rows: list[pd.DataFrame] = []
    seen: set[tuple] = set()

    for key, meta in statuses.items():
        eng, origin, product_id = key
        if engine is not None and eng != str(engine).lower():
            continue
        if meta["job_status"] != "SUCCESS":
            continue
        product, seed = decode_product_key(product_id)
        logical = (str(product), int(origin), eng, "" if seed is None else int(seed))
        if logical in seen:
            continue

        out_path = meta.get("output_path")
        cand = Path(str(out_path)) if out_path else None
        if cand is None or not (cand / "forecast.csv").is_file():
            # Resolve under jobs/
            from pkg.benchmark.calendar import quarter_from_origin

            q = quarter_from_origin(int(origin))
            safe = product_id.replace("/", "_").replace("\\", "_")
            cand = jobs_root / eng / f"{q}__{safe}"
        fc = cand / "forecast.csv" if cand else None
        if fc is None or not fc.is_file():
            continue
        frame = pd.read_csv(fc)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["product"] = str(product)
        frame["product_id"] = str(product_id)
        frame["engine"] = eng
        frame["forecast_origin"] = int(origin)
        frame["seed"] = seed if seed is not None else pd.NA
        frame["job_status"] = "SUCCESS"
        frame["selected_model"] = meta.get("selected_model")
        frame["artifact_path"] = str(fc)
        rows.append(frame)
        seen.add(logical)

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    # Normalize numerics
    for c in ("forecast_origin", "target_date", "horizon"):
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    if "forecast" in panel.columns:
        panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    return panel


REQUIRED_A0_SEEDS = frozenset({41, 42, 43})


def seed_mean_panel(
    panel: pd.DataFrame,
    *,
    required_seeds: frozenset[int] = REQUIRED_A0_SEEDS,
) -> pd.DataFrame:
    """Mean forecast over seeds for neural engines; pass-through for seedless.

    Seed-mean rows are emitted only when every required seed is present for the
    product×origin×engine×horizon×target_date group.
    """
    if panel.empty:
        return panel.copy()
    work = panel.copy()
    if "seed" not in work.columns:
        return work
    keys = [
        c
        for c in ("product", "forecast_origin", "engine", "horizon", "target_date")
        if c in work.columns
    ]
    has_seed = work["seed"].notna()
    seeded = work.loc[has_seed]
    unseeded = work.loc[~has_seed]
    parts = []
    if not seeded.empty:
        seed_key = [c for c in ("product", "forecast_origin", "engine") if c in seeded.columns]
        complete = (
            seeded.groupby(seed_key)["seed"]
            .agg(lambda s: set(int(x) for x in s.dropna()) >= set(required_seeds))
            .rename("_complete")
            .reset_index()
        )
        seeded = seeded.merge(complete, on=seed_key, how="left")
        seeded = seeded.loc[seeded["_complete"].fillna(False)].drop(columns=["_complete"])
        if not seeded.empty:
            g = (
                seeded.groupby(keys, as_index=False)["forecast"]
                .mean()
                .assign(seed=pd.NA, aggregation="seed_mean")
            )
            parts.append(g)
    if not unseeded.empty:
        u = unseeded.loc[:, keys + ["forecast"]].copy()
        u["seed"] = pd.NA
        u["aggregation"] = "native"
        parts.append(u)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def export_run(
    *,
    run_id: str,
    engine: Optional[str],
    out_dir: Path,
) -> dict[str, Any]:
    run_dir = FORECAST_JOBS / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = assemble_success_panel(run_dir, engine=engine)
    mean = seed_mean_panel(panel)
    panel_path = out_dir / "forecasts_panel.parquet"
    csv_path = out_dir / "forecasts_panel.csv"
    mean_pq = out_dir / "forecasts_seed_mean.parquet"
    mean_csv = out_dir / "forecasts_seed_mean.csv"
    if not panel.empty:
        panel.to_parquet(panel_path, index=False)
        panel.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")
    if not mean.empty:
        mean.to_parquet(mean_pq, index=False)
        mean.to_csv(mean_csv, index=False, lineterminator="\n", encoding="utf-8")

    # Status histogram from sqlite
    statuses = _load_statuses(run_dir)
    hist: dict[str, int] = {}
    for meta in statuses.values():
        if engine is not None and False:
            pass
        st = meta["job_status"]
        hist[st] = hist.get(st, 0) + 1
    if engine is not None:
        hist = {}
        for (eng, _o, _p), meta in statuses.items():
            if eng != str(engine).lower():
                continue
            st = meta["job_status"]
            hist[st] = hist.get(st, 0) + 1

    summary = {
        "run_id": run_id,
        "engine_filter": engine,
        "n_success_rows": int(len(panel)),
        "n_success_jobs": int(
            panel.groupby(
                ["product", "forecast_origin", "engine", "seed"], dropna=False
            ).ngroups
        )
        if not panel.empty
        else 0,
        "n_seed_mean_rows": int(len(mean)),
        "status_counts": hist,
        "out_dir": out_dir.as_posix(),
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--engine", default=None, help="Optional engine filter (v2.1|a0)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: src/data/results/product_universe/{run_id}_export/",
    )
    args = ap.parse_args()
    out = args.out_dir or (
        REPO
        / "src"
        / "data"
        / "results"
        / "product_universe"
        / f"{args.run_id}_export"
    )
    summary = export_run(run_id=args.run_id, engine=args.engine, out_dir=out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
