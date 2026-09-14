#!/usr/bin/env python3
"""Assemble Archval55 forecasts_panel from the finished forecast_jobs run.

Writes versioned export under
``src/data/results/ts_v3a_architecture_validation/archval55_export/``
(does not touch archval25 trees).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from pkg.benchmark.universes import load_universe_product_names
from pkg.forecast_jobs.plan import encode_product_key

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO / "src" / "data" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1"
)
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_OUT = RESULTS / "archval55_export"

EARLY_ORIGINS = frozenset({140104, 140201})
LATE_ORIGINS = frozenset({140301, 140402})
OLD_UNIVERSE = "v3a_architecture_validation_products"
MVP_UNIVERSE = "mvp_products"


def _sku_cohort(product: str, old_skus: set[str]) -> str:
    return "old_sku" if str(product) in old_skus else "new_sku"


def _origin_cohort(origin: int) -> str:
    o = int(origin)
    if o in EARLY_ORIGINS:
        return "early_origin"
    if o in LATE_ORIGINS:
        return "late_origin"
    return "other_origin"


def _load_job_status(run_dir: Path) -> dict[tuple[str, int, str], dict]:
    """Map (engine, origin, product_id) → status row."""
    db = run_dir / "state.sqlite"
    out: dict[tuple[str, int, str], dict] = {}
    if not db.is_file():
        return out
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(
        """
        SELECT engine_version, forecast_origin, product_id, status,
               error_type, error_message, selected_model
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
        }
    conn.close()
    return out


def _parse_job_dir(path: Path, jobs_root: Path) -> tuple[str, str, str] | None:
    """Return (engine, quarter, product_id) from .../jobs/{engine}/{quarter}__{pid}/forecast.csv."""
    try:
        rel = path.parent.relative_to(jobs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    engine = str(parts[0]).lower()
    slug = str(parts[1])
    if "__" not in slug:
        return None
    quarter, _, product_id = slug.partition("__")
    return engine, quarter, product_id


def assemble_panel(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    jobs_root = run_dir / "jobs"
    if not jobs_root.is_dir():
        raise FileNotFoundError(f"missing jobs dir: {jobs_root}")

    old_skus = set(load_universe_product_names(OLD_UNIVERSE, validate=True))
    mvp = set(load_universe_product_names(MVP_UNIVERSE, validate=True))
    new_skus = mvp - old_skus
    status_map = _load_job_status(run_dir)

    frames: list[pd.DataFrame] = []
    n_files = 0
    for csv_path in sorted(jobs_root.rglob("forecast.csv")):
        n_files += 1
        parsed = _parse_job_dir(csv_path, jobs_root)
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.copy()
        df["artifact_path"] = str(csv_path.relative_to(run_dir)).replace("\\", "/")

        engine = str(df["engine"].iloc[0]).lower() if "engine" in df.columns else ""
        origin = int(pd.to_numeric(df["forecast_origin"], errors="coerce").iloc[0])
        product = str(df["product"].iloc[0])

        seed = None
        if "seed" in df.columns and pd.notna(df["seed"].iloc[0]):
            seed = int(df["seed"].iloc[0])

        # Prefer CSV product+seed for SQLite identity (path slug may replace '/' with '_').
        if engine in {"a0", "a1"} and seed is not None:
            product_id = encode_product_key(product, seed)
        else:
            product_id = encode_product_key(product, None)

        if parsed is not None and parsed[0]:
            engine = parsed[0] or engine

        st = status_map.get((engine, origin, str(product_id)), {})
        job_status = st.get("job_status") or "UNKNOWN"

        fb_path = csv_path.parent / "fallback.json"
        fallback_reason = None
        if fb_path.is_file():
            try:
                fb = json.loads(fb_path.read_text(encoding="utf-8"))
                extras = fb.get("extras") or {}
                fallback_reason = extras.get("fallback_reason") or fb.get("error_message")
            except Exception:
                fallback_reason = None

        ineligible = job_status == "INELIGIBLE" or bool(
            str(df.get("model", pd.Series([""])).iloc[0]) == "naive_last_value"
            and job_status != "SUCCESS"
        )
        if job_status == "INELIGIBLE":
            ineligible = True

        df["engine"] = engine
        df["product"] = product
        df["product_id"] = product_id
        df["job_status"] = job_status
        df["ineligible"] = ineligible
        df["fallback_reason"] = fallback_reason
        df["sku_cohort"] = _sku_cohort(product, old_skus)
        df["origin_cohort"] = _origin_cohort(origin)
        if "seed" in df.columns:
            df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        frames.append(df)

    if not frames:
        raise RuntimeError(f"no forecast.csv under {jobs_root}")

    panel = pd.concat(frames, ignore_index=True)
    for c in ("forecast_origin", "target_date", "horizon"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(int)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    panel["engine"] = panel["engine"].astype(str).str.lower()
    panel["product"] = panel["product"].astype(str)

    summary = {
        "run_dir": str(run_dir.as_posix()),
        "n_forecast_files": n_files,
        "n_panel_rows": int(len(panel)),
        "n_old_skus": len(old_skus),
        "n_new_skus": len(new_skus),
        "status_counts": (
            dict(
                sqlite3.connect(str(run_dir / "state.sqlite")).execute(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                )
            )
            if (run_dir / "state.sqlite").is_file()
            else {}
        ),
        "panel_job_status_counts": panel.groupby("job_status").size().astype(int).to_dict(),
        "panel_ineligible_rows": int(panel["ineligible"].fillna(False).astype(bool).sum()),
        "early_origins": sorted(EARLY_ORIGINS),
        "late_origins": sorted(LATE_ORIGINS),
    }
    return panel, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    panel, summary = assemble_panel(run_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "forecasts_panel.parquet", index=False)
    panel.to_csv(out_dir / "forecasts_panel.csv", index=False)
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Compact status snapshot for eval scripts (no full DB copy required).
    db = run_dir / "state.sqlite"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        status = {
            "status_counts": dict(
                conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            ),
            "source_db": str(db.as_posix()),
        }
        conn.close()
        (out_dir / "status_reclassification.json").write_text(
            json.dumps(
                {
                    "effective_status_view": status["status_counts"],
                    "source_status_counts": status["status_counts"],
                    "note": "Archval55 finished run; INELIGIBLE is expected short-history fallback.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    (out_dir / "README.md").write_text(
        "# Archval55 export\n\n"
        f"- Source run: `{run_dir.as_posix()}`\n"
        f"- Rows: {len(panel)}\n"
        f"- Job status: {summary.get('status_counts')}\n"
        "- Built by `scripts/export_archval55_panel.py`.\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'forecasts_panel.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
