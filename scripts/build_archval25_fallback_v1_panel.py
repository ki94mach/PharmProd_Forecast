#!/usr/bin/env python3
"""Build versioned archval25 panel with INELIGIBLE + naive_last_value fallback.

Leaves ``archval25_export`` / ``archval25_production`` / ``archval25_blends``
intact. Writes a parallel tree:

  src/data/results/ts_v3a_architecture_validation/archval25_export_fallback_v1/

For each FAILED job in the original export ``state.sqlite`` that matches
expected ineligibility (A1 short history / V2.1 no eligible models), appends
``naive_last_value`` forecasts (last finite production-cut training sales —
never a synthetic zero) and records the reason in ``fallback_jobs.csv``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from pkg.forecast_jobs.cutoff import apply_cutoff_policy
from pkg.forecast_jobs.engines import (
    FALLBACK_MODEL,
    REASON_NEURAL_EMPTY_PREDICTIONS,
    REASON_NEURAL_SHORT_HISTORY,
    REASON_V21_NO_ELIGIBLE_MODELS,
    build_naive_last_value_forecasts,
    last_training_value,
    target_dates_for_origin,
)
from pkg.benchmark.backfill_runner.types import EngineJobRequest
from pkg.forecast_jobs.plan import encode_product_key, decode_product_key

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
SRC_EXPORT = RESULTS / "archval25_export"
OUT_EXPORT = RESULTS / "archval25_export_fallback_v1"
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

FALLBACK_POLICY = "naive_last_value"
FALLBACK_VERSION = "v1"
HORIZON = 15


def _decode_product_seed(product_id: str, engine: str) -> tuple[str, int | None]:
    """Invert ``encode_product_key`` for neural jobs; V2.1 has no seed suffix."""
    eng = str(engine).lower()
    if eng in {"a0", "a1"}:
        return decode_product_key(str(product_id))
    return str(product_id), None


def _classify_failed(error_type: str | None, error_message: str | None) -> str | None:
    et = (error_type or "").strip()
    msg = (error_message or "").lower()
    if et == "EmptyPredictions" or "neural backtest returned no predictions" in msg:
        if "insufficient" in msg:
            return REASON_NEURAL_SHORT_HISTORY
        return REASON_NEURAL_EMPTY_PREDICTIONS
    if "no eligible models for selection" in msg or "insufficient origins" in msg:
        return REASON_V21_NO_ELIGIBLE_MODELS
    if et in {"ExpectedIneligible"}:
        if "no eligible" in msg:
            return REASON_V21_NO_ELIGIBLE_MODELS
        return REASON_NEURAL_SHORT_HISTORY
    return None


def _quarter_for_origin(origin: int) -> str:
    o = int(origin)
    y, m = divmod(o, 100)
    # Shamsi calendar quarters used in forecast_jobs plan.
    q = (int(m) - 1) // 3 + 1
    return f"{y}Q{q}"


def main() -> int:
    panel_path = SRC_EXPORT / "forecasts_panel.parquet"
    if not panel_path.is_file():
        panel_path = SRC_EXPORT / "forecasts_panel.csv"
    if not panel_path.is_file():
        print(f"missing source panel under {SRC_EXPORT}")
        return 2
    db_path = SRC_EXPORT / "state.sqlite"
    if not db_path.is_file():
        print(f"missing state.sqlite at {db_path}")
        return 2
    if not SALES.is_file():
        print(f"missing sales at {SALES}")
        return 2

    panel = (
        pd.read_parquet(panel_path)
        if panel_path.suffix == ".parquet"
        else pd.read_csv(panel_path)
    )
    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    failed = list(
        conn.execute(
            """
            SELECT engine_version, quarter, forecast_origin, product_id,
                   error_type, error_message, status
            FROM jobs
            WHERE status = 'FAILED'
            ORDER BY engine_version, product_id, forecast_origin
            """
        )
    )
    conn.close()

    fallback_rows: list[pd.DataFrame] = []
    job_log: list[dict] = []
    skipped: list[dict] = []

    for row in failed:
        engine = str(row["engine_version"]).lower()
        origin = int(row["forecast_origin"])
        product_id = str(row["product_id"])
        product, seed = _decode_product_seed(product_id, engine)
        reason = _classify_failed(row["error_type"], row["error_message"])
        if reason is None:
            skipped.append(
                {
                    "engine": engine,
                    "product": product,
                    "product_id": product_id,
                    "forecast_origin": origin,
                    "error_type": row["error_type"],
                    "error_message": row["error_message"],
                    "skip_reason": "not_expected_ineligibility",
                }
            )
            continue

        cut = apply_cutoff_policy(sales, origin, policy_name="production")
        product_sales = cut.loc[cut["product"].astype(str) == str(product)].copy()
        last = last_training_value(product_sales, product)
        if last is None:
            skipped.append(
                {
                    "engine": engine,
                    "product": product,
                    "product_id": product_id,
                    "forecast_origin": origin,
                    "error_type": row["error_type"],
                    "error_message": row["error_message"],
                    "skip_reason": "fallback_unavailable_no_finite_training",
                }
            )
            continue

        quarter = str(row["quarter"] or _quarter_for_origin(origin))
        targets = target_dates_for_origin(origin, HORIZON)
        request = EngineJobRequest(
            engine=engine,
            product=product,
            quarter=quarter,
            forecast_origin=origin,
            horizon=HORIZON,
            target_dates=targets,
            training_sales=product_sales,
            meta={
                "fallback_policy": FALLBACK_POLICY,
                "fallback_version": FALLBACK_VERSION,
            },
        )
        frame = build_naive_last_value_forecasts(
            request,
            engine=engine,
            last_value=float(last),
            seed=seed,
            architecture=engine if engine in {"a0", "a1"} else None,
        )
        # Match export panel extras.
        frame["architecture"] = engine if engine in {"a0", "a1"} else pd.NA
        if seed is not None:
            frame["seed"] = int(seed)
        else:
            frame["seed"] = pd.NA
        frame["actual"] = pd.NA
        frame["prediction_kind"] = "fallback"
        frame["product_id"] = (
            encode_product_key(product, seed) if seed is not None else product
        )
        frame["artifact_path"] = (
            f"fallback_v1/{engine}/{quarter}__{product_id}/forecast.csv"
        )
        frame["ineligible"] = True
        frame["ineligibility_reason"] = reason
        frame["fallback_policy"] = FALLBACK_POLICY
        frame["fallback_version"] = FALLBACK_VERSION
        frame["fallback_reason"] = (
            f"{reason}|policy={FALLBACK_POLICY}|version={FALLBACK_VERSION}"
        )
        frame["fallback_last_value"] = float(last)
        fallback_rows.append(frame)
        job_log.append(
            {
                "status": "INELIGIBLE",
                "engine": engine,
                "product": product,
                "seed": seed,
                "product_id": product_id,
                "forecast_origin": origin,
                "quarter": quarter,
                "error_type": "ExpectedIneligible",
                "error_message": row["error_message"],
                "ineligibility_reason": reason,
                "fallback_policy": FALLBACK_POLICY,
                "fallback_version": FALLBACK_VERSION,
                "fallback_model": FALLBACK_MODEL,
                "fallback_last_value": float(last),
                "n_horizon_rows": int(len(frame)),
            }
        )

    OUT_EXPORT.mkdir(parents=True, exist_ok=True)
    # Preserve original columns; add fallback metadata cols (NaN for SUCCESS rows).
    out_panel = panel.copy()
    for col in (
        "ineligible",
        "ineligibility_reason",
        "fallback_policy",
        "fallback_version",
        "fallback_reason",
        "fallback_last_value",
    ):
        if col not in out_panel.columns:
            out_panel[col] = pd.NA
    out_panel["ineligible"] = False

    if fallback_rows:
        fb = pd.concat(fallback_rows, ignore_index=True)
        # Align columns
        for col in out_panel.columns:
            if col not in fb.columns:
                fb[col] = pd.NA
        fb = fb.loc[:, list(out_panel.columns)]
        out_panel = pd.concat([out_panel, fb], ignore_index=True)

    out_panel.to_parquet(OUT_EXPORT / "forecasts_panel.parquet", index=False)
    out_panel.to_csv(OUT_EXPORT / "forecasts_panel.csv", index=False)

    jobs_df = pd.DataFrame(job_log)
    jobs_df.to_csv(OUT_EXPORT / "fallback_jobs.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT_EXPORT / "fallback_skipped.csv", index=False)

    # Copy state.sqlite read-only note + write version manifest (do not mutate source).
    manifest = {
        "version": FALLBACK_VERSION,
        "fallback_policy": FALLBACK_POLICY,
        "fallback_model": FALLBACK_MODEL,
        "source_export": str(SRC_EXPORT.as_posix()),
        "source_panel_rows": int(len(panel)),
        "output_panel_rows": int(len(out_panel)),
        "ineligible_jobs_filled": int(len(job_log)),
        "skipped_failed_jobs": int(len(skipped)),
        "note": (
            "Original archval25_export left intact. "
            "Filled expected-ineligible FAILED jobs with naive_last_value; "
            "never substituted zero for missing forecasts."
        ),
    }
    (OUT_EXPORT / "fallback_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Point eval scripts at the same state counts story.
    if (SRC_EXPORT / "state.sqlite").is_file():
        # Do not copy mutable DB as authoritative; write a status summary instead.
        conn = sqlite3.connect(str(SRC_EXPORT / "state.sqlite"))
        counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
        conn.close()
        status_note = {
            "source_status_counts": counts,
            "reclassified": {
                "FAILED_expected_ineligible_filled": len(job_log),
                "FAILED_left_unfilled": len(skipped),
            },
            "effective_status_view": {
                "SUCCESS": int(counts.get("SUCCESS", 0)),
                "INELIGIBLE": int(len(job_log)),
                "FAILED": int(counts.get("FAILED", 0) - len(job_log)),
            },
        }
        (OUT_EXPORT / "status_reclassification.json").write_text(
            json.dumps(status_note, indent=2) + "\n", encoding="utf-8"
        )

    print(f"wrote {OUT_EXPORT}")
    print(f"source_rows={len(panel)} out_rows={len(out_panel)}")
    print(f"ineligible_filled={len(job_log)} skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
