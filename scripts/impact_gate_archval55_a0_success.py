#!/usr/bin/env python3
"""Impact gate: SUCCESS A0 prediction identity under empty-telemetry fix.

Samples existing SUCCESS A0 forecast.csv vectors from the original run and
re-runs the same product×origin×seed through backtest_product_architectures.
Writes a small report under archval55_a0_empty_diag_v1/impact_gate/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import backtest_product_architectures

REPO = Path(__file__).resolve().parents[1]
ORIG_RUN = REPO / "src" / "data" / "forecast_jobs" / "archval55_v21_a0_a1_fallback_v1"
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval55_a0_empty_diag_v1"
    / "impact_gate"
)
ARCH = ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM


def _vector_hash(forecasts: np.ndarray) -> str:
    arr = np.asarray(forecasts, dtype=np.float64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sample", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    # Prefer SUCCESS neural (non-naive) A0 forecasts.
    rows = []
    for csv_path in sorted((ORIG_RUN / "jobs" / "a0").rglob("forecast.csv")):
        job_json = csv_path.parent / "job.json"
        if not job_json.is_file():
            continue
        meta = json.loads(job_json.read_text(encoding="utf-8"))
        if meta.get("ineligible") or not meta.get("success"):
            continue
        if str(meta.get("selected_model") or "").startswith("naive"):
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "forecast" not in df.columns:
            continue
        rows.append(
            {
                "product": str(meta["product"]),
                "forecast_origin": int(meta["forecast_origin"]),
                "seed": int(meta["seed"]),
                "path": str(csv_path),
                "orig_hash": _vector_hash(df.sort_values("horizon")["forecast"].to_numpy()),
                "orig_forecasts": df.sort_values("horizon")["forecast"].astype(float).tolist(),
            }
        )
        if len(rows) >= int(args.n_sample):
            break

    if not rows:
        raise SystemExit("no SUCCESS A0 neural forecasts found to sample")

    results = []
    for row in rows:
        print(
            f"replaying SUCCESS {row['product']} @ {row['forecast_origin']} "
            f"seed={row['seed']}"
        )
        result = backtest_product_architectures(
            sales,
            row["product"],
            architectures=(ARCH,),
            seeds=(int(row["seed"]),),
            explicit_origins=(int(row["forecast_origin"]),),
            min_successful_seeds=1,
        )
        preds = result.predictions
        if preds is None or preds.empty:
            results.append({**row, "match": False, "n_pred": 0, "replay_hash": None})
            continue
        replay = (
            preds.sort_values("horizon")["prediction"].astype(float).to_numpy()
        )
        rh = _vector_hash(replay)
        # Bit-identical vs artifact is unlikely across TF non-determinism;
        # gate is: SUCCESS path still produces 15 rows and fold status ok.
        # Also compare shape / finite.
        match_len = len(replay) == 15
        status_ok = (
            result.fold_metadata is not None
            and not result.fold_metadata.empty
            and str(result.fold_metadata.iloc[0].get("status")) == "ok"
        )
        results.append(
            {
                "product": row["product"],
                "forecast_origin": row["forecast_origin"],
                "seed": row["seed"],
                "orig_hash": row["orig_hash"],
                "replay_hash": rh,
                "hash_equal": rh == row["orig_hash"],
                "n_pred": int(len(replay)),
                "status_ok": bool(status_ok),
                "finite": bool(np.isfinite(replay).all()),
                "success_path_ok": bool(match_len and status_ok and np.isfinite(replay).all()),
            }
        )

    frame = pd.DataFrame(results)
    frame.to_csv(out / "success_replay.csv", index=False)
    summary = {
        "n_sampled": int(len(frame)),
        "n_success_path_ok": int(frame["success_path_ok"].sum()),
        "n_hash_equal": int(frame["hash_equal"].sum()),
        "policy": (
            "refit_90_only"
            if bool(frame["success_path_ok"].all())
            else "full_a0_refit_required"
        ),
        "note": (
            "TF training is not bit-stable across hosts; identity gate requires "
            "SUCCESS path still yields 15 finite horizons with fold status=ok. "
            "Telemetry-only fix must not convert SUCCESS into empty/unavailable."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["policy"] == "refit_90_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
