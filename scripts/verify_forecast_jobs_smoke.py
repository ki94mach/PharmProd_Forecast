#!/usr/bin/env python3
"""Post-run smoke checks for forecast_jobs (V2.1 / A0 / A1) on the server."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.dates import resolve_production_time_contract

RUN_ID = "smoke_v21_a0_a1"
RUN_ROOT = Path("/opt/app/Forecast/src/data/forecast_jobs") / RUN_ID
ORIGIN = 140402
HORIZON = 15
PRODUCTS = [
    "Recigen",
    "Cinnatropin 10",
    "Aryoseven 1",
    "Melitide",
    "Paglino 10",
]
ENGINES = ("v2.1", "a0", "a1")
JOIN_KEYS = ["product", "forecast_origin", "horizon", "target_date"]
V21_COLS = [
    "product",
    "quarter",
    "forecast_origin",
    "target_date",
    "horizon",
    "forecast",
    "raw_forecast",
    "model",
    "engine",
]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def _load_forecasts(engine: str) -> pd.DataFrame:
    rows = []
    root = RUN_ROOT / "jobs" / engine
    if not root.is_dir():
        _fail(f"missing forecasts dir for {engine}: {root}")
    for path in sorted(root.glob("*/forecast.csv")):
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        _fail(f"no forecast.csv under {root}")
    out = pd.concat(rows, ignore_index=True)
    out["forecast_origin"] = pd.to_numeric(out["forecast_origin"], errors="coerce").astype(int)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype(int)
    out["target_date"] = pd.to_numeric(out["target_date"], errors="coerce").astype(int)
    return out


def check_shared_input() -> dict:
    snap_path = RUN_ROOT / "input_snapshot.json"
    man_path = RUN_ROOT / "manifest.json"
    if not snap_path.is_file():
        _fail("missing input_snapshot.json")
    if not man_path.is_file():
        _fail("missing manifest.json")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if not snap.get("content_sha256"):
        _fail("input snapshot missing content_sha256")
    if man.get("input_snapshot", {}).get("content_sha256") != snap["content_sha256"]:
        _fail("manifest input_snapshot sha mismatch vs input_snapshot.json")
    cfg_path = RUN_ROOT / "config.resolved.json"
    if not cfg_path.is_file():
        _fail("missing config.resolved.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("cutoff_policy", {}).get("name") != "production":
        _fail("cutoff_policy is not production")
    _ok(f"shared input snapshot sha={snap['content_sha256'][:12]}… rows={snap.get('n_rows')}")
    return snap


def check_cutoff_leakage(frames: dict[str, pd.DataFrame]) -> None:
    contract = resolve_production_time_contract(ORIGIN, delivered_horizon=HORIZON)
    partial = int(contract.current_partial_month)
    last_complete = int(contract.last_complete_month)
    sales = pd.read_parquet("/opt/app/Forecast/src/data/benchmarks/v1/raw/sales.parquet")
    # Re-derive training end used by production cutoff
    for eng, df in frames.items():
        if (df["forecast_origin"] != ORIGIN).any():
            _fail(f"{eng}: unexpected forecast_origin values")
        # Delivered targets must start at origin and never include partial
        if (df["target_date"] == partial).any():
            _fail(f"{eng}: delivered targets include bridge/partial month {partial}")
        if (df["target_date"] < ORIGIN).any():
            _fail(f"{eng}: target_date before origin (leakage into history)")
    # Training sales for products must not include partial or later for this origin
    train = sales.loc[
        (sales["product"].astype(str).isin(PRODUCTS))
        & (sales["date"].astype(int) <= last_complete)
    ]
    leaked = sales.loc[
        (sales["product"].astype(str).isin(PRODUCTS))
        & (sales["date"].astype(int) > last_complete)
        & (sales["date"].astype(int) < ORIGIN)
    ]
    # leaked months exist in warehouse (partial) but must not be in train slice above
    if not leaked.empty and (train["date"].astype(int) >= partial).any():
        _fail("training slice includes partial month")
    _ok(
        f"cutoff leakage: last_complete={last_complete} partial={partial} "
        f"excluded from delivered targets"
    )


def check_horizon_mapping(frames: dict[str, pd.DataFrame]) -> None:
    expected_targets = [shamsi_add_months(ORIGIN, i) for i in range(HORIZON)]
    for eng, df in frames.items():
        for product, g in df.groupby(df["product"].astype(str)):
            # For a0/a1 there may be one seed file; for v2.1 one file
            # Check each (product, model/seed) group has exactly 15 horizons
            key_cols = ["product"]
            if "seed" in g.columns:
                key_cols.append("seed")
            elif eng in {"a0", "a1"} and "model" in g.columns:
                key_cols.append("model")
            for _, gg in g.groupby([c for c in key_cols if c in g.columns], dropna=False):
                hs = sorted(int(x) for x in gg["horizon"].tolist())
                if hs != list(range(1, HORIZON + 1)):
                    _fail(f"{eng}/{product}: horizons {hs} != 1..15 (16→15 contract)")
                targets = [int(x) for x in gg.sort_values("horizon")["target_date"].tolist()]
                if targets != expected_targets:
                    _fail(f"{eng}/{product}: target_dates {targets} != {expected_targets}")
    _ok("16→15 horizon mapping: exactly 15 delivered months starting at origin")


def check_output_joins(frames: dict[str, pd.DataFrame]) -> None:
    for eng, df in frames.items():
        missing = [c for c in V21_COLS if c not in df.columns]
        if missing:
            _fail(f"{eng}: missing V2.1 join columns {missing}")
        if "origin" in df.columns or "prediction" in df.columns:
            _fail(f"{eng}: screening aliases origin/prediction still present")
    v21 = frames["v2.1"]
    for eng in ("a0", "a1"):
        merged = v21.merge(
            frames[eng],
            on=JOIN_KEYS,
            suffixes=("_v21", f"_{eng}"),
            how="inner",
        )
        if merged.empty:
            _fail(f"join v2.1↔{eng} produced 0 rows")
        # Expect 5 products × 15 horizons (one seed)
        if len(merged) < 5 * HORIZON:
            _fail(f"join v2.1↔{eng} too small: {len(merged)}")
    _ok("output joins: V2.1 keys align with A0/A1 forecast frames")


def check_sqlite_and_fallback() -> None:
    db = RUN_ROOT / "state.sqlite"
    if not db.is_file():
        _fail("missing state.sqlite")
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT status, COUNT(*) FROM jobs WHERE experiment_id=? GROUP BY status",
            (RUN_ID,),
        ).fetchall()
    finally:
        con.close()
    counts = {s: n for s, n in rows}
    # Expected: 5 products × (1 v2.1 + 1 a0 + 1 a1) = 15
    total = sum(counts.values())
    if total < 15:
        _fail(f"expected >=15 jobs, got {counts}")
    if counts.get("SUCCESS", 0) < 15:
        _fail(f"not all smoke jobs SUCCESS: {counts}")
    if counts.get("RUNNING", 0):
        _fail(f"stale RUNNING rows remain: {counts}")
    # Fallback: at least one v2.1 forecast row should carry a model label
    # (selection / intermittent fallback still writes SUCCESS with model name)
    v21 = _load_forecasts("v2.1")
    if v21["model"].isna().all():
        _fail("v2.1 model column empty (fallback/selection not recorded)")
    _ok(f"checkpoint state SUCCESS={counts.get('SUCCESS')} total={total}; models present")


def check_deterministic_artifact() -> None:
    """Compare two copies of the same Recigen a1 seed41 forecast if re-run marker exists.

    Baseline: forecasts must be finite and stable within the artifact (no NaN).
    """
    a1 = _load_forecasts("a1")
    rec = a1.loc[a1["product"].astype(str) == "Recigen"].sort_values("horizon")
    if len(rec) != HORIZON:
        _fail(f"Recigen a1 expected {HORIZON} rows, got {len(rec)}")
    if rec["forecast"].isna().any():
        _fail("Recigen a1 has NaN forecasts")
    # Write a fingerprint for operator comparison on forced re-run
    fp = ";".join(f"{int(r.horizon)}:{float(r.forecast):.10g}" for r in rec.itertuples())
    fp_path = RUN_ROOT / "smoke_recigen_a1_seed41_fingerprint.txt"
    if fp_path.is_file():
        old = fp_path.read_text(encoding="utf-8").strip()
        if old != fp:
            _fail("deterministic re-run fingerprint mismatch for Recigen a1")
        _ok("deterministic re-run: Recigen a1 fingerprint matches prior run")
    else:
        fp_path.write_text(fp + "\n", encoding="utf-8")
        _ok("deterministic fingerprint recorded (first pass)")


def main() -> int:
    if not RUN_ROOT.is_dir():
        _fail(f"run root missing: {RUN_ROOT}")
    check_shared_input()
    frames = {eng: _load_forecasts(eng) for eng in ENGINES}
    check_cutoff_leakage(frames)
    check_horizon_mapping(frames)
    check_output_joins(frames)
    check_sqlite_and_fallback()
    check_deterministic_artifact()
    print("SMOKE_CHECKS_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
