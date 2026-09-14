#!/usr/bin/env python3
"""Merge original Archval55 export with A0 empty-fix refit overlays.

Keeps original V2.1/A1 (+ untouched A0 SUCCESS) and replaces the 90 empty A0
jobs with forecasts from archval55_a0_empty_fix_v1. Writes
archval55_export_a0fix_v1/ without mutating prior export trees.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from pkg.forecast_jobs.plan import encode_product_key

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
ORIG_EXPORT = RESULTS / "archval55_export"
FIX_RUN = REPO / "src" / "data" / "forecast_jobs" / "archval55_a0_empty_fix_v1"
INELIG = RESULTS / "archval55_decision_v1" / "ineligible_jobs.csv"
OUT = RESULTS / "archval55_export_a0fix_v1"


def _empty_keys(path: Path) -> set[tuple[str, int, int]]:
    jobs = pd.read_csv(path)
    a0 = jobs.loc[
        (jobs["engine"].astype(str).str.lower() == "a0")
        & jobs["reason_code"].astype(str).str.contains("empty", case=False, na=False)
    ]
    return {
        (str(r.product), int(r.forecast_origin), int(r.seed))
        for r in a0.itertuples(index=False)
    }


def _load_fix_panel(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    import importlib.util
    import sys

    mod_path = REPO / "scripts" / "export_archval55_panel.py"
    spec = importlib.util.spec_from_file_location("export_archval55_panel", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    panel, summary = mod.assemble_panel(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fix_run_export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return panel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-export", type=Path, default=ORIG_EXPORT)
    p.add_argument("--fix-run", type=Path, default=FIX_RUN)
    p.add_argument("--ineligible-csv", type=Path, default=INELIG)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    orig_panel_path = Path(args.orig_export) / "forecasts_panel.parquet"
    if not orig_panel_path.is_file():
        # fallback csv
        orig_panel_path = Path(args.orig_export) / "forecasts_panel.csv"
    if orig_panel_path.suffix == ".parquet":
        orig = pd.read_parquet(orig_panel_path)
    else:
        orig = pd.read_csv(orig_panel_path)

    empty_keys = _empty_keys(args.ineligible_csv)
    fix_panel = _load_fix_panel(Path(args.fix_run), out)
    fix_a0 = fix_panel.loc[fix_panel["engine"].astype(str).str.lower() == "a0"].copy()

    def _key_row(r) -> tuple[str, int, int]:
        seed = r.seed if "seed" in r._fields else None
        if seed is None or (isinstance(seed, float) and pd.isna(seed)):
            seed = int(str(r.product_id).rsplit("__seed", 1)[-1]) if "__seed" in str(r.product_id) else -1
        return (str(r.product), int(r.forecast_origin), int(seed))

    # Drop original empty A0 keys (all rows for those product×origin×seed).
    keep_mask = []
    for r in orig.itertuples(index=False):
        eng = str(getattr(r, "engine")).lower()
        if eng != "a0":
            keep_mask.append(True)
            continue
        seed = getattr(r, "seed", None)
        if seed is None or (isinstance(seed, float) and pd.isna(seed)):
            pid = str(getattr(r, "product_id", ""))
            seed = int(pid.rsplit("__seed", 1)[-1]) if "__seed" in pid else -1
        key = (str(r.product), int(r.forecast_origin), int(seed))
        keep_mask.append(key not in empty_keys)
    kept = orig.loc[keep_mask].copy()

    # Keep only allow-listed fix rows (and prefer SUCCESS when present).
    fix_rows = []
    for r in fix_a0.itertuples(index=False):
        seed = getattr(r, "seed", None)
        if seed is None or (isinstance(seed, float) and pd.isna(seed)):
            pid = str(getattr(r, "product_id", ""))
            seed = int(pid.rsplit("__seed", 1)[-1]) if "__seed" in pid else -1
        key = (str(r.product), int(r.forecast_origin), int(seed))
        if key in empty_keys:
            fix_rows.append(r)
    if not fix_rows:
        raise SystemExit("no overlapping fix-run A0 rows for empty keys")
    fix_df = pd.DataFrame(fix_rows)

    # Align columns
    for c in kept.columns:
        if c not in fix_df.columns:
            fix_df[c] = None
    fix_df = fix_df[kept.columns]
    merged = pd.concat([kept, fix_df], ignore_index=True)

    merged.to_parquet(out / "forecasts_panel.parquet", index=False)
    merged.to_csv(out / "forecasts_panel.csv", index=False)

    # Coverage of the 90 keys in merged panel
    covered = set()
    still_empty = []
    for product, origin, seed in sorted(empty_keys):
        sub = merged.loc[
            (merged["engine"].astype(str).str.lower() == "a0")
            & (merged["product"].astype(str) == product)
            & (merged["forecast_origin"].astype(int) == origin)
            & (pd.to_numeric(merged["seed"], errors="coerce") == seed)
        ]
        if sub.empty:
            still_empty.append((product, origin, seed, "missing"))
        elif bool(sub["ineligible"].fillna(False).astype(bool).all()):
            still_empty.append((product, origin, seed, "still_ineligible"))
        else:
            covered.add((product, origin, seed))

    summary = {
        "n_orig_rows": int(len(orig)),
        "n_kept_rows": int(len(kept)),
        "n_fix_rows_added": int(len(fix_df)),
        "n_merged_rows": int(len(merged)),
        "n_empty_keys": len(empty_keys),
        "n_empty_keys_covered_success": len(covered),
        "n_empty_keys_unresolved": len(still_empty),
        "unresolved_sample": still_empty[:20],
        "orig_export": str(args.orig_export.as_posix()),
        "fix_run": str(Path(args.fix_run).as_posix()),
    }
    (out / "merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # copy any manifest-like files from orig for continuity
    for name in ("export_summary.json", "report.md"):
        src = Path(args.orig_export) / name
        if src.is_file():
            shutil.copy2(src, out / f"orig_{name}")
    print(json.dumps(summary, indent=2))
    return 0 if not still_empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
