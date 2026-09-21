"""Audit helpers for portfolio TS lock (HB coverage metadata only for HB origins)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.portfolio_ts_lock.constants import A0_SEEDS, HB_ORIGINS, KEYS


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
    finally:
        conn.close()
    return {str(s): int(n) for s, n in rows}


def hb_coverage_metadata(db_path: Path, engine: str) -> dict[str, Any]:
    """Identifiers + coverage only — no outcomes."""
    conn = sqlite3.connect(str(db_path))
    try:
        q = """
        SELECT forecast_origin, status, COUNT(*) AS n
        FROM jobs
        WHERE forecast_origin IN ({})
        GROUP BY forecast_origin, status
        """.format(
            ",".join(str(o) for o in sorted(HB_ORIGINS))
        )
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    by_origin: dict[str, dict[str, int]] = {}
    for _, r in df.iterrows():
        o = str(int(r["forecast_origin"]))
        by_origin.setdefault(o, {})
        by_origin[o][str(r["status"])] = int(r["n"])
    return {
        "engine": engine,
        "hb_origins": sorted(HB_ORIGINS),
        "status_by_origin": by_origin,
        "note": "coverage metadata only; outcomes not inspected",
    }


def a0_seed_completeness(panel_a0: pd.DataFrame) -> pd.DataFrame:
    p = panel_a0.copy()
    p = p.loc[~pd.to_numeric(p["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))]
    if "model" in p.columns:
        p = p.loc[p["model"].astype(str) != "naive_last_value"]
    p["seed"] = pd.to_numeric(p.get("seed"), errors="coerce")
    rows = []
    for (product, origin), g in p.groupby(["product", "forecast_origin"], sort=False):
        seeds = sorted({int(s) for s in g["seed"].dropna().tolist() if int(s) in A0_SEEDS})
        finite = g.loc[g["seed"].isin(list(A0_SEEDS)) & g["forecast"].notna()]
        per = finite.groupby("seed").size()
        complete = set(seeds) == set(A0_SEEDS) and all(
            int(per.get(s, 0)) >= 15 for s in A0_SEEDS
        )
        rows.append(
            {
                "product": str(product),
                "forecast_origin": int(origin),
                "seeds_present": ",".join(str(s) for s in seeds),
                "complete_triple": bool(set(seeds) == set(A0_SEEDS)),
                "complete_15h_triple": bool(complete),
            }
        )
    return pd.DataFrame(rows)


def horizon_audit(panel: pd.DataFrame) -> dict[str, Any]:
    p = panel.loc[~pd.to_numeric(panel["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))]
    hs = sorted({int(h) for h in pd.to_numeric(p["horizon"], errors="coerce").dropna()})
    bad_h1 = 0
    if not p.empty:
        h1 = p.loc[p["horizon"].astype(int) == 1]
        bad_h1 = int((h1["target_date"].astype(int) != h1["forecast_origin"].astype(int)).sum())
    return {
        "horizons_observed": hs,
        "exactly_1_to_15": hs == list(range(1, 16)),
        "n_h1_target_ne_origin": bad_h1,
    }


def duplicate_audit(panel: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in ("product", "forecast_origin", "engine", "seed", "horizon") if c in panel.columns]
    p = panel.loc[~pd.to_numeric(panel["forecast_origin"], errors="coerce").isin(list(HB_ORIGINS))]
    if not cols:
        return {"n_rows": 0, "n_dup_groups": 0}
    dups = p.groupby(cols, dropna=False).size()
    n_dup = int((dups > 1).sum())
    return {"n_rows": int(len(p)), "n_dup_groups": n_dup, "key": cols}
