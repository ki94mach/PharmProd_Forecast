#!/usr/bin/env python3
"""Evaluate archval25 forecast_jobs panel: V2.1 vs A0 vs A1 (matched rows)."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval25_export"
)
DEFAULT_OUT = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval25_production"
)
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"
FINDINGS = (
    REPO
    / "src"
    / "data"
    / "results"
    / "ts_v3a_architecture_validation"
    / "archval25_v21_a0_a1_report_findings.md"
)

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
ENGINES = ("v2.1", "a0", "a1")


def _md_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    float_cols = float_cols or set()
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif c in float_cols or isinstance(v, float):
                cells.append(f"{float(v):.4g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def attach_actuals(df: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    need = out["actual"].isna() if "actual" in out.columns else pd.Series(True, index=out.index)
    if not need.any():
        return out
    s = sales.rename(columns={"sales": "actual_sales", "date": "target_date"})
    s["product"] = s["product"].astype(str)
    s["target_date"] = pd.to_numeric(s["target_date"], errors="coerce").astype(int)
    merged = out.merge(
        s[["product", "target_date", "actual_sales"]],
        on=["product", "target_date"],
        how="left",
    )
    if "actual" not in merged.columns:
        merged["actual"] = merged["actual_sales"]
    else:
        merged["actual"] = merged["actual"].where(merged["actual"].notna(), merged["actual_sales"])
    return merged.drop(columns=["actual_sales"], errors="ignore")


def seed_ensemble(neural: pd.DataFrame) -> pd.DataFrame:
    """Mean forecast across seeds for A0/A1; keep cells with ≥1 seed."""
    g = (
        neural.groupby(KEYS + ["engine"], as_index=False)
        .agg(
            forecast=("forecast", "mean"),
            actual=("actual", "mean"),
            n_seeds=("seed", "nunique"),
            model=("model", "first"),
        )
    )
    return g


def metrics_for(g: pd.DataFrame, engine: str) -> dict:
    m = g.dropna(subset=["actual", "forecast"])
    if m.empty:
        return {
            "engine": engine,
            "n_rows": 0,
            "n_products": 0,
            "n_origins": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "portfolio_wmape_pct": np.nan,
            "bias": np.nan,
            "mean_horizon_mae": np.nan,
        }
    err = m["forecast"] - m["actual"]
    # Equal-weight mean of per-horizon MAE
    h_maes = []
    for _, hg in m.groupby("horizon"):
        h_maes.append(float(np.mean(np.abs(hg["forecast"] - hg["actual"]))))
    mean_h_mae = float(np.mean(h_maes)) if h_maes else float("nan")
    return {
        "engine": engine,
        "n_rows": int(len(m)),
        "n_products": int(m["product"].nunique()),
        "n_origins": int(m["forecast_origin"].nunique()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
        "mean_horizon_mae": mean_h_mae,
    }


def paired(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict:
    a = left[KEYS + ["forecast", "actual"]].rename(
        columns={"forecast": "f_left", "actual": "actual_left"}
    )
    b = right[KEYS + ["forecast"]].rename(columns={"forecast": "f_right"})
    m = a.merge(b, on=KEYS, how="inner")
    m = m.dropna(subset=["actual_left", "f_left", "f_right"])
    if m.empty:
        return {
            "left": left_name,
            "right": right_name,
            "n": 0,
            "mae_left": np.nan,
            "mae_right": np.nan,
            "mae_diff_left_minus_right": np.nan,
            "wmape_left": np.nan,
            "wmape_right": np.nan,
            "left_wins": 0,
            "right_wins": 0,
            "ties": 0,
        }
    ae_l = (m["f_left"] - m["actual_left"]).abs()
    ae_r = (m["f_right"] - m["actual_left"]).abs()
    diff = ae_l - ae_r
    return {
        "left": left_name,
        "right": right_name,
        "n": int(len(m)),
        "mae_left": float(ae_l.mean()),
        "mae_right": float(ae_r.mean()),
        "mae_diff_left_minus_right": float(diff.mean()),
        "wmape_left": float(portfolio_wmape_pct(m["actual_left"], m["f_left"])),
        "wmape_right": float(portfolio_wmape_pct(m["actual_left"], m["f_right"])),
        "left_wins": int((diff < 0).sum()),
        "right_wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def triple_matched(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows present in all three engines."""
    base = None
    for eng in ENGINES:
        k = frames[eng][KEYS].drop_duplicates()
        base = k if base is None else base.merge(k, on=KEYS, how="inner")
    assert base is not None
    rows = []
    for eng in ENGINES:
        part = frames[eng].merge(base, on=KEYS, how="inner")
        part = part.assign(engine=eng)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT,
        help="Directory with forecasts_panel.parquet (+ optional state.sqlite)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for metrics / report (must not overwrite originals unless intended)",
    )
    p.add_argument(
        "--findings",
        type=Path,
        default=None,
        help="Optional findings markdown path (default: only for original export)",
    )
    args = p.parse_args()
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    findings_path = args.findings
    if findings_path is None and export_dir.resolve() == DEFAULT_EXPORT.resolve():
        findings_path = FINDINGS

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = export_dir / "forecasts_panel.parquet"
    if not panel_path.is_file():
        panel_path = export_dir / "forecasts_panel.csv"
    panel = pd.read_parquet(panel_path) if panel_path.suffix == ".parquet" else pd.read_csv(panel_path)
    panel["product"] = panel["product"].astype(str)
    for c in ("forecast_origin", "target_date", "horizon"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(int)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    panel["engine"] = panel["engine"].astype(str).str.lower()

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)

    panel = attach_actuals(panel, sales)

    frames: dict[str, pd.DataFrame] = {}
    for eng in ENGINES:
        sub = panel.loc[panel["engine"] == eng].copy()
        if eng in {"a0", "a1"}:
            sub = seed_ensemble(sub)
        else:
            sub = (
                sub.groupby(KEYS, as_index=False)
                .agg(forecast=("forecast", "first"), actual=("actual", "first"), model=("model", "first"))
                .assign(engine=eng, n_seeds=1)
            )
        sub["engine"] = eng
        frames[eng] = sub

    # Job status
    db = export_dir / "state.sqlite"
    status_counts = {}
    failed_note = ""
    reclass_path = export_dir / "status_reclassification.json"
    if db.is_file():
        conn = sqlite3.connect(db)
        status_counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
        failed = list(
            conn.execute(
                "SELECT engine_version, COUNT(*) FROM jobs WHERE status='FAILED' GROUP BY engine_version"
            )
        )
        conn.close()
        failed_note = ", ".join(f"{e}={n}" for e, n in failed)
    elif reclass_path.is_file():
        reclass = json.loads(reclass_path.read_text(encoding="utf-8"))
        status_counts = dict(reclass.get("effective_status_view") or {})
        failed_note = "reclassified from source FAILED via fallback_v1"

    # Unmatched (all available rows with actuals)
    summary_rows = [metrics_for(frames[e].dropna(subset=["actual", "forecast"]), e) for e in ENGINES]
    summary = pd.DataFrame(summary_rows)

    # Triple-matched
    tri = triple_matched(frames)
    tri = tri.dropna(subset=["actual", "forecast"])
    matched_summary = []
    for eng in ENGINES:
        matched_summary.append(metrics_for(tri.loc[tri["engine"] == eng], eng))
    matched_df = pd.DataFrame(matched_summary)

    # Paired on matched keys across each pair (using same triple key set for fairness)
    wide = tri.loc[tri["engine"] == "a0", KEYS + ["actual"]].drop_duplicates(KEYS)
    for eng in ENGINES:
        part = (
            tri.loc[tri["engine"] == eng, KEYS + ["forecast"]]
            .rename(columns={"forecast": f"f_{eng}"})
        )
        wide = wide.merge(part, on=KEYS, how="inner")
    assert wide is not None and not wide.empty

    pairs = []
    for left, right in (("v2.1", "a0"), ("v2.1", "a1"), ("a0", "a1")):
        fl = wide[KEYS + ["actual", f"f_{left}"]].rename(columns={f"f_{left}": "forecast"})
        fr = wide[KEYS + ["actual", f"f_{right}"]].rename(columns={f"f_{right}": "forecast"})
        pairs.append(paired(fl, fr, left, right))
    paired_df = pd.DataFrame(pairs)

    # By origin on triple-matched
    by_origin_rows = []
    for origin, g in tri.groupby("forecast_origin"):
        for eng in ENGINES:
            m = metrics_for(g.loc[g["engine"] == eng], eng)
            m["forecast_origin"] = int(origin)
            by_origin_rows.append(m)
    by_origin = pd.DataFrame(by_origin_rows)

    # Coverage
    cov = []
    for eng in ENGINES:
        f = frames[eng]
        cov.append(
            {
                "engine": eng,
                "n_cells": int(len(f)),
                "n_with_actual": int(f["actual"].notna().sum()),
                "n_products": int(f["product"].nunique()),
                "n_origins": int(f["forecast_origin"].nunique()),
            }
        )
    coverage = pd.DataFrame(cov)
    n_triple = int(len(wide))

    # Persist tables
    summary.to_csv(out_dir / "engine_summary_all_rows.csv", index=False)
    matched_df.to_csv(out_dir / "engine_summary_matched.csv", index=False)
    paired_df.to_csv(out_dir / "paired_comparisons_matched.csv", index=False)
    by_origin.to_csv(out_dir / "by_origin_matched.csv", index=False)
    coverage.to_csv(out_dir / "coverage.csv", index=False)
    wide.to_parquet(out_dir / "triple_matched_wide.parquet", index=False)

    # Recommendation
    best = matched_df.sort_values("portfolio_wmape_pct").iloc[0]
    p_v21_a0 = paired_df.loc[(paired_df.left == "v2.1") & (paired_df.right == "a0")].iloc[0]
    p_a0_a1 = paired_df.loc[(paired_df.left == "a0") & (paired_df.right == "a1")].iloc[0]

    findings_excerpt = ""
    if findings_path is not None and Path(findings_path).is_file():
        findings_excerpt = Path(findings_path).read_text(encoding="utf-8").strip()

    run_label = export_dir.name
    report = f"""# Archval25 production-cutoff evaluation (V2.1 vs A0 vs A1)

- Export: `{run_label}`
- Contract: production cutoff (train through last complete = origin−2); deliver 15 horizons from origin
- Panel: `{export_dir.as_posix()}/forecasts_panel.*`
- Job status: {status_counts or "n/a"}{f" (failed by engine: {failed_note})" if failed_note else ""}
- Neural A0/A1: seed-ensemble = mean of available seeds (41/42/43)
- Analysis out: `{out_dir.as_posix()}`

## Coverage

{_md_table(coverage)}

Triple-matched cells (product × origin × horizon × target_date present in **all three** engines, with actuals): **{n_triple}**

## Engine summary (all available rows with actuals)

{_md_table(summary, {"mae", "rmse", "portfolio_wmape_pct", "bias", "mean_horizon_mae"})}

## Engine summary (triple-matched rows only)

Primary comparison — identical SKU×origin×horizon×target_date across V2.1 / A0 / A1.

{_md_table(matched_df, {"mae", "rmse", "portfolio_wmape_pct", "bias", "mean_horizon_mae"})}

## Paired comparisons (triple-matched)

MAE diff = left − right (negative ⇒ left better). Wins = lower abs error on that cell.

{_md_table(paired_df, {"mae_left", "mae_right", "mae_diff_left_minus_right", "wmape_left", "wmape_right"})}

## By origin (triple-matched portfolio WMAPE %)

{_md_table(by_origin.pivot_table(index="forecast_origin", columns="engine", values="portfolio_wmape_pct").reset_index(), set(ENGINES))}

## Headline

- On **{n_triple} matched cells**, best portfolio WMAPE is **{best["engine"]}** at **{best["portfolio_wmape_pct"]:.2f}%**.
- V2.1 vs A0: MAE diff (V2.1−A0) = **{p_v21_a0["mae_diff_left_minus_right"]:.1f}** on {int(p_v21_a0["n"])} cells; WMAPE {p_v21_a0["wmape_left"]:.2f}% vs {p_v21_a0["wmape_right"]:.2f}%.
- A0 vs A1: MAE diff (A0−A1) = **{p_a0_a1["mae_diff_left_minus_right"]:.1f}** on {int(p_a0_a1["n"])} cells; WMAPE {p_a0_a1["wmape_left"]:.2f}% vs {p_a0_a1["wmape_right"]:.2f}%.

## Failures / limitations (include in final report)

{findings_excerpt if findings_excerpt else "_see archval25_v21_a0_a1_report_findings.md_"}

### Eval implication

Triple-matched metrics **exclude** cells where A1 (or V2.1) failed — mostly short-history A1 at origin `140104`. That slightly favors engines that completed those cells when reading “all rows” tables; prefer **matched** tables for architecture ranking.

## Relation to prior screening report

Prior screening (`report.md`, experiment `20260908T103000Z_archval`) compared A0–A5 under screening folds (train through incomplete month). This archval25 run compares **V2.1 / A0 / A1 under the production cutoff** on the same 25-SKU × 4-origin grid. Numbers are not directly interchangeable with screening WMAPE, but A0 ≳ A1 ordering is consistent with the earlier A0 vs A1 paired result.

## Recommendation (report-only)

- Prefer **{best["engine"]}** among V2.1 / A0 / A1 on this production-cutoff matched panel.
- Keep documenting origin-`140104` short-SKU A1 ineligibility (and Xacrel V2.1) as coverage gaps, not model crashes.
"""

    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    # Pointer doc: original tree only for default out; otherwise versioned doc name.
    if out_dir.resolve() == DEFAULT_OUT.resolve():
        docs = REPO / "docs" / "archval25_v21_a0_a1_eval.md"
    else:
        docs = REPO / "docs" / f"{out_dir.name}_eval.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        f"# {out_dir.name} — V2.1 / A0 / A1 evaluation\n\n"
        f"Full report: [`{report_path.as_posix()}`]({report_path.as_posix()}).\n\n"
        f"Matched cells: {n_triple}. Best WMAPE: {best['engine']} "
        f"({best['portfolio_wmape_pct']:.2f}%).\n",
        encoding="utf-8",
    )
    print(f"wrote {report_path}")
    print(matched_df.to_string(index=False))
    print(paired_df.to_string(index=False))
    print(f"triple_matched_n={n_triple}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
