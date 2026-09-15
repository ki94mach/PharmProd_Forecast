#!/usr/bin/env python3
"""Evaluate A2–A5 late-prod vs corrected archval55 A0-fix baselines.

Compares new A2–A5 forecasts to V2.1/A0/A1 from ``archval55_export_a0fix_v1``
on identical late-origin cells. No weight search. Writes only under
``archval55_a2_a5_late_prod_eval_v1/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_A25 = RESULTS / "archval55_a2_a5_late_prod_export_v1"
DEFAULT_BASE = RESULTS / "archval55_export_a0fix_v1"
DEFAULT_OUT = RESULTS / "archval55_a2_a5_late_prod_eval_v1"
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
LATE_ORIGINS = frozenset({140301, 140402})
AX = ("a2", "a3", "a4", "a5")
BASE_ENGINES = ("v2.1", "a0", "a1")


def _load_panel(path: Path) -> pd.DataFrame:
    pq = path / "forecasts_panel.parquet"
    csv = path / "forecasts_panel.csv"
    if pq.is_file():
        df = pd.read_parquet(pq)
    elif csv.is_file():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"missing panel under {path}")
    df["product"] = df["product"].astype(str)
    df["engine"] = df["engine"].astype(str).str.lower()
    for c in ("forecast_origin", "target_date", "horizon"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
    if "ineligible" not in df.columns:
        df["ineligible"] = False
    else:
        df["ineligible"] = df["ineligible"].fillna(False).astype(bool)
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


def _attach_actuals(df: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    need = out["actual"].isna() if "actual" in out.columns else pd.Series(True, index=out.index)
    if not bool(need.any()):
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
        merged["actual"] = merged["actual"].where(
            merged["actual"].notna(), merged["actual_sales"]
        )
    return merged.drop(columns=["actual_sales"], errors="ignore")


def _seed_mean(panel: pd.DataFrame, engine: str) -> pd.DataFrame:
    sub = panel.loc[
        (panel["engine"] == engine)
        & (~panel["ineligible"])
        & (panel.get("model", pd.Series("", index=panel.index)).astype(str) != "naive_last_value")
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=KEYS + ["forecast", "actual", "engine", "n_seeds"])
    g = sub.groupby(KEYS, as_index=False).agg(
        forecast=("forecast", "mean"),
        actual=("actual", "mean"),
        n_seeds=("seed", "nunique") if "seed" in sub.columns else ("forecast", "size"),
    )
    g["engine"] = engine
    return g


def _score(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    m = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    if m.empty:
        return {
            "n_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "portfolio_wmape_pct": np.nan,
            "bias": np.nan,
            "ae_p90": np.nan,
            "ae_p95": np.nan,
        }
    err = m["forecast"] - m["actual"]
    ae = np.abs(err)
    return {
        "n_rows": int(len(m)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
        "ae_p90": float(np.quantile(ae, 0.90)),
        "ae_p95": float(np.quantile(ae, 0.95)),
    }


def _merge_on_keys(left: pd.DataFrame, right: pd.DataFrame, suffixes=("_l", "_r")) -> pd.DataFrame:
    return left.merge(right, on=KEYS, how="inner", suffixes=suffixes)


def _blend(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted blend on intersection of keys; weights must sum to 1."""
    engines = list(weights)
    base = frames[engines[0]][KEYS + ["actual"]].drop_duplicates(KEYS)
    out = base.copy()
    out["forecast"] = 0.0
    for eng, w in weights.items():
        f = frames[eng][KEYS + ["forecast"]].rename(columns={"forecast": f"f_{eng}"})
        out = out.merge(f, on=KEYS, how="inner")
        out["forecast"] = out["forecast"] + float(w) * out[f"f_{eng}"]
    return out


def _sku_origin_wmape(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (product, origin), g in df.groupby(["product", "forecast_origin"]):
        m = g.dropna(subset=["actual", "forecast"])
        if m.empty:
            continue
        rows.append(
            {
                "product": product,
                "forecast_origin": int(origin),
                "portfolio_wmape_pct": float(
                    portfolio_wmape_pct(m["actual"], m["forecast"])
                ),
                "n_rows": int(len(m)),
            }
        )
    return pd.DataFrame(rows)


def _oracle_ceilings(candidates: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Hindsight cell and SKU×origin oracle among candidate frames."""
    names = list(candidates)
    if not names:
        return {}
    base = candidates[names[0]][KEYS + ["actual"]].drop_duplicates(KEYS)
    wide = base.copy()
    for n, df in candidates.items():
        wide = wide.merge(
            df[KEYS + ["forecast"]].rename(columns={"forecast": n}),
            on=KEYS,
            how="inner",
        )
    if wide.empty:
        return {"n_cells": 0}

    def _best_row(row: pd.Series) -> float:
        act = float(row["actual"])
        best = None
        for n in names:
            if pd.isna(row[n]):
                continue
            ae = abs(float(row[n]) - act)
            if best is None or ae < best:
                best = ae
        return float(best) if best is not None else np.nan

    wide["oracle_ae"] = wide.apply(_best_row, axis=1)
    cell_oracle_mae = float(wide["oracle_ae"].mean())
    # SKU×origin: pick method with best WMAPE per group (hindsight)
    so_rows = []
    for (product, origin), g in wide.groupby(["product", "forecast_origin"]):
        best_w = None
        best_name = None
        for n in names:
            m = g.dropna(subset=["actual", n])
            if m.empty:
                continue
            w = float(portfolio_wmape_pct(m["actual"], m[n]))
            if best_w is None or w < best_w:
                best_w = w
                best_name = n
        if best_w is not None:
            so_rows.append(
                {
                    "product": product,
                    "forecast_origin": int(origin),
                    "best_method": best_name,
                    "oracle_wmape_pct": best_w,
                }
            )
    so = pd.DataFrame(so_rows)
    return {
        "n_cells": int(len(wide)),
        "cell_oracle_mae": cell_oracle_mae,
        "n_sku_origin": int(len(so)),
        "mean_sku_origin_oracle_wmape_pct": float(so["oracle_wmape_pct"].mean())
        if not so.empty
        else np.nan,
        "sku_origin_oracle": so,
    }


def _seed_dispersion(panel: pd.DataFrame, engine: str) -> dict[str, float]:
    sub = panel.loc[(panel["engine"] == engine) & (~panel["ineligible"])].copy()
    if sub.empty or "seed" not in sub.columns:
        return {"engine": engine, "n_cells": 0, "mean_seed_std": np.nan}
    g = sub.groupby(KEYS)["forecast"].agg(["std", "count"])
    g = g.loc[g["count"] >= 2]
    return {
        "engine": engine,
        "n_cells": int(len(g)),
        "mean_seed_std": float(g["std"].mean()) if not g.empty else np.nan,
        "p90_seed_std": float(g["std"].quantile(0.9)) if not g.empty else np.nan,
    }


def _error_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    m = _merge_on_keys(
        a.rename(columns={"forecast": "fa"}),
        b.rename(columns={"forecast": "fb"}),
    )
    if m.empty:
        return float("nan")
    ea = m["fa"] - m["actual"]
    eb = m["fb"] - m["actual"]
    if ea.std() == 0 or eb.std() == 0:
        return float("nan")
    return float(ea.corr(eb))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a25-export", type=Path, default=DEFAULT_A25)
    p.add_argument("--baseline-export", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a25 = _load_panel(Path(args.a25_export))
    base = _load_panel(Path(args.baseline_export))
    a25 = a25.loc[a25["forecast_origin"].isin(LATE_ORIGINS)].copy()
    base = base.loc[base["forecast_origin"].isin(LATE_ORIGINS)].copy()

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    a25 = _attach_actuals(a25, sales)
    base = _attach_actuals(base, sales)

    frames: dict[str, pd.DataFrame] = {}
    sub_v21 = base.loc[(base["engine"] == "v2.1") & (~base["ineligible"])].copy()
    frames["v2.1"] = (
        sub_v21.groupby(KEYS, as_index=False)
        .agg(forecast=("forecast", "first"), actual=("actual", "first"))
        .assign(engine="v2.1", n_seeds=1)
    )
    for eng in ("a0", "a1"):
        frames[eng] = _seed_mean(base, eng)
    for eng in AX:
        frames[eng] = _seed_mean(a25, eng)

    # Coverage
    cov_rows = []
    for eng, df in frames.items():
        cov_rows.append(
            {
                "engine": eng,
                "n_rows": int(len(df)),
                "n_products": int(df["product"].nunique()) if not df.empty else 0,
                "n_origins": int(df["forecast_origin"].nunique()) if not df.empty else 0,
            }
        )
    pd.DataFrame(cov_rows).to_csv(out / "coverage.csv", index=False)

    # Standalone metrics
    standalone = []
    for eng, df in frames.items():
        row = _score(df["actual"], df["forecast"])
        row["engine"] = eng
        standalone.append(row)
    pd.DataFrame(standalone).to_csv(out / "standalone_metrics.csv", index=False)

    # Seed dispersion (Ax + A0/A1)
    disp = [_seed_dispersion(base, "a0"), _seed_dispersion(base, "a1")]
    for eng in AX:
        disp.append(_seed_dispersion(a25, eng))
    pd.DataFrame(disp).to_csv(out / "seed_dispersion.csv", index=False)

    # Runtime if present on raw panel
    rt_rows = []
    for eng in AX:
        sub = a25.loc[a25["engine"] == eng]
        if "runtime_seconds" in sub.columns:
            rt_rows.append(
                {
                    "engine": eng,
                    "mean_runtime_s": float(sub["runtime_seconds"].mean()),
                    "n": int(sub["runtime_seconds"].notna().sum()),
                }
            )
    if rt_rows:
        pd.DataFrame(rt_rows).to_csv(out / "runtime_by_engine.csv", index=False)

    # Fixed blends
    blend_rows = []
    blend_frames: dict[str, pd.DataFrame] = {}
    for ax in AX:
        b50 = _blend(frames, {"v2.1": 0.5, ax: 0.5})
        name50 = f"v21_{ax}_blend50"
        blend_frames[name50] = b50
        r = _score(b50["actual"], b50["forecast"])
        r["method"] = name50
        blend_rows.append(r)
        b25 = _blend(frames, {"v2.1": 0.5, "a0": 0.25, ax: 0.25})
        name25 = f"v21_a0_{ax}_blend50_25_25"
        blend_frames[name25] = b25
        r2 = _score(b25["actual"], b25["forecast"])
        r2["method"] = name25
        blend_rows.append(r2)
    pd.DataFrame(blend_rows).to_csv(out / "blend_metrics.csv", index=False)

    # By horizon / origin
    by_h = []
    by_o = []
    for eng, df in {**frames, **blend_frames}.items():
        for h, g in df.groupby("horizon"):
            r = _score(g["actual"], g["forecast"])
            r.update({"method": eng, "horizon": int(h)})
            by_h.append(r)
        for o, g in df.groupby("forecast_origin"):
            r = _score(g["actual"], g["forecast"])
            r.update({"method": eng, "forecast_origin": int(o)})
            by_o.append(r)
    pd.DataFrame(by_h).to_csv(out / "metrics_by_horizon.csv", index=False)
    pd.DataFrame(by_o).to_csv(out / "metrics_by_origin.csv", index=False)

    # Error correlation matrix (seed-mean engines)
    eng_list = [e for e in list(BASE_ENGINES) + list(AX) if e in frames and not frames[e].empty]
    corr = pd.DataFrame(index=eng_list, columns=eng_list, dtype=float)
    for i in eng_list:
        for j in eng_list:
            corr.loc[i, j] = _error_corr(frames[i], frames[j]) if i != j else 1.0
    corr.to_csv(out / "error_correlation.csv")

    # Oracle ceilings among V2.1, A0, and each Ax (+ optional A1)
    oracle_summary = {}
    for ax in AX:
        cands = {
            "v2.1": frames["v2.1"],
            "a0": frames["a0"],
            ax: frames[ax],
        }
        if not frames["a1"].empty:
            cands["a1"] = frames["a1"]
        oc = _oracle_ceilings(cands)
        so = oc.pop("sku_origin_oracle", pd.DataFrame())
        if isinstance(so, pd.DataFrame) and not so.empty:
            so.to_csv(out / f"oracle_sku_origin_{ax}.csv", index=False)
        oracle_summary[ax] = oc
    (out / "oracle_ceilings.json").write_text(
        json.dumps(oracle_summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Drop-top-3 gain SKU sensitivity vs V2.1 for each Ax blend50
    sens_rows = []
    v2 = frames["v2.1"]
    v2_so = _sku_origin_wmape(v2).rename(columns={"portfolio_wmape_pct": "wmape_v2"})
    for ax in AX:
        b = blend_frames[f"v21_{ax}_blend50"]
        b_so = _sku_origin_wmape(b).rename(columns={"portfolio_wmape_pct": "wmape_blend"})
        m = v2_so.merge(b_so, on=["product", "forecast_origin"], how="inner")
        m["lift_pp"] = m["wmape_v2"] - m["wmape_blend"]
        # gainers = positive lift
        top = m.sort_values("lift_pp", ascending=False).head(3)
        drop_skus = set(top["product"])
        kept = b.loc[~b["product"].isin(drop_skus)]
        v2_kept = v2.loc[~v2["product"].isin(drop_skus)]
        # rematch keys
        common = kept.merge(
            v2_kept[KEYS + ["forecast"]].rename(columns={"forecast": "f_v2"}),
            on=KEYS,
            how="inner",
        )
        sc_b = _score(common["actual"], common["forecast"])
        sc_v = _score(common["actual"], common["f_v2"])
        sens_rows.append(
            {
                "ax": ax,
                "dropped_skus": ",".join(sorted(drop_skus)),
                "blend_wmape_full": _score(b["actual"], b["forecast"])[
                    "portfolio_wmape_pct"
                ],
                "v2_wmape_full": _score(v2["actual"], v2["forecast"])[
                    "portfolio_wmape_pct"
                ],
                "blend_wmape_drop3": sc_b["portfolio_wmape_pct"],
                "v2_wmape_drop3": sc_v["portfolio_wmape_pct"],
                "hybrid_still_beats": bool(
                    sc_b["portfolio_wmape_pct"] < sc_v["portfolio_wmape_pct"]
                )
                if sc_b["n_rows"] and sc_v["n_rows"]
                else False,
            }
        )
    pd.DataFrame(sens_rows).to_csv(out / "drop_top3_sensitivity.csv", index=False)

    # Tail errors by SKU×origin for standalone Ax
    tail_rows = []
    for eng in AX:
        so = _sku_origin_wmape(frames[eng])
        if so.empty:
            continue
        tail_rows.append(
            {
                "engine": eng,
                "n_groups": int(len(so)),
                "wmape_p90": float(so["portfolio_wmape_pct"].quantile(0.9)),
                "wmape_p95": float(so["portfolio_wmape_pct"].quantile(0.95)),
                "n_gt_50": int((so["portfolio_wmape_pct"] > 50).sum()),
                "n_gt_100": int((so["portfolio_wmape_pct"] > 100).sum()),
            }
        )
    pd.DataFrame(tail_rows).to_csv(out / "tail_errors.csv", index=False)

    report = [
        "# Archval55 A2–A5 late-prod evaluation",
        "",
        "Baseline: `archval55_export_a0fix_v1` (late origins only).",
        "Challengers: A2–A5 from `archval55_a2_a5_late_prod_export_v1`.",
        "No weight search. Fixed blends only.",
        "",
        "## Standalone WMAPE",
        "",
        pd.DataFrame(standalone).to_markdown(index=False)
        if hasattr(pd.DataFrame(standalone), "to_markdown")
        else pd.DataFrame(standalone).to_string(index=False),
        "",
        "## Fixed blends",
        "",
        pd.DataFrame(blend_rows).to_string(index=False),
        "",
        "## Drop-top-3 sensitivity (V2.1+Ax 50/50)",
        "",
        pd.DataFrame(sens_rows).to_string(index=False),
        "",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "n_a25_rows": int(len(a25)),
        "n_baseline_late_rows": int(len(base)),
        "engines_ax": list(AX),
        "standalone": standalone,
        "blends": blend_rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(out), "standalone": standalone[:6]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
