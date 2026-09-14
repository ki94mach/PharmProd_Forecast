#!/usr/bin/env python3
"""Fixed 50/50 V2.1+neural blend evaluation on archval25 pairwise panels.

Each neural (A0, A1) gets its own V2.1 ∩ neural matched panel so A1 failures
do not remove valid V2.1+A0 rows. No refits; no weight optimization.
"""
from __future__ import annotations

import argparse
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
    / "archval25_blends"
)
SALES = REPO / "src" / "data" / "benchmarks" / "v1" / "raw" / "sales.parquet"

KEYS = ["product", "forecast_origin", "horizon", "target_date"]
NEURALS = ("a0", "a1")
TOP_N_VOLUME = 10


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
        cells: list[str] = []
        for c, v in zip(cols, row):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif c in float_cols or isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.4g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def attach_actuals(df: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    need = (
        out["actual"].isna()
        if "actual" in out.columns
        else pd.Series(True, index=out.index)
    )
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


def seed_ensemble(neural: pd.DataFrame) -> pd.DataFrame:
    return neural.groupby(KEYS + ["engine"], as_index=False).agg(
        forecast=("forecast", "mean"),
        actual=("actual", "mean"),
        n_seeds=("seed", "nunique"),
    )


def load_engine_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    v21 = panel.loc[panel["engine"] == "v2.1"].copy()
    frames["v2.1"] = (
        v21.groupby(KEYS, as_index=False)
        .agg(forecast=("forecast", "first"), actual=("actual", "first"))
        .assign(engine="v2.1")
    )
    for eng in NEURALS:
        sub = panel.loc[panel["engine"] == eng].copy()
        frames[eng] = seed_ensemble(sub).assign(engine=eng)
    return frames


def build_pairwise_panel(
    v21: pd.DataFrame, neural: pd.DataFrame, neural_name: str
) -> pd.DataFrame:
    a = v21[KEYS + ["forecast", "actual"]].rename(
        columns={"forecast": "f_v21", "actual": "actual_v21"}
    )
    b = neural[KEYS + ["forecast", "actual"]].rename(
        columns={"forecast": "f_neural", "actual": "actual_neural"}
    )
    m = a.merge(b, on=KEYS, how="inner")
    m["actual"] = m["actual_v21"].where(m["actual_v21"].notna(), m["actual_neural"])
    m = m.dropna(subset=["actual", "f_v21", "f_neural"]).copy()
    m["f_blend50"] = 0.5 * m["f_v21"] + 0.5 * m["f_neural"]
    m["ae_v21"] = (m["f_v21"] - m["actual"]).abs()
    m["ae_neural"] = (m["f_neural"] - m["actual"]).abs()
    m["ae_blend50"] = (m["f_blend50"] - m["actual"]).abs()
    # Cell oracle: pick better of V2.1 vs neural per row
    use_v21 = m["ae_v21"] <= m["ae_neural"]
    m["f_cell_oracle"] = np.where(use_v21, m["f_v21"], m["f_neural"])
    m["ae_cell_oracle"] = np.where(use_v21, m["ae_v21"], m["ae_neural"])
    m["cell_oracle_source"] = np.where(use_v21, "v2.1", neural_name)
    m["neural"] = neural_name
    m["volume"] = m["actual"].abs()
    return m.drop(columns=["actual_v21", "actual_neural"], errors="ignore")


def add_sku_origin_oracle(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    """Per (product, origin), pick engine with better group WMAPE; broadcast forecast."""
    out = panel.copy()
    choices: list[pd.DataFrame] = []
    for (prod, origin), g in out.groupby(["product", "forecast_origin"], sort=False):
        w_v = portfolio_wmape_pct(g["actual"], g["f_v21"])
        w_n = portfolio_wmape_pct(g["actual"], g["f_neural"])
        pick = "v2.1" if w_v <= w_n else neural_name
        part = g[KEYS].copy()
        part["sku_origin_oracle_source"] = pick
        part["f_sku_origin_oracle"] = np.where(
            pick == "v2.1", g["f_v21"].to_numpy(), g["f_neural"].to_numpy()
        )
        choices.append(part)
    if not choices:
        out["sku_origin_oracle_source"] = "v2.1"
        out["f_sku_origin_oracle"] = out["f_v21"]
        return out
    ch = pd.concat(choices, ignore_index=True)
    out = out.merge(ch, on=KEYS, how="left")
    out["ae_sku_origin_oracle"] = (out["f_sku_origin_oracle"] - out["actual"]).abs()
    return out


def score_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    m = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    if m.empty:
        return {
            "n_rows": 0.0,
            "mae": float("nan"),
            "portfolio_wmape_pct": float("nan"),
            "bias": float("nan"),
        }
    err = m["forecast"] - m["actual"]
    return {
        "n_rows": float(len(m)),
        "mae": float(np.mean(np.abs(err))),
        "portfolio_wmape_pct": float(portfolio_wmape_pct(m["actual"], m["forecast"])),
        "bias": float(np.mean(err)),
    }


def headline_block(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    rows = []
    for label, col in (
        ("v2.1", "f_v21"),
        (neural_name, "f_neural"),
        ("blend50", "f_blend50"),
        ("cell_oracle", "f_cell_oracle"),
        ("sku_origin_oracle", "f_sku_origin_oracle"),
    ):
        s = score_forecast(panel["actual"], panel[col])
        s["method"] = label
        s["neural"] = neural_name
        s["n_panel"] = len(panel)
        s["blend_wins_vs_v21"] = (
            int((panel["ae_blend50"] < panel["ae_v21"]).sum())
            if label == "blend50"
            else np.nan
        )
        rows.append(s)
    return pd.DataFrame(rows)


def by_origin(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    rows = []
    for origin, g in panel.groupby("forecast_origin", sort=True):
        for method, col in (
            ("v2.1", "f_v21"),
            (neural_name, "f_neural"),
            ("blend50", "f_blend50"),
        ):
            s = score_forecast(g["actual"], g[col])
            s["forecast_origin"] = int(origin)
            s["method"] = method
            s["neural"] = neural_name
            rows.append(s)
    return pd.DataFrame(rows)


def by_sku(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    rows = []
    for product, g in panel.groupby("product", sort=True):
        s_v = score_forecast(g["actual"], g["f_v21"])
        s_b = score_forecast(g["actual"], g["f_blend50"])
        s_n = score_forecast(g["actual"], g["f_neural"])
        vol = float(g["volume"].sum())
        rows.append(
            {
                "neural": neural_name,
                "product": product,
                "n_rows": int(len(g)),
                "volume": vol,
                "wmape_v21": s_v["portfolio_wmape_pct"],
                "wmape_neural": s_n["portfolio_wmape_pct"],
                "wmape_blend50": s_b["portfolio_wmape_pct"],
                "delta_wmape_blend_minus_v21": s_b["portfolio_wmape_pct"]
                - s_v["portfolio_wmape_pct"],
                "ae_sum_v21": float(g["ae_v21"].sum()),
                "ae_sum_blend50": float(g["ae_blend50"].sum()),
                "ae_reduction_vs_v21": float(g["ae_v21"].sum() - g["ae_blend50"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("volume", ascending=False).reset_index(drop=True)


def high_volume(sku_df: pd.DataFrame, panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    top = sku_df.head(TOP_N_VOLUME).copy()
    total_ae_v21 = float(panel["ae_v21"].sum())
    total_ae_red = float((panel["ae_v21"] - panel["ae_blend50"]).sum())
    total_vol = float(panel["volume"].sum())
    top["share_of_panel_volume"] = top["volume"] / total_vol if total_vol else np.nan
    top["share_of_v21_abs_error"] = (
        top["ae_sum_v21"] / total_ae_v21 if total_ae_v21 else np.nan
    )
    top["share_of_ae_reduction_vs_v21"] = (
        top["ae_reduction_vs_v21"] / total_ae_red if total_ae_red else np.nan
    )
    top["neural"] = neural_name
    return top


def loo_origins(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    rows = []
    origins = sorted(panel["forecast_origin"].unique().tolist())
    for dropped in origins:
        g = panel.loc[panel["forecast_origin"] != dropped]
        s_v = score_forecast(g["actual"], g["f_v21"])
        s_b = score_forecast(g["actual"], g["f_blend50"])
        rows.append(
            {
                "neural": neural_name,
                "dropped_origin": int(dropped),
                "n_rows": int(len(g)),
                "wmape_v21": s_v["portfolio_wmape_pct"],
                "wmape_blend50": s_b["portfolio_wmape_pct"],
                "delta_wmape_blend_minus_v21": s_b["portfolio_wmape_pct"]
                - s_v["portfolio_wmape_pct"],
                "blend_beats_v21": bool(
                    s_b["portfolio_wmape_pct"] < s_v["portfolio_wmape_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def loo_skus(panel: pd.DataFrame, neural_name: str) -> pd.DataFrame:
    rows = []
    products = sorted(panel["product"].unique().tolist())
    for dropped in products:
        g = panel.loc[panel["product"] != dropped]
        s_v = score_forecast(g["actual"], g["f_v21"])
        s_b = score_forecast(g["actual"], g["f_blend50"])
        rows.append(
            {
                "neural": neural_name,
                "dropped_product": dropped,
                "n_rows": int(len(g)),
                "wmape_v21": s_v["portfolio_wmape_pct"],
                "wmape_blend50": s_b["portfolio_wmape_pct"],
                "delta_wmape_blend_minus_v21": s_b["portfolio_wmape_pct"]
                - s_v["portfolio_wmape_pct"],
                "blend_beats_v21": bool(
                    s_b["portfolio_wmape_pct"] < s_v["portfolio_wmape_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def recommend(results: dict[str, dict]) -> str:
    """Apply plan recommendation rule."""
    beats = {
        n: r["headline"].loc[r["headline"]["method"] == "blend50"].iloc[0]
        for n, r in results.items()
    }
    lifts = {}
    for n, row in beats.items():
        v21 = results[n]["headline"].loc[
            results[n]["headline"]["method"] == "v2.1"
        ].iloc[0]["portfolio_wmape_pct"]
        lifts[n] = float(v21 - row["portfolio_wmape_pct"])  # positive => blend better

    winners = [n for n, lift in lifts.items() if lift > 0]
    lines = ["## Pilot recommendation (fixed 50/50; report-only)", ""]

    if not winners:
        lines.extend(
            [
                "**No neural pilot for fixed 50/50 blending.** Neither V2.1+A0 nor "
                "V2.1+A1 beats pure V2.1 on portfolio WMAPE on its pairwise panel.",
                "",
                "Oracle gaps (cell / SKU×origin) remain research signals for *routing*, "
                "not a reason to pilot a flat 50/50 mix.",
            ]
        )
        for n, lift in lifts.items():
            h = beats[n]
            v21 = results[n]["headline"].loc[
                results[n]["headline"]["method"] == "v2.1"
            ].iloc[0]["portfolio_wmape_pct"]
            lines.append(
                f"- V2.1+{n.upper()}: blend WMAPE={h['portfolio_wmape_pct']:.2f}% "
                f"vs V2.1={v21:.2f}% (lift={lift:+.2f} pp), n={int(h['n_rows'])}"
            )
        return "\n".join(lines)

    # LOO checks
    def loo_ok(n: str) -> tuple[bool, str]:
        lo = results[n]["loo_origin"]
        ls = results[n]["loo_sku"]
        origin_ok = int(lo["blend_beats_v21"].sum())
        origin_n = int(len(lo))
        sku_ok = int(ls["blend_beats_v21"].sum())
        sku_n = int(len(ls))
        ok = origin_ok >= max(3, origin_n - 1) and sku_ok > sku_n / 2
        detail = (
            f"LOO origins blend-wins {origin_ok}/{origin_n}; "
            f"SKU holdouts {sku_ok}/{sku_n}"
        )
        return ok, detail

    # Prefer larger lift; coverage tie-break toward a0 if lifts similar
    ranked = sorted(winners, key=lambda n: (-lifts[n], 0 if n == "a0" else 1))
    pick = ranked[0]
    ok, detail = loo_ok(pick)
    lines.append(
        f"**Larger pilot candidate: V2.1 + {pick.upper()}** "
        f"(fixed 50/50) - WMAPE lift vs V2.1 = **{lifts[pick]:+.2f} pp** "
        f"on n={int(beats[pick]['n_rows'])} pairwise cells."
    )
    lines.append("")
    lines.append(f"Sensitivity: {detail}.")
    if not ok:
        lines.append(
            "Caution: LOO stability is weak; treat as a small pilot only."
        )
    if len(winners) > 1:
        other = [n for n in winners if n != pick]
        lines.append(
            "Also beats V2.1: "
            + ", ".join(f"{n.upper()} (lift {lifts[n]:+.2f} pp)" for n in other)
            + "."
        )
    # Coverage note
    n0 = int(results["a0"]["panel_n"])
    n1 = int(results["a1"]["panel_n"])
    lines.append(
        f"Coverage: A0 pairwise n={n0}, A1 pairwise n={n1} "
        "(A1 gaps at early origins do not shrink the A0 panel)."
    )
    return "\n".join(lines)


def evaluate_one(
    frames: dict[str, pd.DataFrame], neural_name: str
) -> dict:
    panel = build_pairwise_panel(frames["v2.1"], frames[neural_name], neural_name)
    panel = add_sku_origin_oracle(panel, neural_name)
    head = headline_block(panel, neural_name)
    # lift vs v21 for blend
    v21_w = float(head.loc[head["method"] == "v2.1", "portfolio_wmape_pct"].iloc[0])
    for method in ("blend50", "cell_oracle", "sku_origin_oracle", neural_name):
        mask = head["method"] == method
        head.loc[mask, "lift_vs_v21_pp"] = (
            v21_w - head.loc[mask, "portfolio_wmape_pct"]
        )
    origin = by_origin(panel, neural_name)
    sku = by_sku(panel, neural_name)
    hv = high_volume(sku, panel, neural_name)
    lo = loo_origins(panel, neural_name)
    ls = loo_skus(panel, neural_name)
    return {
        "panel": panel,
        "headline": head,
        "by_origin": origin,
        "by_sku": sku,
        "high_volume": hv,
        "loo_origin": lo,
        "loo_sku": ls,
        "panel_n": len(panel),
        "blend_cell_wins": int((panel["ae_blend50"] < panel["ae_v21"]).sum()),
    }


def write_report(
    results: dict[str, dict],
    recommendation: str,
    *,
    export_dir: Path,
    out_dir: Path,
) -> str:
    sections: list[str] = [
        "# Archval25 fixed 50/50 blends (V2.1 + neural)",
        "",
        f"- Source: `{export_dir.name}` predictions (production cutoff).",
        "- Blends: `0.5 * V2.1 + 0.5 * neural` (A0 and A1 separately).",
        "- Pairwise panels: V2.1 ∩ neural only (A1 failures do not drop V2.1+A0 rows).",
        "- No refits; no weight search.",
        f"- Artifacts: `{out_dir.as_posix()}`",
        "",
    ]

    for neural_name in NEURALS:
        r = results[neural_name]
        head = r["headline"]
        sections.extend(
            [
                f"## V2.1 + {neural_name.upper()} (n={r['panel_n']})",
                "",
                f"Blend abs-error wins vs V2.1: **{r['blend_cell_wins']}/{r['panel_n']}** cells.",
                "",
                "### Headline",
                "",
                _md_table(
                    head[
                        [
                            "method",
                            "n_rows",
                            "mae",
                            "portfolio_wmape_pct",
                            "bias",
                            "lift_vs_v21_pp",
                        ]
                    ],
                    {"mae", "portfolio_wmape_pct", "bias", "lift_vs_v21_pp"},
                ),
                "",
                "### By origin (portfolio WMAPE %)",
                "",
            ]
        )
        piv = (
            r["by_origin"]
            .pivot_table(
                index="forecast_origin",
                columns="method",
                values="portfolio_wmape_pct",
                aggfunc="first",
            )
            .reset_index()
        )
        sections.append(_md_table(piv, set(piv.columns) - {"forecast_origin"}))
        sections.extend(
            [
                "",
                f"### High-volume SKUs (top {TOP_N_VOLUME} by |actual|)",
                "",
                _md_table(
                    r["high_volume"][
                        [
                            "product",
                            "volume",
                            "wmape_v21",
                            "wmape_blend50",
                            "delta_wmape_blend_minus_v21",
                            "share_of_panel_volume",
                            "share_of_ae_reduction_vs_v21",
                        ]
                    ],
                    {
                        "volume",
                        "wmape_v21",
                        "wmape_blend50",
                        "delta_wmape_blend_minus_v21",
                        "share_of_panel_volume",
                        "share_of_ae_reduction_vs_v21",
                    },
                ),
                "",
                "### Leave-one-origin-out",
                "",
                _md_table(
                    r["loo_origin"][
                        [
                            "dropped_origin",
                            "n_rows",
                            "wmape_v21",
                            "wmape_blend50",
                            "delta_wmape_blend_minus_v21",
                            "blend_beats_v21",
                        ]
                    ],
                    {
                        "wmape_v21",
                        "wmape_blend50",
                        "delta_wmape_blend_minus_v21",
                    },
                ),
                "",
                "### Leave-one-SKU-out (summary)",
                "",
            ]
        )
        ls = r["loo_sku"]
        sections.append(
            f"- Blend beats V2.1 on **{int(ls['blend_beats_v21'].sum())}/{len(ls)}** "
            f"SKU holdouts; median ΔWMAPE (blend−V2.1) = "
            f"**{float(ls['delta_wmape_blend_minus_v21'].median()):+.3g}** pp."
        )
        sections.append("")

    sections.extend([recommendation, ""])
    sections.extend(
        [
            "## Notes",
            "",
            "- Cell oracle = per-row pick of V2.1 vs neural (lower abs error).",
            "- SKU×origin oracle = per (product, origin) pick by group WMAPE.",
            "- Positive `lift_vs_v21_pp` means lower WMAPE than V2.1 (better).",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = export_dir / "forecasts_panel.parquet"
    if not panel_path.is_file():
        panel_path = export_dir / "forecasts_panel.csv"
    panel = (
        pd.read_parquet(panel_path)
        if panel_path.suffix == ".parquet"
        else pd.read_csv(panel_path)
    )
    panel["product"] = panel["product"].astype(str)
    for c in ("forecast_origin", "target_date", "horizon"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(int)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    panel["engine"] = panel["engine"].astype(str).str.lower()

    sales = pd.read_parquet(SALES)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    panel = attach_actuals(panel, sales)

    frames = load_engine_frames(panel)
    results: dict[str, dict] = {}
    headline_all: list[pd.DataFrame] = []

    for neural_name in NEURALS:
        r = evaluate_one(frames, neural_name)
        results[neural_name] = r
        r["panel"].to_parquet(out_dir / f"panel_v21_{neural_name}.parquet", index=False)
        r["headline"].to_csv(out_dir / f"headline_v21_{neural_name}.csv", index=False)
        r["by_origin"].to_csv(out_dir / f"by_origin_v21_{neural_name}.csv", index=False)
        r["by_sku"].to_csv(out_dir / f"by_sku_v21_{neural_name}.csv", index=False)
        r["high_volume"].to_csv(out_dir / f"high_volume_v21_{neural_name}.csv", index=False)
        r["loo_origin"].to_csv(out_dir / f"loo_origin_v21_{neural_name}.csv", index=False)
        r["loo_sku"].to_csv(out_dir / f"loo_sku_v21_{neural_name}.csv", index=False)
        headline_all.append(r["headline"])

    pd.concat(headline_all, ignore_index=True).to_csv(out_dir / "headline.csv", index=False)

    recommendation = recommend(results)
    report = write_report(
        results, recommendation, export_dir=export_dir, out_dir=out_dir
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"wrote {report_path}")
    for n in NEURALS:
        h = results[n]["headline"]
        print(f"--- {n} n={results[n]['panel_n']} ---")
        print(
            h[
                [
                    "method",
                    "portfolio_wmape_pct",
                    "lift_vs_v21_pp",
                    "mae",
                ]
            ].to_string(index=False)
        )
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
