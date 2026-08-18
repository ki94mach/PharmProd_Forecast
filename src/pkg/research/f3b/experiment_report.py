"""Write docs/f3b_price.md from F3B Step 3 CSV artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.f3b.config import docs_dir, f3b_output_dir
from pkg.research.features.price import FEATURE_NAMES
from pkg.research.harness.report import md_table as _md_table
from pkg.research.harness.report import read_csv as _read
from pkg.research.harness.report import repo_relative

VERDICT_TEXT = {
    "A": (
        "A — price improves both TS and Human. Official consumer-price state "
        "known by origin is genuinely useful contextual information. Retain it "
        "as a promising candidate requiring validation on future/shadow origins."
    ),
    "B": (
        "B — price improves TS only. Retain for TS architecture as a promising "
        "candidate requiring validation on future/shadow origins."
    ),
    "C": (
        "C — price improves Human only. Retain for Human architecture as a "
        "promising candidate requiring validation on future/shadow origins."
    ),
    "D": (
        "D — pooled improvement is weak or zero, but price identifies a "
        "potentially useful regime (change magnitude or recency). Do not "
        "automatically retain it as a scored feature. Keep it as descriptive / "
        "routing research only; do not set routing rules from these five origins."
    ),
    "E": (
        "E — the current price representation fails as a scored feature and "
        "does not identify a usable regime. Reject this representation; do not "
        "tune log/clip/interactions on PRIMARY."
    ),
}

RETAIN = {
    "A": "retain for TS and Human (promising; needs future/shadow origins)",
    "B": "retain for TS only (promising; needs future/shadow origins)",
    "C": "retain for Human only (promising; needs future/shadow origins)",
    "D": "descriptive/routing research only — not a scored feature",
    "E": "rejected — close this price representation",
}


def _row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.loc[df["experiment"] == name]
    return sub.iloc[0] if len(sub) else None


def _yes_no_improve(r: Optional[pd.Series]) -> str:
    if r is None:
        return "not run"
    rel = float(r["rel_wmape_vs_control_pct"])
    if rel > 0:
        return f"yes ({rel:+.2f}% relative WMAPE vs {r['control']})"
    return f"no ({rel:+.2f}% relative WMAPE vs {r['control']})"


def _gates_passed(gates: Optional[pd.DataFrame]) -> str:
    if gates is None or gates.empty or "ok" not in gates.columns:
        return "unknown"
    return "yes" if bool(gates["ok"].all()) else "no"


def _regime_answer(regimes: Optional[pd.DataFrame]) -> str:
    if regimes is None or regimes.empty:
        return "regime table missing."
    rec = regimes.loc[
        (regimes["slice"] == "recency") & (regimes["group"] != "missing")
    ]
    chg = regimes.loc[
        (regimes["slice"] == "change_magnitude") & (regimes["group"] != "missing")
    ]
    bits = []
    if not rec.empty and "wmape_P0_TS" in rec.columns and "wmape_P1_TS" in rec.columns:
        rec = rec.copy()
        rec["d_ts"] = rec["wmape_P0_TS"] - rec["wmape_P1_TS"]
        best = rec.sort_values("d_ts", ascending=False).iloc[0]
        bits.append(
            f"Largest TS WMAPE drop among recency groups is {best['group']} "
            f"(n={int(best['n'])}, ΔWMAPE={float(best['d_ts']):+.2f})."
        )
    if not chg.empty and "wmape_P0_TS" in chg.columns and "wmape_P1_TS" in chg.columns:
        chg = chg.copy()
        chg["d_ts"] = chg["wmape_P0_TS"] - chg["wmape_P1_TS"]
        best = chg.sort_values("d_ts", ascending=False).iloc[0]
        bits.append(
            f"Largest TS WMAPE drop among change-magnitude groups is {best['group']} "
            f"(n={int(best['n'])}, ΔWMAPE={float(best['d_ts']):+.2f})."
        )
    bits.append(
        "These slices are descriptive only; they were not optimized and are not "
        "converted into a routing rule in this experiment."
    )
    return " ".join(bits) if bits else "no finite recency/change groups."


def _importance_answer(imp: Optional[pd.DataFrame]) -> str:
    if imp is None or imp.empty:
        return "feature-importance diagnostic missing."
    price = imp.loc[imp["is_price_feature"] == True]  # noqa: E712
    if price.empty:
        return "no price-feature rows in the diagnostic."
    agg = (
        price.groupby(["experiment", "feature"])
        .agg(mean_gain=("gain", "mean"), folds_used=("weight", lambda s: int((s > 0).sum())))
        .reset_index()
    )
    parts = []
    for exp, g in agg.groupby("experiment"):
        used = g.loc[g["mean_gain"] > 0].sort_values("mean_gain", ascending=False)
        if used.empty:
            parts.append(f"{exp}: none of the three price features received gain.")
            continue
        ranked = ", ".join(
            f"{r.feature} (mean gain={r.mean_gain:.4g}, folds={int(r.folds_used)})"
            for r in used.itertuples()
        )
        unused = [f for f in FEATURE_NAMES if f not in set(used["feature"])]
        extra = f"; unused: {', '.join(unused)}" if unused else ""
        parts.append(f"{exp}: {ranked}{extra}")
    parts.append("Gain is diagnostic only and did not determine the F3B verdict.")
    return " ".join(parts)


def write_gate_failure(
    message: str, *, out_dir: Optional[Path] = None, path: Optional[Path] = None
) -> Path:
    out_dir = out_dir or f3b_output_dir()
    path = path or (docs_dir() / "f3b_price.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# F3B price features\n\n"
        f"**Date:** {date.today().isoformat()}  \n"
        "**Status:** reproduction gate failed — F3B was not interpreted.\n\n"
        f"{message}\n\n"
        "Locked freeze-time benchmark contracts were not modified.\n"
    )
    path.write_text(text, encoding="utf-8")
    (out_dir / "gate_failure.txt").write_text(message, encoding="utf-8")
    return path


def write_f3b_results(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = out_dir or report.get("out_dir") or f3b_output_dir()
    path = path or (docs_dir() / "f3b_price.md")
    overall = report.get("overall")
    if overall is None:
        overall = _read(out_dir, "overall.csv")
    by_o = report.get("by_origin")
    if by_o is None:
        by_o = _read(out_dir, "by_origin.csv")
    by_p = report.get("by_product")
    if by_p is None:
        by_p = _read(out_dir, "by_product.csv")
    by_h = report.get("by_horizon")
    if by_h is None:
        by_h = _read(out_dir, "by_horizon.csv")
    conc = report.get("error_concentration")
    if conc is None:
        conc = _read(out_dir, "error_concentration.csv")
    regimes = report.get("price_regime_analysis")
    if regimes is None:
        regimes = _read(out_dir, "price_regime_analysis.csv")
    watch = report.get("watchlist")
    if watch is None:
        watch = _read(out_dir, "high_volume_watchlist.csv")
    gates = report.get("gates")
    if gates is None:
        gates = _read(out_dir, "reproduction_gates.csv")
    imp = report.get("feature_importance")
    if imp is None:
        imp = _read(out_dir, "feature_importance.csv")
    verdict = report.get("verdict")
    if not verdict:
        vdf = _read(out_dir, "verdict.csv")
        verdict = str(vdf["verdict"].iloc[0]) if vdf is not None and len(vdf) else "E"

    p1t = _row(overall, "P1_TS") if overall is not None else None
    p1h = _row(overall, "P1_HUMAN") if overall is not None else None

    q1 = _gates_passed(gates)
    q2 = _yes_no_improve(p1t)
    q3 = _yes_no_improve(p1h)
    q4 = "n/a"
    if p1t is not None and p1h is not None:
        q4 = (
            f"P1_TS {float(p1t['rel_wmape_vs_control_pct']):+.2f}% vs P0_TS; "
            f"P1_HUMAN {float(p1h['rel_wmape_vs_control_pct']):+.2f}% vs P0_HUMAN"
        )
    q5 = "n/a"
    if p1t is not None:
        q5 = f"P1_TS {int(p1t['origins_improved'])}/{int(p1t['origins_total'])} origins"
    if p1h is not None:
        q5 += f"; P1_HUMAN {int(p1h['origins_improved'])}/{int(p1h['origins_total'])} origins"
    q6 = ""
    if p1t is not None:
        q6 = f"P1_TS product win rate={float(p1t['product_win_rate'])*100:.1f}%"
    if p1h is not None:
        q6 += f"; P1_HUMAN product win rate={float(p1h['product_win_rate'])*100:.1f}%"
    q7 = ""
    if p1t is not None:
        q7 = (
            f"P1_TS median={float(p1t['median_product_improvement_pct']):+.2f}% "
            f"(p25={float(p1t['p25_product_improvement_pct']):+.2f}, "
            f"p75={float(p1t['p75_product_improvement_pct']):+.2f})"
        )
    if p1h is not None:
        q7 += (
            f"; P1_HUMAN median={float(p1h['median_product_improvement_pct']):+.2f}% "
            f"(p25={float(p1h['p25_product_improvement_pct']):+.2f}, "
            f"p75={float(p1h['p75_product_improvement_pct']):+.2f})"
        )
    q8 = "see error concentration and high-volume watchlist"
    if p1t is not None:
        q8 = (
            f"P1_TS top1/top5/top10 deterioration share="
            f"{float(p1t['top1_deterioration_share'])*100:.1f}%/"
            f"{float(p1t['top5_deterioration_share'])*100:.1f}%/"
            f"{float(p1t['top10_deterioration_share'])*100:.1f}%."
        )
    if p1h is not None:
        q8 += (
            f" P1_HUMAN top1/top5/top10="
            f"{float(p1h['top1_deterioration_share'])*100:.1f}%/"
            f"{float(p1h['top5_deterioration_share'])*100:.1f}%/"
            f"{float(p1h['top10_deterioration_share'])*100:.1f}%."
        )
    q11 = RETAIN.get(verdict, verdict)
    q12 = (
        "No — this representation should not be treated as production-ready; "
        "close it and do not tune on PRIMARY."
        if verdict == "E"
        else (
            "Yes as a research candidate only: if retained, confirm on future/"
            "shadow origins. This PRIMARY panel was already used for F1/F2/"
            "ablation/F3A, so F3B is not an unbiased production estimate."
        )
    )

    answers = [
        f"1. **Did the F0 reproduction gates pass?** {q1}",
        f"2. **Does F3B improve TS F0?** {q2}",
        f"3. **Does F3B improve Human F0?** {q3}",
        f"4. **How large is the relative WMAPE change for each?** {q4}",
        f"5. **On how many of 5 origins does each improve?** {q5}",
        f"6. **What percentage of products improve?** {q6}",
        f"7. **What are the median product-level effects?** {q7}",
        f"8. **Are losses concentrated in high-volume products?** {q8}",
        "9. **Does price appear more useful after recent/larger price changes?** "
        + _regime_answer(regimes),
        "10. **Which of the three price features does XGBoost actually use?** "
        + _importance_answer(imp),
        f"11. **Should F3B be retained for TS, Human, both, descriptive/routing research, or rejected?** {q11}",
        f"12. **If promising, does it justify confirmation on future/shadow origins?** {q12}",
    ]

    overall_cols = [
        "experiment",
        "anchor",
        "control",
        "n_features",
        "wmape",
        "rel_wmape_vs_control_pct",
        "rmse",
        "mae",
        "bias",
        "n",
        "origins_improved",
        "origins_total",
        "median_origin_improvement_pct",
        "product_win_rate",
        "median_product_improvement_pct",
        "p25_product_improvement_pct",
        "p75_product_improvement_pct",
        "top1_deterioration_share",
        "top5_deterioration_share",
        "top10_deterioration_share",
    ]
    cand_origin = None
    if by_o is not None and "experiment" in by_o.columns:
        cand_origin = by_o.loc[by_o["experiment"].isin(["P1_TS", "P1_HUMAN"])]
    cand_prod = None
    if by_p is not None and "experiment" in by_p.columns:
        cand_prod = by_p.loc[by_p["experiment"].isin(["P1_TS", "P1_HUMAN"])]
    cand_h = None
    if by_h is not None and "experiment" in by_h.columns:
        cand_h = by_h.loc[by_h["experiment"].isin(["P1_TS", "P1_HUMAN"])]
    cand_watch = None
    if watch is not None and "experiment" in watch.columns:
        cand_watch = watch.loc[watch["experiment"].isin(["P1_TS", "P1_HUMAN"])]
    price_imp = None
    if imp is not None and not imp.empty and "is_price_feature" in imp.columns:
        price_imp = imp.loc[imp["is_price_feature"] == True]  # noqa: E712

    sections = [
        "# F3B — Point-in-time consumer price\n",
        f"**Date:** {date.today().isoformat()}  \n",
        "**Benchmark:** frozen v1 matched PRIMARY (n=1877, 5 origins)  \n",
        f"**CSV artifacts:** `{repo_relative(Path(out_dir))}`\n",
        "\n",
        "F3B is another hypothesis evaluated on an already reused research test panel "
        "(F1, F2, feature-family ablation, F3A). Results are useful for research "
        "direction but should not be treated as unbiased production estimates.\n",
        "\n",
        "## Hypothesis\n\n",
        "Official consumer-price state known by forecast origin may explain residual "
        "error beyond F0. Features describe the last known official price, not a "
        "future planned change.\n",
        "\n",
        "## Frozen scored features (exactly three, predefined before evaluation)\n\n",
        "- `log_consumer_price_asof_origin` = `log1p(consumer_price_asof_origin)`.\n",
        "- `last_consumer_price_change_pct` on the two most recent distinct consumer-price states "
        "with `effective_month < origin`. Missing = NaN, not 0.\n",
        "- `months_since_last_consumer_price_change` via `shamsi_month_diff`.\n",
        "\nA price is visible iff `effective_month < origin`. Origin-month and later prices are invisible.\n",
        "\nNo raw consumer price, distributor/pharmacy, direction indicator, future/target-month "
        "price, inflation, interactions, caps, or alternative transforms were added. Features "
        "were not tuned on PRIMARY.\n",
        "\n## Reproduction gates\n\n",
        _md_table(gates) if gates is not None else "_missing_\n",
        "Locked freeze-time Analysis B WMAPEs were not rewritten. Controls are the "
        "currently reproduced F0 backtest (`P0_TS` / `P0_HUMAN`). This environment "
        "matches the locked Analysis B F0 (TS 37.23014, Human 36.69475); it is not "
        "the F3A-era drifted XGB environment (38.2848 / 36.5602).\n",
        "\n## Scoreboard\n\n",
        _md_table(overall, max_rows=8, cols=overall_cols) if overall is not None else "_missing_\n",
        "\nP1_TS is compared to P0_TS; P1_HUMAN to P0_HUMAN. Positive relative WMAPE means "
        "the price features helped. Promotion is not based on pooled WMAPE alone.\n",
        "\n## Verdict\n\n",
        f"**Case {verdict}.** {VERDICT_TEXT.get(verdict, '').split(' — ', 1)[-1]}\n",
        "\n## Answers\n\n",
    ]
    sections.extend(ln + "\n\n" for ln in answers)
    sections += [
        "## By origin (candidates)\n\n",
        _md_table(cand_origin, max_rows=20) if cand_origin is not None else "_missing_\n",
        "\n## By horizon bucket\n\n",
        _md_table(cand_h, max_rows=24) if cand_h is not None else "_missing_\n",
        "\n## Error concentration\n\n",
        _md_table(conc) if conc is not None else "_missing_\n",
        "\n## High-volume watchlist\n\n",
        _md_table(cand_watch, max_rows=40) if cand_watch is not None else "_missing_\n",
        "\n## Product robustness (P1_TS and P1_HUMAN; top by volume)\n\n",
        _md_table(cand_prod, max_rows=20) if cand_prod is not None else "_missing_\n",
        "\n## Price-regime slices (descriptive only)\n\n",
        "Quartile bins are taken from the pre-model PRIMARY distribution of "
        "`last_consumer_price_change_pct` and `months_since_last_consumer_price_change`. "
        "Direction is `>0` / `=0` / `<0`. These are **not** model features and were not "
        "optimized against WMAPE.\n\n",
        _md_table(regimes, max_rows=40) if regimes is not None else "_missing_\n",
        "\n## XGBoost feature gain (diagnostic, not promotion)\n\n",
        _md_table(price_imp, max_rows=40) if price_imp is not None else "_missing_\n",
        "\n## What was not done\n\n",
        "No CORE_TS variants, no F3A lifecycle, no F1/F2 rejected features, no SHAP, "
        "no XGB tuning, no clip/inflation/distributor/pharmacy/interactions. "
        "Step 1 source parquet and Step 2 feature definitions were not modified. "
        "The next feature family was not started.\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
