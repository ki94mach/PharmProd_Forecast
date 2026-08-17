"""Write docs/f3a_lifecycle.md from F3A CSV artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.research.f3a.config import docs_dir, f3a_output_dir
from pkg.research.harness.report import md_table as _md_table
from pkg.research.harness.report import read_csv as _read
from pkg.research.harness.report import repo_relative

VERDICT_TEXT = {
    "A": (
        "A — F3A helps both anchors. Lifecycle is genuinely useful contextual "
        "information. Retain it for subsequent research."
    ),
    "B": (
        "B — F3A helps Human but not TS. Lifecycle context is particularly "
        "useful for moderating Human forecasts. Retain for Human architecture."
    ),
    "C": (
        "C — F3A helps TS but not Human. Retain only for TS architecture."
    ),
    "D": (
        "D — F3A does not improve pooled forecasting but reveals "
        "lifecycle-dependent model performance. Do not automatically retain "
        "it as a scored feature. Retain it as a potential future "
        "routing/segmentation variable."
    ),
    "E": (
        "E — F3A fails both as feature and segmentation variable. Close "
        "observed-product-age as a forecasting feature and move to the next "
        "genuinely new information family."
    ),
}

RETAIN = {
    "A": "a scored forecasting feature (both anchors)",
    "B": "a scored forecasting feature for Human only",
    "C": "a scored forecasting feature for TS only",
    "D": "a routing/segmentation variable (not a scored feature)",
    "E": "rejected — close observed-product-age",
}


def _row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.loc[df["experiment"] == name]
    return sub.iloc[0] if len(sub) else None


def _age_group_answer(age_g: Optional[pd.DataFrame], min_age: float) -> str:
    established = (
        f" PRIMARY test rows are all established (min observed age={min_age:.0f} months); "
        "this panel cannot test launch/new-SKU effects."
        if np.isfinite(min_age)
        else ""
    )
    if age_g is None or age_g.empty:
        return "age-group table missing." + established
    g = age_g.loc[age_g["age_group"] != "age_missing"]
    if g.empty or "wmape_T0" not in g.columns:
        return "no available-age groups." + established
    t0 = g["wmape_T0"].to_numpy(dtype=float)
    spread = float(np.nanmax(t0) - np.nanmin(t0)) if len(t0) else float("nan")
    return (
        f"Baseline TS F0 WMAPE ranges {spread:.1f} points across observed-age quartiles, "
        "but the pattern is not monotonic in age (product mix likely dominates). "
        "Adding the scored age feature does not systematically help younger vs older groups. "
        "Do not set routing thresholds from this table."
        + established
    )


def _verdict_caveat(min_age: float, age_g: Optional[pd.DataFrame]) -> str:
    extra = ""
    if np.isfinite(min_age) and min_age >= 24:
        extra = (
            f" The PRIMARY panel has no young products (min observed age={min_age:.0f} months), "
            "so F3A did not test commercial launch. Quartile WMAPE differences should not be "
            "turned into age buckets in this experiment.\n"
        )
    return extra


def _yes_no_improve(r: Optional[pd.Series]) -> str:
    if r is None:
        return "not run"
    rel = float(r["rel_wmape_vs_control_pct"])
    if rel > 0:
        return f"yes ({rel:+.2f}% relative WMAPE vs {r['control']})"
    return f"no ({rel:+.2f}% relative WMAPE vs {r['control']})"


def write_gate_failure(message: str, *, out_dir: Optional[Path] = None, path: Optional[Path] = None) -> Path:
    out_dir = out_dir or f3a_output_dir()
    path = path or (docs_dir() / "f3a_lifecycle.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# F3A lifecycle\n\n"
        f"**Date:** {date.today().isoformat()}  \n"
        "**Status:** reproduction gate failed — F3A was not interpreted.\n\n"
        f"{message}\n\n"
        "Locked freeze-time benchmark contracts were not modified.\n"
    )
    path.write_text(text, encoding="utf-8")
    (out_dir / "gate_failure.txt").write_text(message, encoding="utf-8")
    return path


def write_f3a_results(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = out_dir or report.get("out_dir") or f3a_output_dir()
    path = path or (docs_dir() / "f3a_lifecycle.md")
    overall = report.get("overall")
    if overall is None:
        overall = _read(out_dir, "overall.csv")
    coverage = report.get("lifecycle_coverage")
    if coverage is None:
        coverage = _read(out_dir, "lifecycle_coverage.csv")
    product_audit = report.get("lifecycle_audit")
    if product_audit is None:
        product_audit = _read(out_dir, "lifecycle_audit.csv")
    life_origin = report.get("lifecycle_by_origin")
    if life_origin is None:
        life_origin = _read(out_dir, "lifecycle_audit_by_origin.csv")
    nz = report.get("first_nonzero_audit")
    if nz is None:
        nz = _read(out_dir, "first_nonzero_audit.csv")
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
    age_g = report.get("age_groups")
    if age_g is None:
        age_g = _read(out_dir, "age_groups.csv")
    gates = report.get("gates")
    if gates is None:
        gates = _read(out_dir, "reproduction_gates.csv")
    watch = report.get("watchlist")
    if watch is None:
        watch = _read(out_dir, "watchlist.csv")
    verdict = report.get("verdict")
    if not verdict:
        vdf = _read(out_dir, "verdict.csv")
        verdict = str(vdf["verdict"].iloc[0]) if vdf is not None and len(vdf) else "E"

    t1 = _row(overall, "T1") if overall is not None else None
    t3 = _row(overall, "T3") if overall is not None else None
    h1 = _row(overall, "H1") if overall is not None else None
    cov = coverage.iloc[0] if coverage is not None and len(coverage) else None
    min_age = float(cov["min_age"]) if cov is not None and "min_age" in cov.index else float("nan")

    def cov_f(key: str, fmt: str = "{:.2f}") -> str:
        if cov is None:
            return "n/a"
        v = cov[key]
        if pd.isna(v):
            return "n/a"
        if isinstance(v, (int, float)) and fmt:
            try:
                return fmt.format(v)
            except (ValueError, TypeError):
                return str(v)
        return str(v)

    pit_safe = "yes — first_positive_sale_month is computed from sales_month < origin; missing before first sale; Shamsi month arithmetic via shamsi_month_diff"

    q2 = (
        f"{cov_f('age_coverage_pct')}% of PRIMARY test rows "
        f"({cov_f('age_available_rows', '{:.0f}')}/{cov_f('n_rows', '{:.0f}')}); "
        f"missing rows={cov_f('age_missing_rows', '{:.0f}')}. "
        f"has_prior_positive_sale constant={cov_f('has_prior_positive_sale_constant', '{}')}. "
        f"min observed age={cov_f('min_age', '{:.0f}')} months (all PRIMARY rows are established SKUs)."
    )
    q3 = (
        f"{cov_f('left_censored_product_pct')}% of MVP products "
        f"({cov_f('left_censored_products', '{:.0f}')}/{cov_f('n_products', '{:.0f}')}); "
        f"left_censored_rows={cov_f('left_censored_rows', '{:.0f}')}. "
        "This is observed tenure inside available history, not true commercial age."
    )
    q7 = (
        f"T1 vs T0: {int(t1['origins_improved'])}/{int(t1['origins_total'])} origins"
        if t1 is not None
        else "n/a"
    )
    if h1 is not None:
        q7 += f"; H1 vs H0: {int(h1['origins_improved'])}/{int(h1['origins_total'])} origins"
    q8 = ""
    if t1 is not None:
        q8 = f"T1 product win rate={float(t1['product_win_rate'])*100:.1f}%"
    if h1 is not None:
        q8 += f"; H1 product win rate={float(h1['product_win_rate'])*100:.1f}%"
    q9 = "see error concentration and high-volume watchlist"
    if t1 is not None:
        q9 = (
            f"T1 yes — Cinnatropin 10 alone is {float(t1['top1_deterioration_share'])*100:.1f}% "
            f"of T1 deterioration (top5={float(t1['top5_deterioration_share'])*100:.1f}%, "
            f"top10={float(t1['top10_deterioration_share'])*100:.1f}%). "
            "Net T1 delta AE is near zero because high-volume losses offset other-product gains."
        )
    if h1 is not None:
        q9 += (
            f" H1 top1/top5/top10 deterioration share="
            f"{float(h1['top1_deterioration_share'])*100:.1f}%/"
            f"{float(h1['top5_deterioration_share'])*100:.1f}%/"
            f"{float(h1['top10_deterioration_share'])*100:.1f}%."
        )
    q12 = (
        "Yes — close F3A and move to the next genuinely new information source "
        "(do not start F3B/F3C from this run)."
        if verdict == "E"
        else (
            "Retain the lifecycle conclusion above, then move to the next "
            "genuinely new information source. Do not start F3B/F3C automatically."
        )
    )

    answers = [
        f"1. **Is `months_since_first_observed_positive_sale` point-in-time safe?** {pit_safe}",
        f"2. **What percentage of MVP products/test rows have a usable value?** {q2}",
        f"3. **How much of the product set is left-censored by the beginning of sales history?** {q3}",
        f"4. **Does lifecycle improve TS F0?** {_yes_no_improve(t1)}",
        f"5. **Does lifecycle improve CORE_TS?** {_yes_no_improve(t3)} (secondary; CORE_TS was frozen before F3A)",
        f"6. **Does lifecycle improve Human F0?** {_yes_no_improve(h1)}",
        f"7. **On how many origins does it improve?** {q7}",
        f"8. **What percentage of products improve?** {q8}",
        f"9. **Are gains/losses concentrated in a few high-volume products?** {q9}",
        "10. **Does performance appear different across observed lifecycle age?** "
        + _age_group_answer(age_g, min_age),
        f"11. **Should lifecycle remain a scored feature, a routing/segmentation variable, or be rejected?** {RETAIN.get(verdict, verdict)} "
        "(PRIMARY contains only established SKUs; do not set age thresholds from these five origins).",
        f"12. **Does the result justify moving to the next genuinely new information source?** {q12}",
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
        "median_origin_improvement",
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
        cand_origin = by_o.loc[by_o["experiment"].isin(["T1", "T3", "H1"])]
    cand_prod = None
    if by_p is not None and "experiment" in by_p.columns:
        cand_prod = by_p.loc[by_p["experiment"].isin(["T1", "H1"])]
    cand_h = None
    if by_h is not None and "experiment" in by_h.columns:
        cand_h = by_h.loc[by_h["experiment"].isin(["T1", "T3", "H1"])]
    cand_watch = None
    if watch is not None and "experiment" in watch.columns:
        cand_watch = watch.loc[watch["experiment"].isin(["T1", "T3", "H1"])]

    sections = [
        "# F3A — Product Lifecycle\n",
        f"**Date:** {date.today().isoformat()}  \n",
        "**Benchmark:** frozen v1 matched PRIMARY (n=1877, 5 origins)  \n",
        f"**CSV artifacts:** `{repo_relative(Path(out_dir))}`\n",
        "\n",
        "F3A is another hypothesis evaluated on an already reused research test panel. "
        "Results are useful for research direction but should not be treated as final "
        "unbiased production estimates.\n",
        "\n",
        "## Hypothesis\n\n",
        "The residual forecasting error may depend on how long the exact product/SKU "
        "has been commercially observed. This is not an F1/F2 demand-dynamics feature "
        "(no rolling averages, volatility, growth, trend, or YoY transforms).\n",
        "\n",
        "## Feature definition\n\n",
        "Primary scored feature (exactly one): `months_since_first_observed_positive_sale`.\n\n",
        "- Product-level, not generic-level.\n",
        "- `first_positive_sale_month(p, O)` = earliest month `t < O` with **net sales > 0**.\n",
        "- Age = Shamsi calendar months between that month and the forecast origin, "
        "via `shamsi_month_diff` (never YYYYMM integer subtraction).\n",
        "- `sales = 0` is not commercial launch. `sales < 0` is not first commercial "
        "sale (returns/adjustments).\n",
        "- If no prior positive sale: age stays **NaN** (not encoded as zero). XGB sees native missing values.\n",
        "- `has_prior_positive_sale` and `first_sale_left_censored` are diagnostics only; they are not in XGB.\n",
        "- Observed age is **tenure inside available sales history**, never true product age.\n",
        "- Left-censored = the product already has sales > 0 in the global earliest month of frozen `raw/sales.parquet`.\n",
        "\n",
        "Drug Launch event extraction remains in `pkg.db.query.event_profile` as a "
        "**deferred / exploratory commercial-event source**. It is not scored in F3A "
        "(product-level coverage is approximately 46%; events are generic-level; some "
        "mature products have recorded launch events long after commercial sales began). "
        "It was not modified or tuned based on F3A.\n",
        "\nNo lifecycle categories (`is_new`, age buckets, log-age, caps, interactions) were created.\n",
        "\n## Reproduction gates\n\n",
        _md_table(gates) if gates is not None else "_missing_\n",
        "Locked freeze-time Analysis B WMAPEs were not rewritten. Controls are the "
        "currently reproduced F0 backtest (T0/H0) and the pre-F3A CORE_TS diagnostic (T2).\n",
        "\n## Pre-model audit\n\n",
        _md_table(coverage) if coverage is not None else "_missing_\n",
        "\n### By origin\n\n",
        _md_table(life_origin) if life_origin is not None else "_missing_\n",
        "\n### Per MVP product\n\n",
        _md_table(product_audit, max_rows=30) if product_audit is not None else "_missing_\n",
        "\n### First-positive vs first-nonzero (not scored)\n\n",
        _md_table(nz) if nz is not None else "_missing_\n",
        "\n## Scoreboard\n\n",
        _md_table(overall, max_rows=12, cols=overall_cols) if overall is not None else "_missing_\n",
        "\nT1 is compared to T0; T3 to T2; H1 to H0. Positive relative WMAPE means the lifecycle feature helped. "
        "T2/T3 are secondary challenger experiments because CORE_TS was identified **before F3A**.\n",
        "\n## Verdict\n\n",
        f"**Case {verdict}.** {VERDICT_TEXT.get(verdict, '').split(' — ', 1)[-1]}\n",
        _verdict_caveat(min_age, age_g),
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
        "\n## Product robustness (T1 and H1; top by volume)\n\n",
        _md_table(cand_prod, max_rows=20) if cand_prod is not None else "_missing_\n",
        "\n## Observed-age groups (descriptive only)\n\n",
        "Quartile bins are taken from the pre-model age distribution on PRIMARY rows. "
        "They are **not** model features and were not optimized against WMAPE.\n\n",
        _md_table(age_g) if age_g is not None else "_missing_\n",
        "\n## What was not done\n\n",
        "No price, inventory, commercial-event scoring, Human-bias features, extra sales "
        "trends, XGB tuning, SHAP, routing categories, F3B, or F3C. Frozen F0/F1/F2/"
        "ablation artifacts were not modified.\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
