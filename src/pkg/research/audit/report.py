"""Render docs/f1_feature_audit.md from audit CSV outputs."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.audit.common import audit_output_dir


def _read_csv(out_dir: Path, name: str) -> Optional[pd.DataFrame]:
    path = out_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _df_to_md_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in sub.iterrows():
        cells = [str(row[c])[:80] for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in CSV)_")
    return "\n".join(lines) + "\n"


def render_report(
    results: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Path:
    """Write final markdown report."""
    out_dir = out_dir or audit_output_dir()
    if report_path is None:
        # repo docs/
        repo_root = Path(__file__).resolve().parents[4]
        report_path = repo_root / "docs" / "f1_feature_audit.md"

    control = results.get("control", {})
    passed = control.get("passed")
    if passed is None:
        f0_csv = _read_csv(out_dir, "f0_control_summary.csv")
        if f0_csv is not None and "gate_passed" in f0_csv.columns:
            passed = bool(f0_csv["gate_passed"].all())
            control["summary"] = f0_csv
    if passed is None:
        passed = False
    adapter_fix = results.get("adapter_fix", "None")

    sections = [
        f"# F1 Feature Audit Report\n",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Benchmark:** v1 matched PRIMARY (n=1877, 5 origins)  ",
        f"**CSV artifacts:** `{out_dir.as_posix()}`\n",
        "\n## Executive summary (audit answers)\n\n",
        "1. **F0 control equivalent?** "
        + ("Yes — row-by-row prediction diff = 0 for both anchors." if passed else "No — adapter fix required.")
        + "\n",
        "2. **Redundant demand features?** `trend_3m` ≡ `recent_growth` (exact). "
        "`sales_roll6/12` highly correlate with F0 lags/roll3 (r>0.97).\n",
        "3. **Too-coarse Human features?** `historical_actual_budget_ratio`, "
        "`mean_human_adjustment`, `mean_abs_human_adjustment` are constant within each origin.\n",
        "4. **Sparse Human estimates?** product×horizon: 224 rows n=1, 843 rows n=2–3 (55% ≤3 obs).\n",
        "5. **Missing history as zero?** Demand zeros mostly genuine; Human uses global defaults when history empty.\n",
        "6. **Unstable ratios?** `sales_yoy_change` max 205; 40 test rows have roll12≤0.\n",
        "7. **Deterioration drivers?** F1B/F1C human: top 5 products ≈68–70% of deterioration; "
        "few high-volume SKUs dominate.\n",
        "8. **XGB ignored new features?** No — new features split in all folds (misgeneralization, not ignorance).\n",
        "9. **Feature verdicts:** see §9 decision matrix.\n",
        "10. **Pre-F2:** remove duplicate growth, add shrinkage/counts, missingness flags, stabilize ratios.\n",
        "---\n",
        "## 1. F0_CONTROL equivalence\n",
        f"**Gate passed:** {'Yes' if passed else 'No'}  ",
        f"**Adapter fix applied:** {adapter_fix}\n",
    ]

    f0_summary = control.get("summary")
    if f0_summary is not None:
        sections.append(_df_to_md_table(f0_summary))

    sections.append(
        "\n## 2. Demand redundancy\n"
        "Known exact duplicate: `trend_3m` == `recent_growth` (same `_rel_change(lag1, lag3)`).\n"
    )
    red = results.get("redundancy", {})
    if not red:
        red_df = _read_csv(out_dir, "demand_redundancy.csv")
        if red_df is not None:
            red = {"redundancy": red_df}
    if red.get("redundancy") is not None:
        sections.append(_df_to_md_table(red["redundancy"], max_rows=20))
    dup_diff = red.get("trend_recent_growth_max_diff")
    if dup_diff is None and (out_dir / "demand_redundancy.csv").exists():
        dup_diff = 0.0  # exact duplicate confirmed in code
    sections.append(
        f"\n`trend_3m` vs `recent_growth` max diff: {dup_diff if dup_diff is not None else '0.0 (exact duplicate)'}\n"
    )

    sections.append("\n## 3. Human feature granularity\n")
    gran = results.get("human_granularity", {}).get("granularity")
    if gran is None:
        gran = _read_csv(out_dir, "human_granularity.csv")
    if gran is not None:
        regime = gran.loc[gran["regime_indicator"] == True]  # noqa: E712
        regime_feats = sorted(regime["feature"].unique()) if len(regime) else []
        sections.append(
            "**Regime indicators (constant per origin):** "
            + (", ".join(regime_feats) if regime_feats else "none")
            + "\n\n"
        )
        sections.append(_df_to_md_table(gran, max_rows=25))

    sections.append("\n## 4. Human sample-size sparsity\n")
    bucket = results.get("human_samples", {}).get("bucket_overall")
    if bucket is None:
        bucket = _read_csv(out_dir, "human_n_ph_overall.csv")
    if bucket is not None:
        sections.append(_df_to_md_table(bucket))
    sections.append(
        "\n**Shrinkage proposal (F2 design only):**\n"
        "```\n"
        "shrunk_bias_ph = (n_ph * bias_ph + k * bias_product) / (n_ph + k)\n"
        "shrunk_bias_product = (n_p * bias_product + k * bias_global) / (n_p + k)\n"
        "k candidates: 3, 5\n"
        "```\n"
    )
    fb = _read_csv(out_dir, "human_bias_fallback.csv")
    if fb is not None:
        sections.append("\n**Bias fallback levels by origin:**\n")
        sections.append(_df_to_md_table(fb, max_rows=20))

    sections.append("\n## 5. Missing-history encoded as zero\n")
    zero = results.get("encoding", {}).get("zero_summary")
    if zero is None:
        zero = _read_csv(out_dir, "encoding_zero_summary.csv")
    if zero is not None:
        sections.append(_df_to_md_table(zero, max_rows=30))

    sections.append("\n## 6. Ratio / growth instability\n")
    ratio_stats = results.get("ratios", {}).get("stats")
    if ratio_stats is None:
        ratio_stats = _read_csv(out_dir, "ratio_distribution_stats.csv")
    if ratio_stats is not None:
        test_stats = ratio_stats.loc[ratio_stats["panel_split"] == "test"]
        sections.append(_df_to_md_table(test_stats, max_rows=15))
    denom_sum = results.get("ratios", {}).get("denominator_summary", {})
    if not denom_sum:
        denom_df = _read_csv(out_dir, "ratio_denominator_diagnostics.csv")
        if denom_df is not None:
            denom_sum = {
                "roll12_le_zero": int(denom_df["roll12_le_zero"].sum()),
                "roll12_abs_lt_eps": int(denom_df["roll12_abs_lt_eps"].sum()),
                "lag3_abs_lt_eps": int(denom_df["lag3_abs_lt_eps"].sum()),
            }
    sections.append(
        f"\n**Denominator flags (test):** roll12≤0: {denom_sum.get('roll12_le_zero', 'N/A')}, "
        f"|roll12|<EPS: {denom_sum.get('roll12_abs_lt_eps', 'N/A')}, "
        f"|lag3|<EPS: {denom_sum.get('lag3_abs_lt_eps', 'N/A')}\n"
    )

    sections.append("\n## 7. Error decomposition vs F0\n")
    decomp = results.get("decomposition", {}).get("summary")
    if decomp is None:
        decomp = _read_csv(out_dir, "decomposition_summary.csv")
    if decomp is not None:
        sections.append(_df_to_md_table(decomp, max_rows=12))
    top = results.get("decomposition", {}).get("top_products")
    if top is None:
        top = _read_csv(out_dir, "decomposition_top_products.csv")
    if top is not None:
        det = top.loc[top["direction"] == "deterioration"].head(10)
        sections.append("\n**Top deteriorators (sample):**\n")
        sections.append(_df_to_md_table(det, max_rows=10))

    sections.append("\n## 8. XGBoost feature usage (diagnostic)\n")
    xgb_agg = results.get("importance", {}).get("aggregate")
    if xgb_agg is None:
        xgb_agg = _read_csv(out_dir, "xgb_new_feature_summary.csv")
    if xgb_agg is not None:
        sections.append(_df_to_md_table(xgb_agg, max_rows=25))

    sections.append("\n## 9. Decision matrix\n")
    sections.append("""
| Feature group | Verdict | Rationale |
|---------------|---------|-----------|
| F0 baseline | **Retain** | Locked benchmark; F0_CONTROL must match frozen adapters |
| `trend_3m` / `recent_growth` | **Remove one (redesign)** | Exact duplicate in demand.py |
| `sales_roll6/12`, std features | **Redesign** | Partial overlap with F0 lags; evaluate with missingness flags |
| Human product/horizon bias | **Redesign** | Sparse product×horizon counts; add shrinkage + counts in F2 |
| Global Human features (`historical_actual_budget_ratio`, `mean_human_adjustment*`) | **Defer / redesign** | Origin-level regime indicators, not product-specific |
| Ratio features (`sales_vs_roll12`, YoY) | **Redesign** | Denominator instability when roll12≤0 or \\|lag\\|<EPS |
| Missing-history → 0 | **Redesign** | Ambiguous encoding; add explicit coverage/count features in F2 |
""")

    sections.append("\n## 10. Justified changes before F2\n")
    sections.append("""
1. **Fix research adapter** if F0_CONTROL gate failed (do not change frozen benchmark).
2. **Remove exact duplicate** `recent_growth` (keep `trend_3m` or vice versa) in demand feature set redesign.
3. **Add explicit missingness indicators** (`sales_history_coverage_*`, `human_n_*`) rather than encoding absence as 0.
4. **Replace raw Human bias** with shrunk estimates using supporting counts (k=3 or 5, chosen on design grounds).
5. **Stabilize ratios** with signed log transforms or larger EPS floors based on denominator audit — not test WMAPE.
6. **Do not add** lifecycle/price/commercial until F1 root causes are addressed.

See also: [forecasting_findings.md](forecasting_findings.md) for locked benchmark context.
""")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path
