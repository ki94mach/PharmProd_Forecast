"""Write docs/f3d_patient_consumption_profile.md from F3D Step 2 results."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.f3d.config import docs_dir, f3d_output_dir
from pkg.research.harness.report import md_table

VERDICT_TEXT = {
    "A": (
        "A — patient-consumption profile robustly useful for both TS and Human. "
        "Retain as promising research evidence requiring future/shadow-origin confirmation."
    ),
    "B": (
        "B — patient-consumption profile useful for TS only. "
        "Retain for the TS anchor; investigate why Human does not improve."
    ),
    "C": (
        "C — patient-consumption profile useful for Human only. "
        "Retain for the Human anchor; investigate why TS does not improve."
    ),
    "D": (
        "D — weak or regime-specific signal; descriptive only. "
        "Do not automatically retain as a scored production feature."
    ),
    "E": (
        "E — current patient-consumption representation fails. "
        "Do not tune annualization formula, transform, subsets, encoding, or hyperparameters."
    ),
}


def _row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.loc[df["experiment"] == name]
    return sub.iloc[0] if len(sub) else None


def _yes_no(r: Optional[pd.Series], ctrl: str = "") -> str:
    if r is None:
        return "not run"
    rel = float(r["rel_wmape_vs_control_pct"])
    ctrl_label = ctrl or str(r.get("control", ""))
    if rel > 0:
        return f"yes ({rel:+.2f}% relative WMAPE vs {ctrl_label})"
    return f"no ({rel:+.2f}% relative WMAPE vs {ctrl_label})"


def write_f3d_results(report: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3d_patient_consumption_profile.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    overall: pd.DataFrame = report.get("overall", pd.DataFrame())
    verdict: str = report.get("verdict", "unknown")
    gates: pd.DataFrame = report.get("gates", pd.DataFrame())
    consume_type: pd.DataFrame = report.get("by_patient_consume_type", pd.DataFrame())
    quartile: pd.DataFrame = report.get("by_consumption_quartile", pd.DataFrame())
    importance: pd.DataFrame = report.get("feature_importance", pd.DataFrame())

    d1t = _row(overall, "D1_TS_TYPE")
    d2t = _row(overall, "D2_TS_PROFILE")
    d1h = _row(overall, "D1_HUMAN_TYPE")
    d2h = _row(overall, "D2_HUMAN_PROFILE")

    # Origins consistency: count experiments where majority of origins improved
    def _origins_ok(r: Optional[pd.Series]) -> bool:
        if r is None:
            return False
        return int(r.get("origins_improved", 0)) > int(r.get("origins_total", 5)) / 2

    consistent = sum(
        1
        for r in (d1t, d1h)
        if r is not None and _origins_ok(r)
    )

    # Product win rates
    def _win_rate(r: Optional[pd.Series]) -> str:
        if r is None:
            return "n/a"
        return f"{float(r.get('product_win_rate', np.nan)):.1%}"

    def _d2_vs_d0(r: Optional[pd.Series]) -> str:
        if r is None:
            return "n/a"
        v = float(r.get("rel_wmape_vs_d0_pct", np.nan))
        if np.isfinite(v):
            return f"{v:+.4f}%"
        return "n/a"

    lines = [
        "# F3D — Patient Consumption Profile",
        f"**Date:** {date.today()}  ",
        f"**CSV artifacts:** `src/data/results/f3d`",
        "",
        "F3D is a hypothesis evaluated on an already reused research test panel. "
        "Results are useful for research direction but should not be treated as "
        "unbiased production estimates.",
        "",
        "## Business contract",
        "",
        "**Source:** `[Iris_DW].[Dim].[Product]`  ",
        "**Mapping:** exact `ProductTitleEN == product` (no fuzzy)  ",
        "**Annualisation:**",
        "- Continuous: `patient_annual_consumption = PatientConsumePerPeriod × 12`",
        "- SinglePeriod: `patient_annual_consumption = PatientConsumePerPeriod`",
        "",
        "**Scored features:**",
        "- `is_continuous_consumption`: Continuous → 1, SinglePeriod → 0, missing → NaN",
        "- `log_patient_annual_consumption = log1p(patient_annual_consumption)`, NaN if missing/negative",
        "",
        "**Static-feature assumption:** same profile attached to all origins/horizons.  "
        "No historical reconstruction.  Cannot explain temporary events; allows the model "
        "to learn different error behaviour across product types.",
        "",
        "## F0 reproduction",
        "",
        md_table(gates, max_rows=10) if not gates.empty else "_gates not available_",
        "",
        "## Overall results",
        "",
        md_table(overall, max_rows=20) if not overall.empty else "_not available_",
        "",
        "## Verdict questions",
        "",
        f"1. **Did F0 reproduction pass?** "
        + (
            "yes"
            if (not gates.empty and bool(gates["ok"].all()))
            else "see reproduction_gates.csv"
        ),
        f"2. **Does PatientConsumeType improve TS?** {_yes_no(d1t, 'D0_TS')}",
        f"3. **Does PatientConsumeType improve Human?** {_yes_no(d1h, 'D0_HUMAN')}",
        f"4. **Does annual patient consumption improve TS beyond type?** {_yes_no(d2t, 'D1_TS_TYPE')}",
        f"5. **Does annual patient consumption improve Human beyond type?** {_yes_no(d2h, 'D1_HUMAN_TYPE')}",
        f"6. **Are results consistent across origins?** "
        + f"{consistent}/2 anchors have majority-origin improvement.",
        f"7. **What percentage of products improve?** "
        + f"TS type: {_win_rate(d1t)}, Human type: {_win_rate(d1h)}.",
        f"8. **Are gains/losses concentrated in high-volume products?** "
        "See error_concentration.csv and high_volume_watchlist.csv.",
        f"9. **Is performance different for Continuous vs SinglePeriod products?** "
        "See by_patient_consume_type.csv (diagnostic only; no routing rules derived).",
        f"10. **Did XGBoost use the new profile variables?** "
        "See feature_importance.csv (gain, diagnostic only, no SHAP).",
        "",
        f"## D2 vs D0 (full profile vs F0 baseline)",
        "",
        f"- D2_TS_PROFILE vs D0_TS: {_d2_vs_d0(d2t)}",
        f"- D2_HUMAN_PROFILE vs D0_HUMAN: {_d2_vs_d0(d2h)}",
        "",
        "## Performance by PatientConsumeType (diagnostic)",
        "",
        md_table(consume_type, max_rows=10) if not consume_type.empty else "_not available_",
        "",
        "## Performance by consumption quartile (diagnostic)",
        "",
        "Pre-model quartiles of unique-product `patient_annual_consumption`.  "
        "Diagnostic only.  No routing rules derived.  No thresholds chosen using WMAPE.",
        "",
        md_table(quartile, max_rows=10) if not quartile.empty else "_not available_",
        "",
        "## Feature importance (XGBoost gain, diagnostic only)",
        "",
        "No SHAP.  Gain reported per origin/experiment for F3D features.",
        "",
        md_table(
            importance.loc[importance["is_f3d_feature"] == True].head(40)
            if not importance.empty and "is_f3d_feature" in importance.columns
            else importance.head(40),
            max_rows=40,
        )
        if not importance.empty
        else "_not available_",
        "",
        f"## Verdict: {verdict}",
        "",
        VERDICT_TEXT.get(verdict, f"Unknown verdict: {verdict}"),
        "",
        "## Research limitation",
        "",
        "The five PRIMARY origins have already been repeatedly used for previous "
        "feature-family research.  Therefore any positive F3D result must be described as "
        "**promising research evidence requiring future/shadow-origin confirmation**, "
        "not unbiased production performance.",
        "",
        "Do not modify the annualization formula, feature transform, feature subsets, "
        "categorical encoding (`is_continuous_consumption`), or XGBoost parameters "
        "after observing the PRIMARY results.",
        "",
        "## What was not done",
        "",
        "- No ProductType, ProductForm, Field, Provider, Weight, price, or inventory features.",
        "- No SHAP analysis.",
        "- No hyperparameter tuning, early-stopping redesign, or feature-subset search.",
        "- No routing rules derived from consumption thresholds.",
        "- F0/F1/F2/F3A/F3B/F3C artifacts were not modified.",
        "- Frozen benchmark v1 panels were not changed.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
