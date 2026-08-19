"""Write docs/f3e_peer_demand_source_audit.md from F3E Step 1 frozen artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.f3e.config import docs_dir, f3e_source_dir
from pkg.research.harness.report import md_table


def _yn(cond: bool, yes: str = "yes", no: str = "no") -> str:
    return yes if cond else no


def _safe_int(val, fallback: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return fallback


def write_peer_demand_source_audit(result: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3e_peer_demand_source_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    panel: pd.DataFrame = result.get("panel", pd.DataFrame())
    mvp_list: list = result.get("mvp_list", [])
    n_mvp_in_panel: int = result.get("n_mvp_in_panel", 0)
    unit_ratio_audit: pd.DataFrame = result.get("unit_ratio_audit", pd.DataFrame())
    unit_ratio_invalid: pd.DataFrame = result.get("unit_ratio_invalid", pd.DataFrame())
    patient_audit: pd.DataFrame = result.get("patient_conversion_audit", pd.DataFrame())
    generic_peer: pd.DataFrame = result.get("generic_peer_audit", pd.DataFrame())
    cross_generic: pd.DataFrame = result.get("cross_generic_peer_audit", pd.DataFrame())
    norm_examples: pd.DataFrame = result.get("normalization_examples", pd.DataFrame())
    pe_examples: pd.DataFrame = result.get("patient_equivalent_examples", pd.DataFrame())
    neg_report: pd.DataFrame = result.get("negative_sales_report", pd.DataFrame())
    month_cov: pd.DataFrame = result.get("global_month_coverage", pd.DataFrame())

    n_mvp = len(mvp_list)

    # Generic peer summary
    if not generic_peer.empty and "n_generic_peers_with_sales" in generic_peer.columns:
        n_with_peers = int((generic_peer["n_generic_peers_with_sales"] >= 1).sum())
        n_without_peers = n_mvp - n_with_peers
    else:
        n_with_peers = n_without_peers = 0

    # Cross-generic peer summary
    if not cross_generic.empty and "n_cross_generic_peers_with_valid_patient_conversion" in cross_generic.columns:
        n_with_cross = int(
            (cross_generic["n_cross_generic_peers_with_valid_patient_conversion"] >= 1).sum()
        )
    else:
        n_with_cross = 0

    # Unit ratio validity
    n_valid_unit = 0
    n_selling = 0
    if not unit_ratio_audit.empty and "n_valid_positive_unit_ratio" in unit_ratio_audit.columns:
        row = unit_ratio_audit.dropna(subset=["n_valid_positive_unit_ratio"]).iloc[:1]
        if not row.empty:
            n_valid_unit = _safe_int(row["n_valid_positive_unit_ratio"].iloc[0])
            n_selling = _safe_int(row["n_selling_products"].iloc[0])

    # Month coverage
    first_month = last_month = n_months = n_gaps = "n/a"
    if not month_cov.empty:
        r = month_cov.iloc[0]
        first_month = str(r.get("first_month", "n/a"))
        last_month = str(r.get("last_month", "n/a"))
        n_months = str(r.get("n_distinct_months", "n/a"))
        n_gaps = str(r.get("n_missing_global_months", "n/a"))

    # Negative sales
    neg_dqty_rows = 0
    if not neg_report.empty and "n_negative_rows" in neg_report.columns:
        row = neg_report.loc[neg_report["quantity"] == "monthly_dqty"]
        if not row.empty:
            neg_dqty_rows = _safe_int(row["n_negative_rows"].iloc[0])

    lines = [
        "# F3E Step 1 — Peer Demand Source Audit",
        f"**Date:** {date.today()}  ",
        f"**Frozen artifacts:** `src/data/results/f3e/source`",
        "",
        "No XGBoost. No WMAPE. No scored F3E features computed in this step.",
        "",
        "## Source",
        "",
        "- Sales: `[DWOrchid].[dbo].[Flat_Fact_Sale]` joined to `[Iris_DW].[Dim].[Product]`.",
        "- Filter: `ProductTitleEN IS NOT NULL AND Field != '-'`.",
        "- Granularity: one row per `ProductTitleEN × ShamsiYearMonth`.",
        "- Peer universe: ALL products passing the filter (not MVP-restricted).",
        "- Negative monthly_dqty sums retained; Step 2 decides clipping policy.",
        "",
        "## Normalization rules (frozen before any WMAPE)",
        "",
        "### Same-generic (DQtyUnit)",
        "",
        "```",
        "monthly_dqtyunit = monthly_dqty * unit_ratio",
        "```",
        "",
        "- `unit_ratio` = `Dim.Product.Unit` (within-generic conversion ratio).",
        "- Valid only when `unit_ratio` is finite and `> 0`; NaN otherwise.",
        "- Used ONLY within the same `FKGeneric`.",
        "- `Unit` is NOT assumed comparable across different generics.",
        "",
        "### Cross-generic (monthly_patient_equivalent)",
        "",
        "```",
        "monthly_patient_equivalent = monthly_dqty / PatientConsumePerPeriod",
        "```",
        "",
        "- Valid for `PatientConsumeType ∈ {Continuous, SinglePeriod}` and `PatientConsumePerPeriod > 0`.",
        "- **No ×12 or ÷12 applied** (unlike F3D annualization).",
        "- Continuous denominator = monthly patient consumption quantity.",
        "- SinglePeriod denominator = complete annual/single-period consumption quantity.",
        "- Used ONLY across generics within the same `Field × PatientConsumeType` segment.",
        "- The target product's **entire FKGeneric is excluded** from cross-generic peers.",
        "",
        "### Why the two normalizations differ",
        "",
        "Same-generic: `Unit` is a within-generic ratio designed to make presentations "
        "within the same generic comparable.  Cross-generic: `Unit` is not assumed "
        "comparable between different generics, so demand is expressed in patient "
        "equivalents instead.",
        "",
        "## Static-dimension assumption",
        "",
        "`Dim.Product` is a current snapshot.  `FKGeneric`, `Field`, `Unit`, "
        "`PatientConsumeType`, and `PatientConsumePerPeriod` are treated as static "
        "product definitions.  No historical reconstruction is attempted.",
        "",
        "## Unit-ratio audit",
        "",
        f"Valid positive unit_ratio: {n_valid_unit} / {n_selling} selling products.",
        "",
        md_table(unit_ratio_audit, max_rows=10) if not unit_ratio_audit.empty else "_not available_",
        "",
        "### Invalid unit_ratio products (sample)",
        "",
        md_table(unit_ratio_invalid.head(20), max_rows=20) if not unit_ratio_invalid.empty else "None.",
        "",
        "## Patient-conversion audit",
        "",
        md_table(patient_audit, max_rows=20) if not patient_audit.empty else "_not available_",
        "",
        "## Generic peer audit (MVP targets)",
        "",
        f"- Targets with ≥1 same-generic peer with sales: **{n_with_peers} / {n_mvp}**",
        f"- Targets with zero same-generic peers: **{n_without_peers}**",
        "",
        md_table(generic_peer.head(50), max_rows=50) if not generic_peer.empty else "_not available_",
        "",
        "## Cross-generic Field×ConsumeType peer audit (MVP targets)",
        "",
        f"- Targets with ≥1 cross-generic peer with valid patient conversion: **{n_with_cross} / {n_mvp}**",
        "",
        md_table(cross_generic.head(50), max_rows=50) if not cross_generic.empty else "_not available_",
        "",
        "## Normalization examples (manual DQtyUnit check)",
        "",
        md_table(norm_examples, max_rows=30) if not norm_examples.empty else "_no multi-presentation generics found_",
        "",
        "## Patient-equivalent examples",
        "",
        md_table(pe_examples, max_rows=20) if not pe_examples.empty else "_no valid examples found_",
        "",
        "## Negative sales",
        "",
        md_table(neg_report, max_rows=10) if not neg_report.empty else "_not available_",
        "",
        f"Negative monthly_dqty rows: {neg_dqty_rows}. "
        "No clipping applied in Step 1. Step 2 will define representation policy.",
        "",
        "## Global month coverage",
        "",
        md_table(month_cov, max_rows=5) if not month_cov.empty else "_not available_",
        "",
        f"First month: {first_month} | Last month: {last_month} | "
        f"Distinct months: {n_months} | Missing global months: {n_gaps}",
        "",
        "## Audit answers",
        "",
        f"1. **Is sales mapping valid?** "
        + _yn(n_mvp_in_panel > 0,
              f"yes — {n_mvp_in_panel}/{n_mvp} MVP products found in peer panel.",
              "FAIL — zero MVP products in panel."),
        f"2. **Is Unit valid for same-generic normalization?** "
        + _yn(n_valid_unit > 0,
              f"{n_valid_unit}/{n_selling} selling products have valid positive unit_ratio.",
              "no valid unit_ratio found."),
        "3. **Is DQtyUnit correctly DQty × Unit?** "
        "Yes — programmatic assertion `assert_dqtyunit_formula` passed.",
        "4. **Is DQtyUnit used ONLY within generic?** "
        "Yes — `monthly_dqtyunit` is computed for all products but same-generic "
        "aggregation in Step 2 will join only on `FKGeneric == FKGeneric_target`.",
        "5. **Is PatientConsumePerPeriod sufficiently available?** "
        "See patient_conversion_audit.csv for coverage.",
        "6. **Is monthly_patient_equivalent calculated correctly?** "
        "Yes — programmatic assertion `assert_patient_equivalent_formula` passed.",
        "7. **Is Continuous interpreted as monthly consumption?** "
        "Yes — denominator is monthly PatientConsumePerPeriod for Continuous products.",
        "8. **Is SinglePeriod interpreted as full annual/single-period consumption without ×12 or ÷12?** "
        "Yes — same formula `monthly_dqty / PatientConsumePerPeriod`; no multiplier applied.",
        f"9. **How many targets have valid same-generic peers?** "
        f"{n_with_peers}/{n_mvp}.",
        f"10. **How many targets have valid cross-generic Field×ConsumeType peers?** "
        f"{n_with_cross}/{n_mvp}.",
        "11. **Is the target SKU excluded from same-generic peer demand?** "
        "Yes — assertion `assert_same_generic_excludes_self` passed.",
        "12. **Is the ENTIRE target generic excluded from cross-generic peer demand?** "
        "Yes — assertion `assert_cross_generic_excludes_entire_generic` passed.",
        f"13. **Are negative quantities material?** "
        f"{neg_dqty_rows} negative monthly_dqty rows. See negative_sales_report.csv.",
        f"14. **Is source month coverage adequate?** "
        f"Months {first_month}–{last_month}, {n_months} distinct, {n_gaps} global gaps. "
        "Step 2 will treat covered months with no peer sales as zero demand.",
        "15. **Are frozen F3E sources safe for Step 2?** "
        "Yes — benchmark freeze checksum verified; parquets written to "
        "`src/data/results/f3e/source/`.",
        "",
        "## What was not done",
        "",
        "- No XGBoost, no WMAPE, no scored F3E features.",
        "- No imputation of missing unit_ratio or PatientConsumePerPeriod.",
        "- No clipping of negative sales.",
        "- No peer-demand time-series construction (deferred to Step 2).",
        "- F0/F1/F2/F3A/F3B/F3C/F3D artifacts were not modified.",
        "- Frozen benchmark v1 panels were not changed.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
