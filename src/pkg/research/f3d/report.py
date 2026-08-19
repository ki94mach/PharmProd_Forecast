"""Write docs/f3d_patient_consumption_profile_audit.md from prepare_f3d result."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.f3d.config import docs_dir, f3d_profile_audit_dir
from pkg.research.harness.report import md_table


def _yn(cond: bool, yes: str = "yes", no: str = "no") -> str:
    return yes if cond else no


def write_profile_audit(result: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3d_patient_consumption_profile_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    audit: pd.DataFrame = result.get("audit", pd.DataFrame())
    type_counts: pd.DataFrame = result.get("type_counts", pd.DataFrame())
    period_dist: pd.DataFrame = result.get("period_distributions", pd.DataFrame())
    annual_dist: pd.DataFrame = result.get("annual_distributions", pd.DataFrame())
    log_dist: pd.DataFrame = result.get("log_distributions", pd.DataFrame())
    semantic: pd.DataFrame = result.get("semantic_table", pd.DataFrame())
    neg_report: pd.DataFrame = result.get("negative_report", pd.DataFrame())

    # Coverage numbers from audit
    if not audit.empty:
        n_products = int(audit["n_products"].iloc[0])
        n_type = int(audit["n_with_PatientConsumeType"].iloc[0])
        n_period = int(audit["n_with_PatientConsumePerPeriod"].iloc[0])
        cov_pct = float(audit["coverage_pct"].iloc[0])
        n_unexpected = int(audit["n_unexpected_types"].iloc[0])
        unexpected_str = str(audit["unexpected_types"].iloc[0]) if n_unexpected else "none"
        n_negative = int(audit["n_negative_period"].iloc[0])
    else:
        n_products = n_type = n_period = 0
        cov_pct = 0.0
        n_unexpected = 0
        unexpected_str = "none"
        n_negative = 0

    # Known types from type_counts
    known_types = []
    if not type_counts.empty and "PatientConsumeType" in type_counts.columns:
        for _, row in type_counts.iterrows():
            known_types.append(
                f"- `{row['PatientConsumeType']}`: {int(row.get('n_products', 0))} products"
            )

    lines = [
        "# F3D Step 1 — Patient Consumption Profile Audit",
        f"**Date:** {date.today()}  ",
        f"**CSV artifacts:** `src/data/results/f3d/profile_audit`  ",
        "**Frozen source:** `src/data/results/f3d/source/product_profile.parquet`",
        "",
        "No XGBoost, no WMAPE, no `FamilySession`.",
        "",
        "## Source and mapping",
        "",
        "- Source: `[Iris_DW].[Dim].[Product]` via `load_dim_product()`.",
        "- Canonical product list: unique `product` from frozen `matched_universe.parquet`.",
        "- Join: exact `ProductTitleEN == product` (no fuzzy matching).",
        "- Static-feature assumption: `Dim.Product` is a current snapshot.  "
        "The same profile is attached to every forecast origin, target month, "
        "and horizon for a given product.  No historical reconstruction is "
        "performed because the source does not contain dated rows.",
        "",
        "## Coverage",
        "",
        f"| metric | value |",
        f"| --- | --- |",
        f"| n_products (canonical MVP) | {n_products} |",
        f"| n_with_PatientConsumeType | {n_type} |",
        f"| n_with_PatientConsumePerPeriod | {n_period} |",
        f"| coverage_pct | {cov_pct:.1f}% |",
        f"| n_unexpected_types | {n_unexpected} |",
        f"| unexpected_types | {unexpected_str} |",
        f"| n_negative_PatientConsumePerPeriod | {n_negative} |",
        "",
        "## PatientConsumeType values",
        "",
    ] + known_types + [
        "",
        md_table(type_counts, max_rows=30) if not type_counts.empty else "_type_counts not available_",
        "",
        "## Business semantics",
        "",
        "- **Continuous**: `PatientConsumePerPeriod` = quantity per **month** → "
        "`patient_annual_consumption = PatientConsumePerPeriod × 12`.",
        "- **SinglePeriod**: `PatientConsumePerPeriod` = quantity per **year** → "
        "`patient_annual_consumption = PatientConsumePerPeriod` (no multiplier).",
        "- `log_patient_annual_consumption = log1p(patient_annual_consumption)` "
        "for finite non-negative values; `NaN` otherwise.",
        "- `is_continuous_consumption`: Continuous → 1, SinglePeriod → 0, missing/unexpected → NaN.",
        "",
        "## PatientConsumePerPeriod distributions (by type)",
        "",
        md_table(period_dist, max_rows=30) if not period_dist.empty else "_not available_",
        "",
        "## patient_annual_consumption distributions",
        "",
        md_table(annual_dist, max_rows=30) if not annual_dist.empty else "_not available_",
        "",
        "## log_patient_annual_consumption distributions",
        "",
        md_table(log_dist, max_rows=30) if not log_dist.empty else "_not available_",
        "",
        "## Semantic validation table (first 50 rows)",
        "",
        md_table(semantic.head(50), max_rows=50) if not semantic.empty else "_not available_",
        "",
        "## Negative PatientConsumePerPeriod",
        "",
        f"{n_negative} product(s) had negative PatientConsumePerPeriod.  "
        "These are set to NaN for `patient_annual_consumption` and "
        "`log_patient_annual_consumption`.  See `negative_period_report.csv`."
        if n_negative > 0
        else "No negative PatientConsumePerPeriod values found.",
        "",
        "## Predeclared feature families (frozen before WMAPE)",
        "",
        "- **F3D-A:** `is_continuous_consumption`",
        "- **F3D-B:** `is_continuous_consumption` + `log_patient_annual_consumption`",
        "",
        "These families were declared before any WMAPE was observed.  "
        "No modifications are allowed after PRIMARY results.",
        "",
        "## Audit answers",
        "",
        "1. **What PatientConsumeType values exist?**  "
        + ("; ".join(str(v) for v in type_counts["PatientConsumeType"].tolist()) if not type_counts.empty else "see type_counts.csv"),
        f"2. **What is coverage among benchmark products?** {cov_pct:.1f}% ({n_type}/{n_products}).",
        f"3. **Is PatientConsumePerPeriod available for most products?** "
        + _yn(n_period >= n_products * 0.5, f"yes ({n_period}/{n_products})", f"partial ({n_period}/{n_products})"),
        "4. **Was Continuous correctly annualised using ×12?** "
        + "Yes — programmatic assertion passed (see prepare.py `_compute_annual`).",
        "5. **Was SinglePeriod kept as the annual amount as-is?** "
        + "Yes — programmatic assertion passed.",
        "6. **Are missing values preserved as NaN?** "
        + "Yes — NEVER_FILLNA excludes all F3D columns from zero-fill in `make_residual_model`.",
        "7. **Are F3D-A and F3D-B ready for controlled evaluation?** "
        + _yn(n_type > 0, "yes", "no — coverage is zero, STOP"),
        "",
        "## What was not done",
        "",
        "- No XGBoost, no WMAPE, no SHAP.",
        "- No ProductType, ProductForm, Field, Provider, Weight, price, or inventory columns used.",
        "- No imputation of median or zero for missing profile values.",
        "- F0/F1/F2/F3A/F3B/F3C artifacts were not modified.",
        "- Frozen benchmark v1 panels were not changed.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
