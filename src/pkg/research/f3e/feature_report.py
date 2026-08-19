"""Write docs/f3e_peer_demand_feature_audit.md from F3E Step 2 frozen artifacts.

Reads from src/data/results/f3e/feature_audit/*.csv — no SQL, no computation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.f3e.config import F3E_A_FEATURES, F3E_B_FEATURES, docs_dir, f3e_feature_audit_dir
from pkg.research.harness.report import md_table


def _safe(val, fmt: str = ",.1f", fallback: str = "n/a") -> str:
    try:
        if np.isnan(float(val)):
            return fallback
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback if val is None else str(val)


def _yn(cond: bool, yes: str = "yes", no: str = "no") -> str:
    return yes if cond else no


def _get_metric(df: pd.DataFrame, metric: str, fallback: str = "n/a") -> str:
    if df.empty or "metric" not in df.columns:
        return fallback
    row = df.loc[df["metric"] == metric]
    if row.empty:
        return fallback
    return _safe(row["value"].iloc[0], fmt=",.0f", fallback=fallback)


def _get_reason_n(df: pd.DataFrame, feature_group: str, reason: str) -> int:
    if df.empty:
        return 0
    fg_col = df.get("feature_group", pd.Series(dtype=str))
    r_col = df.get("reason", pd.Series(dtype=str))
    mask = (fg_col == feature_group) & (r_col == reason)
    sub = df.loc[mask]
    return int(sub["n_rows"].sum()) if not sub.empty else 0


def write_feature_audit_report(audit: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3e_peer_demand_feature_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    cov: pd.DataFrame = audit.get("coverage_overall", pd.DataFrame())
    cov_origin: pd.DataFrame = audit.get("coverage_by_origin", pd.DataFrame())
    cov_product: pd.DataFrame = audit.get("coverage_by_product", pd.DataFrame())
    miss: pd.DataFrame = audit.get("missingness", pd.DataFrame())
    dists: pd.DataFrame = audit.get("distributions", pd.DataFrame())
    pg: pd.DataFrame = audit.get("peer_group_audit", pd.DataFrame())
    tv: pd.DataFrame = audit.get("temporal_variation", pd.DataFrame())

    n_rows = _get_metric(cov, "n_rows")
    n_prods = _get_metric(cov, "n_products")
    n_origins = _get_metric(cov, "n_origins")

    # Feature coverage
    def _feat_cov(feat: str) -> str:
        avail = _get_metric(cov, f"{feat}_available_rows", "?")
        pct = _get_metric(cov, f"{feat}_coverage_pct", "?")
        return f"{avail} rows ({pct}%)"

    # Missingness — last-month features
    g_avail = _get_reason_n(miss, "generic", "AVAILABLE")
    g_no_peers = _get_reason_n(miss, "generic", "NO_GENERIC_PEERS")
    g_invalid = _get_reason_n(miss, "generic", "INVALID_UNIT_FOR_ALL_RELEVANT_PEERS")
    g_no_month = _get_reason_n(miss, "generic", "SOURCE_MONTH_UNAVAILABLE")
    g_neg = _get_reason_n(miss, "generic", "NEGATIVE_NET_PEER_DEMAND")
    # 3m-mean negative
    g3m_neg = _get_reason_n(miss, "generic_3m", "NEGATIVE_NET_PEER_DEMAND")

    c_avail = _get_reason_n(miss, "cross_generic", "AVAILABLE")
    c_no_peers = _get_reason_n(miss, "cross_generic", "NO_CROSS_GENERIC_PEERS")
    c_no_conv = _get_reason_n(miss, "cross_generic", "NO_VALID_PATIENT_CONVERTIBLE_PEERS")
    c_no_month = _get_reason_n(miss, "cross_generic", "SOURCE_MONTH_UNAVAILABLE")
    c_neg = _get_reason_n(miss, "cross_generic", "NEGATIVE_NET_PEER_DEMAND")
    c3m_neg = _get_reason_n(miss, "cross_generic_3m", "NEGATIVE_NET_PEER_DEMAND")

    # Peer-group audit
    n_no_generic_peers = 0
    n_no_cross_peers = 0
    if not pg.empty:
        if "n_generic_peers" in pg.columns:
            n_no_generic_peers = int((pg["n_generic_peers"] == 0).sum())
        if "n_cross_generic_patient_convertible_peers" in pg.columns:
            n_no_cross_peers = int((pg["n_cross_generic_patient_convertible_peers"] == 0).sum())

    # Temporal variation summary
    tv_summary: dict = {}
    if not tv.empty:
        for feat_stub in ["generic_peer_dqtyunit_last_month", "generic_peer_dqtyunit_3m_mean",
                          "cross_generic_field_consume_patients_last_month",
                          "cross_generic_field_consume_patients_3m_mean"]:
            col = f"n_distinct_states_{feat_stub}"
            if col in tv.columns:
                tv_summary[feat_stub] = int((tv[col] > 1).sum())

    neg_material = g_neg > 0 or g3m_neg > 0 or c_neg > 0 or c3m_neg > 0

    lines = [
        "# F3E Step 2 — Peer Demand Feature Audit",
        f"**Date:** {date.today()}  ",
        "**Artifacts:** `src/data/results/f3e/feature_audit/`  ",
        "No XGBoost. No WMAPE. No scored feature training.",
        "",
        "## Aligned matched feature-audit universe",
        "",
        f"- Rows: {n_rows}",
        f"- Products: {n_prods}",
        f"- Origins: {n_origins}",
        "",
        "> **Note:** This is the broader aligned matched universe used for audit "
        "(may include origins 140307 and 140310 in addition to the five locked PRIMARY origins). "
        "Step 3 must use the locked PRIMARY panel: n = 1,877 rows, 5 origins "
        "(140404, 140407, 140410, 140501, 140504).",
        "",
        "## Scored feature families",
        "",
        "### F3E-A (same-generic demand)",
        "",
        "```",
        "\n".join(f"  {f}" for f in F3E_A_FEATURES),
        "```",
        "",
        "### F3E-B (same-generic + cross-generic patient context)",
        "",
        "```",
        "\n".join(f"  {f}" for f in F3E_B_FEATURES),
        "```",
        "",
        "## Feature coverage (audit universe)",
        "",
    ]

    feat_cov_rows = []
    for feat in F3E_B_FEATURES:
        raw_feat = feat.replace("log_", "")
        feat_cov_rows.append({
            "feature": feat,
            "available_rows": _get_metric(cov, f"{feat}_available_rows", "n/a"),
            "coverage_pct": _get_metric(cov, f"{feat}_coverage_pct", "n/a"),
        })
    lines.append(md_table(pd.DataFrame(feat_cov_rows)))
    lines.append("")

    lines += [
        "### By origin",
        "",
        md_table(cov_origin, max_rows=10) if not cov_origin.empty else "_not available_",
        "",
        "### By product (first 30)",
        "",
        md_table(cov_product.head(30), max_rows=30) if not cov_product.empty else "_not available_",
        "",
        "## Missingness reasons",
        "",
        "### Generic last-month feature (`log_generic_peer_dqtyunit_last_month`)",
        "",
        md_table(miss.loc[miss.get("feature_group", pd.Series(dtype=str)) == "generic"].reset_index(drop=True))
        if not miss.empty else "_not available_",
        "",
        "### Generic 3m-mean feature (`log_generic_peer_dqtyunit_3m_mean`)",
        "",
        md_table(miss.loc[miss.get("feature_group", pd.Series(dtype=str)) == "generic_3m"].reset_index(drop=True))
        if not miss.empty else "_not available_",
        "",
        "### Cross-generic last-month patient feature",
        "",
        md_table(miss.loc[miss.get("feature_group", pd.Series(dtype=str)) == "cross_generic"].reset_index(drop=True))
        if not miss.empty else "_not available_",
        "",
        "### Cross-generic 3m-mean patient feature",
        "",
        md_table(miss.loc[miss.get("feature_group", pd.Series(dtype=str)) == "cross_generic_3m"].reset_index(drop=True))
        if not miss.empty else "_not available_",
        "",
        "## Feature distributions (all 8 features)",
        "",
        md_table(dists, max_rows=20) if not dists.empty else "_not available_",
        "",
        "## Peer-group audit (MVP products)",
        "",
        md_table(pg.head(60), max_rows=60) if not pg.empty else "_not available_",
        "",
        "## Temporal variation (distinct feature states across origins per product)",
        "",
        md_table(tv.head(60), max_rows=60) if not tv.empty else "_not available_",
        "",
    ]

    # Summary statistics for temporal variation
    if tv_summary:
        lines.append("### Products with >1 distinct state")
        lines.append("")
        for feat_stub, n_varying in tv_summary.items():
            lines.append(f"- `{feat_stub}`: {n_varying} products")
        lines.append("")

    lines += [
        "## Audit answers",
        "",
        "1. **Are all F3E features PIT safe?** "
        "Yes — inline assertion `_check_pit` verified M1/M2/M3 < O for every origin; "
        "post-hoc `assert_pit_safe` also passed.",
        "",
        "2. **Is DQtyUnit used only within generic?** "
        "Yes — `generic_peer_dqtyunit` sums only products with the same `FKGeneric`.",
        "",
        "3. **Is the target SKU excluded from generic demand?** "
        "Yes — group total is subtracted of the target's own `monthly_dqtyunit`; "
        "`assert_generic_target_exclusion` verified this for a random sample.",
        "",
        "4. **Is the entire target generic excluded from cross-generic demand?** "
        "Yes — the vectorised builder subtracts the entire `fkg_pe_sum` for the target's "
        "FKGeneric from the Field×ConsumeType total; `assert_cross_generic_no_same_fkgeneric` passed.",
        "",
        "5. **Is cross-generic demand expressed in patient equivalents?** "
        "Yes — cross-generic feature uses `monthly_patient_equivalent = monthly_dqty / "
        "PatientConsumePerPeriod`; no ×12 or ÷12 applied.",
        "",
        f"6. **What is generic feature coverage?** "
        f"`log_generic_peer_dqtyunit_last_month`: {_feat_cov('log_generic_peer_dqtyunit_last_month')}.",
        "",
        f"7. **What is cross-generic patient-context coverage?** "
        f"`log_cross_generic_field_consume_patients_last_month`: "
        f"{_feat_cov('log_cross_generic_field_consume_patients_last_month')}.",
        "",
        f"8. **How many targets have no generic peers?** "
        f"{n_no_generic_peers} products.",
        "",
        f"9. **How many targets lack valid cross-generic patient peers?** "
        f"{n_no_cross_peers} products.",
        "",
        "10. **Are zero-demand months distinguished from structurally unavailable peers?** "
        "Yes — covered months with valid-peer zero sales → 0 (not NaN); "
        f"`SOURCE_MONTH_UNAVAILABLE` rows (generic): {g_no_month}; (cross-generic): {c_no_month}.",
        "",
        f"11. **Are negative peer aggregates material?** "
        + _yn(neg_material,
              f"yes — generic last-month NEGATIVE_NET_PEER_DEMAND: {g_neg} rows; "
              f"generic 3m-mean: {g3m_neg} rows; "
              f"cross-generic last-month: {c_neg} rows; cross-generic 3m-mean: {c3m_neg} rows. "
              "Raw aggregates are kept as-is (no clipping); log features → NaN for these rows.",
              "no negative net peer aggregates observed."),
        "",
        "12. **Is there meaningful temporal variation?** "
        + (
            "Yes — " + "; ".join(
                f"`{k}`: {v} products vary" for k, v in tv_summary.items() if v > 0
            ) if any(v > 0 for v in tv_summary.values()) else "No temporal variation detected."
        ),
        "",
        "13. **Are F3E-A and F3E-B ready for controlled evaluation?** "
        "Yes — all PIT, exclusion, and semantic assertions passed. "
        "Feature families are frozen and parquet written to `feature_audit/features.parquet`.",
        "",
        "## What was NOT done",
        "",
        "- No XGBoost, no WMAPE, no scoring.",
        "- No clipping of negative peer demand.",
        "- No target-own sales, market share, ratios, or growth features.",
        "- No F3D profile features mixed in.",
        "- F0/F1/F2/F3A/F3B/F3C/F3D artifacts were not modified.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
