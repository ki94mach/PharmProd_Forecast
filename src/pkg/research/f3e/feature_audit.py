"""F3E Step 2 — pre-model feature audit (no XGB, no WMAPE).

Reads the enriched PRIMARY DataFrame produced by features.build_f3e_features()
and writes audit CSVs to src/data/results/f3e/feature_audit/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PRIMARY_ORIGINS
from pkg.research.f3e.config import KNOWN_CONSUME_TYPES, f3e_feature_audit_dir
from pkg.research.f3e.features import (
    ALL_FEATURE_NAMES,
    LOG_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    _C_AVAILABLE,
    _C_NEGATIVE,
    _C_NO_CONV,
    _C_NO_MONTH,
    _C_NO_PEERS,
    _G_AVAILABLE,
    _G_INVALID_UNIT,
    _G_NEGATIVE,
    _G_NO_MONTH,
    _G_NO_PEERS,
)

QUANTILE_VALS = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)
QUANTILE_NAMES = ("min", "p1", "p10", "p25", "median", "p75", "p90", "p99", "max")

SCORED_FEATURES = LOG_FEATURE_NAMES


def _pct(n: int, d: int) -> float:
    return float(n) / float(d) * 100.0 if d > 0 else float("nan")


def _finite_mask(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").apply(lambda v: bool(np.isfinite(v)))


def _origin_col(df: pd.DataFrame) -> str:
    for cand in ("budget_origin", "ts_origin", "origin"):
        if cand in df.columns:
            return cand
    raise KeyError(f"No origin column in columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_overall(enriched: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(enriched)
    n_prods = enriched["product"].nunique()
    oc = _origin_col(enriched)
    n_origins = enriched[oc].nunique()

    rows = [{"metric": "n_rows", "value": n_rows},
            {"metric": "n_products", "value": n_prods},
            {"metric": "n_origins", "value": n_origins}]

    for feat in SCORED_FEATURES:
        n_avail = int(_finite_mask(enriched[feat]).sum())
        rows.append({
            "metric": f"{feat}_available_rows",
            "value": n_avail,
        })
        rows.append({
            "metric": f"{feat}_coverage_pct",
            "value": round(_pct(n_avail, n_rows), 2),
        })
    return pd.DataFrame(rows)


def coverage_by_origin(enriched: pd.DataFrame) -> pd.DataFrame:
    oc = _origin_col(enriched)
    rows = []
    for o in sorted(enriched[oc].dropna().unique()):
        g = enriched.loc[enriched[oc].astype(int) == int(o)]
        n = len(g)
        np_ = g["product"].nunique()
        row = {"origin": int(o), "n_rows": n, "n_products": np_}
        for feat in SCORED_FEATURES:
            n_avail = int(_finite_mask(g[feat]).sum())
            row[f"{feat}_available"] = n_avail
            row[f"{feat}_coverage_pct"] = round(_pct(n_avail, n), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_by_product(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product, g in enriched.groupby("product"):
        n = len(g)
        row = {"product": str(product), "n_rows": n}
        for feat in SCORED_FEATURES:
            n_avail = int(_finite_mask(g[feat]).sum())
            row[f"{feat}_available"] = n_avail
            row[f"{feat}_coverage_pct"] = round(_pct(n_avail, n), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Missingness reasons
# ---------------------------------------------------------------------------

def missingness(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(enriched)

    # Generic last-month reason
    g_codes = [_G_AVAILABLE, _G_NO_PEERS, _G_INVALID_UNIT, _G_NO_MONTH, _G_NEGATIVE]
    for code in g_codes:
        n = int((enriched["generic_missing_reason"] == code).sum())
        rows.append({
            "feature": "log_generic_peer_dqtyunit_last_month",
            "feature_group": "generic",
            "reason": code,
            "n_rows": n,
            "pct": round(_pct(n, total), 2),
        })

    # Generic 3m-mean reason
    g3m_col = "generic_3m_missing_reason"
    if g3m_col in enriched.columns:
        for code in g_codes:
            n = int((enriched[g3m_col] == code).sum())
            rows.append({
                "feature": "log_generic_peer_dqtyunit_3m_mean",
                "feature_group": "generic_3m",
                "reason": code,
                "n_rows": n,
                "pct": round(_pct(n, total), 2),
            })

    # Cross-generic last-month reason
    c_codes = [_C_AVAILABLE, _C_NO_PEERS, _C_NO_CONV, _C_NO_MONTH, _C_NEGATIVE]
    for code in c_codes:
        n = int((enriched["cross_generic_missing_reason"] == code).sum())
        rows.append({
            "feature": "log_cross_generic_field_consume_patients_last_month",
            "feature_group": "cross_generic",
            "reason": code,
            "n_rows": n,
            "pct": round(_pct(n, total), 2),
        })

    # Cross-generic 3m-mean reason
    c3m_col = "cross_generic_3m_missing_reason"
    if c3m_col in enriched.columns:
        for code in c_codes:
            n = int((enriched[c3m_col] == code).sum())
            rows.append({
                "feature": "log_cross_generic_field_consume_patients_3m_mean",
                "feature_group": "cross_generic_3m",
                "reason": code,
                "n_rows": n,
                "pct": round(_pct(n, total), 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def distributions(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in ALL_FEATURE_NAMES:
        if feat not in enriched.columns:
            continue
        vals = pd.to_numeric(enriched[feat], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        row = {
            "feature": feat,
            "n_total": len(vals),
            "n_finite": len(finite),
            "n_nan": len(vals) - len(finite),
        }
        if len(finite) > 0:
            qs = np.quantile(finite, QUANTILE_VALS)
            for lbl, q in zip(QUANTILE_NAMES, qs):
                row[lbl] = float(q)
        else:
            for lbl in QUANTILE_NAMES:
                row[lbl] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Peer-group audit (per product)
# ---------------------------------------------------------------------------

def peer_group_audit(
    panel: pd.DataFrame,
    profile: pd.DataFrame,
    mvp_list: list[str],
) -> pd.DataFrame:
    """For each MVP product report peer counts (not scored as features)."""
    prod_profile = (
        profile[["product", "FKGeneric", "Field", "PatientConsumeType", "PatientConsumePerPeriod"]]
        .drop_duplicates("product")
        .set_index("product")
    )

    rows = []
    for mvp in mvp_list:
        row: dict = {"product": mvp}

        if mvp not in prod_profile.index:
            row.update({
                "n_generic_peers": 0,
                "n_generic_peers_with_valid_unit": 0,
                "n_cross_generic_field_consume_peers": 0,
                "n_cross_generic_patient_convertible_peers": 0,
            })
            rows.append(row)
            continue

        pr = prod_profile.loc[mvp]
        fkg = pr["FKGeneric"]
        field = pr["Field"]
        ptype = pr["PatientConsumeType"]

        # Generic peers
        if pd.isna(fkg):
            n_g = 0
            n_g_valid_unit = 0
        else:
            generic_profile = prod_profile[
                (prod_profile["FKGeneric"] == fkg)
                & (prod_profile.index != mvp)
            ]
            n_g = len(generic_profile)
            # Valid unit = has rows with non-null monthly_dqtyunit in panel
            if n_g > 0:
                peer_prods = set(generic_profile.index)
                panel_peers = panel[panel["product"].isin(peer_prods)]
                n_g_valid_unit = int(
                    panel_peers[panel_peers["monthly_dqtyunit"].notna()]["product"].nunique()
                )
            else:
                n_g_valid_unit = 0

        # Cross-generic peers
        if pd.isna(field) or pd.isna(ptype) or ptype not in KNOWN_CONSUME_TYPES:
            n_cg = 0
            n_cg_convertible = 0
        else:
            cross_profile = prod_profile[
                (prod_profile["Field"] == field)
                & (prod_profile["PatientConsumeType"] == ptype)
                & (prod_profile["FKGeneric"] != fkg)
            ]
            n_cg = len(cross_profile)
            if n_cg > 0:
                cross_prods = set(cross_profile.index)
                panel_cross = panel[panel["product"].isin(cross_prods)]
                # Convertible = has valid monthly_patient_equivalent
                n_cg_convertible = int(
                    panel_cross[panel_cross["monthly_patient_equivalent"].notna()]["product"].nunique()
                )
            else:
                n_cg_convertible = 0

        row.update({
            "FKGeneric": fkg,
            "Field": field,
            "PatientConsumeType": ptype,
            "n_generic_peers": n_g,
            "n_generic_peers_with_valid_unit": n_g_valid_unit,
            "n_cross_generic_field_consume_peers": n_cg,
            "n_cross_generic_patient_convertible_peers": n_cg_convertible,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Temporal variation
# ---------------------------------------------------------------------------

def temporal_variation(enriched: pd.DataFrame) -> pd.DataFrame:
    """Per product: how many distinct (non-NaN) states across origins.

    A 'state' is the finite value of a raw feature across multiple origins.
    Products with >1 state have meaningful temporal variation.
    """
    oc = _origin_col(enriched)
    tracked_features = [
        "generic_peer_dqtyunit_last_month",
        "generic_peer_dqtyunit_3m_mean",
        "cross_generic_field_consume_patients_last_month",
        "cross_generic_field_consume_patients_3m_mean",
    ]

    rows = []
    for product, g in enriched.groupby("product"):
        row = {"product": str(product), "n_origins": g[oc].nunique()}
        for feat in tracked_features:
            if feat not in g.columns:
                row[f"n_distinct_states_{feat}"] = 0
                continue
            finite_vals = pd.to_numeric(g[feat], errors="coerce").dropna()
            n_distinct = int(finite_vals.nunique())
            row[f"n_distinct_states_{feat}"] = n_distinct
        rows.append(row)

    df = pd.DataFrame(rows)

    # Summary: products with >1 state per feature
    for feat in tracked_features:
        col = f"n_distinct_states_{feat}"
        if col in df.columns:
            n_varying = int((df[col] > 1).sum())
            df.attrs[f"n_products_varying_{feat}"] = n_varying

    return df


# ---------------------------------------------------------------------------
# Run all audits and write CSVs
# ---------------------------------------------------------------------------

def run_feature_audit(
    enriched: pd.DataFrame,
    panel: pd.DataFrame,
    profile: pd.DataFrame,
    mvp_list: list[str],
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Run all audit functions and write CSVs. Returns dict of DataFrames."""
    out = out_dir or f3e_feature_audit_dir()
    out.mkdir(parents=True, exist_ok=True)

    cov_overall = coverage_overall(enriched)
    cov_origin = coverage_by_origin(enriched)
    cov_product = coverage_by_product(enriched)
    miss = missingness(enriched)
    dists = distributions(enriched)
    pg_audit = peer_group_audit(panel, profile, mvp_list)
    tv = temporal_variation(enriched)

    cov_overall.to_csv(out / "coverage_overall.csv", index=False)
    cov_origin.to_csv(out / "coverage_by_origin.csv", index=False)
    cov_product.to_csv(out / "coverage_by_product.csv", index=False)
    miss.to_csv(out / "missingness_reasons.csv", index=False)
    dists.to_csv(out / "distributions.csv", index=False)
    pg_audit.to_csv(out / "peer_group_audit.csv", index=False)
    tv.to_csv(out / "temporal_variation.csv", index=False)

    return {
        "coverage_overall": cov_overall,
        "coverage_by_origin": cov_origin,
        "coverage_by_product": cov_product,
        "missingness": miss,
        "distributions": dists,
        "peer_group_audit": pg_audit,
        "temporal_variation": tv,
        "out_dir": out,
    }
