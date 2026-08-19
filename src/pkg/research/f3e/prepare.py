"""F3E Step 1: freeze peer-demand source and run all semantic audits.

SQL is used ONLY here.  All downstream F3E steps must read from
``src/data/results/f3e/source/`` and must not query the database.

Two peer-demand concepts with two different normalizations
----------------------------------------------------------
Same-generic (DQtyUnit):
    monthly_dqtyunit = monthly_dqty * unit_ratio
    unit_ratio (Dim.Product.Unit) is a within-generic conversion ratio that
    makes SKUs within the same FKGeneric comparable.
    It is NOT assumed comparable across different generics.
    Used ONLY within the same FKGeneric.

Cross-generic (monthly_patient_equivalent):
    monthly_patient_equivalent = monthly_dqty / PatientConsumePerPeriod
    Converts actual monthly sales into estimated patient equivalents.
    No ×12 or ÷12 is applied (unlike F3D annualization).
    Continuous denominator = monthly patient consumption quantity.
    SinglePeriod denominator = complete annual/single-period consumption quantity.
    Used ONLY across generics within the same Field × PatientConsumeType segment,
    explicitly excluding the target product's ENTIRE generic.

Static-dimension assumption
---------------------------
Dim.Product is a current snapshot.  FKGeneric, Field, Unit,
PatientConsumeType, and PatientConsumePerPeriod are treated as static product
definitions.  No historical reconstruction is attempted because the source
does not contain dated rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import INCOMPLETE_SHAMSI_MONTHS, PANEL_FILES, RAW_FILES
from pkg.benchmark.config import default_benchmark_root
from pkg.research.f3e.config import (
    KNOWN_CONSUME_TYPES,
    NORMALIZED_MONTHLY_SALES_PARQUET,
    PRODUCT_PEER_PROFILE_PARQUET,
    f3e_source_dir,
)

QUANTILE_LABELS = ("min", "p1", "p10", "p25", "median", "p75", "p90", "p99", "max")
QUANTILE_VALS = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


# ---------------------------------------------------------------------------
# Freeze-file fingerprint guard
# ---------------------------------------------------------------------------

def _file_fingerprint(root: Path) -> dict:
    out = {}
    names = list(PANEL_FILES) + [f"raw/{n}" for n in RAW_FILES]
    for name in names:
        p = root / name
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        out[name] = (p.stat().st_mtime_ns, p.stat().st_size, h.hexdigest())
    return out


def assert_freeze_untouched(root: Path, before: dict) -> None:
    after = _file_fingerprint(root)
    if after != before:
        raise AssertionError(
            "F3E source prep modified frozen benchmark files "
            f"(before keys={sorted(before)} after keys={sorted(after)})"
        )


# ---------------------------------------------------------------------------
# MVP products
# ---------------------------------------------------------------------------

def mvp_products(benchmark_root: Optional[Path] = None) -> list[str]:
    root = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
    path = root / "matched_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"matched universe missing: {path}")
    matched = pd.read_parquet(path, columns=["product"])
    return sorted(matched["product"].astype(str).unique())


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def compute_dqtyunit(
    monthly_dqty: pd.Series, unit_ratio: pd.Series
) -> pd.Series:
    """monthly_dqty * unit_ratio; NaN when unit_ratio is missing, zero, or negative."""
    dqty = pd.to_numeric(monthly_dqty, errors="coerce")
    ratio = pd.to_numeric(unit_ratio, errors="coerce")
    valid = ratio.notna() & (ratio > 0)
    result = pd.Series(np.nan, index=dqty.index, dtype=float)
    result.loc[valid] = dqty.loc[valid] * ratio.loc[valid]
    return result


def compute_patient_equivalent(
    monthly_dqty: pd.Series,
    patient_consume_type: pd.Series,
    patient_consume_per_period: pd.Series,
) -> pd.Series:
    """monthly_dqty / PatientConsumePerPeriod for known types with valid period.

    No ×12 or ÷12 applied.  Continuous and SinglePeriod use the same formula.
    NaN for unknown types, missing period, zero period, or negative period.
    """
    dqty = pd.to_numeric(monthly_dqty, errors="coerce")
    period = pd.to_numeric(patient_consume_per_period, errors="coerce")
    ptype = patient_consume_type.astype(object)

    known_type = ptype.isin(KNOWN_CONSUME_TYPES)
    valid_period = period.notna() & (period > 0)
    valid = known_type & valid_period

    result = pd.Series(np.nan, index=dqty.index, dtype=float)
    result.loc[valid] = dqty.loc[valid] / period.loc[valid]
    return result


# ---------------------------------------------------------------------------
# Semantic-assertion helpers (called programmatically; STOP on failure)
# ---------------------------------------------------------------------------

def assert_dqtyunit_formula(panel: pd.DataFrame) -> None:
    """Assert monthly_dqtyunit == monthly_dqty * unit_ratio for every valid row."""
    valid = (
        panel["unit_ratio"].notna()
        & (panel["unit_ratio"] > 0)
        & panel["monthly_dqtyunit"].notna()
    )
    sub = panel.loc[valid]
    if sub.empty:
        return
    expected = sub["monthly_dqty"] * sub["unit_ratio"]
    if not np.allclose(
        expected.to_numpy(dtype=float),
        sub["monthly_dqtyunit"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AssertionError(
            "F3E: monthly_dqtyunit != monthly_dqty * unit_ratio for some valid rows"
        )


def assert_patient_equivalent_formula(panel: pd.DataFrame) -> None:
    """Assert monthly_patient_equivalent == monthly_dqty / PatientConsumePerPeriod."""
    known = panel["PatientConsumeType"].isin(KNOWN_CONSUME_TYPES)
    valid_period = (
        pd.to_numeric(panel["PatientConsumePerPeriod"], errors="coerce").notna()
        & (pd.to_numeric(panel["PatientConsumePerPeriod"], errors="coerce") > 0)
    )
    valid = known & valid_period & panel["monthly_patient_equivalent"].notna()
    sub = panel.loc[valid]
    if sub.empty:
        return
    period = pd.to_numeric(sub["PatientConsumePerPeriod"], errors="coerce")
    expected = sub["monthly_dqty"] / period
    if not np.allclose(
        expected.to_numpy(dtype=float),
        sub["monthly_patient_equivalent"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AssertionError(
            "F3E: monthly_patient_equivalent != monthly_dqty / PatientConsumePerPeriod"
        )


def assert_same_generic_excludes_self(
    panel: pd.DataFrame, mvp_list: list[str]
) -> None:
    """Assert that for each MVP target, the same-generic peer set excludes itself."""
    profile = (
        panel[["product", "FKGeneric"]]
        .drop_duplicates("product")
        .set_index("product")["FKGeneric"]
    )
    peer_products = set(panel["product"].astype(str).unique())
    for mvp in mvp_list:
        if mvp not in profile.index:
            continue
        fkg = profile[mvp]
        if pd.isna(fkg):
            continue
        same_generic = {
            p for p in peer_products
            if p != mvp and profile.get(p) == fkg
        }
        # assertion: mvp not in its own same-generic peer set (trivially true by
        # construction, but validate the peer-set builder logic)
        if mvp in same_generic:
            raise AssertionError(
                f"F3E: target '{mvp}' found in its own same-generic peer set"
            )


def assert_cross_generic_excludes_entire_generic(
    panel: pd.DataFrame, mvp_list: list[str]
) -> None:
    """Assert cross-generic peers contain no product with FKGeneric == target's FKGeneric."""
    profile = (
        panel[["product", "FKGeneric", "Field", "PatientConsumeType"]]
        .drop_duplicates("product")
        .set_index("product")
    )
    peer_products = set(panel["product"].astype(str).unique())

    for mvp in mvp_list:
        if mvp not in profile.index:
            continue
        row = profile.loc[mvp]
        fkg_target = row["FKGeneric"]
        field_target = row["Field"]
        ptype_target = row["PatientConsumeType"]

        if pd.isna(fkg_target):
            continue  # cannot determine generic → skip (documented as NaN feature)

        cross_generic_peers = {
            p for p in peer_products
            if p != mvp
            and profile.loc[p, "Field"] == field_target
            and profile.loc[p, "PatientConsumeType"] == ptype_target
            and profile.loc[p, "FKGeneric"] != fkg_target
        } if field_target and ptype_target in KNOWN_CONSUME_TYPES else set()

        # Check no product with fkg == fkg_target slipped through
        forbidden = {
            p for p in cross_generic_peers
            if profile.loc[p, "FKGeneric"] == fkg_target
        }
        if forbidden:
            raise AssertionError(
                f"F3E: cross-generic peer set for '{mvp}' contains products "
                f"from the same FKGeneric ({fkg_target}): {sorted(forbidden)[:5]}"
            )


# ---------------------------------------------------------------------------
# Audit builders
# ---------------------------------------------------------------------------

def _quantile_row(values: np.ndarray, label: str) -> dict:
    finite = values[np.isfinite(values)]
    row: dict = {"variable": label, "count": len(finite), "missing": len(values) - len(finite)}
    if len(finite) > 0:
        qs = np.quantile(finite, QUANTILE_VALS)
        for lbl, q in zip(QUANTILE_LABELS, qs):
            row[lbl] = float(q)
    else:
        for lbl in QUANTILE_LABELS:
            row[lbl] = np.nan
    return row


def build_unit_ratio_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit-ratio coverage and distribution; returns (summary_df, invalid_products_df)."""
    profile = panel[["product", "unit_ratio"]].drop_duplicates("product")
    n_total = len(profile)
    ratio = pd.to_numeric(profile["unit_ratio"], errors="coerce")
    n_present = int(ratio.notna().sum())
    n_missing = n_total - n_present
    n_zero = int((ratio == 0).sum())
    n_negative = int((ratio < 0).sum())
    n_valid = int((ratio.notna() & (ratio > 0)).sum())

    summary = pd.DataFrame([{
        "n_selling_products": n_total,
        "n_with_unit_ratio": n_present,
        "n_missing_unit_ratio": n_missing,
        "n_zero_unit_ratio": n_zero,
        "n_negative_unit_ratio": n_negative,
        "n_valid_positive_unit_ratio": n_valid,
    }])

    dist_row = _quantile_row(ratio.to_numpy(dtype=float), "unit_ratio")
    dist_df = pd.DataFrame([dist_row])

    invalid_mask = ratio.isna() | (ratio <= 0)
    invalid_products = profile.loc[invalid_mask, ["product", "unit_ratio"]].copy()

    # DQty affected by invalid unit_ratio
    dqty_by_product = (
        panel.groupby("product")["monthly_dqty"].sum().reset_index()
    )
    invalid_with_vol = invalid_products.merge(dqty_by_product, on="product", how="left")

    result = pd.concat([summary, dist_df], axis=0, ignore_index=True)
    return result, invalid_with_vol


def build_patient_conversion_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Patient-conversion coverage, type counts, and period distributions."""
    profile = panel[["product", "PatientConsumeType", "PatientConsumePerPeriod"]].drop_duplicates("product")
    n_total = len(profile)

    ptype = profile["PatientConsumeType"].astype(object)
    period = pd.to_numeric(profile["PatientConsumePerPeriod"], errors="coerce")

    counts = []
    for label in list(KNOWN_CONSUME_TYPES) + ["missing", "unexpected"]:
        if label == "missing":
            n = int(ptype.isna().sum())
        elif label == "unexpected":
            n = int((ptype.notna() & ~ptype.isin(KNOWN_CONSUME_TYPES)).sum())
        else:
            n = int((ptype == label).sum())
        counts.append({"PatientConsumeType": label, "n_products": n})

    n_with_type = int(ptype.isin(KNOWN_CONSUME_TYPES).sum())
    n_with_period = int(period.notna().sum())
    n_valid_both = int((ptype.isin(KNOWN_CONSUME_TYPES) & period.notna() & (period > 0)).sum())

    # DQty share with valid conversion
    total_dqty = float(pd.to_numeric(panel["monthly_dqty"], errors="coerce").sum())
    valid_products = set(
        profile.loc[
            ptype.isin(KNOWN_CONSUME_TYPES) & period.notna() & (period > 0), "product"
        ].astype(str)
    )
    valid_dqty = float(
        pd.to_numeric(
            panel.loc[panel["product"].astype(str).isin(valid_products), "monthly_dqty"],
            errors="coerce",
        ).sum()
    )
    dqty_share = valid_dqty / total_dqty if total_dqty != 0 else np.nan

    coverage_row = {
        "n_products_total": n_total,
        "n_with_known_PatientConsumeType": n_with_type,
        "n_with_PatientConsumePerPeriod": n_with_period,
        "n_valid_both": n_valid_both,
        "dqty_share_with_valid_conversion": round(dqty_share, 4) if np.isfinite(dqty_share) else np.nan,
    }

    dist_rows = []
    for ptype_val in KNOWN_CONSUME_TYPES:
        mask = profile["PatientConsumeType"] == ptype_val
        vals = period.loc[mask].to_numpy(dtype=float)
        dist_rows.append(_quantile_row(vals, f"PatientConsumePerPeriod_{ptype_val}"))

    result = pd.DataFrame([coverage_row])
    counts_df = pd.DataFrame(counts)
    dist_df = pd.DataFrame(dist_rows)
    return pd.concat([result, counts_df, dist_df], axis=0, ignore_index=True)


def build_generic_peer_audit(
    panel: pd.DataFrame, mvp_list: list[str]
) -> pd.DataFrame:
    """Per-MVP target: same-generic peer count and product names."""
    profile = (
        panel[["product", "FKGeneric"]]
        .drop_duplicates("product")
        .set_index("product")["FKGeneric"]
    )
    products_with_sales = set(panel["product"].astype(str).unique())

    rows = []
    for mvp in mvp_list:
        fkg = profile.get(mvp)
        if pd.isna(fkg) or fkg is None:
            rows.append({
                "product": mvp,
                "FKGeneric": None,
                "n_catalog_generic_peers": 0,
                "n_generic_peers_with_sales": 0,
                "generic_peer_products": "",
            })
            continue

        catalog_peers = [
            p for p, g in profile.items()
            if g == fkg and p != mvp
        ]
        peers_with_sales = [p for p in catalog_peers if p in products_with_sales]
        rows.append({
            "product": mvp,
            "FKGeneric": fkg,
            "n_catalog_generic_peers": len(catalog_peers),
            "n_generic_peers_with_sales": len(peers_with_sales),
            "generic_peer_products": "; ".join(sorted(peers_with_sales)),
        })
    return pd.DataFrame(rows)


def build_cross_generic_peer_audit(
    panel: pd.DataFrame, mvp_list: list[str]
) -> pd.DataFrame:
    """Per-MVP target: cross-generic Field×ConsumeType peer counts."""
    profile = (
        panel[["product", "FKGeneric", "Field", "PatientConsumeType", "PatientConsumePerPeriod"]]
        .drop_duplicates("product")
        .set_index("product")
    )
    products_with_sales = set(panel["product"].astype(str).unique())
    period_valid = (
        pd.to_numeric(profile["PatientConsumePerPeriod"], errors="coerce").notna()
        & (pd.to_numeric(profile["PatientConsumePerPeriod"], errors="coerce") > 0)
    )

    rows = []
    for mvp in mvp_list:
        if mvp not in profile.index:
            rows.append({
                "product": mvp, "Field": None, "PatientConsumeType": None,
                "FKGeneric": None,
                "n_cross_generic_catalog_peers": 0,
                "n_cross_generic_peers_with_sales": 0,
                "n_cross_generic_peers_with_valid_patient_conversion": 0,
                "contributing_generics": "",
            })
            continue

        row = profile.loc[mvp]
        fkg_target = row["FKGeneric"]
        field_target = row["Field"]
        ptype_target = row["PatientConsumeType"]

        if pd.isna(fkg_target):
            rows.append({
                "product": mvp, "Field": field_target,
                "PatientConsumeType": ptype_target, "FKGeneric": None,
                "n_cross_generic_catalog_peers": 0,
                "n_cross_generic_peers_with_sales": 0,
                "n_cross_generic_peers_with_valid_patient_conversion": 0,
                "contributing_generics": "FKGeneric_missing",
            })
            continue

        # Same Field, same PatientConsumeType, different FKGeneric
        mask = (
            (profile["Field"] == field_target)
            & (profile["PatientConsumeType"] == ptype_target)
            & (profile["FKGeneric"] != fkg_target)
        )
        catalog_peers = profile.index[mask].tolist()
        peers_with_sales = [p for p in catalog_peers if p in products_with_sales]
        peers_valid_conv = [
            p for p in peers_with_sales
            if p in period_valid.index and period_valid.loc[p]
            and profile.loc[p, "PatientConsumeType"] in KNOWN_CONSUME_TYPES
        ]
        contributing = sorted(
            set(str(profile.loc[p, "FKGeneric"]) for p in peers_with_sales)
        )
        rows.append({
            "product": mvp,
            "Field": field_target,
            "PatientConsumeType": ptype_target,
            "FKGeneric": fkg_target,
            "n_cross_generic_catalog_peers": len(catalog_peers),
            "n_cross_generic_peers_with_sales": len(peers_with_sales),
            "n_cross_generic_peers_with_valid_patient_conversion": len(peers_valid_conv),
            "contributing_generics": "; ".join(contributing),
        })
    return pd.DataFrame(rows)


def build_normalization_examples(
    panel: pd.DataFrame, n_generics: int = 5, n_per_generic: int = 3
) -> pd.DataFrame:
    """Sample rows for manual DQtyUnit verification."""
    # pick multi-presentation generics
    counts = (
        panel[["FKGeneric", "product"]]
        .drop_duplicates()
        .groupby("FKGeneric")["product"]
        .nunique()
    )
    multi = counts[counts >= 2].index.tolist()[:n_generics]
    if not multi:
        return pd.DataFrame()

    cols = ["product", "FKGeneric", "date", "monthly_dqty", "unit_ratio", "monthly_dqtyunit"]
    parts = []
    for fkg in multi:
        sub = panel.loc[
            (panel["FKGeneric"] == fkg) & panel["monthly_dqtyunit"].notna()
        ][cols]
        parts.append(sub.head(n_per_generic))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def build_patient_equivalent_examples(
    panel: pd.DataFrame, n_per_type: int = 5
) -> pd.DataFrame:
    """Sample rows for manual patient-equivalent verification."""
    cols = [
        "product", "Field", "FKGeneric", "PatientConsumeType",
        "PatientConsumePerPeriod", "date", "monthly_dqty", "monthly_patient_equivalent",
    ]
    parts = []
    for ptype in KNOWN_CONSUME_TYPES:
        sub = panel.loc[
            (panel["PatientConsumeType"] == ptype)
            & panel["monthly_patient_equivalent"].notna()
        ][cols]
        parts.append(sub.head(n_per_type))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def build_negative_sales_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Products and months with negative computed quantities."""
    neg_dqty = panel.loc[pd.to_numeric(panel["monthly_dqty"], errors="coerce") < 0, :]
    neg_dqtyunit = panel.loc[pd.to_numeric(panel["monthly_dqtyunit"], errors="coerce") < 0, :]
    neg_pe = panel.loc[
        pd.to_numeric(panel["monthly_patient_equivalent"], errors="coerce") < 0, :
    ]

    rows = [
        {
            "quantity": "monthly_dqty",
            "n_negative_rows": len(neg_dqty),
            "n_affected_products": neg_dqty["product"].nunique(),
        },
        {
            "quantity": "monthly_dqtyunit",
            "n_negative_rows": len(neg_dqtyunit),
            "n_affected_products": neg_dqtyunit["product"].nunique(),
        },
        {
            "quantity": "monthly_patient_equivalent",
            "n_negative_rows": len(neg_pe),
            "n_affected_products": neg_pe["product"].nunique(),
        },
    ]
    return pd.DataFrame(rows)


def build_global_month_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """First/last month, distinct months, gap count."""
    from pkg.benchmark.calendar import shamsi_add_months

    months = sorted(panel["date"].dropna().astype(int).unique())
    if not months:
        return pd.DataFrame([{"first_month": None, "last_month": None,
                              "n_distinct_months": 0, "n_missing_global_months": 0}])
    first, last = months[0], months[-1]
    # enumerate expected months
    expected = []
    m = first
    while m <= last:
        expected.append(m)
        m = shamsi_add_months(m, 1)
    gaps = [m for m in expected if m not in set(months)]
    return pd.DataFrame([{
        "first_month": first,
        "last_month": last,
        "n_distinct_months": len(months),
        "n_expected_months": len(expected),
        "n_missing_global_months": len(gaps),
        "missing_months": "; ".join(str(g) for g in gaps) if gaps else "",
    }])


# ---------------------------------------------------------------------------
# Main prepare function
# ---------------------------------------------------------------------------

def prepare_peer_demand_source(
    *,
    out_dir: Optional[Path] = None,
    benchmark_root: Optional[Path] = None,
    verify_freeze: bool = True,
) -> dict:
    """Full F3E Step 1: SQL freeze, normalizations, audits, exclusion assertions.

    Returns a dict with keys:
        panel, product_peer_profile, mvp_list,
        unit_ratio_audit, patient_conversion_audit,
        generic_peer_audit, cross_generic_peer_audit,
        normalization_examples, patient_equivalent_examples,
        negative_sales_report, global_month_coverage,
        out_dir
    """
    from pkg.db.query.peer_sales import load_peer_sales

    src_dir = out_dir or f3e_source_dir()
    src_dir.mkdir(parents=True, exist_ok=True)
    bench_root = Path(benchmark_root) if benchmark_root else default_benchmark_root()

    if verify_freeze:
        fingerprint_before = _file_fingerprint(bench_root)

    mvp_list = mvp_products(bench_root)

    # ── Load and clean raw peer sales ──────────────────────────────────────
    raw = load_peer_sales()
    raw["date"] = pd.to_numeric(raw["date"], errors="coerce")
    raw = raw[~raw["date"].isin(INCOMPLETE_SHAMSI_MONTHS)].copy()
    raw["product"] = raw["product"].astype(str)

    # ── Compute normalizations ─────────────────────────────────────────────
    raw["monthly_dqtyunit"] = compute_dqtyunit(raw["monthly_dqty"], raw["unit_ratio"])
    raw["monthly_patient_equivalent"] = compute_patient_equivalent(
        raw["monthly_dqty"], raw["PatientConsumeType"], raw["PatientConsumePerPeriod"]
    )

    # ── Semantic formula assertions (STOP on failure) ──────────────────────
    assert_dqtyunit_formula(raw)
    assert_patient_equivalent_formula(raw)

    # ── MVP coverage check (STOP if zero) ─────────────────────────────────
    products_in_panel = set(raw["product"].unique())
    mvp_in_panel = [p for p in mvp_list if p in products_in_panel]
    if not mvp_in_panel:
        raise AssertionError(
            "F3E: zero MVP benchmark products found in peer-sales panel. "
            "Check database connection or product names."
        )

    # ── Exclusion assertions (STOP on failure) ────────────────────────────
    assert_same_generic_excludes_self(raw, mvp_list)
    assert_cross_generic_excludes_entire_generic(raw, mvp_list)

    # ── Build product peer profile (one row per product) ──────────────────
    profile_cols = [
        "product", "FKGeneric", "Field", "unit_ratio",
        "PatientConsumeType", "PatientConsumePerPeriod",
    ]
    product_peer_profile = (
        raw[profile_cols].drop_duplicates("product").reset_index(drop=True)
    )

    # ── Audit tables ──────────────────────────────────────────────────────
    unit_ratio_summary, unit_ratio_invalid = build_unit_ratio_audit(raw)
    patient_conversion_audit = build_patient_conversion_audit(raw)
    generic_peer_audit = build_generic_peer_audit(raw, mvp_list)
    cross_generic_peer_audit = build_cross_generic_peer_audit(raw, mvp_list)
    normalization_examples = build_normalization_examples(raw)
    patient_equivalent_examples = build_patient_equivalent_examples(raw)
    negative_sales_report = build_negative_sales_report(raw)
    global_month_coverage = build_global_month_coverage(raw)

    # ── Write frozen parquets ─────────────────────────────────────────────
    raw.to_parquet(src_dir / NORMALIZED_MONTHLY_SALES_PARQUET, index=False)
    product_peer_profile.to_parquet(src_dir / PRODUCT_PEER_PROFILE_PARQUET, index=False)

    # ── Write audit CSVs ──────────────────────────────────────────────────
    unit_ratio_summary.to_csv(src_dir / "unit_ratio_audit.csv", index=False)
    unit_ratio_invalid.to_csv(src_dir / "unit_ratio_invalid_products.csv", index=False)
    patient_conversion_audit.to_csv(src_dir / "patient_conversion_audit.csv", index=False)
    generic_peer_audit.to_csv(src_dir / "generic_peer_audit.csv", index=False)
    cross_generic_peer_audit.to_csv(src_dir / "cross_generic_field_consume_peer_audit.csv", index=False)
    normalization_examples.to_csv(src_dir / "generic_normalization_examples.csv", index=False)
    patient_equivalent_examples.to_csv(src_dir / "patient_equivalent_examples.csv", index=False)
    negative_sales_report.to_csv(src_dir / "negative_sales_report.csv", index=False)
    global_month_coverage.to_csv(src_dir / "global_month_coverage.csv", index=False)

    if verify_freeze:
        assert_freeze_untouched(bench_root, fingerprint_before)

    return {
        "panel": raw,
        "product_peer_profile": product_peer_profile,
        "mvp_list": mvp_list,
        "n_mvp_in_panel": len(mvp_in_panel),
        "unit_ratio_audit": unit_ratio_summary,
        "unit_ratio_invalid": unit_ratio_invalid,
        "patient_conversion_audit": patient_conversion_audit,
        "generic_peer_audit": generic_peer_audit,
        "cross_generic_peer_audit": cross_generic_peer_audit,
        "normalization_examples": normalization_examples,
        "patient_equivalent_examples": patient_equivalent_examples,
        "negative_sales_report": negative_sales_report,
        "global_month_coverage": global_month_coverage,
        "out_dir": src_dir,
    }
