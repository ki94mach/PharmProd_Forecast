"""Freeze Dim.Product patient-consumption profile into F3D source artefacts.

SQL is used **only here**.  All downstream F3D steps read
``src/data/results/f3d/source/product_profile.parquet`` and must not query
the database.

Steps
-----
1. Load canonical MVP products from frozen ``matched_universe.parquet``.
2. Load ``Dim.Product`` via :func:`~pkg.db.query.dim_product.load_dim_product`.
3. Exact join on ``ProductTitleEN == product`` (no fuzzy matching).
4. Validate mapping (duplicate conflicts, zero overlap → STOP).
5. Compute ``patient_annual_consumption``, ``log_patient_annual_consumption``,
   ``is_continuous_consumption``.
6. Assert annualisation semantics programmatically.
7. Write ``product_profile.parquet`` and profile-audit CSVs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.config import PANEL_FILES, RAW_FILES, default_benchmark_root
from pkg.research.f3d.config import f3d_profile_audit_dir, f3d_source_dir
from pkg.research.features.patient_consumption import (
    KNOWN_TYPES,
    _compute_annual,
    _compute_indicator,
)

PROFILE_PARQUET = "product_profile.parquet"


# ---------------------------------------------------------------------------
# Freeze-file fingerprint guard
# ---------------------------------------------------------------------------

def _file_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
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
            "F3D source prep modified frozen benchmark files "
            f"(before keys={sorted(before)} after keys={sorted(after)})"
        )


# ---------------------------------------------------------------------------
# MVP products from frozen matched_universe
# ---------------------------------------------------------------------------

def mvp_products(benchmark_root: Optional[Path] = None) -> list[str]:
    root = Path(benchmark_root) if benchmark_root is not None else default_benchmark_root()
    path = root / "matched_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"matched universe missing: {path}")
    matched = pd.read_parquet(path, columns=["product"])
    return sorted(matched["product"].astype(str).unique())


# ---------------------------------------------------------------------------
# Mapping & validation
# ---------------------------------------------------------------------------

class DuplicateConflictError(Exception):
    """Raised when a ProductTitleEN maps to conflicting profile rows."""


class ZeroOverlapError(Exception):
    """Raised when no canonical products match any Dim.Product row."""


def _validate_no_conflicts(dim: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by ProductTitleEN; raise if conflicting values exist."""
    key = "ProductTitleEN"
    cols = [key, "PatientConsumeType", "PatientConsumePerPeriod"]
    sub = dim[[c for c in cols if c in dim.columns]].copy()
    sub[key] = sub[key].astype(str)
    # drop truly identical rows
    sub = sub.drop_duplicates()
    dupes = sub[sub.duplicated(key, keep=False)]
    if not dupes.empty:
        conflict_products = dupes[key].unique().tolist()
        raise DuplicateConflictError(
            f"F3D: {len(conflict_products)} ProductTitleEN(s) have conflicting "
            f"PatientConsumeType or PatientConsumePerPeriod: {conflict_products[:10]}"
        )
    return sub.drop_duplicates(key)


def build_product_profile(
    mvp_list: list[str],
    dim_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map Dim.Product to canonical products and compute F3D features.

    Returns
    -------
    profile:
        One row per canonical product with F3D feature columns.
    audit:
        Mapping audit with coverage information.
    """
    # Restrict to MVP products before conflict check so non-MVP rows with
    # conflicting profiles do not cause a STOP.
    mvp_set = set(mvp_list)
    dim_mvp = dim_df[
        dim_df["ProductTitleEN"].astype(str).isin(mvp_set)
    ].copy()
    canonical = _validate_no_conflicts(dim_mvp)
    canonical = canonical.rename(columns={"ProductTitleEN": "product"})
    canonical["product"] = canonical["product"].astype(str)

    # Exact join
    mvp_df = pd.DataFrame({"product": mvp_list})
    merged = mvp_df.merge(canonical, on="product", how="left")

    n_products = len(mvp_list)
    n_matched = int(merged["PatientConsumeType"].notna().sum())
    if n_matched == 0:
        raise ZeroOverlapError(
            "F3D: zero canonical products matched Dim.Product on ProductTitleEN. "
            "Check database connection or product name format."
        )

    # Detect unexpected PatientConsumeType values
    type_series = merged["PatientConsumeType"].dropna().astype(str)
    unexpected_types = sorted(set(type_series.unique()) - KNOWN_TYPES)

    # Compute annualised consumption
    ptypes = merged["PatientConsumeType"].to_numpy(dtype=object)
    pperiods = pd.to_numeric(
        merged.get("PatientConsumePerPeriod", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    annual = np.array([_compute_annual(t, p) for t, p in zip(ptypes, pperiods)])

    # Negative period → NaN + report
    neg_mask = np.isfinite(pperiods) & (pperiods < 0)
    annual[neg_mask] = np.nan

    log_annual = np.where(
        np.isfinite(annual) & (annual >= 0),
        np.log1p(annual),
        np.nan,
    )
    indicator = np.array([_compute_indicator(t) for t in ptypes])

    merged["patient_annual_consumption"] = annual
    merged["log_patient_annual_consumption"] = log_annual
    merged["is_continuous_consumption"] = indicator

    # ── Programmatic semantic assertions ────────────────────────────────────
    continuous_mask = merged["PatientConsumeType"] == "Continuous"
    single_mask = merged["PatientConsumeType"] == "SinglePeriod"

    cont_sub = merged.loc[
        continuous_mask
        & merged["PatientConsumePerPeriod"].notna()
        & (pd.to_numeric(merged["PatientConsumePerPeriod"], errors="coerce") >= 0)
    ]
    if not cont_sub.empty:
        expected = pd.to_numeric(cont_sub["PatientConsumePerPeriod"], errors="coerce") * 12.0
        actual = cont_sub["patient_annual_consumption"]
        if not np.allclose(expected.to_numpy(float), actual.to_numpy(float), equal_nan=True):
            raise AssertionError("F3D: Continuous annualisation (×12) assertion failed")

    sing_sub = merged.loc[
        single_mask
        & merged["PatientConsumePerPeriod"].notna()
        & (pd.to_numeric(merged["PatientConsumePerPeriod"], errors="coerce") >= 0)
    ]
    if not sing_sub.empty:
        expected = pd.to_numeric(sing_sub["PatientConsumePerPeriod"], errors="coerce")
        actual = sing_sub["patient_annual_consumption"]
        if not np.allclose(expected.to_numpy(float), actual.to_numpy(float), equal_nan=True):
            raise AssertionError("F3D: SinglePeriod annual == raw assertion failed")

    audit = pd.DataFrame(
        [
            {
                "n_products": n_products,
                "n_matched_ProductTitleEN": n_matched,
                "n_with_PatientConsumeType": int(
                    merged["PatientConsumeType"].notna().sum()
                ),
                "n_with_PatientConsumePerPeriod": int(
                    pd.to_numeric(
                        merged.get(
                            "PatientConsumePerPeriod",
                            pd.Series(np.nan, index=merged.index),
                        ),
                        errors="coerce",
                    )
                    .notna()
                    .sum()
                ),
                "coverage_pct": round(n_matched / n_products * 100, 2),
                "n_unexpected_types": len(unexpected_types),
                "unexpected_types": "; ".join(unexpected_types) if unexpected_types else "",
                "n_negative_period": int(neg_mask.sum()),
            }
        ]
    )

    return merged, audit


# ---------------------------------------------------------------------------
# Main prepare function
# ---------------------------------------------------------------------------

def prepare_profile_source(
    *,
    out_dir: Optional[Path] = None,
    audit_dir: Optional[Path] = None,
    benchmark_root: Optional[Path] = None,
    verify_freeze: bool = True,
) -> dict:
    """Full F3D Step 1: load SQL, validate, freeze, write audit CSVs.

    Returns a dict with keys:
    ``profile``, ``audit``, ``type_counts``,
    ``period_distributions``, ``annual_distributions``,
    ``log_distributions``, ``semantic_table``,
    ``negative_report``, ``out_dir``.
    """
    from pkg.db.query.dim_product import load_dim_product

    src_dir = out_dir or f3d_source_dir()
    aud_dir = audit_dir or f3d_profile_audit_dir()
    src_dir.mkdir(parents=True, exist_ok=True)
    aud_dir.mkdir(parents=True, exist_ok=True)

    bench_root = Path(benchmark_root) if benchmark_root else default_benchmark_root()

    if verify_freeze:
        fingerprint_before = _file_fingerprint(bench_root)

    mvp_list = mvp_products(bench_root)

    dim_df = load_dim_product()

    profile, audit = build_product_profile(mvp_list, dim_df)

    # ── Type counts ─────────────────────────────────────────────────────────
    type_counts = (
        profile["PatientConsumeType"]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "PatientConsumeType", "count": "n_products"})
    )
    if "PatientConsumeType" not in type_counts.columns:
        type_counts.columns = ["PatientConsumeType", "n_products"]

    # ── Per-type distributions ───────────────────────────────────────────────
    quantile_labels = ("min", "p10", "p25", "median", "p75", "p90", "max")
    quantile_vals = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)

    def _dist_rows(series: pd.Series, group: str, varname: str) -> list[dict]:
        rows = []
        for ptype in ["Continuous", "SinglePeriod", "all"]:
            if ptype == "all":
                sub = series
            else:
                mask = profile["PatientConsumeType"] == ptype
                sub = series.loc[mask]
            vals = pd.to_numeric(sub, errors="coerce").dropna()
            row = {
                "variable": varname,
                "PatientConsumeType": ptype,
                "count": len(vals),
                "missing": len(sub) - len(vals),
            }
            if len(vals) > 0:
                qs = np.quantile(vals, quantile_vals)
                for lbl, q in zip(quantile_labels, qs):
                    row[lbl] = float(q)
            else:
                for lbl in quantile_labels:
                    row[lbl] = np.nan
            rows.append(row)
        return rows

    period_rows = _dist_rows(
        pd.to_numeric(profile.get("PatientConsumePerPeriod", pd.Series(dtype=float)), errors="coerce"),
        "PatientConsumePerPeriod", "PatientConsumePerPeriod",
    )
    annual_rows = _dist_rows(
        profile["patient_annual_consumption"], "patient_annual_consumption", "patient_annual_consumption",
    )
    log_rows = _dist_rows(
        profile["log_patient_annual_consumption"], "log_patient_annual_consumption", "log_patient_annual_consumption",
    )

    period_distributions = pd.DataFrame(period_rows)
    annual_distributions = pd.DataFrame(annual_rows)
    log_distributions = pd.DataFrame(log_rows)

    # ── Semantic validation table ────────────────────────────────────────────
    semantic_cols = [
        "product",
        "PatientConsumeType",
        "PatientConsumePerPeriod",
        "patient_annual_consumption",
        "log_patient_annual_consumption",
        "is_continuous_consumption",
    ]
    semantic_table = profile[[c for c in semantic_cols if c in profile.columns]].copy()

    # ── Negative report ──────────────────────────────────────────────────────
    neg_mask = (
        pd.to_numeric(
            profile.get("PatientConsumePerPeriod", pd.Series(dtype=float)),
            errors="coerce",
        )
        < 0
    ) & pd.to_numeric(
        profile.get("PatientConsumePerPeriod", pd.Series(dtype=float)),
        errors="coerce",
    ).notna()
    negative_report = profile.loc[neg_mask, ["product", "PatientConsumeType", "PatientConsumePerPeriod"]]

    # ── Write frozen parquet ─────────────────────────────────────────────────
    profile.to_parquet(src_dir / PROFILE_PARQUET, index=False)

    # ── Write audit CSVs ─────────────────────────────────────────────────────
    audit.to_csv(aud_dir / "coverage_overall.csv", index=False)
    type_counts.to_csv(aud_dir / "type_counts.csv", index=False)
    period_distributions.to_csv(aud_dir / "period_distributions.csv", index=False)
    annual_distributions.to_csv(aud_dir / "annual_distributions.csv", index=False)
    log_distributions.to_csv(aud_dir / "log_distributions.csv", index=False)
    semantic_table.to_csv(aud_dir / "semantic_table.csv", index=False)
    if not negative_report.empty:
        negative_report.to_csv(aud_dir / "negative_period_report.csv", index=False)

    if verify_freeze:
        assert_freeze_untouched(bench_root, fingerprint_before)

    return {
        "profile": profile,
        "audit": audit,
        "type_counts": type_counts,
        "period_distributions": period_distributions,
        "annual_distributions": annual_distributions,
        "log_distributions": log_distributions,
        "semantic_table": semantic_table,
        "negative_report": negative_report,
        "out_dir": src_dir,
        "audit_dir": aud_dir,
    }
