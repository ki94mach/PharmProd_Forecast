"""Human Budget origin assignment v2: Version → quarter → origin_from_quarter.

Authoritative quarter source is SQL ``Version`` (YYYYQQEE), mapped by
``version_to_qrt``. Do **not** derive quarter or origin from ``min(target_date)``.

Horizon (relative to declared origin)::

    horizon = 12 * (target_year - origin_year) + (target_month - origin_month) + 1
            = shamsi_month_diff(target_date, budget_origin) + 1

Keep only ``1 <= horizon <= 15``. Exclusions are audited; horizons are never
renumbered or filled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.benchmark.calendar import origin_from_quarter, shamsi_month_diff
from pkg.benchmark.config import FORECAST_HORIZON_MONTHS
from pkg.db.query.product_budget import version_to_qrt

REASON_PRE_ORIGIN = "pre_origin"
REASON_BEYOND_H15 = "beyond_h15"
REASON_NON_FINITE_DATE = "non_finite_date"

REQUIRED_HORIZONS = frozenset(range(1, FORECAST_HORIZON_MONTHS + 1))


class BudgetOriginAmbiguityError(ValueError):
    """Raised when Version/quarter assignment cannot be proven."""


class BudgetOriginDuplicateError(ValueError):
    """Raised when product×origin×horizon duplicates remain after aggregation."""


def _version_qq(version: int) -> int:
    return (int(version) // 100) % 100


def assert_version_quarter_consistent(df: pd.DataFrame) -> None:
    """Hard-fail if Version is invalid or disagrees with an existing ``qrt`` column."""
    if df.empty:
        return
    if "version" not in df.columns and "budget_version" not in df.columns:
        raise BudgetOriginAmbiguityError(
            "Version column missing: need 'version' or 'budget_version' "
            "(SQL Version YYYYQQEE is the authoritative quarter source)."
        )
    ver_col = "version" if "version" in df.columns else "budget_version"
    versions = pd.to_numeric(df[ver_col], errors="coerce")
    if versions.isna().any():
        bad = int(versions.isna().sum())
        raise BudgetOriginAmbiguityError(
            f"{bad} row(s) have non-numeric/missing Version; cannot assign quarter."
        )
    qq = versions.astype(int).map(_version_qq)
    bad_qq = ~qq.isin({1, 2, 3, 4})
    if bad_qq.any():
        samples = (
            versions.loc[bad_qq]
            .astype(int)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise BudgetOriginAmbiguityError(
            f"Version QQ must be 1..4; invalid Version samples={samples}"
        )
    derived = versions.astype(int).map(version_to_qrt)
    if "qrt" in df.columns:
        existing = df["qrt"].astype(str).str.strip().str.upper()
        mismatch = existing != derived.astype(str).str.strip().str.upper()
        if mismatch.any():
            sample = df.loc[mismatch, [ver_col, "qrt"]].head(5)
            raise BudgetOriginAmbiguityError(
                "Existing qrt disagrees with version_to_qrt(Version); "
                f"refusing to infer from target_date. sample=\n{sample}"
            )


def horizon_from_origin_target(origin: int, target_date: int) -> int:
    """Horizon relative to declared origin (h1 = origin month)."""
    return int(shamsi_month_diff(int(target_date), int(origin)) + 1)


def assign_budget_origin_from_version(df: pd.DataFrame) -> pd.DataFrame:
    """Add source_budget_quarter, budget_origin, horizon from Version (not min date)."""
    assert_version_quarter_consistent(df)
    out = df.copy()
    ver_col = "version" if "version" in out.columns else "budget_version"
    out["budget_version"] = pd.to_numeric(out[ver_col], errors="coerce").astype(int)
    out["source_budget_quarter"] = out["budget_version"].map(version_to_qrt)
    out["qrt"] = out["source_budget_quarter"]
    out["budget_origin"] = out["source_budget_quarter"].map(origin_from_quarter).astype(int)

    if "target_date" in out.columns:
        td = pd.to_numeric(out["target_date"], errors="coerce")
    elif "date" in out.columns:
        td = pd.to_numeric(out["date"], errors="coerce")
    else:
        raise BudgetOriginAmbiguityError("Need target_date or date (YYYYMM).")
    out["target_date"] = td
    out["date"] = out["target_date"]

    horizons: list[float] = []
    for origin, target in zip(out["budget_origin"], out["target_date"], strict=True):
        if target != target or origin != origin:  # NaN
            horizons.append(float("nan"))
        else:
            horizons.append(float(horizon_from_origin_target(int(origin), int(target))))
    out["horizon"] = horizons
    return out


def filter_horizons_1_15(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep 1..15; return (kept, excluded_with_reason). Never renumber horizons."""
    if df.empty:
        empty = df.copy()
        if "exclusion_reason" not in empty.columns:
            empty["exclusion_reason"] = pd.Series(dtype=str)
        return empty.copy(), empty.copy()

    out = df.copy()
    reasons: list[str] = []
    keep_mask: list[bool] = []
    for h in out["horizon"]:
        if h != h:  # NaN
            reasons.append(REASON_NON_FINITE_DATE)
            keep_mask.append(False)
        elif float(h) < 1:
            reasons.append(REASON_PRE_ORIGIN)
            keep_mask.append(False)
        elif float(h) > FORECAST_HORIZON_MONTHS:
            reasons.append(REASON_BEYOND_H15)
            keep_mask.append(False)
        else:
            reasons.append("")
            keep_mask.append(True)
    out["_keep"] = keep_mask
    out["exclusion_reason"] = reasons
    kept = out.loc[out["_keep"]].drop(columns=["_keep", "exclusion_reason"]).copy()
    kept["horizon"] = kept["horizon"].astype(int)
    excluded = out.loc[~out["_keep"]].drop(columns=["_keep"]).copy()
    return kept, excluded


def assert_no_duplicate_poh(df: pd.DataFrame) -> None:
    """Hard-fail remaining product × budget_origin × horizon duplicates."""
    if df.empty:
        return
    keys = ["product", "budget_origin", "horizon"]
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise BudgetOriginDuplicateError(f"Missing columns for dup check: {missing}")
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].drop_duplicates().head(10)
        raise BudgetOriginDuplicateError(
            "Duplicate product×budget_origin×horizon after aggregation; "
            f"sample=\n{sample}"
        )


def aggregate_forecast_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Documented deterministic resolve: mean forecast on (product, qrt, version, date)."""
    work = df.copy()
    if "version" not in work.columns and "budget_version" in work.columns:
        work["version"] = work["budget_version"]
    if "forecast" not in work.columns and "budget_forecast" in work.columns:
        work["forecast"] = work["budget_forecast"]
    if "date" not in work.columns and "target_date" in work.columns:
        work["date"] = work["target_date"]
    if "qrt" not in work.columns and "source_budget_quarter" in work.columns:
        work["qrt"] = work["source_budget_quarter"]
    need = {"product", "qrt", "version", "date", "forecast"}
    missing = need - set(work.columns)
    if missing:
        raise BudgetOriginAmbiguityError(f"aggregate_forecast_mean missing {missing}")
    agg: dict[str, str] = {"forecast": "mean"}
    if "generic" in work.columns:
        agg["generic"] = "first"
    return work.groupby(["product", "qrt", "version", "date"], as_index=False).agg(agg)


def completeness_by_origin(df: pd.DataFrame) -> dict[str, Any]:
    """Exact {1..15} completeness; incomplete groups list missing horizons (no fill)."""
    by_origin: dict[str, Any] = {}
    if df.empty:
        return {"by_origin": by_origin, "n_complete_groups": 0, "n_incomplete_groups": 0}

    n_complete = 0
    n_incomplete = 0
    incomplete_groups: list[dict[str, Any]] = []

    for origin, g_o in df.groupby(df["budget_origin"].astype(int), sort=True):
        complete = 0
        incomplete = 0
        products_any = int(g_o["product"].nunique())
        origin_incomplete: list[dict[str, Any]] = []
        for product, pg in g_o.groupby("product", sort=False):
            hs = set(int(h) for h in pg["horizon"].dropna().astype(int).tolist())
            missing = sorted(REQUIRED_HORIZONS - hs)
            extra = sorted(hs - REQUIRED_HORIZONS)
            if missing or extra or len(hs) != FORECAST_HORIZON_MONTHS:
                incomplete += 1
                n_incomplete += 1
                entry = {
                    "product": str(product),
                    "budget_origin": int(origin),
                    "horizons_present": sorted(hs),
                    "missing": missing,
                    "extra": extra,
                }
                origin_incomplete.append(entry)
                incomplete_groups.append(entry)
            else:
                complete += 1
                n_complete += 1
        by_origin[str(int(origin))] = {
            "n_products_with_any_row": products_any,
            "n_complete_1_15": complete,
            "n_incomplete": incomplete,
            "incomplete_groups": origin_incomplete,
        }
    return {
        "by_origin": by_origin,
        "n_complete_groups": n_complete,
        "n_incomplete_groups": n_incomplete,
        "incomplete_groups": incomplete_groups,
    }


def build_budget_panel_v2(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Full transform: aggregate → assign → filter → dup check.

    Returns (kept_panel, excluded_audit, stats).
    """
    n_before = int(len(raw))
    assert_version_quarter_consistent(raw)
    agg = aggregate_forecast_mean(raw)
    assigned = assign_budget_origin_from_version(agg)
    kept, excluded = filter_horizons_1_15(assigned)
    assert_no_duplicate_poh(kept)

    panel = kept.copy()
    panel = panel.rename(columns={"forecast": "budget_forecast"})
    cols = [
        "product",
        "source_budget_quarter",
        "qrt",
        "budget_version",
        "budget_origin",
        "target_date",
        "date",
        "horizon",
        "budget_forecast",
    ]
    if "generic" in panel.columns:
        cols.insert(1, "generic")
    panel = panel[[c for c in cols if c in panel.columns]].copy()
    panel["horizon"] = panel["horizon"].astype(int)
    panel["budget_origin"] = panel["budget_origin"].astype(int)
    panel["target_date"] = panel["target_date"].astype(int)
    panel["date"] = panel["target_date"].astype(int)

    excl = excluded.copy()
    if "forecast" in excl.columns:
        excl = excl.rename(columns={"forecast": "budget_forecast"})

    stats = {
        "n_rows_before_filter": n_before,
        "n_rows_after_aggregate": int(len(agg)),
        "n_rows_kept_1_15": int(len(panel)),
        "n_rows_excluded": int(len(excl)),
        "n_excluded_pre_origin": int(
            (excl["exclusion_reason"] == REASON_PRE_ORIGIN).sum()
        )
        if len(excl)
        else 0,
        "n_excluded_beyond_h15": int(
            (excl["exclusion_reason"] == REASON_BEYOND_H15).sum()
        )
        if len(excl)
        else 0,
        "n_excluded_non_finite_date": int(
            (excl["exclusion_reason"] == REASON_NON_FINITE_DATE).sum()
        )
        if len(excl)
        else 0,
        "distinct_budget_origins": sorted(
            int(x) for x in panel["budget_origin"].unique()
        )
        if len(panel)
        else [],
        "completeness": completeness_by_origin(panel),
        "quarter_to_origin_mapping": {
            q: int(origin_from_quarter(q))
            for q in sorted(panel["source_budget_quarter"].unique())
        }
        if len(panel)
        else {},
    }
    return panel, excl, stats


def content_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return content_sha256_bytes(body)
