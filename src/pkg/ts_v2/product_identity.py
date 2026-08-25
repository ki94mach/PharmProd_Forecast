"""Product identity audit helpers for TS V2.

V1 uses ``ProductTitleEN`` as the join key between basket universe and sales.
This module validates whether ``Dim.Product.ID_INT`` (via ``Flat_Fact_Sale.FKProduct``)
is safe to adopt as the V2 SKU key before changing loaders or backtests.

See ``docs/ts_v2_product_identity.md`` for the full audit write-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from pkg.db.query.dim_product import select_basket_products

PRODUCT_ID_COL = "ID_INT"
PRODUCT_TITLE_COL = "ProductTitleEN"
SALES_FK_COL = "FKProduct"


class ProductIdentityBlockedError(Exception):
    """Raised when product-identity data quality fails configured thresholds."""


@dataclass(frozen=True)
class MappingIssue:
    """One many-to-one mapping violation."""

    key: Any
    values: tuple[Any, ...]
    n_rows: int


@dataclass
class ProductIdentityAuditReport:
    """Summary of offline product-identity checks."""

    n_dim_rows: int = 0
    n_basket_rows_before_dedupe: int = 0
    n_basket_rows_after_dedupe: int = 0
    n_distinct_id_int: int = 0
    n_distinct_titles: int = 0
    id_to_titles: list[MappingIssue] = field(default_factory=list)
    title_to_ids: list[MappingIssue] = field(default_factory=list)
    duplicate_basket_by_id: list[MappingIssue] = field(default_factory=list)
    duplicate_basket_by_title: list[MappingIssue] = field(default_factory=list)
    n_sales_rows: Optional[int] = None
    n_sales_null_fk: Optional[int] = None
    n_sales_title_mismatch: Optional[int] = None

    @property
    def has_blocking_issues(self) -> bool:
        return bool(
            self.id_to_titles
            or self.title_to_ids
            or self.duplicate_basket_by_id
            or self.duplicate_basket_by_title
            or (self.n_sales_null_fk or 0) > 0
            or (self.n_sales_title_mismatch or 0) > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_dim_rows": self.n_dim_rows,
            "n_basket_rows_before_dedupe": self.n_basket_rows_before_dedupe,
            "n_basket_rows_after_dedupe": self.n_basket_rows_after_dedupe,
            "n_distinct_id_int": self.n_distinct_id_int,
            "n_distinct_titles": self.n_distinct_titles,
            "n_id_to_titles_violations": len(self.id_to_titles),
            "n_title_to_ids_violations": len(self.title_to_ids),
            "n_duplicate_basket_by_id": len(self.duplicate_basket_by_id),
            "n_duplicate_basket_by_title": len(self.duplicate_basket_by_title),
            "n_sales_rows": self.n_sales_rows,
            "n_sales_null_fk": self.n_sales_null_fk,
            "n_sales_title_mismatch": self.n_sales_title_mismatch,
            "has_blocking_issues": self.has_blocking_issues,
        }


def _active_basket_mask(dim_df: pd.DataFrame) -> pd.Series:
    """Same filters as ``select_basket_products`` but without dedupe."""
    if dim_df is None or dim_df.empty:
        return pd.Series(dtype=bool)
    work = dim_df.copy()
    titles = work[PRODUCT_TITLE_COL].astype("string").str.strip()
    ok_title = titles.notna() & titles.ne("") & titles.ne("nan")
    ok_basket = pd.to_numeric(work["ProductBasket"], errors="coerce") == 1
    fields = work["Field"].astype("string").str.strip()
    ok_field = fields.notna() & fields.ne("") & fields.ne("-")
    if "StatusCode" not in work.columns:
        return pd.Series(False, index=work.index)
    status = work["StatusCode"].astype("string").str.strip()
    ok_status = status == "Active"
    return ok_title & ok_basket & ok_field & ok_status


def _many_to_one_issues(
    df: pd.DataFrame,
    key_col: str,
    value_col: str,
) -> list[MappingIssue]:
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return []
    sub = df[[key_col, value_col]].copy()
    sub[key_col] = sub[key_col].astype(str)
    sub[value_col] = sub[value_col].astype("string").str.strip()
    sub = sub.loc[sub[key_col].notna() & sub[value_col].notna() & sub[value_col].ne("")]
    if sub.empty:
        return []
    grouped = (
        sub.groupby(key_col, sort=False)[value_col]
        .agg(lambda s: tuple(sorted(set(s.astype(str)))))
        .reset_index()
    )
    bad = grouped[grouped[value_col].map(len) > 1]
    issues: list[MappingIssue] = []
    for _, row in bad.iterrows():
        issues.append(
            MappingIssue(
                key=row[key_col],
                values=tuple(row[value_col]),
                n_rows=int(sub.loc[sub[key_col] == row[key_col]].shape[0]),
            )
        )
    return issues


def _duplicate_key_issues(df: pd.DataFrame, key_col: str) -> list[MappingIssue]:
    if df.empty or key_col not in df.columns:
        return []
    sub = df[[key_col]].copy()
    sub[key_col] = sub[key_col].astype(str)
    sub = sub.loc[sub[key_col].notna() & sub[key_col].ne("")]
    if sub.empty:
        return []
    counts = sub.groupby(key_col, sort=False).size()
    dupes = counts[counts > 1]
    issues: list[MappingIssue] = []
    for key, n in dupes.items():
        issues.append(MappingIssue(key=key, values=(key,), n_rows=int(n)))
    return issues


def audit_id_to_titles(dim_df: pd.DataFrame) -> list[MappingIssue]:
    """One ``ID_INT`` mapping to multiple ``ProductTitleEN`` values."""
    if PRODUCT_ID_COL not in dim_df.columns:
        return []
    return _many_to_one_issues(dim_df, PRODUCT_ID_COL, PRODUCT_TITLE_COL)


def audit_title_to_ids(dim_df: pd.DataFrame) -> list[MappingIssue]:
    """One ``ProductTitleEN`` mapping to multiple ``ID_INT`` values."""
    if PRODUCT_ID_COL not in dim_df.columns:
        return []
    return _many_to_one_issues(dim_df, PRODUCT_TITLE_COL, PRODUCT_ID_COL)


def audit_basket_duplicates(dim_df: pd.DataFrame) -> tuple[list[MappingIssue], list[MappingIssue]]:
    """Duplicated active basket rows by ``ID_INT`` and by ``ProductTitleEN``."""
    mask = _active_basket_mask(dim_df)
    basket = dim_df.loc[mask].copy()
    by_id = _duplicate_key_issues(basket, PRODUCT_ID_COL) if PRODUCT_ID_COL in basket.columns else []
    by_title = _duplicate_key_issues(basket, PRODUCT_TITLE_COL)
    return by_id, by_title


def audit_sales_fk_and_titles(
    sales_df: pd.DataFrame,
    *,
    fk_col: str = SALES_FK_COL,
    fact_title_col: str = "ProductTitleEN",
    dim_title_col: str = "dim_product_title",
) -> tuple[int, int]:
    """Return ``(n_null_fk, n_title_mismatch)`` for a FK-joined sales extract."""
    if sales_df.empty:
        return 0, 0
    n_null_fk = 0
    if fk_col in sales_df.columns:
        n_null_fk = int(sales_df[fk_col].isna().sum())
    n_mismatch = 0
    if fact_title_col in sales_df.columns and dim_title_col in sales_df.columns:
        fact = sales_df[fact_title_col].astype("string").str.strip()
        dim = sales_df[dim_title_col].astype("string").str.strip()
        comparable = fact.notna() & dim.notna() & fact.ne("") & dim.ne("")
        n_mismatch = int((fact[comparable].str.casefold() != dim[comparable].str.casefold()).sum())
    return n_null_fk, n_mismatch


def run_product_identity_audit(
    dim_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
) -> ProductIdentityAuditReport:
    """Run all offline product-identity checks on dimension (and optional sales) frames."""
    report = ProductIdentityAuditReport()
    if dim_df is None or dim_df.empty:
        return report

    report.n_dim_rows = int(len(dim_df))
    basket_mask = _active_basket_mask(dim_df)
    basket_before = dim_df.loc[basket_mask]
    basket_after = select_basket_products(dim_df)
    report.n_basket_rows_before_dedupe = int(len(basket_before))
    report.n_basket_rows_after_dedupe = int(len(basket_after))

    if PRODUCT_ID_COL in dim_df.columns:
        ids = dim_df[PRODUCT_ID_COL].dropna().astype(str)
        report.n_distinct_id_int = int(ids.nunique())
    if PRODUCT_TITLE_COL in dim_df.columns:
        titles = dim_df[PRODUCT_TITLE_COL].astype("string").str.strip()
        titles = titles.loc[titles.notna() & titles.ne("") & titles.ne("nan")]
        report.n_distinct_titles = int(titles.nunique())

    report.id_to_titles = audit_id_to_titles(dim_df)
    report.title_to_ids = audit_title_to_ids(dim_df)
    by_id, by_title = audit_basket_duplicates(dim_df)
    report.duplicate_basket_by_id = by_id
    report.duplicate_basket_by_title = by_title

    if sales_df is not None:
        report.n_sales_rows = int(len(sales_df))
        null_fk, mismatch = audit_sales_fk_and_titles(sales_df)
        report.n_sales_null_fk = null_fk
        report.n_sales_title_mismatch = mismatch

    return report


def assert_product_identity_ready(
    report: ProductIdentityAuditReport,
    *,
    allow_title_to_ids: bool = False,
    allow_id_to_titles: bool = False,
    allow_duplicate_basket: bool = False,
    allow_sales_null_fk: bool = False,
    allow_sales_title_mismatch: bool = False,
) -> None:
    """Raise ``ProductIdentityBlockedError`` when audit violations exceed policy."""
    problems: list[str] = []
    if report.id_to_titles and not allow_id_to_titles:
        problems.append(f"{len(report.id_to_titles)} ID_INT -> multiple titles")
    if report.title_to_ids and not allow_title_to_ids:
        problems.append(f"{len(report.title_to_ids)} titles -> multiple ID_INT")
    if (report.duplicate_basket_by_id or report.duplicate_basket_by_title) and not allow_duplicate_basket:
        n = len(report.duplicate_basket_by_id) + len(report.duplicate_basket_by_title)
        problems.append(f"{n} duplicate active basket keys")
    if (report.n_sales_null_fk or 0) > 0 and not allow_sales_null_fk:
        problems.append(f"{report.n_sales_null_fk} sales rows with null FKProduct")
    if (report.n_sales_title_mismatch or 0) > 0 and not allow_sales_title_mismatch:
        problems.append(f"{report.n_sales_title_mismatch} fact/dim title mismatches")
    if problems:
        raise ProductIdentityBlockedError("; ".join(problems))
