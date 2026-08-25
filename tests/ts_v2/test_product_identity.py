"""Tests for TS V2 product identity audit helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.product_identity import (
    ProductIdentityBlockedError,
    assert_product_identity_ready,
    audit_basket_duplicates,
    audit_id_to_titles,
    audit_sales_fk_and_titles,
    audit_title_to_ids,
    run_product_identity_audit,
)


def _basket_row(
    *,
    id_int: int,
    title: str,
    persian: str = "فارسی",
    field: str = "غدد",
    basket: int = 1,
    status: str = "Active",
) -> dict:
    return {
        "ID_INT": id_int,
        "ID": f"guid-{id_int}",
        "ProductTitleEN": title,
        "Title": persian,
        "ProductBasket": basket,
        "Field": field,
        "StatusCode": status,
        "Provider": "P1",
        "OrchidBoxQuantity": 1,
        "BoxQuantity": 1,
    }


class TestProductIdentityAudit(unittest.TestCase):
    def test_clean_one_to_one_mapping(self):
        dim = pd.DataFrame(
            [
                _basket_row(id_int=1, title="Alpha"),
                _basket_row(id_int=2, title="Beta"),
            ]
        )
        report = run_product_identity_audit(dim)
        self.assertFalse(report.has_blocking_issues)
        self.assertEqual(report.n_basket_rows_before_dedupe, 2)
        self.assertEqual(report.n_basket_rows_after_dedupe, 2)

    def test_title_to_multiple_ids(self):
        dim = pd.DataFrame(
            [
                _basket_row(id_int=1, title="Dup"),
                _basket_row(id_int=2, title="Dup"),
            ]
        )
        issues = audit_title_to_ids(dim)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].key, "Dup")
        self.assertEqual(set(issues[0].values), {"1", "2"})

    def test_id_to_multiple_titles(self):
        dim = pd.DataFrame(
            [
                {
                    **_basket_row(id_int=1, title="TitleA"),
                    "ProductTitleEN": "TitleA",
                },
                {
                    **_basket_row(id_int=1, title="TitleB"),
                    "ProductTitleEN": "TitleB",
                    "ProductBasket": 0,
                },
            ]
        )
        # Two dim rows same ID_INT different titles (one out of basket)
        issues = audit_id_to_titles(dim)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].key, "1")

    def test_duplicate_active_basket_by_id(self):
        dim = pd.DataFrame(
            [
                _basket_row(id_int=10, title="SameIdA"),
                _basket_row(id_int=10, title="SameIdB"),
            ]
        )
        by_id, by_title = audit_basket_duplicates(dim)
        self.assertEqual(len(by_id), 1)
        self.assertEqual(by_id[0].key, "10")
        self.assertEqual(len(by_title), 0)

    def test_duplicate_active_basket_by_title_before_dedupe(self):
        dim = pd.DataFrame(
            [
                _basket_row(id_int=1, title="SharedTitle"),
                _basket_row(id_int=2, title="SharedTitle"),
            ]
        )
        by_id, by_title = audit_basket_duplicates(dim)
        self.assertEqual(len(by_id), 0)
        self.assertEqual(len(by_title), 1)
        report = run_product_identity_audit(dim)
        self.assertEqual(report.n_basket_rows_before_dedupe, 2)
        self.assertEqual(report.n_basket_rows_after_dedupe, 1)

    def test_sales_title_mismatch(self):
        sales = pd.DataFrame(
            [
                {
                    "FKProduct": 1,
                    "ProductTitleEN": "FactTitle",
                    "dim_product_title": "DimTitle",
                }
            ]
        )
        null_fk, mismatch = audit_sales_fk_and_titles(sales)
        self.assertEqual(null_fk, 0)
        self.assertEqual(mismatch, 1)

    def test_assert_blocks_on_violations(self):
        dim = pd.DataFrame([_basket_row(id_int=1, title="X"), _basket_row(id_int=2, title="X")])
        report = run_product_identity_audit(dim)
        with self.assertRaises(ProductIdentityBlockedError):
            assert_product_identity_ready(report)

    def test_assert_allows_explicit_waiver(self):
        dim = pd.DataFrame([_basket_row(id_int=1, title="X"), _basket_row(id_int=2, title="X")])
        report = run_product_identity_audit(dim)
        assert_product_identity_ready(
            report,
            allow_title_to_ids=True,
            allow_duplicate_basket=True,
        )


if __name__ == "__main__":
    unittest.main()
