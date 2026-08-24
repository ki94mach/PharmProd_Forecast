"""Basket product universe tests (no live SQL)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.db.query.dim_product import select_basket_products
from pkg.utils import drop_unmapped_departments, update_department_info


def _dim_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "StatusCode" not in df.columns:
        df["StatusCode"] = "Active"
    return df


class TestSelectBasketProducts(unittest.TestCase):
    def test_includes_product_basket_one(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "InBasket",
                    "ProductBasket": 1,
                    "Title": "داخل",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "OutBasket",
                    "ProductBasket": 0,
                    "Title": "خارج",
                    "Provider": "P2",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "NullBasket",
                    "ProductBasket": None,
                    "Title": "خالی",
                    "Provider": "P3",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertEqual(list(basket["ProductTitleEN"]), ["InBasket"])

    def test_drops_blank_product_title_en(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "  ",
                    "ProductBasket": 1,
                    "Title": "خالی",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": None,
                    "ProductBasket": 1,
                    "Title": "ندارد",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertTrue(basket.empty)

    def test_dedupes_product_title_en(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "Dup",
                    "ProductBasket": 1,
                    "Title": "اول",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "Dup",
                    "ProductBasket": 1,
                    "Title": "دوم",
                    "Provider": "P2",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 10,
                    "BoxQuantity": 10,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertEqual(len(basket), 1)
        self.assertEqual(basket.iloc[0]["Title"], "اول")

    def test_accepts_string_and_float_basket_flags(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "A",
                    "ProductBasket": "1",
                    "Title": "آ",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "B",
                    "ProductBasket": 1.0,
                    "Title": "ب",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertEqual(set(basket["ProductTitleEN"]), {"A", "B"})

    def test_excludes_dash_field(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "DashField",
                    "ProductBasket": 1,
                    "Title": "خط تیره",
                    "Provider": "P1",
                    "Field": "-",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "KeepField",
                    "ProductBasket": 1,
                    "Title": "معتبر",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertEqual(list(basket["ProductTitleEN"]), ["KeepField"])

    def test_excludes_non_active_status_code(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "InactiveSku",
                    "ProductBasket": 1,
                    "Title": "غیرفعال",
                    "Provider": "P1",
                    "Field": "غدد",
                    "StatusCode": "Inactive",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "ActiveSku",
                    "ProductBasket": 1,
                    "Title": "فعال",
                    "Provider": "P1",
                    "Field": "غدد",
                    "StatusCode": "Active",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        basket = select_basket_products(dim)
        self.assertEqual(list(basket["ProductTitleEN"]), ["ActiveSku"])


class TestBasketUniverseVsSales(unittest.TestCase):
    def test_no_sales_sku_stays_in_universe(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "NeverSold",
                    "ProductBasket": 1,
                    "Title": "بدون فروش",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "HasSales",
                    "ProductBasket": 1,
                    "Title": "با فروش",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        sales = pd.DataFrame(
            [
                {
                    "product": "HasSales",
                    "product_fa": "با فروش",
                    "date": 140501,
                    "provider": "P1",
                    "dep": "غدد",
                    "boxq": 1,
                    "sales": 10,
                }
            ]
        )
        basket = select_basket_products(dim)
        universe = set(basket["ProductTitleEN"].astype(str))
        sales_products = set(sales["product"].astype(str))
        self.assertIn("NeverSold", universe)
        self.assertNotIn("NeverSold", sales_products)
        self.assertIn("HasSales", universe)
        self.assertIn("HasSales", sales_products)

    def test_sales_only_non_basket_sku_excluded(self):
        dim = _dim_rows(
            [
                {
                    "ProductTitleEN": "InBasket",
                    "ProductBasket": 1,
                    "Title": "داخل",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
                {
                    "ProductTitleEN": "SalesOnly",
                    "ProductBasket": 0,
                    "Title": "فقط فروش",
                    "Provider": "P1",
                    "Field": "غدد",
                    "OrchidBoxQuantity": 1,
                    "BoxQuantity": 1,
                },
            ]
        )
        sales = pd.DataFrame(
            [
                {
                    "product": "SalesOnly",
                    "product_fa": "فقط فروش",
                    "date": 140501,
                    "provider": "P1",
                    "dep": "غدد",
                    "boxq": 1,
                    "sales": 5,
                },
                {
                    "product": "InBasket",
                    "product_fa": "داخل",
                    "date": 140501,
                    "provider": "P1",
                    "dep": "غدد",
                    "boxq": 1,
                    "sales": 8,
                },
            ]
        )
        basket = select_basket_products(dim)
        universe = set(basket["ProductTitleEN"].astype(str))
        sales_products = set(sales["product"].astype(str))
        self.assertIn("SalesOnly", sales_products)
        self.assertNotIn("SalesOnly", universe)
        self.assertIn("InBasket", universe)


class TestDropUnmappedDepartments(unittest.TestCase):
    def test_drops_null_file_name_and_keeps_mapped(self):
        pivot = pd.DataFrame(
            [
                {
                    "product_fa": "خوب",
                    "dep": "غدد",
                    "provider": "P1",
                    "status": "عدد",
                    "file_name": "1405Q1_Endo",
                },
                {
                    "product_fa": "بد",
                    "dep": "unknown-dept",
                    "provider": "P1",
                    "status": "عدد",
                    "file_name": None,
                },
            ]
        )
        cleaned = drop_unmapped_departments(pivot)
        self.assertEqual(list(cleaned["product_fa"]), ["خوب"])
        self.assertTrue(cleaned["file_name"].notna().all())

    def test_maps_known_persian_field_via_department_dict(self):
        dep_dict = update_department_info("1405Q1")
        pivot = pd.DataFrame(
            [
                {
                    "product_fa": "محصول",
                    "dep": "غدد",
                    "provider": "P1",
                    "status": "عدد",
                    "file_name": dep_dict["غدد"],
                }
            ]
        )
        cleaned = drop_unmapped_departments(pivot)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["file_name"], "1405Q1_Endo")


class TestTruncateSalesBeforeOrigin(unittest.TestCase):
    def test_keeps_months_strictly_before_origin(self):
        from pkg.sales_forecasting import SalesForecasting

        sales = pd.DataFrame(
            [
                {"product": "A", "date": 140410, "sales": 1},
                {"product": "A", "date": 140411, "sales": 2},
                {"product": "A", "date": 140412, "sales": 3},
                {"product": "A", "date": 140501, "sales": 4},
                {"product": "B", "date": 140502, "sales": 5},
            ]
        )
        out = SalesForecasting.truncate_sales_before_origin(sales, 140501)
        self.assertEqual(sorted(out["date"].tolist()), [140410, 140411, 140412])
        self.assertNotIn(140501, set(out["date"]))
        self.assertNotIn("B", set(out["product"]))

    def test_empty_frame_passthrough(self):
        from pkg.sales_forecasting import SalesForecasting

        empty = pd.DataFrame(columns=["product", "date", "sales"])
        out = SalesForecasting.truncate_sales_before_origin(empty, 140501)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
