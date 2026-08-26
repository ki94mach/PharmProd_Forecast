"""Product activity filter tests (no live SQL)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import last_complete_6m
from pkg.product_activity_filter import (
    filter_basket_active,
    filter_forecast_active,
    inactive_products,
    products_with_distributor_inventory,
    products_with_recent_sales,
)

ORIGIN = 140501  # window: 140407 .. 140412


def _forecast(*products: str) -> pd.DataFrame:
    rows = []
    for p in products:
        rows.append(
            {
                "product": p,
                "product_fa": p,
                "date": ORIGIN,
                "provider": "P",
                "model": "x",
                "dep": "d",
                "status": "s",
                "forecast": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _sales(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _inv(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestLastComplete6mWindow(unittest.TestCase):
    def test_window_bounds(self):
        start, end = last_complete_6m(ORIGIN)
        self.assertEqual(start, 140407)
        self.assertEqual(end, 140412)


class TestRecentSales(unittest.TestCase):
    def test_sales_in_window_counts(self):
        sale_df = _sales(
            [
                {"product": "KeepSales", "date": 140410, "sales": 5},
                {"product": "OldOnly", "date": 140301, "sales": 100},
                {"product": "ZeroInWindow", "date": 140410, "sales": 0},
            ]
        )
        active = products_with_recent_sales(sale_df, ORIGIN)
        self.assertEqual(active, {"KeepSales"})

    def test_empty_sales(self):
        self.assertEqual(products_with_recent_sales(pd.DataFrame(), ORIGIN), set())


class TestDistributorInventory(unittest.TestCase):
    def test_latest_snapshot_only(self):
        inv = _inv(
            [
                {
                    "product": "HadStock",
                    "snapshot_date": "2025-01-01",
                    "distributor_inventory_qty": 50,
                },
                {
                    "product": "HadStock",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 0,
                },
                {
                    "product": "HasStock",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 10,
                },
                {
                    "product": "ZeroStock",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 0,
                },
            ]
        )
        has_inv = products_with_distributor_inventory(inv)
        self.assertEqual(has_inv, {"HasStock"})
        self.assertNotIn("HadStock", has_inv)


class TestInactiveFilter(unittest.TestCase):
    def test_sales_no_inv_keep(self):
        forecast = _forecast("A", "B")
        sale_df = _sales([{"product": "A", "date": 140410, "sales": 3}])
        inv = _inv(
            [
                {
                    "product": "Other",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 1,
                }
            ]
        )
        inactive = inactive_products(sale_df, inv, ORIGIN, {"A", "B"})
        self.assertEqual(inactive, {"B"})
        out, dropped = filter_forecast_active(forecast, sale_df, inv, ORIGIN)
        self.assertEqual(dropped, {"B"})
        self.assertEqual(set(out["product"]), {"A"})

    def test_no_sales_positive_inv_keep(self):
        forecast = _forecast("StockOnly")
        sale_df = _sales([])
        inv = _inv(
            [
                {
                    "product": "StockOnly",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 2,
                }
            ]
        )
        inactive = inactive_products(sale_df, inv, ORIGIN, {"StockOnly"})
        self.assertEqual(inactive, set())
        out, dropped = filter_forecast_active(forecast, sale_df, inv, ORIGIN)
        self.assertEqual(dropped, set())
        self.assertEqual(set(out["product"]), {"StockOnly"})

    def test_no_sales_missing_or_zero_inv_drop(self):
        forecast = _forecast("Missing", "Zero")
        sale_df = _sales([])
        inv = _inv(
            [
                {
                    "product": "Zero",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 0,
                }
            ]
        )
        inactive = inactive_products(
            sale_df, inv, ORIGIN, {"Missing", "Zero"}
        )
        self.assertEqual(inactive, {"Missing", "Zero"})
        out, dropped = filter_forecast_active(forecast, sale_df, inv, ORIGIN)
        self.assertEqual(dropped, {"Missing", "Zero"})
        self.assertTrue(out.empty)

    def test_sales_outside_window_treated_as_no_recent(self):
        sale_df = _sales([{"product": "Stale", "date": 140301, "sales": 99}])
        inv = _inv([])
        inactive = inactive_products(sale_df, inv, ORIGIN, {"Stale"})
        self.assertEqual(inactive, {"Stale"})


class TestFilterBasketActive(unittest.TestCase):
    def test_drops_inactive_before_ts(self):
        basket = pd.DataFrame(
            {
                "ProductTitleEN": ["KeepSales", "DropMe", "KeepInv"],
                "OrchidBoxQuantity": ["بسته", "عدد", "بسته"],
            }
        )
        sale_df = _sales([{"product": "KeepSales", "date": 140410, "sales": 2}])
        inv = _inv(
            [
                {
                    "product": "KeepInv",
                    "snapshot_date": "2025-06-01",
                    "distributor_inventory_qty": 5,
                }
            ]
        )
        out, skipped = filter_basket_active(basket, sale_df, inv, ORIGIN)
        self.assertEqual(skipped, {"DropMe"})
        self.assertEqual(set(out["ProductTitleEN"]), {"KeepSales", "KeepInv"})


if __name__ == "__main__":
    unittest.main()
