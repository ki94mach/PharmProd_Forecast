"""Tests for TS V2 missing-month vs zero-sales gap audit."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.gap_audit import (
    audit_inventory_relationship,
    audit_product_gaps,
    classify_month_states,
    run_gap_audit,
    summarize_gap_audit,
    write_gap_audit_report,
)


def _sales(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["product", "date", "sales"])


class TestClassifyMonthStates(unittest.TestCase):
    def test_missing_explicit_zero_and_positive(self):
        observed = pd.Series(
            {140410: 10.0, 140411: 0.0},
            name="SkuA",
        )
        detail = classify_month_states(observed, [140410, 140411, 140412])
        states = dict(zip(detail["date"], detail["month_state"]))
        self.assertEqual(states[140410], "positive")
        self.assertEqual(states[140411], "explicit_zero")
        self.assertEqual(states[140412], "missing")


class TestAuditProductGaps(unittest.TestCase):
    def test_gap_between_observed_months(self):
        observed = pd.Series({140410: 10.0, 140412: 30.0}, name="SkuA")
        row = audit_product_gaps(observed, product="SkuA")
        assert row is not None
        self.assertEqual(row.n_expected_months, 3)
        self.assertEqual(row.n_missing_months, 1)
        self.assertEqual(row.n_positive_months, 2)
        self.assertEqual(row.longest_missing_run, 1)
        self.assertEqual(row.first_observed_month, 140410)
        self.assertEqual(row.last_observed_month, 140412)

    def test_explicit_zero_not_counted_as_missing(self):
        observed = pd.Series({140410: 10.0, 140411: 0.0, 140412: 5.0}, name="SkuA")
        row = audit_product_gaps(observed, product="SkuA")
        assert row is not None
        self.assertEqual(row.n_missing_months, 0)
        self.assertEqual(row.n_explicit_zero_months, 1)
        self.assertEqual(row.longest_explicit_zero_run, 1)
        self.assertAlmostEqual(row.pct_explicit_zero_observed, 1 / 3)

    def test_longest_zero_run(self):
        observed = pd.Series(
            {140410: 0.0, 140411: 0.0, 140412: 0.0, 140501: 10.0},
            name="SkuA",
        )
        row = audit_product_gaps(observed, product="SkuA")
        assert row is not None
        self.assertEqual(row.longest_explicit_zero_run, 3)
        self.assertEqual(row.n_demand_months, 1)
        self.assertIsNone(row.average_inter_demand_interval)

    def test_calendar_extends_through_origin_minus_one(self):
        sales = _sales(
            [
                ("SkuA", 140410, 10.0),
                ("SkuA", 140412, 20.0),
                ("SkuA", 140501, 999.0),
            ]
        )
        report = run_gap_audit(sales, origin=140501, include_month_detail=True)
        row = report.products.loc[report.products["product"] == "SkuA"].iloc[0]
        self.assertEqual(int(row["calendar_end"]), 140412)
        self.assertEqual(int(row["n_missing_months"]), 1)
        self.assertNotIn(140501, report.month_detail["date"].tolist())

    def test_activity_start_trims_leading_months(self):
        sales = _sales(
            [
                ("SkuA", 140409, 3.0),
                ("SkuA", 140410, 10.0),
                ("SkuA", 140411, 10.0),
            ]
        )
        report = run_gap_audit(sales, apply_activity_start=True)
        row = report.products.loc[report.products["product"] == "SkuA"].iloc[0]
        self.assertEqual(int(row["first_active_month"]), 140410)
        self.assertEqual(int(row["calendar_start"]), 140410)


class TestPortfolioSummary(unittest.TestCase):
    def test_summarize_counts(self):
        sales = _sales(
            [
                ("A", 140410, 10.0),
                ("A", 140412, 0.0),
                ("B", 140410, 0.0),
                ("B", 140411, 0.0),
            ]
        )
        report = run_gap_audit(sales)
        p = report.portfolio
        self.assertEqual(p["n_products"], 2)
        self.assertGreater(p["total_missing_months"], 0)
        self.assertGreater(p["total_explicit_zero_months"], 0)


class TestInventoryCrossTab(unittest.TestCase):
    def test_exploratory_inventory_rates(self):
        month_detail = pd.DataFrame(
            [
                {"product": "A", "date": 140501, "month_state": "missing", "sales": None},
                {"product": "A", "date": 140502, "month_state": "positive", "sales": 10.0},
            ]
        )
        dist = pd.DataFrame(
            [
                {
                    "product": "A",
                    "snapshot_date": pd.Timestamp("2024-03-19"),
                    "distributor_inventory_qty": 0.0,
                },
                {
                    "product": "A",
                    "snapshot_date": pd.Timestamp("2024-04-19"),
                    "distributor_inventory_qty": 100.0,
                },
            ]
        )
        # Shamsi 140501 month-end and 140502 month-end depend on calendar helper;
        # use run_gap_audit path with mocked dates via full pipeline instead.
        out = audit_inventory_relationship(month_detail, distributor_inventory=dist)
        self.assertFalse(out.empty)
        self.assertIn("month_state", out.columns)
        self.assertIn("pct_distributor_qty_eq0", out.columns)


class TestWriteReport(unittest.TestCase):
    def test_writes_csv_and_markdown(self):
        sales = _sales([("A", 140410, 10.0), ("A", 140412, 5.0)])
        report = run_gap_audit(sales, include_month_detail=True)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_gap_audit_report(report, tmp)
            self.assertTrue(paths["products"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["month_detail"].exists())


if __name__ == "__main__":
    unittest.main()
