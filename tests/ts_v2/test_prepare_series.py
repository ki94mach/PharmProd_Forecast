"""Monthly sales-series preparation tests (raw units, origin-safe)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import assert_no_post_origin_leakage, prepare_monthly_series


def _sales(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["product", "date", "sales"])


class TestPrepareMonthlySeries(unittest.TestCase):
    def test_duplicate_monthly_rows_are_summed(self):
        sales = _sales(
            [
                ("SkuA", 140410, 10.0),
                ("SkuA", 140410, 5.0),
                ("SkuA", 140411, 20.0),
                ("SkuA", 140412, 30.0),
            ]
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertEqual(float(prepared.values.loc[140410]), 15.0)
        self.assertEqual(float(prepared.values.loc[140411]), 20.0)

    def test_gaps_in_monthly_history_are_flagged(self):
        sales = _sales(
            [
                ("SkuA", 140410, 10.0),
                ("SkuA", 140412, 30.0),
            ]
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertTrue(bool(prepared.is_missing_month.loc[140411]))
        self.assertFalse(bool(prepared.is_missing_month.loc[140410]))
        self.assertFalse(bool(prepared.is_missing_month.loc[140412]))
        self.assertEqual(float(prepared.values.loc[140411]), 0.0)  # policy "zero"
        self.assertGreaterEqual(prepared.n_gap_months, 1)
        self.assertEqual(list(prepared.dates[:3]), [140410, 140411, 140412])

        missing_cfg = TSForecastConfig(
            activity_start_min_sales=None,
            missing_month_policy="missing",
        )
        as_nan = prepare_monthly_series(sales, "SkuA", 140501, config=missing_cfg)
        self.assertTrue(pd.isna(as_nan.values.loc[140411]))
        self.assertTrue(bool(as_nan.is_missing_month.loc[140411]))
        self.assertEqual(float(as_nan.values.loc[140410]), 10.0)

    def test_explicit_zero_sales_are_observed_not_gaps(self):
        sales = _sales(
            [
                ("SkuA", 140410, 10.0),
                ("SkuA", 140411, 0.0),
                ("SkuA", 140412, 30.0),
            ]
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertEqual(float(prepared.values.loc[140411]), 0.0)
        self.assertFalse(bool(prepared.is_missing_month.loc[140411]))
        self.assertNotEqual(
            bool(prepared.is_missing_month.loc[140411]),
            True,
        )

    def test_forecast_origin_filtering(self):
        sales = _sales(
            [
                ("SkuA", 140411, 10.0),
                ("SkuA", 140412, 20.0),
                ("SkuA", 140501, 999.0),
                ("SkuA", 140502, 888.0),
            ]
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertEqual(prepared.forecast_origin, 140501)
        self.assertEqual(prepared.last_training_month, 140412)
        self.assertNotIn(140501, prepared.dates)
        self.assertNotIn(140502, prepared.dates)
        self.assertTrue(all(d < 140501 for d in prepared.dates))
        self.assertEqual(float(prepared.values.loc[140412]), 20.0)

    def test_no_preprocessing_using_post_origin_observations(self):
        sales = _sales(
            [
                ("SkuA", 140411, 10.0),
                ("SkuA", 140412, 20.0),
                ("SkuA", 140501, 1_000_000.0),
            ]
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        assert_no_post_origin_leakage(sales, prepared)
        # Raw units: origin-month 1e6 must not scale or shift training values.
        self.assertEqual(float(prepared.values.loc[140411]), 10.0)
        self.assertEqual(float(prepared.values.loc[140412]), 20.0)
        self.assertLessEqual(float(prepared.values.max()), 20.0)

    def test_activity_start_threshold_skips_tiny_leading_months(self):
        sales = _sales(
            [
                ("SkuA", 140409, 3.0),
                ("SkuA", 140410, 6.0),
                ("SkuA", 140411, 10.0),
                ("SkuA", 140412, 10.0),
            ]
        )
        default = prepare_monthly_series(sales, "SkuA", 140501)
        self.assertEqual(default.first_active_month, 140410)
        self.assertNotIn(140409, default.dates)

        disabled = prepare_monthly_series(
            sales,
            "SkuA",
            140501,
            config=TSForecastConfig(activity_start_min_sales=None),
        )
        self.assertEqual(disabled.first_active_month, 140409)
        self.assertIn(140409, disabled.dates)

    def test_monthly_grid_runs_through_training_end(self):
        sales = _sales([("SkuA", 140410, 10.0)])
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertEqual(prepared.dates[0], 140410)
        self.assertEqual(prepared.dates[-1], 140412)
        self.assertEqual(prepared.n_observations, len(prepared.dates))
        self.assertEqual(prepared.n_observations, len(prepared.values))


if __name__ == "__main__":
    unittest.main()
