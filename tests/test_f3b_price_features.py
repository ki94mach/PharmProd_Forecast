"""Unit tests for F3B Step 2 PIT consumer-price features (no XGB, no freeze writes)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_month_diff
from pkg.research.features import FEATURE_GROUPS
from pkg.research.features.price import (
    FEATURE_NAMES,
    SCORED_FEATURES,
    add_price_features,
    assert_price_point_in_time,
    build_price_features,
)


def _hist(rows: list[tuple]) -> pd.DataFrame:
    cols = ["product", "effective_month", "consumer_price"]
    if rows and len(rows[0]) == 4:
        cols = ["product", "effective_month", "effective_date", "consumer_price"]
    return pd.DataFrame(rows, columns=cols)


def _panel(product: str = "P", origin: int = 140404) -> pd.DataFrame:
    return pd.DataFrame({"product": [product], "origin": [origin]})


class TestScoredNames(unittest.TestCase):
    def test_scored_features_are_exactly_three(self):
        self.assertEqual(
            FEATURE_NAMES,
            (
                "log_consumer_price_asof_origin",
                "last_consumer_price_change_pct",
                "months_since_last_consumer_price_change",
            ),
        )
        self.assertEqual(FEATURE_NAMES, SCORED_FEATURES)
        self.assertEqual(FEATURE_GROUPS["price"], FEATURE_NAMES)


class TestSpecifiedLeakageExample(unittest.TestCase):
    def test_origin_140404_sees_120_not_150(self):
        hist = _hist(
            [
                ("P", 140301, 100.0),
                ("P", 140403, 120.0),
                ("P", 140405, 150.0),
            ]
        )
        out = add_price_features(_panel(), hist, origin_col="origin")
        self.assertAlmostEqual(float(out["consumer_price_asof_origin"].iloc[0]), 120.0)
        self.assertAlmostEqual(
            float(out["log_consumer_price_asof_origin"].iloc[0]), math.log1p(120.0)
        )
        self.assertAlmostEqual(float(out["last_consumer_price_change_pct"].iloc[0]), 0.20)
        self.assertEqual(float(out["months_since_last_consumer_price_change"].iloc[0]), 1.0)
        self.assertEqual(float(out["last_price_effective_month"].iloc[0]), 140403.0)
        self.assertEqual(float(out["previous_consumer_price"].iloc[0]), 100.0)
        self.assertEqual(float(out["last_change_month"].iloc[0]), 140403.0)
        self.assertNotEqual(float(out["consumer_price_asof_origin"].iloc[0]), 150.0)


class TestOriginMonthNotVisible(unittest.TestCase):
    def test_origin_month_observation_is_not_visible(self):
        hist = _hist(
            [
                ("P", 140403, 100.0),
                ("P", 140404, 200.0),
            ]
        )
        out = add_price_features(_panel(origin=140404), hist, origin_col="origin")
        self.assertAlmostEqual(float(out["consumer_price_asof_origin"].iloc[0]), 100.0)
        self.assertNotEqual(float(out["consumer_price_asof_origin"].iloc[0]), 200.0)
        self.assertEqual(float(out["last_price_effective_month"].iloc[0]), 140403.0)


class TestIdenticalCollapse(unittest.TestCase):
    def test_consecutive_identical_prices_are_one_state(self):
        hist = _hist(
            [
                ("P", 140301, 100.0),
                ("P", 140302, 120.0),
                ("P", 140303, 120.0),
            ]
        )
        out = add_price_features(_panel(), hist, origin_col="origin")
        self.assertAlmostEqual(float(out["consumer_price_asof_origin"].iloc[0]), 120.0)
        self.assertAlmostEqual(float(out["last_consumer_price_change_pct"].iloc[0]), 0.20)
        self.assertEqual(float(out["previous_consumer_price"].iloc[0]), 100.0)
        self.assertEqual(float(out["last_change_month"].iloc[0]), 140302.0)
        self.assertEqual(float(out["last_price_effective_month"].iloc[0]), 140303.0)
        self.assertEqual(float(out["n_price_states_before_origin"].iloc[0]), 2.0)


class TestMissingNotZero(unittest.TestCase):
    def test_single_prior_price_change_features_are_nan_not_zero(self):
        hist = _hist([("P", 140301, 100.0)])
        out = add_price_features(_panel(), hist, origin_col="origin")
        self.assertAlmostEqual(float(out["consumer_price_asof_origin"].iloc[0]), 100.0)
        self.assertTrue(math.isnan(float(out["last_consumer_price_change_pct"].iloc[0])))
        self.assertTrue(
            math.isnan(float(out["months_since_last_consumer_price_change"].iloc[0]))
        )
        self.assertNotEqual(float(out["last_consumer_price_change_pct"].iloc[0]), 0.0)
        self.assertNotEqual(
            float(out["months_since_last_consumer_price_change"].iloc[0]), 0.0
        )

    def test_no_history_is_nan(self):
        hist = _hist([("Q", 140301, 50.0)])
        out = add_price_features(_panel(), hist, origin_col="origin")
        self.assertTrue(math.isnan(float(out["consumer_price_asof_origin"].iloc[0])))
        self.assertTrue(math.isnan(float(out["log_consumer_price_asof_origin"].iloc[0])))


class TestShamsiMonthDiff(unittest.TestCase):
    def test_months_since_uses_shamsi_month_diff(self):
        hist = _hist(
            [
                ("P", 140311, 80.0),
                ("P", 140312, 100.0),
            ]
        )
        out = add_price_features(_panel(origin=140402), hist, origin_col="origin")
        expected = shamsi_month_diff(140402, 140312)
        self.assertEqual(expected, 2)
        self.assertNotEqual(expected, 140402 - 140312)
        self.assertEqual(
            float(out["months_since_last_consumer_price_change"].iloc[0]), float(expected)
        )


class TestLeakageAssert(unittest.TestCase):
    def test_post_origin_month_attached_raises(self):
        feat = pd.DataFrame(
            {
                "last_price_effective_month": [140405.0],
                "last_change_month": [140403.0],
            }
        )
        with self.assertRaises(AssertionError):
            assert_price_point_in_time(feat, np.array([140404]))

    def test_post_origin_change_month_raises(self):
        feat = pd.DataFrame(
            {
                "last_price_effective_month": [140403.0],
                "last_change_month": [140404.0],
            }
        )
        with self.assertRaises(AssertionError):
            assert_price_point_in_time(feat, np.array([140404]))

    def test_builder_visible_months_strictly_before_origin(self):
        hist = _hist(
            [
                ("P", 140403, 120.0),
                ("P", 140405, 150.0),
                ("P", 140406, 160.0),
            ]
        )
        panel = pd.DataFrame({"product": ["P", "P"], "origin": [140404, 140407]})
        feat = build_price_features(panel, hist, origin_col="origin")
        mask = feat["last_price_effective_month"].notna()
        self.assertTrue(
            (feat.loc[mask, "last_price_effective_month"] < panel.loc[mask, "origin"]).all()
        )
        self.assertAlmostEqual(float(feat["consumer_price_asof_origin"].iloc[0]), 120.0)
        self.assertAlmostEqual(float(feat["consumer_price_asof_origin"].iloc[1]), 160.0)


class TestFreezeNotTouched(unittest.TestCase):
    def test_feature_fns_do_not_write_benchmark_dir(self):
        from pkg.benchmark.config import default_benchmark_root

        root = default_benchmark_root()
        if not root.exists():
            self.skipTest("frozen benchmark not present")
        before = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        hist = _hist([("P", 140401, 1.0), ("P", 140403, 1.2)])
        add_price_features(_panel(), hist, origin_col="origin")
        after = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
