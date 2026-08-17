"""Unit tests for F3A lifecycle features (no freeze mutation)."""
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
from pkg.research.ablation.config import CORE_TS
from pkg.research.features.lifecycle import (
    FEATURE_NAMES,
    LAUNCH_EVENT_NAMES,
    SCORED_FEATURE,
    add_lifecycle_features,
    build_lifecycle_features,
    product_lifecycle_catalog,
)
from pkg.research.f3a.config import ALL_EXPERIMENTS, T2, T3


def _sales_frame(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["product", "date", "sales"])


class TestShamsiAge(unittest.TestCase):
    def test_age_uses_shamsi_month_diff_not_yyyymm_subtraction(self):
        hist = _sales_frame([("A", 140201, 10.0)])
        panel = pd.DataFrame({"product": ["A"], "origin": [140404]})
        out = add_lifecycle_features(panel, hist, origin_col="origin")
        expected = shamsi_month_diff(140404, 140201)
        self.assertEqual(expected, 27)
        self.assertNotEqual(expected, 140404 - 140201)
        self.assertEqual(float(out[SCORED_FEATURE].iloc[0]), 27.0)

    def test_one_month_after_first_sale_is_age_one(self):
        hist = _sales_frame([("A", 140401, 5.0)])
        panel = pd.DataFrame({"product": ["A"], "origin": [140402]})
        out = add_lifecycle_features(panel, hist, origin_col="origin")
        self.assertEqual(float(out[SCORED_FEATURE].iloc[0]), 1.0)


class TestPIT(unittest.TestCase):
    def test_future_sales_do_not_change_features(self):
        origin = 140404
        hist = _sales_frame(
            [
                ("P", 140401, 10.0),
                ("P", 140402, 12.0),
                ("P", 140403, 11.0),
            ]
        )
        panel = pd.DataFrame({"product": ["P"], "origin": [origin]})
        a = add_lifecycle_features(panel, hist, origin_col="origin")
        hist2 = pd.concat(
            [hist, _sales_frame([("P", origin, 9999.0), ("P", 140405, 8888.0)])],
            ignore_index=True,
        )
        b = add_lifecycle_features(panel, hist2, origin_col="origin")
        self.assertEqual(float(a[SCORED_FEATURE].iloc[0]), float(b[SCORED_FEATURE].iloc[0]))
        self.assertEqual(
            float(a["first_positive_sale_month"].iloc[0]),
            float(b["first_positive_sale_month"].iloc[0]),
        )

    def test_first_sale_after_origin_is_missing(self):
        hist = _sales_frame([("P", 140405, 10.0)])
        early = pd.DataFrame({"product": ["P"], "origin": [140401]})
        late = pd.DataFrame({"product": ["P"], "origin": [140407]})
        a = add_lifecycle_features(early, hist, origin_col="origin")
        b = add_lifecycle_features(late, hist, origin_col="origin")
        self.assertTrue(math.isnan(float(a[SCORED_FEATURE].iloc[0])))
        self.assertEqual(float(a["has_prior_positive_sale"].iloc[0]), 0.0)
        self.assertFalse(math.isnan(float(b[SCORED_FEATURE].iloc[0])))
        self.assertEqual(float(b["has_prior_positive_sale"].iloc[0]), 1.0)
        self.assertEqual(float(b["first_positive_sale_month"].iloc[0]), 140405.0)

    def test_missing_age_is_nan_not_zero(self):
        hist = _sales_frame([("Q", 140401, 1.0)])
        panel = pd.DataFrame({"product": ["P"], "origin": [140404]})
        out = add_lifecycle_features(panel, hist, origin_col="origin")
        self.assertTrue(math.isnan(float(out[SCORED_FEATURE].iloc[0])))
        self.assertNotEqual(float(out[SCORED_FEATURE].iloc[0]), 0.0)

    def test_zero_and_negative_sales_are_not_first_positive(self):
        hist = _sales_frame(
            [
                ("P", 140401, 0.0),
                ("P", 140402, -8.0),
                ("P", 140403, 12.0),
            ]
        )
        panel = pd.DataFrame({"product": ["P"], "origin": [140404]})
        out = add_lifecycle_features(panel, hist, origin_col="origin")
        self.assertEqual(float(out["first_positive_sale_month"].iloc[0]), 140403.0)
        self.assertEqual(float(out["first_nonzero_sale_month"].iloc[0]), 140402.0)
        self.assertEqual(float(out[SCORED_FEATURE].iloc[0]), 1.0)

    def test_first_positive_strictly_before_origin(self):
        hist = _sales_frame([("P", 140401, 4.0), ("P", 140403, 5.0)])
        panel = pd.DataFrame({"product": ["P", "P"], "origin": [140402, 140404]})
        feat = build_lifecycle_features(panel, hist, origin_col="origin")
        mask = feat["first_positive_sale_month"].notna()
        self.assertTrue(
            (
                feat.loc[mask, "first_positive_sale_month"]
                < panel.loc[mask, "origin"]
            ).all()
        )

    def test_determinism(self):
        hist = _sales_frame([("P", 140401, 5.0), ("P", 140402, 7.0)])
        panel = pd.DataFrame({"product": ["P"], "origin": [140404]})
        a = add_lifecycle_features(panel, hist, origin_col="origin")
        b = add_lifecycle_features(panel, hist, origin_col="origin")
        pd.testing.assert_frame_equal(a, b)


class TestLeftCensoring(unittest.TestCase):
    def test_left_censored_uses_global_earliest_month(self):
        hist = _sales_frame(
            [
                ("Old", 140101, 10.0),
                ("Old", 140201, 8.0),
                ("New", 140201, 3.0),
            ]
        )
        cat = product_lifecycle_catalog(hist)
        old = cat.loc[cat["product"] == "Old"].iloc[0]
        new = cat.loc[cat["product"] == "New"].iloc[0]
        self.assertEqual(int(old["earliest_available_sales_month"]), 140101)
        self.assertEqual(int(old["first_sale_left_censored"]), 1)
        self.assertEqual(int(new["first_sale_left_censored"]), 0)

    def test_later_launch_not_left_censored(self):
        hist = _sales_frame(
            [
                ("A", 140101, 1.0),
                ("B", 140201, 9.0),
            ]
        )
        panel = pd.DataFrame({"product": ["B"], "origin": [140404]})
        out = add_lifecycle_features(panel, hist, origin_col="origin")
        self.assertEqual(float(out["first_sale_left_censored"].iloc[0]), 0.0)


class TestScoredFeatureSet(unittest.TestCase):
    def test_exactly_one_scored_feature(self):
        self.assertEqual(FEATURE_NAMES, (SCORED_FEATURE,))
        self.assertEqual(len(FEATURE_NAMES), 1)

    def test_no_launch_event_names_in_scored_set(self):
        self.assertTrue(set(LAUNCH_EVENT_NAMES).isdisjoint(FEATURE_NAMES))

    def test_first_nonzero_not_scored(self):
        self.assertNotIn("first_nonzero_sale_month", FEATURE_NAMES)
        self.assertNotIn("has_prior_positive_sale", FEATURE_NAMES)
        self.assertNotIn("first_sale_left_censored", FEATURE_NAMES)

    def test_core_ts_imported_from_ablation(self):
        from pkg.research.f3a.config import CORE_TS as F3A_CORE

        self.assertEqual(F3A_CORE, CORE_TS)
        self.assertEqual(T2.features(), tuple(CORE_TS))
        self.assertEqual(T3.features(), tuple(CORE_TS) + (SCORED_FEATURE,))

    def test_human_keeps_f0_sales_lags(self):
        from pkg.benchmark.config import BUDGET_RESID_FEATURES
        from pkg.research.f3a.config import H1

        feats = H1.features()
        for lag in (
            "sales_lag_1",
            "sales_lag_2",
            "sales_lag_3",
            "sales_lag_12",
            "sales_roll3",
        ):
            self.assertIn(lag, feats)
        self.assertEqual(feats[: len(BUDGET_RESID_FEATURES)], tuple(BUDGET_RESID_FEATURES))

    def test_registry_complete(self):
        self.assertEqual(set(ALL_EXPERIMENTS), {"T0", "T1", "T2", "T3", "H0", "H1"})


class TestNeverFillnaAge(unittest.TestCase):
    def test_residual_model_skips_age(self):
        from pkg.research.f3a.config import NEVER_FILLNA
        from pkg.research.f3a.evaluate import make_f3a_residual_model

        self.assertIn(SCORED_FEATURE, NEVER_FILLNA)
        model = make_f3a_residual_model("ts", ["ts_forecast", "horizon", SCORED_FEATURE])
        train = pd.DataFrame(
            {
                "ts_forecast": [10.0, 12.0],
                "horizon": [1, 2],
                "sales": [11.0, 13.0],
                SCORED_FEATURE: [np.nan, 5.0],
            }
        )
        test = pd.DataFrame(
            {
                "ts_forecast": [10.0],
                "horizon": [1],
                "sales": [11.0],
                SCORED_FEATURE: [np.nan],
            }
        )
        preds = model(train, test)
        self.assertEqual(len(preds), 1)
        self.assertTrue(np.isfinite(preds[0]))


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
        hist = _sales_frame([("P", 140401, 1.0)])
        panel = pd.DataFrame({"product": ["P"], "origin": [140404]})
        add_lifecycle_features(panel, hist, origin_col="origin")
        after = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
