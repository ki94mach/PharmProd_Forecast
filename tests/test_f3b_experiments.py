"""Unit tests for F3B Step 3 experiment specs (no freeze writes, no XGB tuning)."""
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

from pkg.benchmark.config import BUDGET_RESID_FEATURES, TS_RESID_FEATURES
from pkg.research.f3b.config import (
    ALL_EXPERIMENTS,
    FILLNA_EXTRA,
    NEVER_FILLNA,
    P0_HUMAN,
    P0_TS,
    P1_HUMAN,
    P1_TS,
)
from pkg.research.f3b.evaluate import make_f3b_residual_model
from pkg.research.features.price import FEATURE_NAMES, SCORED_FEATURES, add_price_features


class TestExperimentSpecs(unittest.TestCase):
    def test_exactly_four_experiments(self):
        self.assertEqual(set(ALL_EXPERIMENTS), {"P0_TS", "P1_TS", "P0_HUMAN", "P1_HUMAN"})

    def test_p1_is_f0_plus_three_price_features(self):
        self.assertEqual(P1_TS.features(), tuple(TS_RESID_FEATURES) + FEATURE_NAMES)
        self.assertEqual(P1_HUMAN.features(), tuple(BUDGET_RESID_FEATURES) + FEATURE_NAMES)
        self.assertEqual(P0_TS.features(), tuple(TS_RESID_FEATURES))
        self.assertEqual(P0_HUMAN.features(), tuple(BUDGET_RESID_FEATURES))

    def test_no_core_ts_or_lifecycle(self):
        joined = " ".join(" ".join(e.features()) for e in ALL_EXPERIMENTS.values())
        self.assertNotIn("months_since_first_observed_positive_sale", joined)
        self.assertNotIn("CORE_TS", ALL_EXPERIMENTS)
        self.assertNotIn("T2", ALL_EXPERIMENTS)
        self.assertNotIn("T3", ALL_EXPERIMENTS)
        for name in FEATURE_NAMES:
            self.assertNotIn(name, P0_TS.features())
            self.assertNotIn(name, P0_HUMAN.features())

    def test_no_extra_price_columns_scored(self):
        extra = {
            "consumer_price_asof_origin",
            "distributor_price",
            "pharmacy_price",
            "price_direction",
        }
        for exp in (P1_TS, P1_HUMAN):
            self.assertTrue(extra.isdisjoint(exp.features()))
            self.assertEqual(exp.features()[-3:], FEATURE_NAMES)

    def test_scored_features_unchanged(self):
        self.assertEqual(
            SCORED_FEATURES,
            (
                "log_consumer_price_asof_origin",
                "last_consumer_price_change_pct",
                "months_since_last_consumer_price_change",
            ),
        )
        self.assertEqual(FEATURE_NAMES, SCORED_FEATURES)

    def test_evaluate_f3b_module_exists(self):
        self.assertTrue((_SRC / "pkg" / "research" / "evaluate_f3b.py").exists())

    def test_p0_uses_frozen_adapter_p1_does_not(self):
        self.assertTrue(P0_TS.use_frozen_adapter)
        self.assertTrue(P0_HUMAN.use_frozen_adapter)
        self.assertFalse(P1_TS.use_frozen_adapter)
        self.assertFalse(P1_HUMAN.use_frozen_adapter)
        self.assertEqual(P1_TS.train_universe, "ts")
        self.assertEqual(P1_HUMAN.train_universe, "budget")


class TestNeverFillnaPrice(unittest.TestCase):
    def test_never_fillna_includes_scored_price(self):
        for name in FEATURE_NAMES:
            self.assertIn(name, NEVER_FILLNA)
        self.assertNotIn(FEATURE_NAMES[0], FILLNA_EXTRA)

    def test_residual_model_leaves_price_nan(self):
        col = FEATURE_NAMES[0]
        model = make_f3b_residual_model("ts", ["ts_forecast", "horizon", col])
        train = pd.DataFrame(
            {
                "ts_forecast": [10.0, 12.0],
                "horizon": [1, 2],
                "sales": [11.0, 13.0],
                col: [np.nan, 5.0],
            }
        )
        test = pd.DataFrame(
            {
                "ts_forecast": [10.0],
                "horizon": [1],
                "sales": [11.0],
                col: [np.nan],
            }
        )
        preds = model(train, test)
        self.assertEqual(len(preds), 1)
        self.assertTrue(np.isfinite(preds[0]))
        self.assertTrue(math.isnan(float(test[col].iloc[0])))


class TestFreezeNotTouched(unittest.TestCase):
    def test_enricher_does_not_write_benchmark_dir(self):
        from pkg.benchmark.config import default_benchmark_root

        root = default_benchmark_root()
        if not root.exists():
            self.skipTest("frozen benchmark not present")
        before = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        hist = pd.DataFrame(
            {
                "product": ["P"],
                "effective_month": [140403],
                "consumer_price": [120.0],
            }
        )
        panel = pd.DataFrame({"product": ["P"], "origin": [140404]})
        add_price_features(panel, hist, origin_col="origin")
        after = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
