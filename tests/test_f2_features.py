"""Unit tests for F2 demand/Human features (no freeze mutation)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.research.features.demand_f2 import add_demand_f2_features, signed_log
from pkg.research.features.human_f2 import add_human_f2_features, shrink


def _sales_frame(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["product", "date", "sales"])


def _budget_frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["product", "target_date", "horizon", "budget_forecast", "sales"]
    return pd.DataFrame(rows, columns=cols)


class TestSignedLog(unittest.TestCase):
    def test_signed_log_basic(self):
        self.assertAlmostEqual(signed_log(0.0), 0.0)
        self.assertGreater(signed_log(10.0), 0.0)
        self.assertLess(signed_log(-10.0), 0.0)
        self.assertEqual(signed_log(float("nan")), 0.0)


class TestShrinkage(unittest.TestCase):
    def test_n0_equals_parent(self):
        self.assertAlmostEqual(shrink(0, 999.0, 10.0, k=5), 10.0)

    def test_small_n_strong_shrink(self):
        raw, parent, k = 100.0, 0.0, 5.0
        s1 = shrink(1, raw, parent, k)
        self.assertAlmostEqual(s1, (1 * raw + k * parent) / (1 + k))
        self.assertLess(abs(s1 - parent), abs(s1 - raw))

    def test_large_n_approaches_raw(self):
        raw, parent, k = 100.0, 0.0, 5.0
        s = shrink(10_000, raw, parent, k)
        self.assertAlmostEqual(s, raw, places=1)


class TestDemandPIT(unittest.TestCase):
    def test_future_sales_do_not_change_features(self):
        origin = 140404
        hist = _sales_frame(
            [
                ("P", 140401, 10.0),
                ("P", 140402, 12.0),
                ("P", 140403, 11.0),
            ]
        )
        panel = pd.DataFrame(
            {"product": ["P"], "origin": [origin], "horizon": [1], "date": [140405]}
        )
        a = add_demand_f2_features(panel, hist, origin_col="origin")
        hist2 = pd.concat(
            [hist, _sales_frame([("P", origin, 9999.0), ("P", 140405, 8888.0)])],
            ignore_index=True,
        )
        b = add_demand_f2_features(panel, hist2, origin_col="origin")
        for col in (
            "sales_std6",
            "trend_log_3m",
            "yoy_log_change",
            "sales_history_months",
        ):
            self.assertEqual(float(a[col].iloc[0]), float(b[col].iloc[0]), msg=col)

    def test_determinism(self):
        hist = _sales_frame([("P", 140401, 5.0), ("P", 140402, 7.0), ("P", 140403, 6.0)])
        panel = pd.DataFrame({"product": ["P"], "origin": [140404], "horizon": [2]})
        a = add_demand_f2_features(panel, hist, origin_col="origin")
        b = add_demand_f2_features(panel, hist, origin_col="origin")
        pd.testing.assert_frame_equal(a, b)


class TestHumanPIT(unittest.TestCase):
    def test_future_budget_outcomes_do_not_change_features(self):
        origin = 140407
        past = _budget_frame(
            [
                ("P", 140401, 3, 100.0, 90.0),
                ("P", 140402, 3, 110.0, 95.0),
                ("Q", 140403, 1, 50.0, 40.0),
            ]
        )
        panel = pd.DataFrame(
            {
                "product": ["P"],
                "origin": [origin],
                "horizon": [3],
                "target_date": [140410],
            }
        )
        a = add_human_f2_features(panel, past, origin_col="origin")
        leaked = pd.concat(
            [
                past,
                _budget_frame(
                    [
                        ("P", origin, 3, 1.0, 9999.0),
                        ("P", 140410, 3, 1.0, 9999.0),
                    ]
                ),
            ],
            ignore_index=True,
        )
        b = add_human_f2_features(panel, leaked, origin_col="origin")
        for col in (
            "human_n_product",
            "human_n_product_horizon",
            "human_bias_product_shrunk",
            "human_bias_product_horizon_shrunk",
            "human_mae_product_shrunk",
        ):
            self.assertAlmostEqual(
                float(a[col].iloc[0]), float(b[col].iloc[0]), places=9, msg=col
            )

    def test_determinism(self):
        bud = _budget_frame([("P", 140401, 1, 10.0, 8.0)])
        panel = pd.DataFrame({"product": ["P"], "origin": [140404], "horizon": [1]})
        a = add_human_f2_features(panel, bud, origin_col="origin")
        b = add_human_f2_features(panel, bud, origin_col="origin")
        pd.testing.assert_frame_equal(a, b)

    def test_no_history_shrinks_to_global(self):
        bud = _budget_frame([("Q", 140401, 1, 10.0, 20.0)])  # other product
        panel = pd.DataFrame({"product": ["P"], "origin": [140404], "horizon": [2]})
        out = add_human_f2_features(panel, bud, origin_col="origin", extras=True)
        self.assertEqual(float(out["human_n_product"].iloc[0]), 0.0)
        self.assertAlmostEqual(
            float(out["human_bias_product_shrunk"].iloc[0]),
            float(out["global_bias"].iloc[0]),
        )


class TestEvalIdentityHelpers(unittest.TestCase):
    def test_row_keys_match_helper(self):
        from pkg.benchmark.evaluate import BacktestResult
        from pkg.research.evaluate_features import assert_same_eval_rows

        keys = pd.DataFrame(
            {
                "product": ["A", "B"],
                "qrt": ["1404Q2", "1404Q2"],
                "target_date": [140405, 140406],
                "test_origin": [140404, 140404],
            }
        )
        a = BacktestResult(
            model_name="f0",
            overall=pd.DataFrame(),
            predictions=keys.copy(),
        )
        b = BacktestResult(
            model_name="f2",
            overall=pd.DataFrame(),
            predictions=keys.copy(),
        )
        assert_same_eval_rows(a, b)
        bad = keys.copy()
        bad.loc[0, "product"] = "Z"
        c = BacktestResult(model_name="x", overall=pd.DataFrame(), predictions=bad)
        with self.assertRaises(AssertionError):
            assert_same_eval_rows(a, c)


class TestFreezeNotTouchedByFeatureFns(unittest.TestCase):
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
        panel = pd.DataFrame({"product": ["P"], "origin": [140404], "horizon": [1]})
        add_demand_f2_features(panel, hist, origin_col="origin")
        after = {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.glob("*.parquet")
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
