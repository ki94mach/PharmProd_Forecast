"""V2 forecast-origin / target-date contract tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.data import (
    assert_training_before_origin,
    filter_training_frame,
    filter_training_history,
)
from pkg.ts_v2.dates import (
    SHAMSI_TO_PANDAS_YYYYMM_OFFSET,
    make_forecast_window,
    pandas_yyyymm_to_shamsi,
    shamsi_to_pandas_yyyymm,
    target_month,
)
from pkg.ts_v2.types import ForecastWindow


class TestForecastWindowContract(unittest.TestCase):
    def test_origin_140501_targets_through_140603(self):
        window = make_forecast_window(140501)
        self.assertEqual(window.forecast_origin, 140501)
        self.assertEqual(window.training_end, 140412)
        self.assertEqual(window.target_dates[0], 140501)
        self.assertEqual(window.target_dates[1], 140502)
        self.assertEqual(window.target_dates[-1], 140603)
        self.assertEqual(target_month(140501, 1), 140501)
        self.assertEqual(target_month(140501, 15), 140603)

    def test_origin_140512_rolls_into_1406(self):
        window = make_forecast_window(140512)
        self.assertEqual(window.forecast_origin, 140512)
        self.assertEqual(window.training_end, 140511)
        self.assertEqual(window.target_dates[0], 140512)
        self.assertEqual(window.target_dates[1], 140601)
        self.assertEqual(window.target_dates[2], 140602)
        # horizon 15 = origin + 14 months → 140702
        self.assertEqual(window.target_dates[14], 140702)
        self.assertTrue(all(d >= 140601 for d in window.target_dates[1:]))

    def test_exactly_fifteen_target_dates(self):
        window = make_forecast_window(140501)
        self.assertIsInstance(window, ForecastWindow)
        self.assertEqual(len(window.target_dates), 15)
        self.assertEqual(len(window.horizons), 15)
        self.assertEqual(window.horizons, tuple(range(1, 16)))
        self.assertEqual(len(set(window.target_dates)), 15)

    def test_training_observations_strictly_earlier_than_origin(self):
        window = make_forecast_window(140501)
        sales = pd.DataFrame(
            {
                "date": [140410, 140411, 140412, 140501, 140502],
                "sales": [1, 2, 3, 4, 5],
            }
        )
        train = filter_training_frame(sales, window)
        self.assertTrue((train["date"] < window.forecast_origin).all())
        self.assertNotIn(140501, set(train["date"].tolist()))
        self.assertEqual(set(train["date"].tolist()), {140410, 140411, 140412})
        self.assertTrue((train["date"] <= window.training_end).all())

    def test_origin_month_cannot_enter_model_history(self):
        window = make_forecast_window(140501)
        history = pd.Series(
            [10.0, 20.0, 30.0, 999.0],
            index=[140410, 140411, 140412, 140501],
            name="sales",
        )
        filtered = filter_training_history(history, window)
        self.assertNotIn(140501, filtered.index)
        self.assertEqual(list(filtered.index), [140410, 140411, 140412])
        assert_training_before_origin(filtered, window)

        with self.assertRaises(ValueError):
            assert_training_before_origin(history, window)


class TestShamsiPandasOffsetCentralized(unittest.TestCase):
    def test_offset_roundtrip(self):
        self.assertEqual(SHAMSI_TO_PANDAS_YYYYMM_OFFSET, 62100)
        self.assertEqual(shamsi_to_pandas_yyyymm(140501), 140501 + 62100)
        self.assertEqual(pandas_yyyymm_to_shamsi(140501 + 62100), 140501)


if __name__ == "__main__":
    unittest.main()
