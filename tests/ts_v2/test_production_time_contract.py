"""Production time contract: horizon mapping, leakage, and bridge discard."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.data import (
    assert_training_before_origin,
    filter_training_frame,
    filter_training_history,
)
from pkg.ts_v2.dates import (
    delivered_target_dates,
    internal_target_dates,
    make_forecast_window,
    make_screening_forecast_window,
    map_internal_predictions_to_delivered,
    resolve_production_time_contract,
)
from pkg.ts_v2.models.base import run_model
from pkg.ts_v2.types import ForecastResult, ForecastWindow


class TestProductionTimeContract(unittest.TestCase):
    def test_resolve_140501_months(self):
        c = resolve_production_time_contract(140501)
        self.assertEqual(c.forecast_start, 140501)
        self.assertEqual(c.current_partial_month, 140412)
        self.assertEqual(c.last_complete_month, 140411)
        self.assertEqual(c.delivered_horizon, 15)
        self.assertEqual(c.internal_steps, 16)

    def test_uses_shamsi_add_months_not_integer_subtraction(self):
        # 140501 - 1 as integer would be 140500 (invalid); calendar rolls to 140412.
        c = resolve_production_time_contract(140501)
        self.assertEqual(c.current_partial_month, shamsi_add_months(140501, -1))
        self.assertEqual(c.last_complete_month, shamsi_add_months(140501, -2))
        self.assertNotEqual(c.current_partial_month, 140501 - 1)

    def test_horizon_mapping_internal_and_delivered(self):
        c = resolve_production_time_contract(140501)
        internal = internal_target_dates(c)
        delivered = delivered_target_dates(c)
        self.assertEqual(len(internal), 16)
        self.assertEqual(len(delivered), 15)
        self.assertEqual(internal[0], 140412)
        self.assertEqual(internal[1], 140501)
        self.assertEqual(delivered[0], 140501)
        self.assertEqual(delivered[-1], 140603)
        self.assertEqual(internal[1:], delivered)

    def test_year_roll_partial_and_complete(self):
        c = resolve_production_time_contract(140501)
        self.assertEqual(c.current_partial_month, 140412)
        self.assertEqual(c.last_complete_month, 140411)
        c2 = resolve_production_time_contract(140101)
        self.assertEqual(c2.current_partial_month, 140012)
        self.assertEqual(c2.last_complete_month, 140011)

    def test_map_internal_predictions_drops_bridge(self):
        internal = tuple(float(i) for i in range(16))
        delivered = map_internal_predictions_to_delivered(
            internal, delivered_horizon=15
        )
        self.assertEqual(len(delivered), 15)
        self.assertEqual(delivered[0], 1.0)
        self.assertEqual(delivered[-1], 15.0)


class TestProductionForecastWindow(unittest.TestCase):
    def test_make_forecast_window_production_fields(self):
        window = make_forecast_window(140501)
        self.assertEqual(window.forecast_origin, 140501)
        self.assertEqual(window.training_end, 140411)
        self.assertEqual(window.current_partial_month, 140412)
        self.assertEqual(len(window.target_dates), 15)
        self.assertEqual(window.target_dates[0], 140501)
        self.assertEqual(window.internal_target_dates[0], 140412)
        self.assertEqual(len(window.internal_target_dates), 16)
        self.assertNotIn(140412, window.target_dates)

    def test_screening_window_unchanged(self):
        window = make_screening_forecast_window(140501)
        self.assertEqual(window.forecast_origin, 140501)
        self.assertEqual(window.training_end, 140412)
        self.assertIsNone(window.current_partial_month)
        self.assertEqual(window.internal_target_dates, window.target_dates)
        self.assertEqual(len(window.target_dates), 15)

    def test_training_excludes_partial_and_origin(self):
        window = make_forecast_window(140501)
        sales = pd.DataFrame(
            {
                "date": [140410, 140411, 140412, 140501, 140502],
                "sales": [1, 2, 3, 4, 5],
            }
        )
        train = filter_training_frame(sales, window)
        self.assertEqual(set(train["date"].tolist()), {140410, 140411})
        self.assertNotIn(140412, set(train["date"].tolist()))
        self.assertNotIn(140501, set(train["date"].tolist()))
        self.assertTrue((train["date"] <= window.training_end).all())

        history = pd.Series(
            [10.0, 20.0, 30.0, 40.0],
            index=[140410, 140411, 140412, 140501],
            name="sales",
        )
        filtered = filter_training_history(history, window)
        self.assertEqual(list(filtered.index), [140410, 140411])
        assert_training_before_origin(filtered, window)
        with self.assertRaises(ValueError):
            assert_training_before_origin(history, window)


class _BridgeEchoModel:
    """Predicts identity values for each requested target date."""

    name = "bridge_echo"

    def fit(self, train_series: pd.Series) -> "_BridgeEchoModel":
        self._n = len(train_series)
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        preds = tuple(float(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=preds,
            target_dates=tuple(int(d) for d in target_dates),
            horizons=tuple(range(1, horizon + 1)),
            metadata={"n_train": self._n},
        )


class TestRunModelBridgeDiscard(unittest.TestCase):
    def test_run_model_strips_bridge_under_production_window(self):
        window = make_forecast_window(140501)
        train = pd.Series(
            [1.0] * 12,
            index=[shamsi_add_months(140411, -i) for i in range(11, -1, -1)],
        )
        outcome = run_model(_BridgeEchoModel(), train, window)
        self.assertIsInstance(outcome, ForecastResult)
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        self.assertEqual(outcome.predictions[0], float(140501))
        self.assertNotEqual(outcome.predictions[0], float(140412))
        self.assertEqual(outcome.horizons, tuple(range(1, 16)))

    def test_run_model_screening_no_bridge(self):
        window = make_screening_forecast_window(140501)
        train = pd.Series(
            [1.0] * 12,
            index=[shamsi_add_months(140412, -i) for i in range(11, -1, -1)],
        )
        outcome = run_model(_BridgeEchoModel(), train, window)
        self.assertIsInstance(outcome, ForecastResult)
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.predictions[0], float(140501))


if __name__ == "__main__":
    unittest.main()
