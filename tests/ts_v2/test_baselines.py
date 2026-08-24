"""Hand-calculated tests for naive, seasonal naive, and drift baselines."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.models import (
    DriftModel,
    ForecastResult,
    ModelFailure,
    NaiveModel,
    SeasonalNaiveModel,
    get_model,
    is_failure,
    is_success,
    run_model,
)


def _series(values: list[float], start: int = 140301) -> pd.Series:
    """Build a Shamsi monthly series starting at ``start`` (YYYYMM)."""
    from pkg.benchmark.calendar import shamsi_add_months

    idx = [shamsi_add_months(start, i) for i in range(len(values))]
    return pd.Series(values, index=idx, name="sales")


def _window(horizon: int = 15, origin: int = 140501):
    return make_forecast_window(origin, config=TSForecastConfig(forecast_horizon=horizon))


class TestNaiveModel(unittest.TestCase):
    def test_all_horizons_equal_last_value_horizon_15(self):
        train = _series([2.0, 5.0, 8.0])
        outcome = run_model(NaiveModel(), train, _window(15))
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates[0], 140501)
        self.assertEqual(outcome.predictions, tuple(8.0 for _ in range(15)))

    def test_one_observation_history(self):
        train = _series([7.5])
        outcome = run_model(NaiveModel(), train, _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(7.5 for _ in range(15)))

    def test_all_zero_history(self):
        train = _series([0.0, 0.0, 0.0])
        outcome = run_model(NaiveModel(), train, _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(0.0 for _ in range(15)))

    def test_does_not_clip_or_round(self):
        train = _series([-1.25, 2.4])
        outcome = run_model(NaiveModel(), train, _window(3))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, (2.4, 2.4, 2.4))

    def test_empty_history_unavailable(self):
        outcome = run_model(NaiveModel(), pd.Series(dtype=float), _window(15))
        self.assertTrue(is_failure(outcome))
        assert isinstance(outcome, ModelFailure)
        self.assertEqual(outcome.error_type, "ModelUnavailable")


class TestSeasonalNaiveModel(unittest.TestCase):
    def test_repeats_last_cycle_and_wraps_past_12(self):
        cycle = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        train = _series(cycle)
        outcome = run_model(SeasonalNaiveModel(), train, _window(15))
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        expected = tuple(cycle + [1.0, 2.0, 3.0])
        self.assertEqual(outcome.predictions, expected)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, _window(15).target_dates)

    def test_uses_latest_complete_cycle_not_older_year(self):
        old = [100.0] * 12
        latest = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        train = _series(old + latest)
        outcome = run_model(SeasonalNaiveModel(), train, _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(latest + [1.0, 2.0, 3.0]))

    def test_insufficient_history_is_unavailable_not_naive(self):
        train = _series([1.0] * 11)
        seasonal = run_model(SeasonalNaiveModel(), train, _window(15))
        naive = run_model(NaiveModel(), train, _window(15))
        self.assertTrue(is_failure(seasonal))
        assert isinstance(seasonal, ModelFailure)
        self.assertEqual(seasonal.error_type, "ModelUnavailable")
        self.assertIn("12", seasonal.reason)
        self.assertTrue(is_success(naive))
        assert isinstance(naive, ForecastResult)
        self.assertEqual(naive.predictions, tuple(1.0 for _ in range(15)))

    def test_all_zero_history_with_full_cycle(self):
        train = _series([0.0] * 12)
        outcome = run_model(SeasonalNaiveModel(), train, _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(0.0 for _ in range(15)))

    def test_one_observation_unavailable(self):
        outcome = run_model(SeasonalNaiveModel(), _series([9.0]), _window(15))
        self.assertTrue(is_failure(outcome))
        assert isinstance(outcome, ModelFailure)
        self.assertEqual(outcome.error_type, "ModelUnavailable")


class TestDriftModel(unittest.TestCase):
    def test_two_points_hand_calculated_horizon_15(self):
        # y = [10, 20], T=2, slope = (20-10)/1 = 10
        # yhat[h] = 20 + 10*h
        train = _series([10.0, 20.0])
        outcome = run_model(DriftModel(), train, _window(15))
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        expected = tuple(20.0 + 10.0 * h for h in range(1, 16))
        self.assertEqual(outcome.predictions, expected)
        self.assertEqual(outcome.predictions[-1], 170.0)
        self.assertEqual(len(outcome.predictions), 15)

    def test_three_points_hand_calculated(self):
        # y = [10, 12, 14], T=3, slope = (14-10)/2 = 2
        # yhat[h] = 14 + 2*h
        train = _series([10.0, 12.0, 14.0])
        outcome = run_model(DriftModel(), train, _window(4))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, (16.0, 18.0, 20.0, 22.0))

    def test_one_observation_unavailable_not_naive(self):
        train = _series([7.0])
        drift = run_model(DriftModel(), train, _window(15))
        naive = run_model(NaiveModel(), train, _window(15))
        self.assertTrue(is_failure(drift))
        assert isinstance(drift, ModelFailure)
        self.assertEqual(drift.error_type, "ModelUnavailable")
        self.assertTrue(is_success(naive))

    def test_all_zero_history(self):
        train = _series([0.0, 0.0, 0.0])
        outcome = run_model(DriftModel(), train, _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(0.0 for _ in range(15)))

    def test_negative_drift_not_clipped(self):
        # y = [10, 4], slope = (4-10)/1 = -6; yhat[h] = 4 - 6h
        train = _series([10.0, 4.0])
        outcome = run_model(DriftModel(), train, _window(3))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, (-2.0, -8.0, -14.0))


class TestBaselineRegistry(unittest.TestCase):
    def test_get_model_names(self):
        for name in ("naive", "seasonal_naive", "drift"):
            model = get_model(name)
            self.assertEqual(model.name, name)
            outcome = run_model(model, _series([1.0] * 12), _window(15))
            self.assertTrue(is_success(outcome))
            assert isinstance(outcome, ForecastResult)
            self.assertEqual(len(outcome.predictions), 15)


if __name__ == "__main__":
    unittest.main()
