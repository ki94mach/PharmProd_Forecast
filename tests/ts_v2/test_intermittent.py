"""Unit tests for Croston SBA, TSB, and intermittency diagnostics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.intermittency import intermittency_stats
from pkg.ts_v2.models import (
    CrostonSBAModel,
    ForecastResult,
    TSBModel,
    is_success,
    run_model,
)
from pkg.ts_v2.models.intermittent import fit_croston_sba, fit_tsb


def _series(values: list[float], start: int = 140301) -> pd.Series:
    idx = [shamsi_add_months(start, i) for i in range(len(values))]
    return pd.Series(values, index=idx, name="sales")


def _window(horizon: int = 15, origin: int = 140501):
    return make_forecast_window(origin, config=TSForecastConfig(forecast_horizon=horizon))


class TestIntermittencyDiagnostics(unittest.TestCase):
    def test_zero_proportion_and_adi(self):
        # demand at indices 0, 3, 6 → intervals 3, 3 → ADI = 3
        values = pd.Series([5.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0])
        stats = intermittency_stats(values)
        self.assertEqual(stats.n_demand_months, 3)
        self.assertAlmostEqual(stats.zero_month_proportion, 5.0 / 8.0)
        self.assertAlmostEqual(stats.average_inter_demand_interval, 3.0)

    def test_prepared_series_exposes_intermittency(self):
        sales = pd.DataFrame(
            [
                ("SkuA", 140410, 10.0),
                ("SkuA", 140411, 0.0),
                ("SkuA", 140412, 0.0),
            ],
            columns=["product", "date", "sales"],
        )
        cfg = TSForecastConfig(activity_start_min_sales=None)
        prepared = prepare_monthly_series(sales, "SkuA", 140501, config=cfg)
        self.assertEqual(prepared.n_demand_months, 1)
        self.assertAlmostEqual(prepared.zero_month_proportion, 2.0 / 3.0)
        self.assertIsNone(prepared.average_inter_demand_interval)


class TestCrostonSBA(unittest.TestCase):
    def test_hand_calculated_two_demands(self):
        # y = [10, 0, 0, 20], α=β=0.5
        # first demand idx0: z=10, p=1
        # second demand idx3: q=3, z=10+0.5*(20-10)=15, p=1+0.5*(3-1)=2
        # rate = (1-0.25)*15/2 = 0.75*7.5 = 5.625
        values = np.array([10.0, 0.0, 0.0, 20.0])
        rate, z, p = fit_croston_sba(values, alpha=0.5, beta=0.5)
        self.assertAlmostEqual(z, 15.0)
        self.assertAlmostEqual(p, 2.0)
        self.assertAlmostEqual(rate, 5.625)

        cfg = TSForecastConfig(croston_alpha=0.5, croston_beta=0.5, forecast_horizon=15)
        outcome = run_model(CrostonSBAModel(cfg), _series(values.tolist()), _window(15))
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, _window(15).target_dates)
        self.assertTrue(all(abs(x - 5.625) < 1e-9 for x in outcome.predictions))

    def test_sparse_demand_horizon_15(self):
        values = [0.0, 8.0] + [0.0] * 10 + [12.0] + [0.0] * 5
        cfg = TSForecastConfig(croston_alpha=0.1, forecast_horizon=15)
        outcome = run_model(CrostonSBAModel(cfg), _series(values), _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(len(set(outcome.predictions)), 1)  # flat rate
        self.assertGreater(outcome.predictions[0], 0.0)

    def test_long_zero_sequences(self):
        values = [5.0] + [0.0] * 20 + [7.0]
        rate, _, p = fit_croston_sba(np.array(values), alpha=0.1, beta=0.1)
        self.assertGreater(p, 1.0)
        outcome = run_model(
            CrostonSBAModel(TSForecastConfig(croston_alpha=0.1)),
            _series(values),
            _window(15),
        )
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertAlmostEqual(outcome.predictions[0], rate)

    def test_all_zero_demand(self):
        values = [0.0] * 12
        outcome = run_model(CrostonSBAModel(), _series(values), _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(0.0 for _ in range(15)))

    def test_intermittent_positive_demand(self):
        # Regular intermittent: demand every 3 months
        values = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0]
        outcome = run_model(
            CrostonSBAModel(TSForecastConfig(croston_alpha=0.2)),
            _series(values),
            _window(15),
        )
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.target_dates[0], 140501)
        self.assertEqual(len(outcome.predictions), 15)
        # Mean demand ~ 10/3 ≈ 3.33; SBA slightly lower than z/p
        self.assertGreater(outcome.predictions[0], 2.0)
        self.assertLess(outcome.predictions[0], 5.0)


class TestTSB(unittest.TestCase):
    def test_hand_calculated_simple(self):
        # After first demand at t=0 (z=10, p=1):
        # t=1 zero: p = 1 + 0.5*(0-1) = 0.5
        # t=2 demand 20: z = 10+0.5*(20-10)=15, p=0.5+0.5*(1-0.5)=0.75
        # rate = 0.75 * 15 = 11.25
        values = np.array([10.0, 0.0, 20.0])
        rate, z, p = fit_tsb(values, alpha=0.5, beta=0.5)
        self.assertAlmostEqual(z, 15.0)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(rate, 11.25)

        cfg = TSForecastConfig(tsb_alpha=0.5, tsb_beta=0.5, forecast_horizon=15)
        outcome = run_model(TSBModel(cfg), _series(values.tolist()), _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, _window(15).target_dates)
        self.assertTrue(all(abs(x - 11.25) < 1e-9 for x in outcome.predictions))

    def test_sparse_and_long_zeros(self):
        values = [6.0] + [0.0] * 15 + [9.0] + [0.0] * 8
        outcome = run_model(
            TSBModel(TSForecastConfig(tsb_alpha=0.1, tsb_beta=0.1)),
            _series(values),
            _window(15),
        )
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        # Long zeros drive probability down → rate below last size.
        self.assertLess(outcome.predictions[0], 9.0)
        self.assertGreaterEqual(outcome.predictions[0], 0.0)

    def test_all_zero_demand(self):
        outcome = run_model(TSBModel(), _series([0.0] * 10), _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions, tuple(0.0 for _ in range(15)))

    def test_intermittent_positive_horizon_15(self):
        values = [0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0]
        outcome = run_model(TSBModel(), _series(values), _window(15))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(len(set(round(x, 12) for x in outcome.predictions)), 1)
        self.assertGreater(outcome.predictions[0], 0.0)


if __name__ == "__main__":
    unittest.main()
