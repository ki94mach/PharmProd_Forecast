"""Expanding-window backtest engine: leakage, coverage, horizon-equal MAE."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.backtest import (
    assert_backtest_no_leakage,
    backtest_product,
    run_backtest,
)
from pkg.ts_v2.backtest_origins import discover_origins
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series, product_monthly_sales
from pkg.ts_v2.metrics import horizon_mae, selection_mae_from_horizons
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.registry import REGISTRY


class DummyLastValueModel(BaseForecastModel):
    name = "dummy_last_value"

    def __init__(self) -> None:
        self._last: float | None = None

    def fit(self, train_series: pd.Series) -> "DummyLastValueModel":
        clean = train_series.dropna()
        self._last = float(clean.iloc[-1]) if not clean.empty else 0.0
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        last = float(self._last)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(last for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


class DummyHorizonBiasModel(BaseForecastModel):
    """Prediction = actual + horizon (for controlled metric tests)."""

    name = "dummy_horizon_bias"

    def fit(self, train_series: pd.Series) -> "DummyHorizonBiasModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(float(h) for h in range(1, horizon + 1)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


def _monthly_sales_frame(
    product: str,
    start_ym: int,
    n_months: int,
    *,
    base: float = 100.0,
    step: float = 1.0,
) -> pd.DataFrame:
    from pkg.benchmark.calendar import shamsi_add_months

    rows = []
    cur = start_ym
    for i in range(n_months):
        rows.append(
            {
                "product": product,
                "date": cur,
                "sales": base + step * i,
            }
        )
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestBacktestOrigins(unittest.TestCase):
    def test_discover_origins_respects_available_actuals(self):
        sales = _monthly_sales_frame("A", 140401, 18)
        series = product_monthly_sales(sales, "A")
        cfg = TSForecastConfig(min_train_months=12, forecast_horizon=15)
        covers = discover_origins(series, config=cfg)
        self.assertTrue(len(covers) >= 1)
        last_cover = min(covers, key=lambda c: c.origin.shamsi_yyyymm)
        self.assertLess(last_cover.max_evaluated_horizon, 15)

    def test_full_horizon_origins_sorted_first(self):
        sales = _monthly_sales_frame("A", 140401, 30)
        series = product_monthly_sales(sales, "A")
        cfg = TSForecastConfig(min_train_months=12, forecast_horizon=15)
        covers = discover_origins(series, config=cfg)
        idx_full = [i for i, c in enumerate(covers) if c.full_horizon_coverage]
        idx_partial = [i for i, c in enumerate(covers) if not c.full_horizon_coverage]
        if idx_full and idx_partial:
            self.assertLess(max(idx_full), min(idx_partial))


class TestBacktestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_model("dummy_last_value", DummyLastValueModel, replace=True)

    @classmethod
    def tearDownClass(cls) -> None:
        REGISTRY.unregister("dummy_last_value")
        REGISTRY.unregister("dummy_horizon_bias")

    def setUp(self) -> None:
        self.cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            candidate_models=("dummy_last_value",),
        )
        self.sales = _monthly_sales_frame("SKU1", 140401, 30)

    def test_run_backtest_prediction_columns(self):
        result = run_backtest(
            self.sales,
            ["SKU1"],
            model_names=("dummy_last_value",),
            config=self.cfg,
        )
        self.assertFalse(result.predictions.empty)
        self.assertEqual(
            list(result.predictions.columns),
            [
                "product",
                "model",
                "origin",
                "target_date",
                "horizon",
                "actual",
                "prediction",
            ],
        )
        self.assertTrue(result.predictions["actual"].notna().all())
        self.assertTrue(result.predictions["prediction"].notna().all())

    def test_expanding_window_train_grows(self):
        result = run_backtest(
            self.sales,
            ["SKU1"],
            model_names=("dummy_last_value",),
            config=self.cfg,
        )
        origins = sorted(result.predictions["origin"].unique())
        self.assertGreaterEqual(len(origins), 2)
        prepared_lengths = []
        for origin in origins[:3]:
            prep = prepare_monthly_series(
                self.sales, "SKU1", int(origin), config=self.cfg
            )
            prepared_lengths.append(prep.n_observations)
        self.assertEqual(prepared_lengths, sorted(prepared_lengths))

    def test_leakage_assertion_per_origin(self):
        result = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel()],
            config=self.cfg,
        )
        prepared = {}
        for origin in result.predictions["origin"].unique():
            prepared[int(origin)] = prepare_monthly_series(
                self.sales, "SKU1", int(origin), config=self.cfg
            )
        assert_backtest_no_leakage(result.predictions, prepared)

    def test_training_strictly_before_origin(self):
        result = run_backtest(
            self.sales,
            ["SKU1"],
            model_names=("dummy_last_value",),
            config=self.cfg,
        )
        for origin, group in result.predictions.groupby("origin"):
            prep = prepare_monthly_series(
                self.sales, "SKU1", int(origin), config=self.cfg
            )
            max_train = max(prep.dates)
            min_target = int(group["target_date"].min())
            self.assertLess(max_train, int(origin))
            self.assertLessEqual(int(origin), min_target)

    def test_short_history_reduced_coverage_reported(self):
        short_sales = _monthly_sales_frame("SHORT", 140401, 16)
        cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        result = backtest_product(
            short_sales,
            "SHORT",
            [DummyLastValueModel()],
            config=cfg,
        )
        row = result.coverage.iloc[0]
        self.assertGreater(row["number_of_origins"], 0)
        self.assertLess(row["max_evaluated_horizon"], 15)
        self.assertEqual(row["n_full_horizon_origins"], 0)

    def test_coverage_fields_present(self):
        result = run_backtest(
            self.sales,
            ["SKU1"],
            model_names=("dummy_last_value",),
            config=self.cfg,
        )
        cov = result.coverage.iloc[0]
        self.assertEqual(cov["product"], "SKU1")
        self.assertEqual(cov["model"], "dummy_last_value")
        self.assertGreater(cov["number_of_origins"], 0)
        self.assertGreater(cov["number_of_predictions"], 0)
        self.assertGreater(cov["max_evaluated_horizon"], 0)
        self.assertIsInstance(cov["evaluated_horizons"], (tuple, list))

    def test_selection_mae_equal_horizon_weight_not_row_weight(self):
        cfg = TSForecastConfig(
            forecast_horizon=2,
            min_train_months=3,
            activity_start_min_sales=None,
        )
        sales = _monthly_sales_frame("M", 140401, 10, base=10.0, step=0.0)
        register_model("dummy_horizon_bias", DummyHorizonBiasModel, replace=True)
        result = run_backtest(
            sales,
            ["M"],
            model_names=("dummy_horizon_bias",),
            config=cfg,
        )
        preds = result.predictions
        h_mae = horizon_mae(preds)
        expected_sel = selection_mae_from_horizons(h_mae)
        reported = float(result.metrics.iloc[0]["selection_mae"])
        self.assertAlmostEqual(reported, expected_sel, places=6)
        row_mae = (preds["actual"] - preds["prediction"]).abs().mean()
        self.assertNotAlmostEqual(reported, float(row_mae), places=3)

    def test_actuals_are_raw_units_not_scaled(self):
        self.sales.loc[0, "sales"] = 123.45
        result = run_backtest(
            self.sales,
            ["SKU1"],
            model_names=("dummy_last_value",),
            config=self.cfg,
        )
        match = result.predictions.loc[
            result.predictions["target_date"] == 140401, "actual"
        ]
        if not match.empty:
            self.assertAlmostEqual(float(match.iloc[0]), 123.45, places=2)

    def test_model_failure_does_not_abort_sku(self):
        class FailModel(BaseForecastModel):
            name = "fail_model"

            def fit(self, train_series: pd.Series) -> "FailModel":
                raise RuntimeError("boom")

            def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
                raise AssertionError("no")

        result = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel(), FailModel()],
            config=self.cfg,
        )
        self.assertFalse(result.predictions.empty)
        self.assertFalse(result.failures.empty)
        self.assertIn("dummy_last_value", set(result.predictions["model"]))
        self.assertIn("fail_model", set(result.failures["model"]))


class TestBacktestMetrics(unittest.TestCase):
    def test_horizon_diagnostics_and_selection_score(self):
        from pkg.ts_v2.metrics import aggregate_metrics, selection_mae_from_horizons

        preds = pd.DataFrame(
            [
                {"horizon": 1, "actual": 10.0, "prediction": 12.0},
                {"horizon": 1, "actual": 20.0, "prediction": 18.0},
                {"horizon": 2, "actual": 10.0, "prediction": 10.0},
            ]
        )
        m = aggregate_metrics(preds)
        h_mae = m["horizon_mae"]
        self.assertAlmostEqual(float(h_mae[1]), 2.0)
        self.assertAlmostEqual(float(h_mae[2]), 0.0)
        self.assertAlmostEqual(selection_mae_from_horizons(h_mae), 1.0)
        self.assertAlmostEqual(float(m["overall_rmse"]), (8.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(float(m["overall_bias"]), 0.0)
        self.assertAlmostEqual(float(m["horizon_bias"][1]), 0.0)


if __name__ == "__main__":
    unittest.main()
