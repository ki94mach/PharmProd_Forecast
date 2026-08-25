"""Ensemble strategy evaluation from OOF backtest predictions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.backtest import backtest_product
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.ensemble import (
    ALL_ENSEMBLE_STRATEGIES,
    STRATEGY_BEST_SINGLE,
    STRATEGY_INVERSE_MAE_TOP3,
    STRATEGY_MEAN_TOP3,
    STRATEGY_MEDIAN_TOP3,
    assert_ensemble_no_future_ranking_leakage,
    build_ensemble_predictions,
    compare_ensemble_strategies,
    _combine_predictions,
    _expanding_model_scores,
)
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.registry import REGISTRY


class FixedPredictionModel(BaseForecastModel):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = float(value)

    def fit(self, train_series: pd.Series) -> "FixedPredictionModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._value for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


def _sales(product: str, start: int, n: int, sales: float = 100.0) -> pd.DataFrame:
    from pkg.benchmark.calendar import shamsi_add_months

    rows = []
    cur = start
    for _ in range(n):
        rows.append({"product": product, "date": cur, "sales": sales})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestEnsembleCombine(unittest.TestCase):
    def test_mean_top3(self):
        pred, used = _combine_predictions(
            {"a": 10.0, "b": 20.0, "c": 30.0},
            ["a", "b", "c"],
            {"a": 1.0, "b": 2.0, "c": 3.0},
            STRATEGY_MEAN_TOP3,
        )
        self.assertAlmostEqual(pred, 20.0)
        self.assertEqual(used, ("a", "b", "c"))

    def test_inverse_mae_weights(self):
        pred, used = _combine_predictions(
            {"a": 10.0, "b": 20.0},
            ["a", "b"],
            {"a": 1.0, "b": 2.0},
            STRATEGY_INVERSE_MAE_TOP3,
        )
        # weights 1 and 0.5 -> (10*1 + 20*0.5)/1.5 = 13.333...
        self.assertAlmostEqual(pred, 40.0 / 3.0, places=6)
        self.assertEqual(used, ("a", "b"))


class TestEnsembleFromBacktest(unittest.TestCase):
    def tearDown(self) -> None:
        for name in ("m10", "m20", "m30", "m100"):
            REGISTRY.unregister(name)

    def test_mean_top3_matches_manual_mean(self):
        register_model("m10", lambda: FixedPredictionModel("m10", 10.0), replace=True)
        register_model("m20", lambda: FixedPredictionModel("m20", 20.0), replace=True)
        register_model("m30", lambda: FixedPredictionModel("m30", 30.0), replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=3,
            min_train_months=3,
            activity_start_min_sales=None,
            ensemble_top_k=3,
            candidate_models=("m10", "m20", "m30"),
        )
        sales = _sales("E", 140401, 10)
        bt = backtest_product(
            sales,
            "E",
            [FixedPredictionModel("m10", 10.0), FixedPredictionModel("m20", 20.0), FixedPredictionModel("m30", 30.0)],
            config=cfg,
        )
        ens = build_ensemble_predictions(bt, "E", STRATEGY_MEAN_TOP3, config=cfg)
        self.assertFalse(ens.empty)
        self.assertTrue(np.allclose(ens["prediction"].to_numpy(), 20.0))

    def test_expanding_scores_ignore_current_and_future_origins(self):
        register_model("m10", lambda: FixedPredictionModel("m10", 10.0), replace=True)
        register_model("m20", lambda: FixedPredictionModel("m20", 20.0), replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=2,
            min_train_months=3,
            activity_start_min_sales=None,
            candidate_models=("m10", "m20"),
        )
        sales = _sales("L", 140401, 12, sales=100.0)
        bt = backtest_product(
            sales,
            "L",
            [FixedPredictionModel("m10", 10.0), FixedPredictionModel("m20", 50.0)],
            config=cfg,
        )
        origins = sorted(int(o) for o in bt.predictions["origin"].unique())
        self.assertGreaterEqual(len(origins), 2)
        second = origins[1]
        scores = _expanding_model_scores(bt.predictions, product="L", origin=second, config=cfg)
        self.assertIn("m10", scores)
        assert_ensemble_no_future_ranking_leakage(bt, "L", second, config=cfg)

    def test_ranking_uses_only_prior_origin_performance(self):
        """A model that wins only on the latest origin must not rank first earlier."""
        from pkg.ts_v2.types import BacktestResult

        rows = [
            # origin 140403: model_a perfect, model_b poor
            {"product": "R", "model": "model_a", "origin": 140403, "target_date": 140403, "horizon": 1, "actual": 100.0, "prediction": 100.0},
            {"product": "R", "model": "model_b", "origin": 140403, "target_date": 140403, "horizon": 1, "actual": 100.0, "prediction": 0.0},
            # origin 140404: model_a poor, model_b perfect
            {"product": "R", "model": "model_a", "origin": 140404, "target_date": 140404, "horizon": 1, "actual": 100.0, "prediction": 0.0},
            {"product": "R", "model": "model_b", "origin": 140404, "target_date": 140404, "horizon": 1, "actual": 100.0, "prediction": 100.0},
        ]
        bt = BacktestResult(
            predictions=pd.DataFrame(rows),
            coverage=pd.DataFrame(),
            metrics=pd.DataFrame(),
            failures=pd.DataFrame(),
        )
        cfg = TSForecastConfig(forecast_horizon=1, ensemble_top_k=1)
        scores = _expanding_model_scores(bt.predictions, product="R", origin=140404, config=cfg)
        self.assertLess(scores["model_a"], scores["model_b"])

        best = build_ensemble_predictions(bt, "R", STRATEGY_BEST_SINGLE, config=cfg)
        row = best.loc[best["origin"] == 140404].iloc[0]
        self.assertEqual(row["contributing_models"], ("model_a",))
        self.assertEqual(float(row["prediction"]), 0.0)

    def test_compare_report_covers_all_strategies(self):
        register_model("m10", lambda: FixedPredictionModel("m10", 10.0), replace=True)
        register_model("m20", lambda: FixedPredictionModel("m20", 20.0), replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=2,
            min_train_months=3,
            activity_start_min_sales=None,
            selection_strategy="best_model",
            candidate_models=("m10", "m20"),
        )
        sales = _sales("P", 140401, 10)
        bt = backtest_product(
            sales,
            "P",
            [FixedPredictionModel("m10", 10.0), FixedPredictionModel("m20", 20.0)],
            config=cfg,
        )
        report = compare_ensemble_strategies(bt, ["P"], config=cfg)
        self.assertEqual(
            set(report.strategy_predictions.keys()),
            set(ALL_ENSEMBLE_STRATEGIES),
        )
        self.assertEqual(len(report.strategy_metrics), len(ALL_ENSEMBLE_STRATEGIES))
        self.assertIn("product", report.sku_comparison.columns)
        self.assertIn("best_strategy_by_mae", report.sku_comparison.columns)
        self.assertIn("current_default_strategy", report.sku_comparison.columns)
        summary = report.summary.set_index("strategy")
        self.assertTrue(bool(summary.loc[STRATEGY_BEST_SINGLE, "is_production_default"]))


if __name__ == "__main__":
    unittest.main()
