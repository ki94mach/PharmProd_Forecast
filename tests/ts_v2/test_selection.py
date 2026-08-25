"""Per-SKU model selection from out-of-fold backtest metrics."""
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
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.registry import REGISTRY
from pkg.ts_v2.selection import (
    pick_winner_with_tiebreak,
    select_models,
    select_product_model,
    simplicity_rank,
)


class DummyLastValueModel(BaseForecastModel):
    name = "dummy_last_value"

    def __init__(self) -> None:
        self._last: float = 0.0

    def fit(self, train_series: pd.Series) -> "DummyLastValueModel":
        clean = train_series.dropna()
        self._last = float(clean.iloc[-1]) if not clean.empty else 0.0
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._last for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


class DummyTrainMeanModel(BaseForecastModel):
    """In-sample favorite: predicts training mean (poor on trending OOF)."""

    name = "dummy_train_mean"

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, train_series: pd.Series) -> "DummyTrainMeanModel":
        clean = train_series.dropna()
        self._mean = float(clean.mean()) if not clean.empty else 0.0
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._mean for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


class DummyConstantScoreModel(BaseForecastModel):
    """Emits a fixed prediction regardless of history (for tie-break tests)."""

    name = "dummy_constant"

    def __init__(self, prediction: float, model_name: str) -> None:
        self._pred = float(prediction)
        self.name = str(model_name)

    def fit(self, train_series: pd.Series) -> "DummyConstantScoreModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._pred for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


class FailAlwaysModel(BaseForecastModel):
    name = "fail_always"

    def fit(self, train_series: pd.Series) -> "FailAlwaysModel":
        raise RuntimeError("always fails")

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        raise AssertionError("no")


def _monthly_sales_frame(
    product: str,
    start_ym: int,
    n_months: int,
    *,
    step: float = 5.0,
) -> pd.DataFrame:
    from pkg.benchmark.calendar import shamsi_add_months

    rows = []
    cur = start_ym
    for i in range(n_months):
        rows.append({"product": product, "date": cur, "sales": 50.0 + step * i})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestSimplicityAndTieBreak(unittest.TestCase):
    def test_simplicity_order(self):
        order = (
            "seasonal_naive",
            "naive",
            "drift",
            "ets",
            "auto_arima",
            "croston_sba",
            "tsb",
            "prophet",
        )
        self.assertLess(simplicity_rank("naive", order), simplicity_rank("prophet", order))
        self.assertLess(
            simplicity_rank("seasonal_naive", order),
            simplicity_rank("naive", order),
        )

    def test_pick_winner_prefers_simpler_within_tolerance(self):
        cfg_order = ("naive", "prophet")
        winner, tied = pick_winner_with_tiebreak(
            {"naive": 1.0, "prophet": 1.0 + 1e-9},
            tolerance=1e-6,
            simplicity_order=cfg_order,
        )
        self.assertEqual(winner, "naive")
        self.assertTrue(tied)

    def test_pick_winner_uses_lower_score_outside_tolerance(self):
        winner, _ = pick_winner_with_tiebreak(
            {"naive": 2.0, "prophet": 1.0},
            tolerance=1e-6,
            simplicity_order=("naive", "prophet"),
        )
        self.assertEqual(winner, "prophet")


class TestSelectFromBacktest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            selection_tie_tolerance=1e-6,
            candidate_models=("dummy_last_value", "dummy_train_mean"),
        )
        self.sales = _monthly_sales_frame("SKU1", 140401, 30)

    def tearDown(self) -> None:
        for name in (
            "dummy_last_value",
            "dummy_train_mean",
            "fail_always",
            "naive_tie",
            "drift_tie",
        ):
            REGISTRY.unregister(name)

    def test_selects_lowest_selection_mae(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        register_model("dummy_train_mean", DummyTrainMeanModel, replace=True)
        bt = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel(), DummyTrainMeanModel()],
            config=self.cfg,
        )
        sel = select_product_model(bt, "SKU1", config=self.cfg)
        self.assertEqual(sel.selected_model, "dummy_last_value")
        self.assertLess(
            sel.candidate_scores["dummy_last_value"],
            sel.candidate_scores["dummy_train_mean"],
        )
        self.assertEqual(sel.metric, "mae")
        self.assertIn(1, sel.horizon_maes)

    def test_selection_uses_oof_not_in_sample_fit(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        register_model("dummy_train_mean", DummyTrainMeanModel, replace=True)
        bt = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel(), DummyTrainMeanModel()],
            config=self.cfg,
        )
        sel = select_product_model(bt, "SKU1", config=self.cfg)

        # Final-window in-sample MAE would favor train-mean on the last origin.
        from pkg.ts_v2.data import prepare_monthly_series

        last_origin = int(bt.predictions["origin"].max())
        prep = prepare_monthly_series(self.sales, "SKU1", last_origin, config=self.cfg)
        train_vals = prep.values.dropna().to_numpy(dtype=float)
        train_mean = float(np.mean(train_vals))
        in_sample_mae_mean = float(np.mean(np.abs(train_vals - train_mean)))
        in_sample_mae_last = float(np.mean(np.abs(train_vals - train_vals[-1])))
        self.assertLess(in_sample_mae_mean, in_sample_mae_last)

        self.assertEqual(sel.selected_model, "dummy_last_value")
        self.assertLess(
            sel.candidate_scores["dummy_last_value"],
            sel.candidate_scores["dummy_train_mean"],
        )

    def test_rmse_not_used_for_selection(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        register_model("dummy_train_mean", DummyTrainMeanModel, replace=True)
        bt = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel(), DummyTrainMeanModel()],
            config=self.cfg,
        )
        row_last = bt.metrics.loc[bt.metrics["model"] == "dummy_last_value"].iloc[0]
        row_mean = bt.metrics.loc[bt.metrics["model"] == "dummy_train_mean"].iloc[0]
        self.assertLess(float(row_last["selection_mae"]), float(row_mean["selection_mae"]))
        # If RMSE were used, winner could differ when RMSE ordering disagrees — here
        # selection_mae ordering must match the picked model.
        sel = select_product_model(bt, "SKU1", config=self.cfg)
        self.assertEqual(sel.selected_model, "dummy_last_value")
        if float(row_mean["overall_rmse"]) < float(row_last["overall_rmse"]):
            self.assertNotEqual(sel.selected_model, "dummy_train_mean")

    def test_failed_model_reported_unavailable(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        register_model("fail_always", FailAlwaysModel, replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            candidate_models=("dummy_last_value", "fail_always"),
        )
        bt = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel(), FailAlwaysModel()],
            config=cfg,
        )
        sel = select_product_model(bt, "SKU1", config=cfg)
        self.assertEqual(sel.selected_model, "dummy_last_value")
        self.assertIn("fail_always", sel.unavailable)
        self.assertIn("always fails", sel.unavailable["fail_always"])

    def test_insufficient_coverage_excluded(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            min_selection_origins=999,
            candidate_models=("dummy_last_value",),
        )
        bt = backtest_product(
            self.sales,
            "SKU1",
            [DummyLastValueModel()],
            config=cfg,
        )
        with self.assertRaises(ValueError):
            select_product_model(bt, "SKU1", config=cfg)

    def test_tie_break_prefers_naive_over_drift(self):
        register_model("naive_tie", lambda: DummyConstantScoreModel(100.0, "naive_tie"), replace=True)
        register_model("drift_tie", lambda: DummyConstantScoreModel(100.0, "drift_tie"), replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=3,
            min_train_months=3,
            activity_start_min_sales=None,
            selection_tie_tolerance=1.0,
            selection_simplicity_order=("naive_tie", "drift_tie"),
            candidate_models=("naive_tie", "drift_tie"),
        )
        sales = _monthly_sales_frame("T", 140401, 10, step=0.0)
        bt = backtest_product(
            sales,
            "T",
            [
                DummyConstantScoreModel(100.0, "naive_tie"),
                DummyConstantScoreModel(100.0, "drift_tie"),
            ],
            config=cfg,
        )
        sel = select_product_model(bt, "T", config=cfg)
        self.assertEqual(sel.selected_model, "naive_tie")
        self.assertTrue(sel.tie_break_applied)

    def test_select_models_multiple_skus(self):
        register_model("dummy_last_value", DummyLastValueModel, replace=True)
        sales = pd.concat(
            [
                _monthly_sales_frame("A", 140401, 20),
                _monthly_sales_frame("B", 140401, 20),
            ],
            ignore_index=True,
        )
        cfg = TSForecastConfig(
            forecast_horizon=5,
            min_train_months=6,
            activity_start_min_sales=None,
            candidate_models=("dummy_last_value",),
        )
        from pkg.ts_v2.backtest import run_backtest

        bt = run_backtest(sales, ["A", "B"], model_names=("dummy_last_value",), config=cfg)
        results = select_models(bt, config=cfg)
        self.assertEqual(set(results.keys()), {"A", "B"})
        self.assertEqual(results["A"].selected_model, "dummy_last_value")


if __name__ == "__main__":
    unittest.main()
