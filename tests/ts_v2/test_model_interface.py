"""Dummy-model tests proving backtest/engine share the V2 model interface."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.backtest import forecast_fold, make_folds
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, parse_origin
from pkg.ts_v2.engine import forecast_series
from pkg.ts_v2.models import (
    BaseForecastModel,
    ForecastResult,
    ModelFailure,
    available_models,
    is_failure,
    is_success,
    register_model,
    run_model,
)
from pkg.ts_v2.models.registry import REGISTRY


class DummyLastValueModel(BaseForecastModel):
    """Repeats the last training value for every requested target date."""

    name = "dummy_last_value"

    def __init__(self) -> None:
        self._last: float | None = None

    def fit(self, train_series: pd.Series) -> "DummyLastValueModel":
        clean = train_series.dropna()
        if clean.empty:
            raise ValueError("dummy_last_value requires non-empty training history")
        self._last = float(clean.iloc[-1])
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        last = float(self._last)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(last for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
            metadata={"last_value": last},
        )


class DummySkipFirstMonthModel(BaseForecastModel):
    """Illegal: emits horizon-1 values (V1-style skip of the first month)."""

    name = "dummy_skip_first"

    def fit(self, train_series: pd.Series) -> "DummySkipFirstMonthModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates[1:])
        preds = tuple(1.0 for _ in dates)
        return ForecastResult(
            model_name=self.name,
            predictions=preds,
            target_dates=dates,
            horizons=tuple(range(1, len(preds) + 1)),
        )


class DummyBoomModel(BaseForecastModel):
    name = "dummy_boom"

    def fit(self, train_series: pd.Series) -> "DummyBoomModel":
        raise RuntimeError("solver exploded")

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        raise AssertionError("predict should not run")


class TestDummyModelInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.Series(
            [10.0, 20.0, 30.0],
            index=[140410, 140411, 140412],
            name="sales",
        )
        self.cfg = TSForecastConfig(forecast_horizon=15)
        self.window = make_forecast_window(140501, config=self.cfg)

    def test_dummy_predictions_match_horizon_and_target_dates(self):
        outcome = run_model(DummyLastValueModel(), self.train, self.window)
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, self.window.target_dates)
        self.assertEqual(outcome.target_dates[0], 140501)
        self.assertEqual(outcome.target_dates[-1], 140603)
        self.assertEqual(outcome.horizons, tuple(range(1, 16)))
        self.assertTrue(all(p == 30.0 for p in outcome.predictions))
        self.assertEqual(outcome.model_name, "dummy_last_value")

    def test_dummy_does_not_round_or_smooth(self):
        train = pd.Series([10.4], index=[140412])
        outcome = run_model(DummyLastValueModel(), train, self.window)
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(outcome.predictions[0], 10.4)

    def test_engine_and_backtest_share_run_model(self):
        model_a = DummyLastValueModel()
        model_b = DummyLastValueModel()
        from_engine = forecast_series(model_a, self.train, self.window)
        fold = make_folds([parse_origin(140501)], config=self.cfg)[0]
        from_backtest = forecast_fold(model_b, self.train, fold, config=self.cfg)
        self.assertTrue(is_success(from_engine))
        self.assertTrue(is_success(from_backtest))
        assert isinstance(from_engine, ForecastResult)
        assert isinstance(from_backtest, ForecastResult)
        self.assertEqual(from_engine.predictions, from_backtest.predictions)
        self.assertEqual(from_engine.target_dates, from_backtest.target_dates)
        self.assertEqual(from_engine.target_dates, self.window.target_dates)

    def test_skipping_first_month_is_a_typed_failure(self):
        outcome = run_model(DummySkipFirstMonthModel(), self.train, self.window)
        self.assertTrue(is_failure(outcome))
        assert isinstance(outcome, ModelFailure)
        self.assertEqual(outcome.model_name, "dummy_skip_first")
        self.assertIn("target_dates", outcome.reason)

    def test_internal_exception_is_model_failure_not_crash(self):
        outcome = run_model(DummyBoomModel(), self.train, self.window)
        self.assertTrue(is_failure(outcome))
        assert isinstance(outcome, ModelFailure)
        self.assertEqual(outcome.error_type, "RuntimeError")
        self.assertIn("solver exploded", outcome.reason)

    def test_registry_configures_candidates_centrally(self):
        self.assertEqual(available_models(), ())
        register_model("dummy_last_value", DummyLastValueModel)
        self.addCleanup(lambda: REGISTRY.unregister("dummy_last_value"))
        self.assertIn("dummy_last_value", available_models())
        cfg = TSForecastConfig(
            forecast_horizon=3,
            candidate_models=("dummy_last_value",),
        )
        from pkg.ts_v2.models import models_from_config

        models = models_from_config(cfg.candidate_models)
        self.assertEqual(len(models), 1)
        window = make_forecast_window(140501, config=cfg)
        outcome = run_model(models[0], self.train, window)
        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 3)
        self.assertEqual(outcome.target_dates[0], 140501)


if __name__ == "__main__":
    unittest.main()
