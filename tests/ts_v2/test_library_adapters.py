"""Contract tests for AutoARIMA / ETS / Prophet V2 adapters (libraries mocked)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import TSForecastConfig, use_seasonal
from pkg.ts_v2.dates import make_forecast_window, shamsi_months_to_ms_index, shamsi_to_month_start_timestamp
from pkg.ts_v2.models import (
    AutoARIMAModel,
    ETSModelAdapter,
    ForecastResult,
    ProphetModel,
    ets_kwargs,
    is_success,
    run_model,
)
from pkg.ts_v2.models.prophet import build_prophet_future


def _train(n: int = 30, start: int = 140201) -> pd.Series:
    values = [10.0 + (i % 12) for i in range(n)]
    idx = [shamsi_add_months(start, i) for i in range(n)]
    return pd.Series(values, index=idx, name="sales")


def _window_15():
    return make_forecast_window(140501, config=TSForecastConfig(forecast_horizon=15))


class TestUseSeasonal(unittest.TestCase):
    def test_v1_strict_greater_than_24(self):
        self.assertFalse(use_seasonal(24))
        self.assertTrue(use_seasonal(25))
        cfg = TSForecastConfig(seasonal_enable_after_months=24)
        self.assertFalse(use_seasonal(24, cfg))
        self.assertTrue(use_seasonal(25, cfg))


class TestEtsKwargsShared(unittest.TestCase):
    def test_seasonal_add_when_long(self):
        kw = ets_kwargs(25)
        self.assertEqual(kw["seasonal"], "add")
        self.assertEqual(kw["seasonal_periods"], 12)
        self.assertEqual(kw["error"], "add")
        self.assertEqual(kw["trend"], "add")

    def test_no_seasonal_when_short(self):
        kw = ets_kwargs(24)
        self.assertNotIn("seasonal", kw)
        self.assertEqual(kw, {"error": "add", "trend": "add"})


class TestAutoARIMAAdapter(unittest.TestCase):
    def test_exactly_15_predictions_and_target_dates(self):
        window = _window_15()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.arange(15, dtype=float)

        with patch("pkg.ts_v2.models.auto_arima._fit_auto_arima", return_value=mock_model):
            outcome = run_model(AutoARIMAModel(), _train(30), window)

        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        self.assertEqual(outcome.target_dates[0], 140501)
        self.assertEqual(outcome.target_dates[-1], 140603)
        mock_model.predict.assert_called_once_with(n_periods=15)

    def test_never_requests_sixteen_periods(self):
        window = _window_15()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones(15)

        with patch("pkg.ts_v2.models.auto_arima._fit_auto_arima", return_value=mock_model):
            run_model(AutoARIMAModel(), _train(30), window)

        args, kwargs = mock_model.predict.call_args
        n_periods = kwargs.get("n_periods", args[0] if args else None)
        self.assertEqual(n_periods, 15)
        self.assertNotEqual(n_periods, 16)


class TestETSAdapter(unittest.TestCase):
    def test_exactly_15_predictions_and_target_dates(self):
        window = _window_15()
        mock_result = MagicMock()
        mock_result.forecast.return_value = pd.Series(np.arange(15, dtype=float))
        mock_ets = MagicMock()
        mock_ets.return_value.fit.return_value = mock_result

        with patch(
            "statsmodels.tsa.exponential_smoothing.ets.ETSModel",
            mock_ets,
        ):
            outcome = run_model(ETSModelAdapter(), _train(30), window)

        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        mock_result.forecast.assert_called_once_with(steps=15)

        # Seasonal long history: seasonal="add" must be passed to ETSModel.
        call_kwargs = mock_ets.call_args.kwargs
        self.assertEqual(call_kwargs.get("seasonal"), "add")
        self.assertEqual(call_kwargs.get("seasonal_periods"), 12)

    def test_short_history_omits_seasonal(self):
        window = _window_15()
        mock_result = MagicMock()
        mock_result.forecast.return_value = pd.Series(np.ones(15))
        mock_ets = MagicMock()
        mock_ets.return_value.fit.return_value = mock_result

        with patch(
            "statsmodels.tsa.exponential_smoothing.ets.ETSModel",
            mock_ets,
        ):
            run_model(ETSModelAdapter(), _train(20), window)

        call_kwargs = mock_ets.call_args.kwargs
        self.assertNotIn("seasonal", call_kwargs)
        mock_result.forecast.assert_called_once_with(steps=15)


class TestProphetAdapter(unittest.TestCase):
    def test_exactly_15_predictions_and_target_dates(self):
        window = _window_15()
        train = _train(30)
        target_ts = shamsi_months_to_ms_index(window.target_dates)
        train_ds = shamsi_months_to_ms_index([int(x) for x in train.index])

        mock_model = MagicMock()

        def _fit(frame):
            return mock_model

        def _predict(future):
            # Return yhat for every future ds; adapter selects target rows.
            return pd.DataFrame(
                {
                    "ds": future["ds"].values,
                    "yhat": np.linspace(1.0, float(len(future)), len(future)),
                }
            )

        mock_model.fit = MagicMock(side_effect=lambda frame: mock_model)
        mock_model.predict = MagicMock(side_effect=_predict)

        mock_prophet_cls = MagicMock(return_value=mock_model)

        with patch("prophet.Prophet", mock_prophet_cls):
            # ProphetModel imports Prophet inside fit().
            outcome = run_model(ProphetModel(), train, window)

        self.assertTrue(is_success(outcome))
        assert isinstance(outcome, ForecastResult)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        self.assertEqual(outcome.target_dates[0], 140501)
        self.assertEqual(outcome.target_dates[-1], 140603)

        future = mock_model.predict.call_args.args[0]
        last_target = shamsi_to_month_start_timestamp(140603)
        self.assertLessEqual(pd.Timestamp(future["ds"].max()), last_target)
        self.assertEqual(pd.Timestamp(future["ds"].max()), last_target)
        # Regression: must not pad like V1 (history + 16 beyond targets).
        n_hist = len(train_ds)
        self.assertLessEqual(len(future), n_hist + 15)
        # Prophet constructed with linear growth and fixed CPS.
        ctor_kwargs = mock_prophet_cls.call_args.kwargs
        self.assertEqual(ctor_kwargs.get("growth"), "linear")
        self.assertEqual(ctor_kwargs.get("changepoint_prior_scale"), 0.05)
        self.assertTrue(ctor_kwargs.get("yearly_seasonality"))  # n=30 > 24

    def test_future_never_extends_beyond_target_window(self):
        train_ds = shamsi_months_to_ms_index(
            [shamsi_add_months(140301, i) for i in range(12)]
        )
        targets = (140501, 140502, 140503)
        future = build_prophet_future(train_ds, targets)
        last = shamsi_to_month_start_timestamp(140503)
        self.assertEqual(pd.Timestamp(future["ds"].max()), last)
        self.assertFalse((future["ds"] > last).any())
        # Contrast with illegal V1-style padding past the window.
        illegal_max = last + pd.DateOffset(months=16)
        self.assertLess(pd.Timestamp(future["ds"].max()), illegal_max)

    def test_short_history_disables_yearly_seasonality(self):
        window = _window_15()
        train = _train(20)
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value=mock_model)

        def _predict(future):
            return pd.DataFrame(
                {
                    "ds": future["ds"].values,
                    "yhat": np.ones(len(future)),
                }
            )

        mock_model.predict = MagicMock(side_effect=_predict)
        mock_prophet_cls = MagicMock(return_value=mock_model)

        with patch("prophet.Prophet", mock_prophet_cls):
            outcome = run_model(ProphetModel(), train, window)

        self.assertTrue(is_success(outcome))
        ctor_kwargs = mock_prophet_cls.call_args.kwargs
        self.assertFalse(ctor_kwargs.get("yearly_seasonality"))


if __name__ == "__main__":
    unittest.main()
