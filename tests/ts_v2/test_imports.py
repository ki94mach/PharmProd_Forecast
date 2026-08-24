"""Import and config smoke tests for pkg.ts_v2 (no models, no DB)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestTsV2Imports(unittest.TestCase):
    def test_package_exports_config(self):
        from pkg.ts_v2 import DEFAULT_CONFIG, TSForecastConfig

        self.assertIsInstance(DEFAULT_CONFIG, TSForecastConfig)

    def test_submodules_import(self):
        import pkg.ts_v2.backtest as backtest
        import pkg.ts_v2.config as config
        import pkg.ts_v2.data as data
        import pkg.ts_v2.dates as dates
        import pkg.ts_v2.engine as engine
        import pkg.ts_v2.models as models
        import pkg.ts_v2.selection as selection
        import pkg.ts_v2.types as types

        self.assertTrue(hasattr(config, "TSForecastConfig"))
        self.assertTrue(hasattr(dates, "make_forecast_window"))
        self.assertTrue(hasattr(data, "filter_training_history"))
        self.assertTrue(hasattr(models, "available_models"))
        self.assertTrue(hasattr(backtest, "make_folds"))
        self.assertTrue(hasattr(selection, "select_best_model"))
        self.assertTrue(hasattr(engine, "forecast_products"))
        self.assertTrue(hasattr(types, "ForecastOrigin"))
        self.assertTrue(hasattr(types, "ForecastWindow"))

    def test_models_registry_empty(self):
        from pkg.ts_v2.models import available_models

        self.assertEqual(available_models(), ())


class TestTsV2Config(unittest.TestCase):
    def test_default_config_values(self):
        from pkg.ts_v2 import DEFAULT_CONFIG

        self.assertEqual(DEFAULT_CONFIG.forecast_horizon, 15)
        self.assertEqual(DEFAULT_CONFIG.selection_metric, "mae")
        self.assertEqual(DEFAULT_CONFIG.seasonal_period, 12)
        self.assertEqual(DEFAULT_CONFIG.min_train_months, 12)
        self.assertTrue(DEFAULT_CONFIG.nonnegative_forecasts)

    def test_config_is_frozen(self):
        from pkg.ts_v2 import DEFAULT_CONFIG

        with self.assertRaises(Exception):
            DEFAULT_CONFIG.forecast_horizon = 12  # type: ignore[misc]


class TestTsV2DatesAndSelection(unittest.TestCase):
    def test_parse_origin_and_target_month(self):
        from pkg.ts_v2.dates import make_forecast_window, parse_origin, target_month

        origin = parse_origin(140501)
        self.assertEqual(origin.shamsi_yyyymm, 140501)
        self.assertEqual(target_month(origin, 1), 140501)
        self.assertEqual(target_month(origin, 15), 140603)
        window = make_forecast_window(origin)
        self.assertEqual(window.target_dates[-1], 140603)

    def test_make_folds_uses_horizon(self):
        from pkg.ts_v2.backtest import make_folds
        from pkg.ts_v2.config import TSForecastConfig
        from pkg.ts_v2.dates import parse_origin

        cfg = TSForecastConfig(forecast_horizon=3)
        folds = make_folds([parse_origin(140501)], config=cfg)
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].train_end_exclusive, 140501)
        self.assertEqual(tuple(folds[0].horizons), (1, 2, 3))
        self.assertIsNotNone(folds[0].window)
        self.assertEqual(folds[0].window.target_dates, (140501, 140502, 140503))

    def test_select_best_model_prefers_lowest_score(self):
        from pkg.ts_v2.dates import parse_origin
        from pkg.ts_v2.selection import select_best_model

        result = select_best_model(
            {"naive": 2.0, "ets": 1.5, "arima": 1.8},
            product="Demo",
            origin=parse_origin(140501),
        )
        self.assertEqual(result.best_model_name, "ets")
        self.assertEqual(result.metric, "mae")


if __name__ == "__main__":
    unittest.main()
