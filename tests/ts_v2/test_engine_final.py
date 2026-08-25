"""Final production refit after backtest selection (fresh models only)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.backtest import backtest_product
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window, parse_origin
from pkg.ts_v2.engine import (
    assert_final_forecast_contract,
    forecast_with_backtest,
    refit_and_forecast_product,
)
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.registry import REGISTRY, models_from_config
from pkg.ts_v2.selection import select_product_model


class InstanceTrackingModel(BaseForecastModel):
    """Records fit length and instance id for reuse tests."""

    name = "track_model"
    instances: list["InstanceTrackingModel"] = []

    def __init__(self, value: float = 1.0) -> None:
        self.instance_id = id(self)
        self.value = float(value)
        self.fit_len: int | None = None
        InstanceTrackingModel.instances.append(self)

    def fit(self, train_series: pd.Series) -> "InstanceTrackingModel":
        clean = train_series.dropna()
        self.fit_len = int(len(clean))
        if not clean.empty:
            self.value = float(clean.iloc[-1])
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self.value for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
            metadata={"instance_id": self.instance_id, "fit_len": self.fit_len},
        )


class FixedModel(BaseForecastModel):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = float(value)

    def fit(self, train_series: pd.Series) -> "FixedModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._value for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


def _sales(product: str, start: int, n: int, step: float = 3.0) -> pd.DataFrame:
    from pkg.benchmark.calendar import shamsi_add_months

    rows = []
    cur = start
    for i in range(n):
        rows.append({"product": product, "date": cur, "sales": 40.0 + step * i})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestFinalForecast(unittest.TestCase):
    def setUp(self) -> None:
        InstanceTrackingModel.instances.clear()
        register_model("track_model", lambda: InstanceTrackingModel(), replace=True)
        register_model("fix_a", lambda: FixedModel("fix_a", 10.0), replace=True)
        register_model("fix_b", lambda: FixedModel("fix_b", 20.0), replace=True)
        register_model("fix_c", lambda: FixedModel("fix_c", 30.0), replace=True)
        self.cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            selection_strategy="best_model",
            candidate_models=("track_model",),
        )
        self.sales = _sales("SKU-F", 140401, 30)
        from pkg.benchmark.calendar import shamsi_add_months
        from pkg.ts_v2.data import product_monthly_sales

        last_obs = int(product_monthly_sales(self.sales, "SKU-F").index.max())
        self.origin = parse_origin(shamsi_add_months(last_obs, 1))

    def tearDown(self) -> None:
        for name in ("track_model", "fix_a", "fix_b", "fix_c"):
            REGISTRY.unregister(name)

    def test_final_fit_uses_more_history_than_cv_folds(self):
        cv_models = models_from_config(self.cfg.candidate_models)
        cv_ids = [id(m) for m in cv_models]
        bt = backtest_product(self.sales, "SKU-F", cv_models, config=self.cfg)
        del cv_models

        selection = select_product_model(bt, "SKU-F", config=self.cfg)
        final = refit_and_forecast_product(
            self.sales,
            "SKU-F",
            self.origin,
            selection,
            backtest=bt,
            config=self.cfg,
        )

        cv_train_lengths = []
        for origin in bt.predictions["origin"].unique():
            prep = prepare_monthly_series(
                self.sales, "SKU-F", int(origin), config=self.cfg
            )
            cv_train_lengths.append(prep.n_observations)

        self.assertGreater(
            final.n_training_observations,
            min(cv_train_lengths),
        )
        self.assertGreater(
            final.n_training_observations,
            max(cv_train_lengths),
        )
        refit_ids = final.metadata["refit_model_ids"]
        self.assertTrue(all(i not in cv_ids for i in refit_ids))

    def test_forecast_contract_assertions(self):
        cv_models = models_from_config(self.cfg.candidate_models)
        bt = backtest_product(self.sales, "SKU-F", cv_models, config=self.cfg)
        selection = select_product_model(bt, "SKU-F", config=self.cfg)
        final = refit_and_forecast_product(
            self.sales,
            "SKU-F",
            self.origin,
            selection,
            config=self.cfg,
        )
        prepared = prepare_monthly_series(
            self.sales, "SKU-F", self.origin, config=self.cfg
        )
        window = make_forecast_window(self.origin, config=self.cfg)
        assert_final_forecast_contract(prepared, window, final.forecast, config=self.cfg)
        self.assertEqual(len(final.horizon_forecasts), 15)
        self.assertEqual(
            tuple(h.target_shamsi_yyyymm for h in final.horizon_forecasts),
            window.target_dates,
        )
        self.assertLess(
            int(final.training_end),
            int(final.forecast_origin),
        )

    def test_cv_models_not_reused_in_full_pipeline(self):
        result = forecast_with_backtest(
            self.sales,
            ["SKU-F"],
            self.origin,
            config=self.cfg,
        )
        cv_ids = set(result.extras["cv_model_ids"])
        final = result.final_forecasts["SKU-F"]
        for mid in final.metadata["refit_model_ids"]:
            self.assertNotIn(mid, cv_ids)

    def test_metadata_includes_cv_score_and_coverage(self):
        result = forecast_with_backtest(
            self.sales,
            ["SKU-F"],
            self.origin,
            config=self.cfg,
        )
        final = result.final_forecasts["SKU-F"]
        self.assertEqual(final.selected_strategy, "best_model")
        self.assertEqual(final.selected_model, "track_model")
        self.assertTrue(math.isfinite(final.cv_score))
        self.assertIn("number_of_origins", final.cv_coverage)
        self.assertIn("evaluated_horizons", final.cv_coverage)
        self.assertIsNotNone(final.training_start)
        self.assertIsNotNone(final.training_end)

    def test_ensemble_refits_each_constituent_fresh(self):
        cfg = TSForecastConfig(
            forecast_horizon=5,
            min_train_months=6,
            activity_start_min_sales=None,
            selection_strategy="top3_mean",
            ensemble_top_k=3,
            candidate_models=("fix_a", "fix_b", "fix_c"),
        )
        sales = _sales("ENS", 140401, 14)
        origin = parse_origin(140412)
        result = forecast_with_backtest(sales, ["ENS"], origin, config=cfg)
        final = result.final_forecasts["ENS"]
        self.assertEqual(final.selected_strategy, "top3_mean")
        self.assertEqual(set(final.constituent_models), {"fix_a", "fix_b", "fix_c"})
        self.assertEqual(len(final.metadata["refit_model_ids"]), 3)
        self.assertEqual(len(final.forecast.predictions), 5)
        self.assertAlmostEqual(float(final.raw_forecast.predictions[0]), 20.0)
        self.assertAlmostEqual(float(final.constrained_forecast.predictions[0]), 20.0)


if __name__ == "__main__":
    unittest.main()
