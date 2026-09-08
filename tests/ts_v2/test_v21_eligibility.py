"""V2.1 intermittent eligibility and persistence tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.backtest import backtest_product
from pkg.ts_v2.config import DEFAULT_CONFIG, DEFAULT_CONFIG_V21, TSForecastConfig
from pkg.ts_v2.eligibility import (
    build_candidate_eligibility,
    evaluate_intermittent_demand,
)
from pkg.ts_v2.engine import forecast_with_backtest
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.intermittent import CrostonSBAModel, TSBModel
from pkg.ts_v2.models.registry import REGISTRY
from pkg.ts_v2.persistence import (
    CANDIDATE_ELIGIBILITY_CSV_NAME,
    FORECAST_CSV_NAME,
    METADATA_JSON_NAME,
    SELECTION_CSV_NAME,
    TS_VERSION_V21,
    persist_completed_v21_run,
)
from pkg.ts_v2.selection import select_product_model


class _ConstantModel(BaseForecastModel):
    def __init__(self, name: str, value: float) -> None:
        self.name = str(name)
        self._value = float(value)

    def fit(self, train_series: pd.Series) -> "_ConstantModel":
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(self._value for _ in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


def _restore_intermittent_builtins() -> None:
    register_model("croston_sba", CrostonSBAModel, replace=True)
    register_model("tsb", TSBModel, replace=True)


def _sales_smooth(product: str = "RecigenLike", start: int = 140201, n: int = 36) -> pd.DataFrame:
    """Positive demand every month → zero_fraction=0, ADI=1."""
    rows = []
    cur = start
    for i in range(n):
        rows.append({"product": product, "date": cur, "sales": 10.0 + (i % 5)})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _sales_intermittent(product: str = "Sparse", start: int = 140201, n: int = 36) -> pd.DataFrame:
    """High zero share and ADI > 1.32."""
    rows = []
    cur = start
    for i in range(n):
        sales = 20.0 if i % 4 == 0 else 0.0
        rows.append({"product": product, "date": cur, "sales": sales})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestIntermittentEligibilityRules(unittest.TestCase):
    def test_recigen_like_ineligible(self):
        eligible, reason = evaluate_intermittent_demand(
            zero_month_proportion=0.0,
            adi=1.0,
            min_zero_fraction=0.10,
            min_adi=1.32,
        )
        self.assertFalse(eligible)
        self.assertIn("not_intermittent", reason)
        self.assertIn("0.10", reason)
        self.assertIn("1.32", reason)
        self.assertIn("0.00", reason)
        self.assertIn("1.00", reason)

    def test_high_zero_or_adi_eligible(self):
        ok_z, _ = evaluate_intermittent_demand(
            zero_month_proportion=0.25,
            adi=1.0,
            min_zero_fraction=0.10,
            min_adi=1.32,
        )
        self.assertTrue(ok_z)
        ok_adi, _ = evaluate_intermittent_demand(
            zero_month_proportion=0.0,
            adi=2.0,
            min_zero_fraction=0.10,
            min_adi=1.32,
        )
        self.assertTrue(ok_adi)

    def test_v2_default_config_has_empty_gate(self):
        self.assertEqual(DEFAULT_CONFIG.intermittent_model_names, ())
        sales = _sales_smooth()
        values = sales.set_index("date")["sales"]
        elig = build_candidate_eligibility(values, DEFAULT_CONFIG)
        self.assertTrue(elig["croston_sba"].eligible)
        self.assertTrue(elig["tsb"].eligible)
        self.assertEqual(elig["croston_sba"].reason, "")

    def test_v21_defaults_gate_croston_tsb(self):
        self.assertEqual(
            DEFAULT_CONFIG_V21.intermittent_model_names,
            ("croston_sba", "tsb"),
        )
        sales = _sales_smooth()
        values = sales.set_index("date")["sales"]
        elig = build_candidate_eligibility(values, DEFAULT_CONFIG_V21)
        self.assertFalse(elig["croston_sba"].eligible)
        self.assertFalse(elig["tsb"].eligible)
        self.assertTrue(elig["naive"].eligible)


class TestV21EngineSelectionGate(unittest.TestCase):
    def setUp(self) -> None:
        register_model("gate_naive", lambda: _ConstantModel("gate_naive", 10.0), replace=True)
        register_model("croston_sba", lambda: _ConstantModel("croston_sba", 999.0), replace=True)
        register_model("tsb", lambda: _ConstantModel("tsb", 998.0), replace=True)
        self.cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            candidate_models=("gate_naive", "croston_sba", "tsb"),
            selection_simplicity_order=("gate_naive", "croston_sba", "tsb"),
            intermittent_model_names=("croston_sba", "tsb"),
            intermittent_min_zero_fraction=0.10,
            intermittent_min_adi=1.32,
        )

    def tearDown(self) -> None:
        REGISTRY.unregister("gate_naive")
        _restore_intermittent_builtins()

    def test_smooth_series_excludes_intermittent_from_scores(self):
        sales = _sales_smooth("SKU")
        models = [
            _ConstantModel("gate_naive", 10.0),
            _ConstantModel("croston_sba", 999.0),
            _ConstantModel("tsb", 998.0),
        ]
        bt = backtest_product(sales, "SKU", models, config=self.cfg)
        self.assertFalse(bt.failures.empty)
        gated = bt.failures[bt.failures["model"].isin(["croston_sba", "tsb"])]
        self.assertFalse(gated.empty)
        self.assertTrue((gated["error_type"] == "NotIntermittent").all())
        self.assertTrue(gated["reason"].str.contains("not_intermittent").all())

        elig = build_candidate_eligibility(
            sales.set_index("date")["sales"], self.cfg
        )
        sel = select_product_model(
            bt, "SKU", config=self.cfg, candidate_eligibility=elig
        )
        self.assertNotIn("croston_sba", sel.candidate_scores)
        self.assertNotIn("tsb", sel.candidate_scores)
        self.assertIn("croston_sba", sel.unavailable)
        self.assertIn("tsb", sel.unavailable)
        self.assertEqual(sel.selected_model, "gate_naive")
        self.assertEqual(sel.fallback_reason, "intermittent_models_excluded")

    def test_intermittent_series_allows_croston_tsb(self):
        sales = _sales_intermittent("SKU")
        values = sales.set_index("date")["sales"]
        elig = build_candidate_eligibility(values, self.cfg)
        self.assertTrue(elig["croston_sba"].eligible)
        self.assertTrue(elig["tsb"].eligible)

        models = [
            _ConstantModel("gate_naive", 10.0),
            _ConstantModel("croston_sba", 999.0),
            _ConstantModel("tsb", 998.0),
        ]
        bt = backtest_product(sales, "SKU", models, config=self.cfg)
        if not bt.failures.empty:
            gated = bt.failures[bt.failures["error_type"] == "NotIntermittent"]
            self.assertTrue(gated.empty)
        self.assertFalse(bt.predictions.empty)
        self.assertTrue(bt.predictions["model"].isin(["croston_sba", "tsb"]).any())


class TestV21Persistence(unittest.TestCase):
    def setUp(self) -> None:
        register_model(
            "persist_naive", lambda: _ConstantModel("persist_naive", 12.0), replace=True
        )
        register_model(
            "croston_sba", lambda: _ConstantModel("croston_sba", 100.0), replace=True
        )
        register_model("tsb", lambda: _ConstantModel("tsb", 100.0), replace=True)
        self.cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
            candidate_models=("persist_naive", "croston_sba", "tsb"),
            selection_simplicity_order=("persist_naive", "croston_sba", "tsb"),
            intermittent_model_names=("croston_sba", "tsb"),
            intermittent_min_zero_fraction=0.10,
            intermittent_min_adi=1.32,
        )

    def tearDown(self) -> None:
        REGISTRY.unregister("persist_naive")
        _restore_intermittent_builtins()

    def test_persist_writes_eligibility_selection_raw_final(self):
        sales = _sales_smooth("SKU")
        origin = 140501
        result = forecast_with_backtest(sales, ["SKU"], origin, config=self.cfg)
        self.assertIn("SKU", result.selections)
        sel = result.selections["SKU"]
        self.assertEqual(sel.fallback_reason, "intermittent_models_excluded")
        self.assertFalse(sel.candidate_eligibility["croston_sba"].eligible)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = persist_completed_v21_run(
                result,
                origin,
                config=self.cfg,
                base_dir=Path(tmp),
                run_id="test_v21_run",
            )
            self.assertTrue((run_dir / FORECAST_CSV_NAME).is_file())
            self.assertTrue((run_dir / CANDIDATE_ELIGIBILITY_CSV_NAME).is_file())
            self.assertTrue((run_dir / SELECTION_CSV_NAME).is_file())

            meta = json.loads((run_dir / METADATA_JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(meta["ts_version"], TS_VERSION_V21)

            forecast = pd.read_csv(run_dir / FORECAST_CSV_NAME)
            self.assertIn("raw_forecast", forecast.columns)
            self.assertIn("forecast", forecast.columns)

            elig_df = pd.read_csv(run_dir / CANDIDATE_ELIGIBILITY_CSV_NAME)
            croston = elig_df[elig_df["model"] == "croston_sba"].iloc[0]
            self.assertFalse(bool(croston["eligible"]))
            self.assertIn("not_intermittent", str(croston["reason"]))

            sel_df = pd.read_csv(run_dir / SELECTION_CSV_NAME)
            self.assertEqual(
                sel_df.iloc[0]["fallback_reason"], "intermittent_models_excluded"
            )
            self.assertEqual(sel_df.iloc[0]["selected_model"], "persist_naive")


if __name__ == "__main__":
    unittest.main()
