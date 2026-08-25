"""Immutable V2 forecast-run persistence tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.dates import make_forecast_window, parse_origin
from pkg.ts_v2.engine import forecast_with_backtest
from pkg.ts_v2.models import BaseForecastModel, ForecastResult, register_model
from pkg.ts_v2.models.registry import REGISTRY
from pkg.ts_v2.persistence import (
    COMPLETE_MARKER,
    RunCheckpointError,
    RunImmutableError,
    begin_v2_run,
    build_forecast_dataframe,
    complete_run_dir,
    config_hash,
    finalize_v2_run,
    is_complete_run,
    persist_completed_run,
    quarter_from_origin,
    validate_forecast_dataframe,
    write_checkpoint_artifacts,
)
from pkg.ts_v2.types import (
    ConstrainedForecastResult,
    EngineResult,
    ForecastResult as RawForecastResult,
    HorizonForecast,
    ProductFinalForecast,
)


class TrackModel(BaseForecastModel):
    name = "track_model"

    def fit(self, train_series):
        return self

    def predict(self, horizon, target_dates):
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(100.0 + i for i in range(horizon)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


def _sales(product, start, n):
    from pkg.benchmark.calendar import shamsi_add_months

    rows = []
    cur = start
    for i in range(n):
        rows.append({"product": product, "date": cur, "sales": 50.0 + i})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _engine_result(origin_ym: int, products=("A", "B")) -> EngineResult:
    origin = parse_origin(origin_ym)
    win = make_forecast_window(origin, config=TSForecastConfig(forecast_horizon=15))
    finals = {}
    for product in products:
        hfs = []
        for h, target in zip(win.horizons, win.target_dates):
            val = float(h)
            hfs.append(
                HorizonForecast(
                    product=product,
                    origin=origin,
                    horizon=int(h),
                    target_shamsi_yyyymm=int(target),
                    raw_forecast=val,
                    constrained_forecast=val,
                    model_name="track_model",
                )
            )
        raw = RawForecastResult(
            model_name="track_model",
            predictions=tuple(float(i) for i in range(1, 16)),
            target_dates=win.target_dates,
            horizons=win.horizons,
        )
        constrained = ConstrainedForecastResult(
            model_name="track_model",
            raw_predictions=raw.predictions,
            constrained_predictions=raw.predictions,
            target_dates=win.target_dates,
            horizons=win.horizons,
        )
        finals[product] = ProductFinalForecast(
            product=product,
            forecast_origin=int(origin_ym),
            selected_strategy="best_model",
            selected_model="track_model",
            constituent_models=("track_model",),
            training_start=140401,
            training_end=int(win.training_end),
            n_training_observations=20,
            cv_score=1.0,
            cv_coverage={"number_of_origins": 3},
            horizon_forecasts=tuple(hfs),
            raw_forecast=raw,
            constrained_forecast=constrained,
        )
    return EngineResult(final_forecasts=finals)


class TestQuarterAndMetadata(unittest.TestCase):
    def test_quarter_from_origin(self):
        self.assertEqual(quarter_from_origin(140501), "1405Q1")
        self.assertEqual(quarter_from_origin(140510), "1405Q4")

    def test_config_hash_stable(self):
        cfg = TSForecastConfig(forecast_horizon=15)
        self.assertEqual(config_hash(cfg), config_hash(cfg))
        self.assertNotEqual(
            config_hash(cfg),
            config_hash(TSForecastConfig(forecast_horizon=12)),
        )


class TestForecastDataframe(unittest.TestCase):
    def test_one_row_per_sku_horizon_with_explicit_origin(self):
        engine = _engine_result(140501, products=("P1",))
        df = build_forecast_dataframe(engine, run_id="run-1")
        self.assertEqual(len(df), 15)
        self.assertEqual(set(df["forecast_origin"]), {140501})
        self.assertTrue((df["run_id"] == "run-1").all())
        validate_forecast_dataframe(df, expected_horizon=15)


class TestRunImmutability(unittest.TestCase):
    def test_completed_run_cannot_be_overwritten(self):
        base = Path(tempfile.mkdtemp())
        origin = 140501
        cfg = TSForecastConfig(forecast_horizon=15)
        engine = _engine_result(origin)
        run_id = "20250825T100000Z_deadbeef"
        created = datetime(2025, 8, 25, 10, 0, 0, tzinfo=timezone.utc)

        path = persist_completed_run(
            engine,
            origin,
            config=cfg,
            base_dir=base,
            run_id=run_id,
            product_titles={"A": "Product A", "B": "Product B"},
            created_at=created,
        )
        self.assertTrue(is_complete_run(path))
        self.assertTrue((path / "forecast.csv").is_file())
        self.assertTrue((path / "run_metadata.json").is_file())
        self.assertTrue((path / "backtest_scores.csv").is_file())
        self.assertTrue((path / COMPLETE_MARKER).is_file())

        with self.assertRaises(RunImmutableError):
            persist_completed_run(
                engine,
                origin,
                config=cfg,
                base_dir=base,
                run_id=run_id,
                created_at=created,
            )

        meta = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["ts_version"], "v2")
        self.assertEqual(meta["run_id"], run_id)
        self.assertEqual(meta["forecast_origin"], origin)
        self.assertEqual(meta["status"], "complete")
        self.assertIn("config_hash", meta)

        forecast = pd.read_csv(path / "forecast.csv")
        self.assertEqual(set(forecast["product_title"]), {"Product A", "Product B"})
        for _product, g in forecast.groupby("product"):
            self.assertEqual(len(g), 15)

    def test_checkpoint_requires_matching_config_hash(self):
        base = Path(tempfile.mkdtemp())
        origin = 140501
        cfg_a = TSForecastConfig(forecast_horizon=15)
        cfg_b = TSForecastConfig(forecast_horizon=12)
        run_id = "20250825T100000Z_cafebabe"
        checkpoint = begin_v2_run(
            origin,
            cfg_a,
            base_dir=base,
            run_id=run_id,
            created_at=datetime(2025, 8, 25, 10, 0, 0, tzinfo=timezone.utc),
        )
        with self.assertRaises(RunCheckpointError):
            begin_v2_run(
                origin,
                cfg_b,
                base_dir=base,
                run_id=run_id,
                resume=True,
            )
        resumed = begin_v2_run(
            origin,
            cfg_a,
            base_dir=base,
            run_id=run_id,
            resume=True,
        )
        self.assertEqual(resumed.config_hash, checkpoint.config_hash)

    def test_finalize_moves_incomplete_to_complete(self):
        base = Path(tempfile.mkdtemp())
        origin = 140501
        cfg = TSForecastConfig(forecast_horizon=15)
        engine = _engine_result(origin)
        run_id = "20250825T110000Z_abcd1234"
        checkpoint = begin_v2_run(
            origin,
            cfg,
            base_dir=base,
            run_id=run_id,
            created_at=datetime(2025, 8, 25, 11, 0, 0, tzinfo=timezone.utc),
        )
        inc_dir = checkpoint.run_dir
        self.assertTrue(inc_dir.is_dir())

        final_path = finalize_v2_run(checkpoint, engine, base_dir=base)
        self.assertFalse(inc_dir.exists())
        self.assertEqual(
            final_path,
            complete_run_dir(base, quarter_from_origin(origin), run_id),
        )
        self.assertTrue(is_complete_run(final_path))

        with self.assertRaises(RunImmutableError):
            write_checkpoint_artifacts(checkpoint, engine)


class TestPersistenceIntegration(unittest.TestCase):
    def tearDown(self) -> None:
        REGISTRY.unregister("track_model")

    def test_engine_pipeline_persists_fifteen_horizons_per_sku(self):
        register_model("track_model", TrackModel, replace=True)
        cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=6,
            activity_start_min_sales=None,
            candidate_models=("track_model",),
        )
        sales = pd.concat(
            [_sales("S1", 140401, 20), _sales("S2", 140401, 20)],
            ignore_index=True,
        )
        from pkg.benchmark.calendar import shamsi_add_months
        from pkg.ts_v2.data import product_monthly_sales

        last = int(product_monthly_sales(sales, "S1").index.max())
        origin = parse_origin(shamsi_add_months(last, 1))

        engine = forecast_with_backtest(sales, ["S1", "S2"], origin, config=cfg)

        base = Path(tempfile.mkdtemp())
        path = persist_completed_run(
            engine,
            origin.shamsi_yyyymm,
            config=cfg,
            base_dir=base,
            product_titles={"S1": "Sku One", "S2": "Sku Two"},
        )
        forecast = pd.read_csv(path / "forecast.csv")
        for product in ("S1", "S2"):
            g = forecast.loc[forecast["product"] == product]
            self.assertEqual(len(g), 15)
            self.assertEqual(set(g["forecast_origin"]), {origin.shamsi_yyyymm})
            self.assertEqual(g["horizon"].tolist(), list(range(1, 16)))


if __name__ == "__main__":
    unittest.main()
