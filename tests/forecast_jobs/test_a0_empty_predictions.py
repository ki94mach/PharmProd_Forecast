"""Regression tests for A0 empty-prediction telemetry and job mapping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.backfill_runner.types import EngineJobRequest
from pkg.benchmark.calendar import shamsi_add_months
from pkg.forecast_jobs.engines import (
    REASON_NEURAL_EMPTY_PREDICTIONS,
    REASON_NEURAL_SHORT_HISTORY,
    run_neural_job,
    target_dates_for_origin,
)
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    backtest_product_architectures,
)
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.types import TrainMetadata, build_train_metadata


def _monthly_sales_frame(
    product: str,
    start_ym: int,
    n_months: int,
    *,
    base: float = 100.0,
    step: float = 1.0,
) -> pd.DataFrame:
    rows = []
    cur = start_ym
    for i in range(n_months):
        rows.append(
            {"product": product, "date": cur, "sales": base + step * i}
        )
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _stub_metadata(*, architecture: str, seed: int, origin: int) -> TrainMetadata:
    train_end = shamsi_add_months(origin, -2)
    return build_train_metadata(
        architecture=architecture,
        parameters={"lookback": 12, "hidden_units": 32, "horizon": 15},
        random_seed=seed,
        n_train_samples=8,
        n_validation_samples=2,
        forecast_origin=origin,
        training_start=shamsi_add_months(origin, -14),
        training_end=train_end,
        validation_start=shamsi_add_months(origin, -4),
        validation_end=train_end,
        epochs_ran=1,
        best_epoch=1,
        best_val_loss=0.1,
        parameter_count=100,
        train_window_count=8,
        validation_window_count=2,
        scaler_params={"mean": 0.0, "scale": 1.0},
    )


class _OkStub:
    instances: list["_OkStub"] = []
    last_seeds: list[int] = []

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        self.config = config or NeuralExperimentConfig(
            architecture_name=ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
            max_epochs=1,
        )
        self.metadata_: Optional[TrainMetadata] = None
        _OkStub.instances.append(self)

    def fit(self, train_series, window, *, seed):
        _OkStub.last_seeds.append(int(seed))
        self.metadata_ = _stub_metadata(
            architecture=self.config.architecture_name,
            seed=int(seed),
            origin=int(window.forecast_origin),
        )
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        return ForecastResult(
            model_name="ok_stub",
            predictions=tuple(float(10 + int(h)) for h in window.horizons),
            target_dates=tuple(int(t) for t in window.target_dates),
            horizons=tuple(int(h) for h in window.horizons),
        )


class _BoomStub(_OkStub):
    def fit(self, train_series, window, *, seed):
        raise RuntimeError("simulated training boom")


def _patch_factory(cls):
    def _factory(architecture, config=None):
        return cls(config)

    return _factory


class TestSilentEmptyTelemetry(unittest.TestCase):
    def test_short_history_emits_unavailable_metadata_not_bare_empty(self):
        # Too short for min_train_months=12 → discover/prepare skip path.
        sales = _monthly_sales_frame("S", 140301, 8)
        v2 = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        neural = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            lookback=3,
            max_epochs=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(_OkStub),
        ):
            result = backtest_product_architectures(
                sales,
                "S",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=neural,
                v2_config=v2,
                explicit_origins=(140401,),
            )
        self.assertTrue(result.predictions.empty)
        self.assertFalse(result.fold_metadata.empty)
        self.assertEqual(result.fold_metadata.iloc[0]["status"], STATUS_UNAVAILABLE)
        self.assertEqual(
            result.fold_metadata.iloc[0]["unavailable_reason"],
            "insufficient_history",
        )

    def test_long_history_stub_delivers_15_horizons(self):
        sales = _monthly_sales_frame("L", 139501, 80)
        origin = 140401
        v2 = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        neural = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
            lookback=12,
            max_epochs=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        _OkStub.instances.clear()
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(_OkStub),
        ):
            result = backtest_product_architectures(
                sales,
                "L",
                architectures=(ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,),
                seeds=(41,),
                config=neural,
                v2_config=v2,
                explicit_origins=(origin,),
            )
        self.assertEqual(len(result.predictions), 15)
        self.assertEqual(int(result.predictions["horizon"].min()), 1)
        self.assertEqual(int(result.predictions["horizon"].max()), 15)
        self.assertEqual(result.fold_metadata.iloc[0]["status"], STATUS_OK)
        # Production cutoff: training_end must be origin-2 (via window contract).
        window_train_end = shamsi_add_months(origin, -2)
        # Stub metadata uses origin-2 as training_end.
        self.assertEqual(
            int(result.fold_metadata.iloc[0]["training_end"]),
            window_train_end,
        )


class TestNeuralJobMapping(unittest.TestCase):
    def _request(self, sales: pd.DataFrame, product: str, origin: int) -> EngineJobRequest:
        targets = target_dates_for_origin(origin, 15)
        last_complete = shamsi_add_months(origin, -2)
        train = sales.loc[
            (sales["product"] == product) & (sales["date"] <= last_complete)
        ].copy()
        return EngineJobRequest(
            engine="a0",
            product=product,
            quarter="1404Q1",
            forecast_origin=origin,
            horizon=15,
            target_dates=targets,
            training_sales=train,
            meta={
                "fallback_policy": "naive_last_value",
                "fallback_version": "v1",
            },
        )

    def test_short_history_maps_to_neural_short_history(self):
        sales = _monthly_sales_frame("S", 140301, 8)
        req = self._request(sales, "S", 140401)
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(_OkStub),
        ):
            # Force backtest path with short series via run_neural_job.
            with patch(
                "pkg.forecast_jobs.engines.backtest_product_architectures",
                wraps=None,
            ) as mocked:
                from pkg.ts_v3a.backtest import (
                    NeuralOuterBacktestResult,
                    _empty_predictions,
                )

                meta = pd.DataFrame(
                    [
                        {
                            "status": STATUS_UNAVAILABLE,
                            "unavailable_reason": "insufficient_history",
                        }
                    ]
                )
                mocked.return_value = NeuralOuterBacktestResult(
                    predictions=_empty_predictions(),
                    fold_metadata=meta,
                    metrics=pd.DataFrame(),
                    ensemble_predictions=_empty_predictions(),
                    ensemble_origin_status=pd.DataFrame(),
                    seed_metrics=pd.DataFrame(),
                    ensemble_metrics=pd.DataFrame(),
                    stability=pd.DataFrame(),
                )
                result = run_neural_job(
                    req, architecture="a0", seed=41, full_sales=sales
                )
        self.assertTrue(result.ineligible)
        self.assertEqual(
            result.extras.get("ineligibility_reason"),
            REASON_NEURAL_SHORT_HISTORY,
        )
        self.assertEqual(
            result.extras.get("unavailable_reason"), "insufficient_history"
        )

    def test_runtime_error_fold_is_failed_not_ineligible(self):
        sales = _monthly_sales_frame("E", 139501, 80)
        req = self._request(sales, "E", 140401)
        from pkg.ts_v3a.backtest import NeuralOuterBacktestResult, _empty_predictions

        meta = pd.DataFrame(
            [
                {
                    "status": STATUS_ERROR,
                    "unavailable_reason": None,
                    "error_type": "RuntimeError",
                    "error_message": "simulated training boom",
                }
            ]
        )
        with patch(
            "pkg.forecast_jobs.engines.backtest_product_architectures",
            return_value=NeuralOuterBacktestResult(
                predictions=_empty_predictions(),
                fold_metadata=meta,
                metrics=pd.DataFrame(),
                ensemble_predictions=_empty_predictions(),
                ensemble_origin_status=pd.DataFrame(),
                seed_metrics=pd.DataFrame(),
                ensemble_metrics=pd.DataFrame(),
                stability=pd.DataFrame(),
            ),
        ):
            result = run_neural_job(
                req, architecture="a0", seed=41, full_sales=sales
            )
        self.assertFalse(result.success)
        self.assertFalse(bool(getattr(result, "ineligible", False)))
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertNotEqual(
            (result.extras or {}).get("ineligibility_reason"),
            REASON_NEURAL_EMPTY_PREDICTIONS,
        )

    def test_three_seeds_deterministic_with_stub(self):
        sales = _monthly_sales_frame("D", 139501, 80)
        origin = 140401
        v2 = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        neural = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
            lookback=12,
            max_epochs=1,
            random_seeds=(41, 42, 43),
            min_successful_seeds=1,
        )
        _OkStub.last_seeds = []
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(_OkStub),
        ):
            result = backtest_product_architectures(
                sales,
                "D",
                architectures=(ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,),
                seeds=(41, 42, 43),
                config=neural,
                v2_config=v2,
                explicit_origins=(origin,),
            )
        self.assertEqual(sorted(_OkStub.last_seeds), [41, 42, 43])
        seeds = sorted(int(s) for s in result.predictions["seed"].unique())
        self.assertEqual(seeds, [41, 42, 43])
        # Same stub predict → identical forecast vector per seed.
        piv = result.predictions.pivot_table(
            index="horizon", columns="seed", values="prediction"
        )
        self.assertTrue((piv[41] == piv[42]).all())
        self.assertTrue((piv[42] == piv[43]).all())
        targets = target_dates_for_origin(origin, 15)
        got = tuple(
            sorted(
                int(t)
                for t in result.predictions.loc[
                    result.predictions["seed"] == 41, "target_date"
                ]
            )
        )
        self.assertEqual(got, tuple(sorted(targets)))


if __name__ == "__main__":
    unittest.main()
