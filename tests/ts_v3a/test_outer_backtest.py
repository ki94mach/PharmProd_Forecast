"""Outer expanding-CV engine: V2 contract, leakage, A0 tiers, metrics."""
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

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    backtest_product_architectures,
    run_outer_backtest,
)
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.eligibility import IneligibleForTrainingError, SampleEligibility
from pkg.ts_v3a.metrics import mean_horizon_mae, selection_mae_from_horizons
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import (
    resolve_legacy_architecture,
)
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.types import TargetMode, TrainMetadata, build_train_metadata
from pkg.ts_v3a.windows import InsufficientHistoryError


def _monthly_sales_frame(
    product: str,
    start_ym: int,
    n_months: int,
    *,
    base: float = 100.0,
    step: float = 1.0,
    product_id: Optional[Any] = None,
) -> pd.DataFrame:
    rows = []
    cur = start_ym
    for i in range(n_months):
        row = {
            "product": product,
            "date": cur,
            "sales": base + step * i,
        }
        if product_id is not None:
            row["product_id"] = product_id
        rows.append(row)
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _stub_metadata(
    *,
    architecture: str,
    seed: int,
    origin: int,
    lookback: int = 12,
    hidden: int = 32,
    second: int = 16,
    training_start: Optional[int] = None,
    training_end: Optional[int] = None,
    validation_start: Optional[int] = None,
    validation_end: Optional[int] = None,
) -> TrainMetadata:
    # All bounds must be strictly before the outer origin.
    train_end = training_end if training_end is not None else shamsi_add_months(origin, -1)
    val_end = validation_end if validation_end is not None else train_end
    train_start = training_start if training_start is not None else shamsi_add_months(origin, -12)
    val_start = validation_start if validation_start is not None else shamsi_add_months(origin, -3)
    return build_train_metadata(
        architecture=architecture,
        parameters={
            "lookback": lookback,
            "hidden_units": hidden,
            "second_hidden_units": second,
            "horizon": 15,
        },
        random_seed=seed,
        n_train_samples=8,
        n_validation_samples=2,
        forecast_origin=origin,
        training_start=train_start,
        training_end=train_end,
        validation_start=val_start,
        validation_end=val_end,
        epochs_ran=1,
        best_epoch=1,
        best_val_loss=0.1,
        parameter_count=100,
        train_window_count=8,
        validation_window_count=2,
        scaler_params={"mean": 0.0, "scale": 1.0},
    )


class RecordingStubModel:
    """Records fit inputs; emits constant raw-unit forecasts."""

    name = "recording_stub"
    last_train_series: Optional[pd.Series] = None
    last_window: Optional[ForecastWindow] = None
    last_seed: Optional[int] = None
    instances: list["RecordingStubModel"] = []
    predict_value: float = 7.5
    raise_on_fit: Optional[BaseException] = None

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        self.config = config or NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
        )
        self.metadata_: Optional[TrainMetadata] = None
        self._scaler: Optional[FoldScaler] = None
        RecordingStubModel.instances.append(self)

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "RecordingStubModel":
        if self.raise_on_fit is not None:
            raise self.raise_on_fit
        RecordingStubModel.last_train_series = train_series.copy()
        RecordingStubModel.last_window = window
        RecordingStubModel.last_seed = int(seed)
        # Simulate fold-local scaler fitted only on pre-origin history indices.
        hist = train_series.to_numpy(dtype=float)
        self._scaler = FoldScaler("standard").fit_on_series(hist)
        # Prove outer months are not in the series passed to fit.
        if len(train_series) and int(train_series.index.max()) >= int(window.forecast_origin):
            raise AssertionError("outer actuals leaked into training series")
        self.metadata_ = _stub_metadata(
            architecture=self.config.architecture_name,
            seed=seed,
            origin=int(window.forecast_origin),
            lookback=int(self.config.lookback),
            hidden=int(self.config.hidden_units),
            second=int(self.config.second_hidden_units),
        )
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        h = len(window.horizons)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(float(self.predict_value) for _ in range(h)),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(x) for x in window.horizons),
        )


class HorizonBiasStubModel(RecordingStubModel):
    """prediction = horizon so horizon MAE equals horizon when actual=0."""

    name = "horizon_bias_stub"

    def predict(self, window: ForecastWindow) -> ForecastResult:
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(float(h) for h in window.horizons),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(x) for x in window.horizons),
        )


def _patch_factory(model_cls: type) -> Any:
    def _factory(architecture, config=None):
        cfg = config or NeuralExperimentConfig(
            architecture_name=str(
                architecture.value if hasattr(architecture, "value") else architecture
            )
        )
        if cfg.architecture_name != (
            architecture.value if hasattr(architecture, "value") else architecture
        ):
            from pkg.ts_v3a.architectures import coerce_architecture_name

            cfg = cfg.with_architecture(coerce_architecture_name(architecture))
        return model_cls(cfg)

    return _factory


class TestOuterContract(unittest.TestCase):
    """Tests 1–3: training cutoff and h1..h15 target contract."""

    def setUp(self) -> None:
        RecordingStubModel.instances.clear()
        RecordingStubModel.last_train_series = None
        RecordingStubModel.last_window = None
        RecordingStubModel.raise_on_fit = None
        self.v2_cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        self.neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        # 12 train + 15 targets = 27 months from 140301 → origin 140401 has full H.
        self.sales = _monthly_sales_frame("SKU1", 140301, 40, product_id=101)
        self.origin = 140401

    def test_max_training_date_lt_outer_origin(self):
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            result = backtest_product_architectures(
                self.sales,
                "SKU1",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
            )
        self.assertFalse(result.predictions.empty)
        series = RecordingStubModel.last_train_series
        self.assertIsNotNone(series)
        assert series is not None
        self.assertLess(int(series.index.max()), self.origin)
        prepared = prepare_monthly_series(
            self.sales, "SKU1", self.origin, config=self.v2_cfg
        )
        self.assertLess(max(prepared.dates), self.origin)

    def test_target_h1_equals_outer_origin(self):
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            result = backtest_product_architectures(
                self.sales,
                "SKU1",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
            )
        window = RecordingStubModel.last_window
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(int(window.target_dates[0]), self.origin)
        self.assertEqual(int(window.horizons[0]), 1)
        h1 = result.predictions.loc[result.predictions["horizon"] == 1]
        self.assertFalse(h1.empty)
        self.assertTrue((h1["target_date"].astype(int) == self.origin).all())

    def test_targets_h1_to_h15_in_order(self):
        window = make_forecast_window(self.origin, config=self.v2_cfg)
        self.assertEqual(window.horizons, tuple(range(1, 16)))
        expected_dates = tuple(
            shamsi_add_months(self.origin, h - 1) for h in range(1, 16)
        )
        self.assertEqual(window.target_dates, expected_dates)
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            backtest_product_architectures(
                self.sales,
                "SKU1",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
            )
        got = RecordingStubModel.last_window
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.horizons, tuple(range(1, 16)))
        self.assertEqual(got.target_dates, expected_dates)


class TestOuterLeakage(unittest.TestCase):
    """Tests 4–5: outer actuals cannot affect scaling or early stopping."""

    def setUp(self) -> None:
        RecordingStubModel.instances.clear()
        RecordingStubModel.last_train_series = None
        self.v2_cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        self.neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        self.origin = 140401
        # Distinct huge post-origin actuals that would dominate a leaky scaler.
        self.sales = _monthly_sales_frame("SKU1", 140301, 40, base=10.0, step=0.0)
        post = self.sales["date"] >= self.origin
        self.sales.loc[post, "sales"] = 1_000_000.0

    def test_outer_actuals_cannot_affect_scaling(self):
        fit_calls: list[np.ndarray] = []
        original_fit = FoldScaler.fit_on_series

        def tracking_fit(self, history):
            arr = np.asarray(history, dtype=float).reshape(-1)
            fit_calls.append(arr.copy())
            return original_fit(self, history)

        with patch.object(FoldScaler, "fit_on_series", tracking_fit), patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            backtest_product_architectures(
                self.sales,
                "SKU1",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
            )
        self.assertTrue(fit_calls)
        for arr in fit_calls:
            self.assertFalse(np.any(arr >= 500_000.0))
        series = RecordingStubModel.last_train_series
        self.assertIsNotNone(series)
        assert series is not None
        self.assertLess(float(series.max()), 500_000.0)
        self.assertLess(int(series.index.max()), self.origin)

    def test_outer_actuals_cannot_affect_early_stopping(self):
        """Validation bounds recorded on ok folds are strictly before origin."""
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            result = backtest_product_architectures(
                self.sales,
                "SKU1",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
            )
        ok = result.fold_metadata.loc[result.fold_metadata["status"] == STATUS_OK]
        self.assertFalse(ok.empty)
        for _, row in ok.iterrows():
            if row["validation_end"] is not None and not pd.isna(row["validation_end"]):
                self.assertLess(int(row["validation_end"]), self.origin)
            if row["training_end"] is not None and not pd.isna(row["training_end"]):
                self.assertLess(int(row["training_end"]), self.origin)


class TestA0FoldIsolation(unittest.TestCase):
    """Test 6: A0 at N=30 uses 25–36 tier even if full SKU later reaches N=60."""

    def test_a0_n30_tier_despite_later_n60_history(self):
        # Full series length 60; at origin with exactly 30 pre-origin months.
        start = 140101
        sales = _monthly_sales_frame("A0SKU", start, 60, base=20.0, step=1.0)
        # Month index 30 is the origin (0..29 train → N=30).
        origin = int(sales.iloc[30]["date"])
        v2_cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        prepared = prepare_monthly_series(sales, "A0SKU", origin, config=v2_cfg)
        self.assertEqual(prepared.n_observations, 30)
        expected = resolve_legacy_architecture(30)
        self.assertEqual(
            (expected.l1, expected.l2, expected.lookback, expected.max_epochs),
            (128, 64, 6, 250),
        )
        large = resolve_legacy_architecture(60)
        self.assertEqual(large.lookback, 12)

        class A0TierStub(RecordingStubModel):
            name = "a0_tier_stub"

            def fit(self, train_series, window, *, seed):
                n = len(train_series)
                spec = resolve_legacy_architecture(n)
                self._resolved_spec = spec
                RecordingStubModel.last_train_series = train_series.copy()
                self.config = NeuralExperimentConfig(
                    architecture_name=ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
                    lookback=int(spec.lookback),
                    hidden_units=int(spec.l1),
                    second_hidden_units=int(spec.l2),
                    max_epochs=1,
                    early_stopping_patience=1,
                    number_layers=2,
                )
                self.metadata_ = _stub_metadata(
                    architecture=self.config.architecture_name,
                    seed=seed,
                    origin=int(window.forecast_origin),
                    lookback=int(spec.lookback),
                    hidden=int(spec.l1),
                    second=int(spec.l2),
                )
                return self

        neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(A0TierStub),
        ):
            result = backtest_product_architectures(
                sales,
                "A0SKU",
                architectures=(ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,),
                seeds=(41,),
                config=neural_cfg,
                v2_config=v2_cfg,
                explicit_origins=(origin,),
            )
        meta = result.fold_metadata.iloc[0]
        self.assertEqual(int(meta["available_history_length"]), 30)
        self.assertEqual(int(meta["lookback"]), 6)
        self.assertEqual(int(meta["hidden_units"]), 128)
        self.assertEqual(int(meta["second_hidden_units"]), 64)
        self.assertNotEqual(int(meta["lookback"]), large.lookback)


class TestUnavailableFolds(unittest.TestCase):
    """Test 7: unavailable folds create no fake predictions."""

    def test_unavailable_folds_create_no_fake_predictions(self):
        sales = _monthly_sales_frame("U", 140301, 40, base=10.0, step=1.0)
        v2_cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        eligibility = SampleEligibility(
            mathematically_constructible=False,
            eligible_for_training=False,
            n_windows=0,
            n_train=0,
            n_validation=0,
            min_internal_train_windows=8,
            min_internal_validation_windows=2,
            reason="insufficient_history_for_windows:need>=13, got=12",
        )

        class FailStub(RecordingStubModel):
            def fit(self, train_series, window, *, seed):
                raise InsufficientHistoryError(
                    "need more history",
                    n_history=len(train_series),
                    lookback=12,
                    horizon=1,
                    mode=TargetMode.RECURSIVE,
                )

        # Also cover IneligibleForTrainingError path in a second seed via factory.
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(FailStub),
        ):
            result = backtest_product_architectures(
                sales,
                "U",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=neural_cfg,
                v2_config=v2_cfg,
                explicit_origins=(140401,),
            )
        self.assertTrue(result.predictions.empty)
        self.assertFalse(result.fold_metadata.empty)
        self.assertTrue((result.fold_metadata["status"] == STATUS_UNAVAILABLE).all())
        self.assertTrue(
            (result.fold_metadata["unavailable_reason"] == "insufficient_history").all()
        )
        # No fabricated zero predictions.
        self.assertEqual(len(result.predictions), 0)

        class IneligStub(RecordingStubModel):
            def fit(self, train_series, window, *, seed):
                raise IneligibleForTrainingError(eligibility)

        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(IneligStub),
        ):
            result2 = backtest_product_architectures(
                sales,
                "U",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=neural_cfg,
                v2_config=v2_cfg,
                explicit_origins=(140401,),
            )
        self.assertTrue(result2.predictions.empty)
        self.assertEqual(
            result2.fold_metadata.iloc[0]["unavailable_reason"],
            eligibility.reason,
        )
        self.assertEqual(int(result2.metrics.iloc[0]["unavailable_fold_count"]), 1)


class TestOuterMetrics(unittest.TestCase):
    """Tests 8–9: raw-unit metrics and equal-weight horizon aggregation."""

    def setUp(self) -> None:
        self.v2_cfg = TSForecastConfig(
            forecast_horizon=2,
            min_train_months=3,
            activity_start_min_sales=None,
        )
        self.neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            horizon=2,
            lookback=3,
            random_seeds=(41,),
            min_successful_seeds=1,
            min_internal_train_windows=2,
            min_internal_validation_windows=1,
        )

    def test_metrics_from_raw_unit_forecasts(self):
        # actuals are exact base values; stub predicts 7.5 in raw units.
        sales = _monthly_sales_frame("R", 140401, 12, base=123.45, step=0.0)
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(RecordingStubModel),
        ):
            RecordingStubModel.predict_value = 7.5
            result = backtest_product_architectures(
                sales,
                "R",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(140404,),
            )
        self.assertFalse(result.predictions.empty)
        # Actuals remain warehouse raw units (not scaled ~0/1).
        self.assertTrue(
            (result.predictions["actual"].astype(float) == 123.45).all()
        )
        self.assertTrue(
            (result.predictions["prediction"].astype(float) == 7.5).all()
        )
        mae = float(result.metrics.iloc[0]["mean_horizon_MAE"])
        self.assertAlmostEqual(mae, abs(123.45 - 7.5), places=5)

    def test_horizon_aggregation_equal_weight(self):
        # Uneven horizon row counts: h1 has more origins than h2.
        # actual=0 so |err|=prediction=horizon → MAE_h = h.
        sales = _monthly_sales_frame("M", 140401, 10, base=0.0, step=0.0)
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(HorizonBiasStubModel),
        ):
            # Two origins: both evaluate h1; only first has room for h2 actual.
            # discover with explicit origins that have different coverage.
            result = backtest_product_architectures(
                sales,
                "M",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41,),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(140404, 140409),
            )
        preds = result.predictions
        self.assertFalse(preds.empty)
        # Manually build uneven frame to lock equal-weight math if coverage is even.
        uneven = pd.DataFrame(
            [
                {"horizon": 1, "actual": 0.0, "prediction": 1.0},
                {"horizon": 1, "actual": 0.0, "prediction": 1.0},
                {"horizon": 1, "actual": 0.0, "prediction": 1.0},
                {"horizon": 2, "actual": 0.0, "prediction": 2.0},
            ]
        )
        from pkg.ts_v2.metrics import horizon_mae as v2_horizon_mae

        h_mae = v2_horizon_mae(uneven)
        expected = selection_mae_from_horizons(h_mae)
        self.assertAlmostEqual(expected, 1.5, places=6)  # mean(1, 2)
        row_mae = (uneven["actual"] - uneven["prediction"]).abs().mean()
        self.assertAlmostEqual(float(row_mae), 5.0 / 4.0, places=6)
        self.assertNotAlmostEqual(float(expected), float(row_mae), places=3)
        self.assertAlmostEqual(mean_horizon_mae(uneven), expected, places=6)

        # Engine-reported metric also uses mean_horizon_MAE column.
        self.assertIn("mean_horizon_MAE", result.metrics.columns)
        engine_mae = float(result.metrics.iloc[0]["mean_horizon_MAE"])
        engine_h = v2_horizon_mae(preds)
        self.assertAlmostEqual(
            engine_mae, float(selection_mae_from_horizons(engine_h)), places=6
        )


class TestOuterBacktestSmoke(unittest.TestCase):
    """Optional TF e2e: one short A1 fold through the real NeuralTrainer path."""

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow not available")

    def test_real_a1_outer_fold_smoke(self):
        # Enough history for lookback=12 recursive eligibility (~N>=22) and H=15 actuals.
        sales = _monthly_sales_frame("TF", 140201, 50, base=10.0, step=1.0, product_id=9)
        origin = 140401
        v2_cfg = TSForecastConfig(
            forecast_horizon=15,
            min_train_months=12,
            activity_start_min_sales=None,
        )
        neural_cfg = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            lookback=12,
            hidden_units=8,
            max_epochs=2,
            early_stopping_patience=1,
            batch_size=4,
            random_seeds=(41,),
            min_successful_seeds=1,
        )
        result = run_outer_backtest(
            sales,
            ["TF"],
            architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
            seeds=(41,),
            config=neural_cfg,
            v2_config=v2_cfg,
            explicit_origins=(origin,),
        )
        self.assertFalse(result.fold_metadata.empty)
        status = result.fold_metadata.iloc[0]["status"]
        if status == STATUS_OK:
            self.assertFalse(result.predictions.empty)
            self.assertEqual(int(result.predictions["horizon"].min()), 1)
            self.assertLessEqual(int(result.predictions["horizon"].max()), 15)
            self.assertTrue(
                (result.predictions["origin"].astype(int) == origin).all()
            )
            self.assertIn("mean_horizon_MAE", result.metrics.columns)
            meta = result.fold_metadata.iloc[0]
            self.assertLess(int(meta["training_end"]), origin)
            if meta["validation_end"] is not None and not pd.isna(meta["validation_end"]):
                self.assertLess(int(meta["validation_end"]), origin)
        else:
            # Still must not invent predictions.
            self.assertTrue(result.predictions.empty)


if __name__ == "__main__":
    unittest.main()
