"""Multi-seed ensemble evaluation: isolation, mean, min_successful_seeds, reproducibility."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.types import ForecastResult, ForecastWindow
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    backtest_product_architectures,
    run_outer_backtest,
)
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.metrics import (
    ENSEMBLE_UNAVAILABLE_REASON,
    PREDICTION_KIND_ENSEMBLE,
    PREDICTION_KIND_SEED,
    build_seed_ensemble_predictions,
    ensemble_metrics_table,
    seed_metrics_table,
)
from pkg.ts_v3a.types import TrainMetadata, build_train_metadata
from pkg.ts_v3a.windows import InsufficientHistoryError
from pkg.ts_v3a.types import TargetMode


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
        row = {"product": product, "date": cur, "sales": base + step * i}
        if product_id is not None:
            row["product_id"] = product_id
        rows.append(row)
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _stub_metadata(*, architecture: str, seed: int, origin: int) -> TrainMetadata:
    train_end = shamsi_add_months(origin, -1)
    return build_train_metadata(
        architecture=architecture,
        parameters={"lookback": 12, "hidden_units": 32, "second_hidden_units": 16},
        random_seed=seed,
        n_train_samples=8,
        n_validation_samples=2,
        forecast_origin=origin,
        training_start=shamsi_add_months(origin, -12),
        training_end=train_end,
        validation_start=shamsi_add_months(origin, -3),
        validation_end=train_end,
        epochs_ran=1,
        best_epoch=1,
        best_val_loss=0.1,
        parameter_count=100,
        train_window_count=8,
        validation_window_count=2,
    )


class SeedValueStub:
    """Predicts a deterministic function of seed (and optional architecture tag)."""

    instances: list["SeedValueStub"] = []
    fail_seeds: set[int] = set()
    # Optional map seed -> prediction value; default seed-41=1, 42=2, 43=3
    seed_values: dict[int, float] = {41: 1.0, 42: 2.0, 43: 3.0}

    def __init__(self, config: Optional[NeuralExperimentConfig] = None) -> None:
        self.config = config or NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            min_successful_seeds=1,
        )
        self.metadata_: Optional[TrainMetadata] = None
        self.name = self.config.architecture_name
        SeedValueStub.instances.append(self)

    def fit(
        self,
        train_series: pd.Series,
        window: ForecastWindow,
        *,
        seed: int,
    ) -> "SeedValueStub":
        if int(seed) in SeedValueStub.fail_seeds:
            raise InsufficientHistoryError(
                "forced seed failure",
                n_history=len(train_series),
                lookback=12,
                horizon=1,
                mode=TargetMode.RECURSIVE,
            )
        self.metadata_ = _stub_metadata(
            architecture=self.config.architecture_name,
            seed=int(seed),
            origin=int(window.forecast_origin),
        )
        self._seed = int(seed)
        return self

    def predict(self, window: ForecastWindow) -> ForecastResult:
        base = float(SeedValueStub.seed_values.get(self._seed, float(self._seed)))
        # Keep architecture outputs distinguishable without mixing keys.
        if self.config.architecture_name == ArchitectureName.A2_MIMO_LSTM.value:
            base = base + 100.0
        h = len(window.horizons)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(base for _ in range(h)),
            target_dates=tuple(int(d) for d in window.target_dates),
            horizons=tuple(int(x) for x in window.horizons),
        )


def _patch_factory(model_cls: type = SeedValueStub):
    def _factory(architecture, config=None):
        from pkg.ts_v3a.architectures import coerce_architecture_name

        name = coerce_architecture_name(architecture)
        cfg = config or NeuralExperimentConfig(architecture_name=name.value)
        if cfg.architecture_name != name.value:
            cfg = cfg.with_architecture(name)
        return model_cls(cfg)

    return _factory


class TestMultiSeedIsolation(unittest.TestCase):
    def setUp(self) -> None:
        SeedValueStub.instances.clear()
        SeedValueStub.fail_seeds = set()
        SeedValueStub.seed_values = {41: 1.0, 42: 2.0, 43: 3.0}
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
            random_seeds=(41, 42, 43),
            min_successful_seeds=3,
            min_internal_train_windows=2,
            min_internal_validation_windows=1,
        )
        sales_a = _monthly_sales_frame("A", 140401, 12, base=10.0, product_id=1)
        sales_b = _monthly_sales_frame("B", 140401, 12, base=20.0, product_id=2)
        self.sales = pd.concat([sales_a, sales_b], ignore_index=True)
        self.origins = (140404, 140406)

    def test_seed_forecasts_never_mixed_across_sku_origin_architecture(self):
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(SeedValueStub),
        ):
            result = run_outer_backtest(
                self.sales,
                ["A", "B"],
                architectures=(
                    ArchitectureName.A1_SMALL_RECURSIVE_LSTM,
                    ArchitectureName.A2_MIMO_LSTM,
                ),
                seeds=(41, 42, 43),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=self.origins,
                min_successful_seeds=3,
            )
        preds = result.seed_predictions
        self.assertFalse(preds.empty)
        self.assertTrue((preds["prediction_kind"] == PREDICTION_KIND_SEED).all())

        # Each (product, architecture, origin, seed) maps to a unique constant.
        for product in ("A", "B"):
            for arch, offset in (
                (ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value, 0.0),
                (ArchitectureName.A2_MIMO_LSTM.value, 100.0),
            ):
                for origin in self.origins:
                    for seed, base in ((41, 1.0), (42, 2.0), (43, 3.0)):
                        sub = preds.loc[
                            (preds["product"] == product)
                            & (preds["architecture"] == arch)
                            & (preds["origin"] == origin)
                            & (preds["seed"] == seed)
                        ]
                        self.assertFalse(sub.empty)
                        expected = base + offset
                        self.assertTrue((sub["prediction"].astype(float) == expected).all())

        ens = result.ensemble_predictions
        self.assertFalse(ens.empty)
        self.assertTrue((ens["prediction_kind"] == PREDICTION_KIND_ENSEMBLE).all())
        # Ensemble for A1 is mean(1,2,3)=2; for A2 mean(101,102,103)=102.
        for product in ("A", "B"):
            a1 = ens.loc[
                (ens["product"] == product)
                & (ens["architecture"] == ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value)
            ]
            a2 = ens.loc[
                (ens["product"] == product)
                & (ens["architecture"] == ArchitectureName.A2_MIMO_LSTM.value)
            ]
            self.assertTrue((a1["prediction"].astype(float) == 2.0).all())
            self.assertTrue((a2["prediction"].astype(float) == 102.0).all())
            # Origins remain separated.
            self.assertEqual(set(a1["origin"].astype(int)), set(self.origins))


class TestEnsembleMean(unittest.TestCase):
    def setUp(self) -> None:
        SeedValueStub.instances.clear()
        SeedValueStub.fail_seeds = set()
        SeedValueStub.seed_values = {41: 1.0, 42: 2.0, 43: 3.0}
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
            random_seeds=(41, 42, 43),
            min_successful_seeds=3,
            min_internal_train_windows=2,
            min_internal_validation_windows=1,
        )
        self.sales = _monthly_sales_frame("M", 140401, 12, base=0.0)
        self.origin = 140404

    def test_ensemble_mean_is_exact(self):
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(SeedValueStub),
        ):
            result = backtest_product_architectures(
                self.sales,
                "M",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41, 42, 43),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
                min_successful_seeds=3,
            )
        ens = result.ensemble_predictions
        self.assertFalse(ens.empty)
        self.assertTrue((ens["prediction"].astype(float) == 2.0).all())
        self.assertTrue((ens["n_successful_seeds"].astype(int) == 3).all())
        status = result.ensemble_origin_status
        self.assertEqual(status.iloc[0]["status"], STATUS_OK)
        self.assertEqual(int(status.iloc[0]["n_successful_seeds"]), 3)


class TestMinSuccessfulSeeds(unittest.TestCase):
    def setUp(self) -> None:
        SeedValueStub.instances.clear()
        SeedValueStub.seed_values = {41: 1.0, 42: 2.0, 43: 3.0}
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
            random_seeds=(41, 42, 43),
            min_successful_seeds=3,
            min_internal_train_windows=2,
            min_internal_validation_windows=1,
        )
        self.sales = _monthly_sales_frame("U", 140401, 12, base=5.0)
        self.origin = 140404

    def test_failed_seed_blocks_ensemble_when_min_is_three(self):
        SeedValueStub.fail_seeds = {43}
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(SeedValueStub),
        ):
            result = backtest_product_architectures(
                self.sales,
                "U",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41, 42, 43),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
                min_successful_seeds=3,
            )
        # Seed OOF still present for successful seeds.
        self.assertFalse(result.seed_predictions.empty)
        self.assertEqual(set(result.seed_predictions["seed"].astype(int)), {41, 42})
        # Ensemble unavailable — no silent average of 2 seeds.
        self.assertTrue(result.ensemble_predictions.empty)
        st = result.ensemble_origin_status.iloc[0]
        self.assertEqual(st["status"], STATUS_UNAVAILABLE)
        self.assertEqual(st["unavailable_reason"], ENSEMBLE_UNAVAILABLE_REASON)
        self.assertEqual(int(st["n_successful_seeds"]), 2)

    def test_lower_min_allows_partial_ensemble(self):
        SeedValueStub.fail_seeds = {43}
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(SeedValueStub),
        ):
            result = backtest_product_architectures(
                self.sales,
                "U",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41, 42, 43),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(self.origin,),
                min_successful_seeds=2,
            )
        ens = result.ensemble_predictions
        self.assertFalse(ens.empty)
        # mean(1, 2) == 1.5
        self.assertTrue((ens["prediction"].astype(float) == 1.5).all())
        self.assertEqual(result.ensemble_origin_status.iloc[0]["status"], STATUS_OK)


class TestMetricsReproducibleFromOOF(unittest.TestCase):
    def setUp(self) -> None:
        SeedValueStub.instances.clear()
        SeedValueStub.fail_seeds = set()
        SeedValueStub.seed_values = {41: 1.0, 42: 2.0, 43: 3.0}
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
            random_seeds=(41, 42, 43),
            min_successful_seeds=3,
            min_internal_train_windows=2,
            min_internal_validation_windows=1,
        )
        self.sales = _monthly_sales_frame("R", 140401, 12, base=10.0)

    def test_architecture_metrics_reproducible_from_saved_oof_rows(self):
        with patch(
            "pkg.ts_v3a.backtest.create_neural_model",
            side_effect=_patch_factory(SeedValueStub),
        ):
            result = backtest_product_architectures(
                self.sales,
                "R",
                architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
                seeds=(41, 42, 43),
                config=self.neural_cfg,
                v2_config=self.v2_cfg,
                explicit_origins=(140404,),
                min_successful_seeds=3,
            )

        rebuilt_seed = seed_metrics_table(
            result.seed_predictions,
            result.fold_metadata,
            forecast_horizon=2,
            v2_config=self.v2_cfg,
        )
        pd.testing.assert_frame_equal(
            result.seed_metrics.reset_index(drop=True),
            rebuilt_seed.reset_index(drop=True),
            check_dtype=False,
        )

        rebuilt_ens_pred, rebuilt_status = build_seed_ensemble_predictions(
            result.seed_predictions,
            result.fold_metadata,
            seeds=(41, 42, 43),
            min_successful_seeds=3,
        )
        pd.testing.assert_frame_equal(
            result.ensemble_predictions.reset_index(drop=True),
            rebuilt_ens_pred.reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            result.ensemble_origin_status.reset_index(drop=True),
            rebuilt_status.reset_index(drop=True),
            check_dtype=False,
        )

        rebuilt_ens_met = ensemble_metrics_table(
            rebuilt_ens_pred,
            rebuilt_status,
            products=["R"],
            architectures=[ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value],
            forecast_horizon=2,
            v2_config=self.v2_cfg,
        )
        pd.testing.assert_frame_equal(
            result.ensemble_metrics.reset_index(drop=True),
            rebuilt_ens_met.reset_index(drop=True),
            check_dtype=False,
        )
        # metrics aliases ensemble_metrics
        pd.testing.assert_frame_equal(
            result.metrics.reset_index(drop=True),
            result.ensemble_metrics.reset_index(drop=True),
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
