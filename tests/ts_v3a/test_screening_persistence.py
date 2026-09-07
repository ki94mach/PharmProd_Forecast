"""Immutable V3A screening persistence: hash, conflicts, immutability, round-trip."""
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
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import STATUS_ERROR, STATUS_OK, NeuralOuterBacktestResult
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.metrics import PREDICTION_KIND_ENSEMBLE, PREDICTION_KIND_SEED
from pkg.ts_v3a.persistence import (
    COMPLETE_MARKER,
    FAILURE_COLUMNS,
    ExperimentCheckpointError,
    ExperimentConfigConflictError,
    ExperimentImmutableError,
    begin_screening_experiment,
    build_failures_dataframe,
    build_horizon_metrics_dataframe,
    finalize_screening_experiment,
    is_complete_experiment,
    load_screening_experiment,
    persist_completed_screening_experiment,
    screening_config_hash,
    write_screening_checkpoint,
)


def _sample_result() -> NeuralOuterBacktestResult:
    seed_preds = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "origin": 140401,
                "target_date": 140401,
                "horizon": 1,
                "actual": 10.0,
                "prediction": 9.0,
                "prediction_kind": PREDICTION_KIND_SEED,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 42,
                "origin": 140401,
                "target_date": 140401,
                "horizon": 1,
                "actual": 10.0,
                "prediction": 11.0,
                "prediction_kind": PREDICTION_KIND_SEED,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "origin": 140401,
                "target_date": 140402,
                "horizon": 2,
                "actual": 12.0,
                "prediction": 12.5,
                "prediction_kind": PREDICTION_KIND_SEED,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 42,
                "origin": 140401,
                "target_date": 140402,
                "horizon": 2,
                "actual": 12.0,
                "prediction": 11.5,
                "prediction_kind": PREDICTION_KIND_SEED,
            },
        ]
    )
    fold_metadata = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "origin": 140401,
                "available_history_length": 24,
                "lookback": 12,
                "hidden_units": 32,
                "second_hidden_units": 16,
                "parameter_count": 100,
                "train_window_count": 8,
                "validation_window_count": 2,
                "best_epoch": 1,
                "epochs_ran": 1,
                "best_val_loss": 0.1,
                "training_start": 140201,
                "training_end": 140312,
                "validation_start": 140301,
                "validation_end": 140312,
                "runtime_seconds": 0.01,
                "status": STATUS_OK,
                "unavailable_reason": None,
                "error_type": None,
                "error_message": None,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 42,
                "origin": 140401,
                "available_history_length": 24,
                "lookback": 12,
                "hidden_units": 32,
                "second_hidden_units": 16,
                "parameter_count": 100,
                "train_window_count": 8,
                "validation_window_count": 2,
                "best_epoch": 1,
                "epochs_ran": 1,
                "best_val_loss": 0.2,
                "training_start": 140201,
                "training_end": 140312,
                "validation_start": 140301,
                "validation_end": 140312,
                "runtime_seconds": 0.01,
                "status": STATUS_OK,
                "unavailable_reason": None,
                "error_type": None,
                "error_message": None,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 43,
                "origin": 140401,
                "available_history_length": 24,
                "lookback": None,
                "hidden_units": None,
                "second_hidden_units": None,
                "parameter_count": None,
                "train_window_count": None,
                "validation_window_count": None,
                "best_epoch": None,
                "epochs_ran": None,
                "best_val_loss": None,
                "training_start": None,
                "training_end": None,
                "validation_start": None,
                "validation_end": None,
                "runtime_seconds": 0.0,
                "status": STATUS_ERROR,
                "unavailable_reason": None,
                "error_type": "RuntimeError",
                "error_message": "boom",
            },
        ]
    )
    ens_preds = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "origin": 140401,
                "target_date": 140401,
                "horizon": 1,
                "actual": 10.0,
                "prediction": 10.0,
                "n_successful_seeds": 2,
                "prediction_kind": PREDICTION_KIND_ENSEMBLE,
            },
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "origin": 140401,
                "target_date": 140402,
                "horizon": 2,
                "actual": 12.0,
                "prediction": 12.0,
                "n_successful_seeds": 2,
                "prediction_kind": PREDICTION_KIND_ENSEMBLE,
            },
        ]
    )
    seed_metrics = pd.DataFrame(
        [
            {
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "mean_horizon_MAE": 0.75,
                "overall_rmse": 1.0,
                "overall_bias": -0.25,
                "overall_wmape": 0.05,
                "number_of_origins": 1,
                "number_of_predictions": 2,
                "evaluated_horizons": (1, 2),
                "max_evaluated_horizon": 2,
                "unavailable_fold_count": 0,
                "mae_h1": 1.0,
                "mae_h2": 0.5,
            }
        ]
    )
    ensemble_metrics = pd.DataFrame(
        [
            {
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "mean_horizon_MAE": 0.0,
                "overall_rmse": 0.0,
                "overall_bias": 0.0,
                "overall_wmape": 0.0,
                "number_of_origins": 1,
                "number_of_predictions": 2,
                "evaluated_horizons": (1, 2),
                "max_evaluated_horizon": 2,
                "unavailable_fold_count": 0,
                "n_origins_with_ensemble": 1,
                "unavailable_ensemble_origin_count": 0,
                "mae_h1": 0.0,
                "mae_h2": 0.0,
            }
        ]
    )
    return NeuralOuterBacktestResult(
        predictions=seed_preds,
        fold_metadata=fold_metadata,
        metrics=ensemble_metrics,
        ensemble_predictions=ens_preds,
        ensemble_origin_status=pd.DataFrame(),
        seed_metrics=seed_metrics,
        ensemble_metrics=ensemble_metrics,
        stability=pd.DataFrame(),
    )


def _base_kwargs(base: Path, **overrides):
    kwargs = dict(
        neural_config=NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            max_epochs=1,
            early_stopping_patience=1,
            random_seeds=(41, 42),
            min_successful_seeds=2,
        ),
        v2_config=TSForecastConfig(
            forecast_horizon=2,
            min_train_months=3,
            activity_start_min_sales=None,
        ),
        architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
        seeds=(41, 42),
        origins=(140401,),
        min_successful_seeds=2,
        base_dir=base,
        experiment_id="20260101T000000Z_persist1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return kwargs


class TestConfigHash(unittest.TestCase):
    def test_config_hash_stability(self):
        neural = NeuralExperimentConfig(
            architecture_name=ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
            random_seeds=(41, 42, 43),
            min_successful_seeds=3,
        )
        v2 = TSForecastConfig(forecast_horizon=15, min_train_months=12)
        h1 = screening_config_hash(
            neural_config=neural,
            v2_config=v2,
            architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
            seeds=(41, 42, 43),
            origins=(140401,),
        )
        h2 = screening_config_hash(
            neural_config=neural,
            v2_config=v2,
            architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
            seeds=(41, 42, 43),
            origins=(140401,),
        )
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

        h_seed = screening_config_hash(
            neural_config=neural,
            v2_config=v2,
            architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
            seeds=(41, 42, 99),
            origins=(140401,),
        )
        self.assertNotEqual(h1, h_seed)

        h_horizon = screening_config_hash(
            neural_config=neural,
            v2_config=TSForecastConfig(forecast_horizon=12, min_train_months=12),
            architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM,),
            seeds=(41, 42, 43),
            origins=(140401,),
        )
        self.assertNotEqual(h1, h_horizon)

        h_arch = screening_config_hash(
            neural_config=neural,
            v2_config=v2,
            architectures=(ArchitectureName.A2_MIMO_LSTM,),
            seeds=(41, 42, 43),
            origins=(140401,),
        )
        self.assertNotEqual(h1, h_arch)


class TestIncompatibleConfig(unittest.TestCase):
    def test_incompatible_config_rejection(self):
        base = Path(tempfile.mkdtemp())
        result = _sample_result()
        kwargs = _base_kwargs(base)
        checkpoint = begin_screening_experiment(**kwargs)
        write_screening_checkpoint(checkpoint, result)

        bad = dict(kwargs)
        bad["seeds"] = (41, 99)
        bad["resume"] = True
        with self.assertRaises(ExperimentConfigConflictError):
            begin_screening_experiment(**bad)

        # New experiment id with different hash against existing incomplete same id
        # is already covered above via resume. Also: finalize then refuse different config.
        path = finalize_screening_experiment(checkpoint, result, base_dir=base)
        self.assertTrue(is_complete_experiment(path))

        with self.assertRaises(ExperimentImmutableError):
            begin_screening_experiment(**kwargs)


class TestImmutability(unittest.TestCase):
    def test_immutable_completed_experiment(self):
        base = Path(tempfile.mkdtemp())
        result = _sample_result()
        kwargs = _base_kwargs(base, experiment_id="20260101T000000Z_immut")
        path = persist_completed_screening_experiment(result, **kwargs)
        self.assertTrue((path / COMPLETE_MARKER).is_file())
        self.assertTrue((path / "manifest.json").is_file())
        self.assertTrue((path / "oof_predictions.parquet").is_file())

        with self.assertRaises(ExperimentImmutableError):
            persist_completed_screening_experiment(result, **kwargs)

        checkpoint = begin_screening_experiment(
            **{
                **kwargs,
                "experiment_id": "20260101T000000Z_immut2",
            }
        )
        finalize_screening_experiment(checkpoint, result, base_dir=base)
        # Writing into a completed path via same id fails at begin.
        with self.assertRaises(ExperimentImmutableError):
            begin_screening_experiment(
                **{**kwargs, "experiment_id": "20260101T000000Z_immut2"}
            )


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_prediction_persistence(self):
        base = Path(tempfile.mkdtemp())
        result = _sample_result()
        kwargs = _base_kwargs(base, experiment_id="20260101T000000Z_round")
        path = persist_completed_screening_experiment(result, **kwargs)
        loaded = load_screening_experiment("20260101T000000Z_round", base_dir=base)
        self.assertEqual(loaded.experiment_dir, path)

        # Parquet round-trip may widen dtypes; compare values after aligning.
        original = result.predictions.reset_index(drop=True)
        reloaded = loaded.oof_predictions.reset_index(drop=True)
        self.assertEqual(list(original.columns), list(reloaded.columns))
        pd.testing.assert_frame_equal(
            original.astype(str),
            reloaded.astype(str),
            check_dtype=False,
        )

        # Manifest carries config_hash and version.
        with (path / "manifest.json").open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["v3a_version"], "v3a")
        self.assertIn("config_hash", manifest)
        self.assertEqual(manifest["config_hash"], kwargs and screening_config_hash(
            neural_config=kwargs["neural_config"],
            v2_config=kwargs["v2_config"],
            architectures=kwargs["architectures"],
            seeds=kwargs["seeds"],
            origins=kwargs["origins"],
            min_successful_seeds=kwargs["min_successful_seeds"],
        ))

        # Failures include the error fold; empty-ok headers still present as columns.
        self.assertFalse(loaded.failures.empty)
        self.assertEqual(list(loaded.failures.columns), list(FAILURE_COLUMNS))
        self.assertEqual(int(loaded.failures.iloc[0]["seed"]), 43)

        # Horizon metrics align with architecture mae_h*.
        arch = loaded.architecture_metrics.iloc[0]
        h = loaded.horizon_metrics
        for _, row in h.iterrows():
            col = f"mae_h{int(row['horizon'])}"
            if col in arch.index:
                self.assertAlmostEqual(float(row["mae"]), float(arch[col]), places=6)


class TestEmptyFailuresHeaders(unittest.TestCase):
    def test_failures_headers_when_empty(self):
        fold = pd.DataFrame(
            [
                {
                    "product": "X",
                    "architecture": "a1",
                    "seed": 41,
                    "origin": 140401,
                    "status": STATUS_OK,
                    "unavailable_reason": None,
                    "error_type": None,
                    "error_message": None,
                }
            ]
        )
        failures = build_failures_dataframe(fold)
        self.assertTrue(failures.empty)
        self.assertEqual(list(failures.columns), list(FAILURE_COLUMNS))

    def test_horizon_metrics_builder_empty(self):
        empty = build_horizon_metrics_dataframe(pd.DataFrame())
        self.assertTrue(empty.empty)
        self.assertIn("mae", empty.columns)


class TestIncompleteResume(unittest.TestCase):
    def test_resume_requires_matching_hash(self):
        base = Path(tempfile.mkdtemp())
        kwargs = _base_kwargs(base, experiment_id="20260101T000000Z_resume")
        begin_screening_experiment(**kwargs)
        resumed = begin_screening_experiment(**{**kwargs, "resume": True})
        self.assertEqual(resumed.experiment_id, "20260101T000000Z_resume")
        self.assertTrue(resumed.experiment_dir.as_posix().endswith(
            "/.incomplete/20260101T000000Z_resume"
        ) or str(resumed.experiment_dir).endswith(
            f".incomplete{Path('/').sep}20260101T000000Z_resume"
        ))

        with self.assertRaises(ExperimentCheckpointError):
            begin_screening_experiment(**kwargs)  # exists, no resume


if __name__ == "__main__":
    unittest.main()
