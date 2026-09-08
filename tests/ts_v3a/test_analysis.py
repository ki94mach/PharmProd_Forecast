"""Tests for V3A offline analysis helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v3a.analysis.metrics import history_bucket, portfolio_wmape
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import STATUS_ERROR, STATUS_OK, NeuralOuterBacktestResult
from pkg.ts_v3a.config import NeuralExperimentConfig
from pkg.ts_v3a.metrics import PREDICTION_KIND_SEED
from pkg.ts_v3a.persistence import persist_completed_screening_experiment


def _sample_result() -> NeuralOuterBacktestResult:
    preds = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": seed,
                "origin": 140401,
                "target_date": 140401,
                "horizon": 1,
                "actual": 10.0,
                "prediction": 9.0 + seed * 0.1,
                "prediction_kind": PREDICTION_KIND_SEED,
            }
            for seed in (41, 42)
        ]
    )
    fold = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": s,
                "origin": 140401,
                "available_history_length": 24,
                "lookback": 12,
                "hidden_units": 32,
                "second_hidden_units": 16,
                "parameter_count": 100 if s != 43 else None,
                "train_window_count": 8 if s != 43 else None,
                "validation_window_count": 2 if s != 43 else None,
                "best_epoch": 1 if s != 43 else None,
                "epochs_ran": 1 if s != 43 else None,
                "best_val_loss": 0.1 if s != 43 else None,
                "training_start": 140201 if s != 43 else None,
                "training_end": 140312 if s != 43 else None,
                "validation_start": 140301 if s != 43 else None,
                "validation_end": 140312 if s != 43 else None,
                "runtime_seconds": 0.01 if s != 43 else 0.0,
                "scaler_mean": 10.0 if s != 43 else None,
                "scaler_scale": 2.0 if s != 43 else None,
                "status": STATUS_OK if s != 43 else STATUS_ERROR,
                "unavailable_reason": None,
                "error_type": None if s != 43 else "RuntimeError",
                "error_message": None if s != 43 else "boom",
            }
            for s in (41, 42, 43)
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "product": "SKU1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "mean_horizon_MAE": 1.0,
                "overall_rmse": 1.0,
                "overall_bias": 0.0,
                "overall_wmape": 0.1,
                "number_of_origins": 1,
                "number_of_predictions": 2,
                "evaluated_horizons": (1,),
                "max_evaluated_horizon": 1,
                "unavailable_fold_count": 0,
                "n_origins_with_ensemble": 1,
                "unavailable_ensemble_origin_count": 0,
                "mae_h1": 1.0,
            }
        ]
    )
    return NeuralOuterBacktestResult(
        predictions=preds,
        fold_metadata=fold,
        metrics=metrics,
        ensemble_predictions=pd.DataFrame(),
        ensemble_origin_status=pd.DataFrame(),
        seed_metrics=metrics.copy(),
        ensemble_metrics=metrics,
        stability=pd.DataFrame(),
    )


class TestHistoryBucket(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(history_bucket(40), ">36")
        self.assertEqual(history_bucket(30), "25-36")
        self.assertEqual(history_bucket(20), "13-24")


class TestPortfolioWmape(unittest.TestCase):
    def test_pooled_not_mean(self):
        actual = pd.Series([100.0, 10.0])
        pred = pd.Series([110.0, 20.0])
        self.assertAlmostEqual(portfolio_wmape(actual, pred), 2000.0 / 110.0, places=5)


class TestCheckpointDeleted(unittest.TestCase):
    def test_finalize_drops_checkpoint(self):
        base = Path(tempfile.mkdtemp())
        result = _sample_result()
        path = persist_completed_screening_experiment(
            result,
            neural_config=NeuralExperimentConfig(),
            v2_config=TSForecastConfig(),
            architectures=[ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value],
            seeds=(41, 42, 43),
            origins=(140401,),
            min_successful_seeds=2,
            base_dir=base,
            experiment_id="20260101T000000Z_ckptgone",
        )
        self.assertTrue((path / ".complete").is_file())
        self.assertFalse((path / "checkpoint.json").exists())
        with (path / "manifest.json").open(encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
