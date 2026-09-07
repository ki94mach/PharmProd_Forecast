"""Tests for V3A smoke/screening CLI (aliases, dry-run, stubbed e2e)."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import STATUS_OK, NeuralOuterBacktestResult
from pkg.ts_v3a.metrics import PREDICTION_KIND_SEED
from pkg.ts_v3a.screen import (
    build_parser,
    main,
    parse_architecture_aliases,
    print_fold_runtime_summary,
    run_screen,
)


def _stub_result() -> NeuralOuterBacktestResult:
    preds = pd.DataFrame(
        [
            {
                "product_id": None,
                "product": "P1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "origin": 140401,
                "target_date": 140401,
                "horizon": 1,
                "actual": 10.0,
                "prediction": 9.5,
                "prediction_kind": PREDICTION_KIND_SEED,
            }
        ]
    )
    fold = pd.DataFrame(
        [
            {
                "product_id": None,
                "product": "P1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "seed": 41,
                "origin": 140401,
                "available_history_length": 24,
                "lookback": 12,
                "hidden_units": 32,
                "second_hidden_units": 16,
                "parameter_count": 100,
                "train_window_count": 10,
                "validation_window_count": 3,
                "best_epoch": 2,
                "epochs_ran": 3,
                "best_val_loss": 0.1,
                "training_start": 140201,
                "training_end": 140312,
                "validation_start": 140301,
                "validation_end": 140312,
                "runtime_seconds": 1.25,
                "status": STATUS_OK,
                "unavailable_reason": None,
                "error_type": None,
                "error_message": None,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "product": "P1",
                "architecture": ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
                "mean_horizon_MAE": 0.5,
                "overall_rmse": 0.5,
                "overall_bias": -0.5,
                "overall_wmape": 0.05,
                "number_of_origins": 1,
                "number_of_predictions": 1,
                "evaluated_horizons": (1,),
                "max_evaluated_horizon": 1,
                "unavailable_fold_count": 0,
                "n_origins_with_ensemble": 1,
                "unavailable_ensemble_origin_count": 0,
                "mae_h1": 0.5,
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


class TestArchitectureAliases(unittest.TestCase):
    def test_short_aliases_resolve(self):
        names = parse_architecture_aliases("a0,a2")
        self.assertEqual(
            names,
            (
                ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,
                ArchitectureName.A2_MIMO_LSTM,
            ),
        )

    def test_full_names_accepted(self):
        names = parse_architecture_aliases("small_recursive_lstm")
        self.assertEqual(names, (ArchitectureName.A1_SMALL_RECURSIVE_LSTM,))

    def test_reject_a6_and_garbage(self):
        with self.assertRaises(ValueError):
            parse_architecture_aliases("a6")
        with self.assertRaises(ValueError):
            parse_architecture_aliases("not_an_arch")
        with self.assertRaises(ValueError):
            parse_architecture_aliases("attention_bidirectional_lstm")


class TestDryRun(unittest.TestCase):
    def test_dry_run_no_train(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--products",
                "P1",
                "--origins",
                "140401",
                "--architectures",
                "a1",
                "--seeds",
                "41",
                "--dry-run",
            ]
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf), patch(
            "pkg.ts_v3a.screen.run_outer_backtest"
        ) as mock_bt, patch(
            "pkg.ts_v3a.screen.persist_completed_screening_experiment"
        ) as mock_persist:
            code = run_screen(args)
        self.assertEqual(code, 0)
        mock_bt.assert_not_called()
        mock_persist.assert_not_called()
        out = buf.getvalue()
        self.assertIn("config_hash=", out)
        self.assertIn("dry_run=1", out)
        self.assertNotIn("selected architecture", out.lower())
        self.assertNotIn("winner", out.lower())


class TestStubbedE2E(unittest.TestCase):
    def test_stubbed_run_prints_summary_and_persists(self):
        base = Path(tempfile.mkdtemp())
        stub = _stub_result()
        sales = pd.DataFrame(
            {
                "product": ["P1", "P1"],
                "date": [140301, 140302],
                "sales": [1.0, 2.0],
            }
        )
        parser = build_parser()
        args = parser.parse_args(
            [
                "--products",
                "P1",
                "--origins",
                "140401",
                "--architectures",
                "a1",
                "--seeds",
                "41",
                "--output",
                str(base),
                "--experiment-id",
                "20260101T000000Z_screencli",
                "--min-successful-seeds",
                "1",
            ]
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf), patch(
            "pkg.ts_v3a.screen.load_screening_sales", return_value=sales
        ), patch(
            "pkg.ts_v3a.screen.run_outer_backtest", return_value=stub
        ) as mock_bt, patch(
            "pkg.ts_v3a.screen.persist_completed_screening_experiment",
            return_value=base / "20260101T000000Z_screencli",
        ) as mock_persist:
            code = run_screen(args)
        self.assertEqual(code, 0)
        mock_bt.assert_called_once()
        mock_persist.assert_called_once()
        out = buf.getvalue()
        self.assertIn("architecture", out)
        self.assertIn("history_length", out)
        self.assertIn("train_windows", out)
        self.assertIn("runtime_s", out)
        self.assertIn("status", out)
        self.assertIn("seed_oof_prediction_rows=1", out)
        self.assertIn("Architecture selection is not performed", out)
        self.assertNotIn("winner", out.lower())
        self.assertNotIn("selected_model", out.lower())

    def test_main_missing_products_returns_2(self):
        code = main(["--origins", "140401"])
        self.assertEqual(code, 2)


class TestPrintHelpers(unittest.TestCase):
    def test_fold_summary_headers(self):
        stub = _stub_result()
        buf = io.StringIO()
        print_fold_runtime_summary(stub.fold_metadata, file=buf)
        lines = buf.getvalue().strip().splitlines()
        self.assertIn("architecture", lines[0])
        self.assertIn("P1", lines[1])
        self.assertIn("1.250", lines[1])


if __name__ == "__main__":
    unittest.main()
