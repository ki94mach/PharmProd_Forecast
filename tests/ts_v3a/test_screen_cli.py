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
from pkg.ts_v3a.persistence import ScreeningExperimentCheckpoint
from pkg.ts_v3a.screen import (
    build_parser,
    main,
    parse_architecture_aliases,
    print_fold_runtime_summary,
    run_screen,
)


def _stub_result(product: str = "P1") -> NeuralOuterBacktestResult:
    preds = pd.DataFrame(
        [
            {
                "product_id": None,
                "product": product,
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
                "product": product,
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
                "scaler_mean": 5.0,
                "scaler_scale": 1.0,
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
                "product": product,
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


def _fake_checkpoint(base: Path, experiment_id: str) -> ScreeningExperimentCheckpoint:
    from pkg.ts_v3a.config import DEFAULT_CONFIG
    from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT

    inc = base / ".incomplete" / experiment_id
    inc.mkdir(parents=True, exist_ok=True)
    return ScreeningExperimentCheckpoint(
        experiment_id=experiment_id,
        config_hash="deadbeef",
        created_at="2026-01-01T00:00:00Z",
        experiment_dir=inc,
        neural_config=DEFAULT_CONFIG,
        v2_config=V2_DEFAULT,
        architectures=(ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,),
        seeds=(41,),
        origins=(140401,),
        product_universe_id="cli_smoke",
        product_universe_hash=None,
        min_successful_seeds=1,
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
            "pkg.ts_v3a.screen.backtest_product_architectures"
        ) as mock_bt, patch(
            "pkg.ts_v3a.screen.begin_screening_experiment"
        ) as mock_begin:
            code = run_screen(args)
        self.assertEqual(code, 0)
        mock_bt.assert_not_called()
        mock_begin.assert_not_called()
        out = buf.getvalue()
        self.assertIn("config_hash=", out)
        self.assertIn("dry_run=1", out)
        self.assertNotIn("selected architecture", out.lower())
        self.assertNotIn("winner", out.lower())


class TestStubbedE2E(unittest.TestCase):
    def test_stubbed_run_checkpoints_per_product_and_finalizes(self):
        base = Path(tempfile.mkdtemp())
        stub = _stub_result("P1")
        sales = pd.DataFrame(
            {
                "product": ["P1", "P1"],
                "date": [140301, 140302],
                "sales": [1.0, 2.0],
            }
        )
        eid = "20260101T000000Z_screencli"
        checkpoint = _fake_checkpoint(base, eid)
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
                eid,
                "--min-successful-seeds",
                "1",
            ]
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf), patch(
            "pkg.ts_v3a.screen.load_screening_sales", return_value=sales
        ), patch(
            "pkg.ts_v3a.screen.begin_screening_experiment", return_value=checkpoint
        ), patch(
            "pkg.ts_v3a.screen.backtest_product_architectures", return_value=stub
        ) as mock_bt, patch(
            "pkg.ts_v3a.screen.write_screening_checkpoint"
        ) as mock_ckpt, patch(
            "pkg.ts_v3a.screen.finalize_screening_experiment",
            return_value=base / eid,
        ) as mock_final, patch(
            "pkg.ts_v3a.screen.assemble_neural_backtest_result", return_value=stub
        ):
            code = run_screen(args)
        self.assertEqual(code, 0)
        mock_bt.assert_called_once()
        mock_ckpt.assert_called()
        self.assertEqual(mock_ckpt.call_args.kwargs.get("completed_products"), ["P1"])
        mock_final.assert_called_once()
        out = buf.getvalue()
        self.assertIn("checkpoint_saved", out)
        self.assertIn("seed_oof_prediction_rows=1", out)
        self.assertIn("Architecture selection is not performed", out)
        self.assertNotIn("winner", out.lower())

    def test_resume_skips_completed_products(self):
        base = Path(tempfile.mkdtemp())
        sales = pd.DataFrame(
            {
                "product": ["P1", "P1", "P2", "P2"],
                "date": [140301, 140302, 140301, 140302],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        eid = "20260101T000000Z_resumecli"
        checkpoint = _fake_checkpoint(base, eid)
        prior = _stub_result("P1")
        next_partial = _stub_result("P2")
        parser = build_parser()
        args = parser.parse_args(
            [
                "--products",
                "P1,P2",
                "--origins",
                "140401",
                "--architectures",
                "a1",
                "--seeds",
                "41",
                "--output",
                str(base),
                "--experiment-id",
                eid,
                "--min-successful-seeds",
                "1",
                "--resume",
            ]
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf), patch(
            "pkg.ts_v3a.screen.load_screening_sales", return_value=sales
        ), patch(
            "pkg.ts_v3a.screen.begin_screening_experiment", return_value=checkpoint
        ) as mock_begin, patch(
            "pkg.ts_v3a.screen.list_completed_products", return_value=["P1"]
        ), patch(
            "pkg.ts_v3a.screen.load_incomplete_screening_result", return_value=prior
        ), patch(
            "pkg.ts_v3a.screen.backtest_product_architectures",
            return_value=next_partial,
        ) as mock_bt, patch(
            "pkg.ts_v3a.screen.write_screening_checkpoint"
        ) as mock_ckpt, patch(
            "pkg.ts_v3a.screen.finalize_screening_experiment",
            return_value=base / eid,
        ), patch(
            "pkg.ts_v3a.screen.assemble_neural_backtest_result",
            return_value=_stub_result("P2"),
        ):
            code = run_screen(args)
        self.assertEqual(code, 0)
        self.assertTrue(mock_begin.call_args.kwargs.get("resume"))
        mock_bt.assert_called_once()
        self.assertEqual(mock_bt.call_args.args[1], "P2")
        self.assertEqual(
            mock_ckpt.call_args.kwargs.get("completed_products"), ["P1", "P2"]
        )
        out = buf.getvalue()
        self.assertIn("resumed_products=['P1']", out)
        self.assertIn("already_done=1", out)

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
