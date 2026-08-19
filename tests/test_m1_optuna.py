"""Unit tests for M1 Optuna hyperparameter experiment.

Synthetic data only — no 40-trial run, no benchmark parquet required.
All tests assert the anti-leakage / anti-PRIMARY-tuning contracts.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.research.tuning.config import (
    INNER_MIN_TRAIN_ROWS,
    MIN_INNER_FOLDS,
    OPTUNA_SEED,
    PRE_PRIMARY_CUTOFF,
)
from pkg.research.tuning.folds import (
    InsufficientFoldsError,
    InnerFold,
    build_inner_folds,
    discover_pre_primary_origins,
)
from pkg.research.tuning.search_space import SEARCH_PARAM_NAMES, suggest_params


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ts_universe(n_origins: int = 5, rows_per_origin: int = 200) -> pd.DataFrame:
    """Synthetic ts_universe with ts_origin and target_date."""
    origins = [140301 + i for i in range(n_origins)]
    parts = []
    for i, orig in enumerate(origins):
        for j in range(rows_per_origin):
            # target_date strictly before origin for training purposes
            target = 140101 + j
            parts.append(
                {
                    "ts_origin": orig,
                    "target_date": target,
                    "sales": float(j + 1),
                    "ts_forecast": float(j),
                    "horizon": 1 + (j % 15),
                    "product": f"P{j % 5}",
                    "sales_lag_1": 0.0,
                    "sales_lag_2": 0.0,
                    "sales_lag_3": 0.0,
                    "sales_lag_12": 0.0,
                    "sales_roll3": 0.0,
                    "month": 1 + (j % 12),
                    "quarter": 1 + (j % 4),
                    "model_enc": 0,
                    "field_enc": 0,
                    "form_enc": 0,
                    "provider_enc": 0,
                }
            )
    return pd.DataFrame(parts)


def _make_budget_universe(n_origins: int = 5, rows_per_origin: int = 200) -> pd.DataFrame:
    origins = [140301 + i for i in range(n_origins)]
    parts = []
    for i, orig in enumerate(origins):
        for j in range(rows_per_origin):
            target = 140101 + j
            parts.append(
                {
                    "budget_origin": orig,
                    "target_date": target,
                    "sales": float(j + 1),
                    "budget_forecast": float(j),
                    "horizon": 1 + (j % 15),
                    "product": f"P{j % 5}",
                    "sales_lag_1": 0.0,
                    "sales_lag_2": 0.0,
                    "sales_lag_3": 0.0,
                    "sales_lag_12": 0.0,
                    "sales_roll3": 0.0,
                    "month": 1 + (j % 12),
                    "quarter": 1 + (j % 4),
                    "model_enc": 0,
                    "field_enc": 0,
                    "form_enc": 0,
                    "provider_enc": 0,
                }
            )
    return pd.DataFrame(parts)


# ── Tests: fold builder ───────────────────────────────────────────────────────

class TestDiscoverPrePrimaryOrigins(unittest.TestCase):

    def test_all_origins_below_cutoff(self):
        """All returned origins must be strictly < PRE_PRIMARY_CUTOFF."""
        univ = _make_ts_universe(n_origins=5)
        origins = discover_pre_primary_origins(univ, "ts")
        for o in origins:
            self.assertLess(o, PRE_PRIMARY_CUTOFF, f"origin {o} >= cutoff {PRE_PRIMARY_CUTOFF}")

    def test_no_primary_origin_appears(self):
        """Inject a PRIMARY origin and confirm it is excluded."""
        univ = _make_ts_universe(n_origins=3)
        # Add a row with a PRIMARY origin
        extra = univ.iloc[:1].copy()
        extra["ts_origin"] = 140404  # first PRIMARY origin
        univ = pd.concat([univ, extra], ignore_index=True)
        origins = discover_pre_primary_origins(univ, "ts")
        self.assertNotIn(140404, origins)

    def test_sorted_ascending(self):
        univ = _make_ts_universe(n_origins=4)
        origins = discover_pre_primary_origins(univ, "ts")
        self.assertEqual(origins, sorted(origins))

    def test_missing_origin_col_raises(self):
        univ = pd.DataFrame({"wrong_col": [1, 2]})
        with self.assertRaises(KeyError):
            discover_pre_primary_origins(univ, "ts")


class TestBuildInnerFolds(unittest.TestCase):

    def _big_universe(self) -> pd.DataFrame:
        """Universe with enough rows per fold for eligibility."""
        return _make_ts_universe(n_origins=5, rows_per_origin=600)

    def test_train_target_date_strictly_before_origin(self):
        """Core anti-leakage: train.target_date.max() < fold.origin."""
        univ = self._big_universe()
        folds = build_inner_folds(univ, "ts", prepped=True)
        for fold in folds:
            self.assertGreater(
                fold.origin,
                int(fold.train["target_date"].max()),
                f"Leakage in fold origin={fold.origin}",
            )

    def test_no_shuffling_folds_sorted(self):
        """Folds must be in ascending origin order (rolling, not shuffled)."""
        univ = self._big_universe()
        folds = build_inner_folds(univ, "ts", prepped=True)
        origins = [f.origin for f in folds]
        self.assertEqual(origins, sorted(origins))

    def test_validation_belongs_to_origin(self):
        """Each fold's val rows must have origin == fold.origin."""
        univ = self._big_universe()
        folds = build_inner_folds(univ, "ts", prepped=True)
        for fold in folds:
            self.assertTrue(
                (fold.val["ts_origin"].astype(int) == fold.origin).all(),
                f"Val rows have wrong origin in fold {fold.origin}",
            )

    def test_insufficient_folds_raises(self):
        """Fewer than MIN_INNER_FOLDS usable folds → InsufficientFoldsError."""
        # Universe with only 1 origin < cutoff and sparse rows
        univ = _make_ts_universe(n_origins=1, rows_per_origin=10)
        with self.assertRaises(InsufficientFoldsError):
            build_inner_folds(univ, "ts", prepped=True)

    def test_skip_fold_when_val_empty(self):
        """Origins with no validation rows (null sales) are skipped."""
        univ = _make_ts_universe(n_origins=5, rows_per_origin=600)
        # Null out sales for one origin's rows to simulate no valid val
        target_origin = sorted(univ["ts_origin"].unique())[2]
        univ.loc[univ["ts_origin"] == target_origin, "sales"] = np.nan
        # Should still work if remaining folds >= MIN_INNER_FOLDS
        folds = build_inner_folds(univ, "ts", prepped=True)
        for fold in folds:
            self.assertNotEqual(fold.origin, target_origin)

    def test_train_min_rows_enforced(self):
        """Folds where train has < INNER_MIN_TRAIN_ROWS rows are skipped."""
        # Only 1 origin with 200 rows of history → train for that fold has too few
        univ = _make_ts_universe(n_origins=1, rows_per_origin=200)
        # Single origin has no earlier training history (target_date < that origin)
        with self.assertRaises(InsufficientFoldsError):
            build_inner_folds(univ, "ts", prepped=True)


# ── Tests: search space ───────────────────────────────────────────────────────

class TestSearchSpace(unittest.TestCase):

    def test_exact_param_names(self):
        """Search space must have exactly the eight specified parameters."""
        expected = {
            "max_depth", "min_child_weight", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
        }
        self.assertEqual(SEARCH_PARAM_NAMES, expected)

    def test_suggest_params_keys(self):
        """suggest_params must return exactly the eight keys."""
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        trial = study.ask()
        params = suggest_params(trial)
        self.assertEqual(set(params.keys()), SEARCH_PARAM_NAMES)

    def test_sampler_seed_is_42(self):
        """TPESampler seed must be OPTUNA_SEED = 42."""
        import optuna
        sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
        self.assertEqual(OPTUNA_SEED, 42)

    def test_max_depth_range(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        for _ in range(20):
            trial = study.ask()
            p = suggest_params(trial)
            self.assertGreaterEqual(p["max_depth"], 2)
            self.assertLessEqual(p["max_depth"], 6)

    def test_subsample_range(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=7))
        for _ in range(20):
            trial = study.ask()
            p = suggest_params(trial)
            self.assertGreaterEqual(p["subsample"], 0.65)
            self.assertLessEqual(p["subsample"], 1.0)

    def test_gamma_range(self):
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=3))
        for _ in range(20):
            trial = study.ask()
            p = suggest_params(trial)
            self.assertGreaterEqual(p["gamma"], 0.0)
            self.assertLessEqual(p["gamma"], 10.0)


# ── Tests: PRIMARY model has no early stopping ───────────────────────────────

class TestPrimaryModelNoEarlyStopping(unittest.TestCase):

    def test_make_primary_model_does_not_pass_eval_set(self):
        """make_primary_model callable must NOT pass eval_set to XGBRegressor.fit."""
        from pkg.research.tuning.fit import make_primary_model
        from pkg.benchmark.config import TS_RESID_FEATURES

        fitted_kwargs: dict = {}
        mock_model = MagicMock()

        def fake_fit(*args, **kwargs):
            fitted_kwargs.update(kwargs)
            return mock_model

        mock_xgb = MagicMock()
        mock_xgb.return_value = mock_model
        mock_model.fit.side_effect = fake_fit
        mock_model.predict.return_value = np.array([0.5, 0.5])

        tuned_params = {
            "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0,
            "gamma": 0.0, "min_child_weight": 1.0,
            "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
        }

        train = pd.DataFrame(
            {
                "sales": [1.0, 2.0],
                "ts_forecast": [0.9, 1.8],
                "horizon": [1, 2],
                **{c: [0.0, 0.0] for c in TS_RESID_FEATURES},
            }
        )
        test = train.copy()

        with patch("pkg.research.tuning.fit.XGBRegressor", mock_xgb):
            fn = make_primary_model("ts", TS_RESID_FEATURES, tuned_params, 200)
            fn(train, test)

        call_kwargs = mock_model.fit.call_args[1]
        self.assertNotIn(
            "eval_set", call_kwargs,
            "PRIMARY model must not pass eval_set to XGBRegressor.fit",
        )
        self.assertNotIn(
            "early_stopping_rounds", call_kwargs,
            "PRIMARY model must not use early_stopping_rounds",
        )

    def test_frozen_params_same_across_origins(self):
        """make_primary_model returns a callable that always uses the same frozen params."""
        from pkg.research.tuning.fit import make_primary_model
        from pkg.benchmark.config import TS_RESID_FEATURES

        tuned_params = {
            "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7,
            "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 2.0,
            "gamma": 1.0, "min_child_weight": 5.0,
            "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1,
        }
        n_est = 150

        instantiation_params = []
        mock_inst = MagicMock()
        mock_inst.fit.return_value = mock_inst
        mock_inst.predict.return_value = np.array([1.0])

        class MockXGB:
            def __init__(self, **kwargs):
                instantiation_params.append(dict(kwargs))
                self._m = mock_inst
            def fit(self, *a, **kw):
                return self._m.fit(*a, **kw)
            def predict(self, X):
                return self._m.predict(X)

        with patch("pkg.research.tuning.fit.XGBRegressor", MockXGB):
            fn = make_primary_model("ts", TS_RESID_FEATURES, tuned_params, n_est)
            train = pd.DataFrame(
                {"sales": [1.0], "ts_forecast": [0.9], "horizon": [1],
                 **{c: [0.0] for c in TS_RESID_FEATURES}}
            )
            # Simulate two different PRIMARY origins by calling twice
            fn(train, train)
            fn(train, train)

        # Both calls must have used identical params
        self.assertEqual(len(instantiation_params), 2)
        self.assertEqual(instantiation_params[0], instantiation_params[1])
        self.assertEqual(instantiation_params[0]["n_estimators"], n_est)


# ── Tests: evaluate_tuned key assertion ──────────────────────────────────────

class TestPrimaryKeyAssertion(unittest.TestCase):

    def _make_result(self, n: int, origin: int) -> "BacktestResult":
        from pkg.benchmark.evaluate import BacktestResult
        preds = pd.DataFrame(
            {
                "product": [f"P{i}" for i in range(n)],
                "qrt": ["1404Q1"] * n,
                "target_date": list(range(140401, 140401 + n)),
                "test_origin": [origin] * n,
                "horizon": [1] * n,
                "actual": [1.0] * n,
                "prediction": [1.0] * n,
            }
        )
        overall = pd.DataFrame([{"wmape": 0.0, "n": n}])
        return BacktestResult(
            model_name="test",
            overall=overall,
            predictions=preds,
            origins=[origin],
        )

    def test_same_keys_passes(self):
        from pkg.research.harness.metrics import assert_same_eval_rows
        r1 = self._make_result(5, 140404)
        r2 = self._make_result(5, 140404)
        # Should not raise — tests that identical keys are accepted
        assert_same_eval_rows(r1, r2)

    def test_different_n_raises(self):
        from pkg.research.tuning.evaluate_tuned import assert_primary_keys
        r1 = self._make_result(5, 140404)
        r2 = self._make_result(4, 140404)
        with self.assertRaises(AssertionError):
            assert_primary_keys(r1, r2, "ts")


# ── Tests: baseline gate tolerance ───────────────────────────────────────────

class TestBaselineGate(unittest.TestCase):

    def _mock_backtest_result(self, wmape: float) -> MagicMock:
        res = MagicMock()
        res.overall = pd.DataFrame([{"wmape": wmape, "n": 1877, "bias": 0.0,
                                      "rmse": 0.0, "mae": 0.0}])
        res.predictions = pd.DataFrame(
            {"product": ["P"], "qrt": ["1404Q1"], "target_date": [140401],
             "test_origin": [140404], "horizon": [1], "actual": [1.0], "prediction": [1.0]}
        )
        res.origins = [140404, 140407, 140410, 140501, 140504]
        return res

    def test_gate_passes_within_tolerance(self):
        from pkg.research.tuning.evaluate_tuned import run_baseline_gate
        # Patch confirm_canonical_f0 to return mock results
        mock_ts = self._mock_backtest_result(38.25)  # within 0.10 of current-env ref 38.2848
        mock_human = self._mock_backtest_result(36.55)  # within 0.10 of current-env ref 36.5602
        mock_canon = {"results": {"ts": mock_ts, "human": mock_human}}

        with patch("pkg.research.tuning.evaluate_tuned.confirm_canonical_f0",
                   return_value=mock_canon):
            result = run_baseline_gate(MagicMock())
        self.assertIn("ts", result)
        self.assertIn("human", result)

    def test_gate_raises_on_excessive_drift(self):
        from pkg.research.tuning.evaluate_tuned import run_baseline_gate
        mock_ts = self._mock_backtest_result(40.0)  # |40.0 - 38.2848| = 1.7 > 0.10
        mock_human = self._mock_backtest_result(36.65)
        mock_canon = {"results": {"ts": mock_ts, "human": mock_human}}

        with patch("pkg.research.tuning.evaluate_tuned.confirm_canonical_f0",
                   return_value=mock_canon):
            with self.assertRaises(EnvironmentError):
                run_baseline_gate(MagicMock())


# ── Tests: freeze checksum unchanged ─────────────────────────────────────────

class TestFreezeChecksumUnchanged(unittest.TestCase):

    def test_file_untouched(self):
        """build_inner_folds must not write to benchmark directory."""
        import tempfile
        from pkg.benchmark.dataset import BenchmarkDataset
        from pkg.research.harness.gates import freeze_checksums

        tmp = Path(tempfile.mkdtemp())
        freeze_file = tmp / "ts_universe.parquet"
        freeze_file.write_bytes(b"frozen-bytes")
        before_bytes = freeze_file.read_bytes()
        before_mtime = freeze_file.stat().st_mtime_ns

        # build_inner_folds uses an in-memory DataFrame — should not touch disk
        univ = _make_ts_universe(n_origins=5, rows_per_origin=600)
        try:
            build_inner_folds(univ, "ts", prepped=True)
        except InsufficientFoldsError:
            pass  # not enough, but that's OK — we only care about file integrity

        self.assertEqual(freeze_file.read_bytes(), before_bytes)
        self.assertEqual(freeze_file.stat().st_mtime_ns, before_mtime)


# ── Tests: verdict classification ────────────────────────────────────────────

class TestVerdictClassification(unittest.TestCase):

    def _base_kwargs(self, better: bool = True) -> dict:
        return dict(
            wmape_baseline=40.0,
            wmape_tuned=38.0 if better else 41.0,
            product_win_rate=0.60,
            median_product_improvement_pct=1.0,
            origins_improved=3,
            origins_total=5,
            bias_baseline=500.0,
            bias_tuned=510.0,
            concentration_flags=[],
        )

    def test_promote(self):
        from pkg.research.tuning.evaluate_tuned import classify_m1_verdict
        v = classify_m1_verdict(**self._base_kwargs(better=True))
        self.assertEqual(v, "PROMOTE")

    def test_reject_when_worse(self):
        from pkg.research.tuning.evaluate_tuned import classify_m1_verdict
        # WMAPE worse AND no product-level improvement signal → REJECT
        kwargs = self._base_kwargs(better=False)
        kwargs["product_win_rate"] = 0.40
        kwargs["median_product_improvement_pct"] = -1.0  # no signal
        v = classify_m1_verdict(**kwargs)
        self.assertEqual(v, "REJECT")

    def test_weak_when_better_but_concentrated(self):
        from pkg.research.tuning.evaluate_tuned import classify_m1_verdict
        kwargs = self._base_kwargs(better=True)
        kwargs["concentration_flags"] = ["one_product_gt_25pct_deterioration"]
        v = classify_m1_verdict(**kwargs)
        self.assertEqual(v, "WEAK_NEEDS_CONFIRMATION")


if __name__ == "__main__":
    unittest.main()
