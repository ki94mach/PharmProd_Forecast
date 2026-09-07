"""Unit tests for the TS V2 backfill evaluation (read-only analysis package).

Covers the invariants the report depends on: join cardinality, actual
consistency across sources, the no-zero-fill rule, agreement with
``pkg.benchmark.evaluate.wmape``, zero-denominator and negative-actual
behaviour, and the relative-improvement identity.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.evaluate import wmape as benchmark_wmape
from pkg.research.v2_eval.config import MATCH_KEYS, SHIFTED_ORIGINS
from pkg.research.v2_eval.diagnostics import (
    error_concentration,
    forecast_shape_flags,
    product_outcomes,
)
from pkg.research.v2_eval.config import DiagnosticThresholds
from pkg.research.v2_eval.load import BackfillInputs
from pkg.research.v2_eval.match import (
    build_matched_panel,
    fully_actualised_pairs,
    horizon_group,
    validate_matched_panel,
)
from pkg.research.v2_eval.metrics import (
    bias_movement,
    mean_horizon_mae,
    metric_pair_rows,
    product_metrics,
    relative_improvement_pct,
    rmse,
    signed_bias,
    wmape_pct,
)

THRESHOLDS = DiagnosticThresholds()


def _v2_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["job_slug"] = frame["quarter"] + "__" + frame["product_id"]
    if "raw_forecast" not in frame:
        frame["raw_forecast"] = frame["forecast"]
    if "model" not in frame:
        frame["model"] = "naive"
    frame["engine"] = "v2"
    return frame


def _fake_inputs(
    v2: pd.DataFrame,
    legacy: pd.DataFrame,
    sales: pd.DataFrame,
) -> BackfillInputs:
    empty = pd.DataFrame()
    return BackfillInputs(
        v2_forecasts=v2,
        jobs=empty,
        job_logs=empty,
        results=empty,
        manifest={},
        run_meta=None,
        stale_failure_files=0,
        backtest_files=0,
        legacy=legacy,
        sales=sales,
        product_attrs=empty,
        vintages=empty,
        universe=empty,
        input_hashes={},
    )


def _tiny_panel() -> BackfillInputs:
    """Two products x one origin x three horizons, one row deliberately
    unmatched on each side and one target month without an actual."""
    v2 = _v2_frame(
        [
            {"product_id": "A", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140101, "horizon": 1, "forecast": 100.0},
            {"product_id": "A", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140102, "horizon": 2, "forecast": 110.0},
            {"product_id": "A", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140103, "horizon": 3, "forecast": 120.0},
            # No actual exists for this month.
            {"product_id": "A", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140104, "horizon": 4, "forecast": 130.0},
            {"product_id": "B", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140101, "horizon": 1, "forecast": 40.0},
            {"product_id": "B", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140102, "horizon": 2, "forecast": 50.0},
            {"product_id": "B", "quarter": "1401Q1", "forecast_origin": 140101,
             "target_date": 140103, "horizon": 3, "forecast": 60.0},
        ]
    )
    legacy = pd.DataFrame(
        [
            {"product_id": "A", "forecast_origin": 140101, "target_date": 140101,
             "horizon": 1, "legacy_prediction": 150.0, "actual_legacy_panel": 90.0},
            {"product_id": "A", "forecast_origin": 140101, "target_date": 140102,
             "horizon": 2, "legacy_prediction": 160.0, "actual_legacy_panel": 100.0},
            {"product_id": "A", "forecast_origin": 140101, "target_date": 140103,
             "horizon": 3, "legacy_prediction": 170.0, "actual_legacy_panel": 110.0},
            {"product_id": "B", "forecast_origin": 140101, "target_date": 140101,
             "horizon": 1, "legacy_prediction": 30.0, "actual_legacy_panel": 45.0},
            {"product_id": "B", "forecast_origin": 140101, "target_date": 140102,
             "horizon": 2, "legacy_prediction": 35.0, "actual_legacy_panel": 55.0},
            {"product_id": "B", "forecast_origin": 140101, "target_date": 140103,
             "horizon": 3, "legacy_prediction": 45.0, "actual_legacy_panel": 65.0},
            # Legacy-only product; can never match.
            {"product_id": "C", "forecast_origin": 140101, "target_date": 140101,
             "horizon": 1, "legacy_prediction": 10.0, "actual_legacy_panel": 12.0},
        ]
    )
    legacy["legacy_quarter"] = "1401Q1"
    legacy["legacy_model"] = "arima"
    sales = pd.DataFrame(
        [
            {"product_id": "A", "target_date": 140101, "actual": 90.0},
            {"product_id": "A", "target_date": 140102, "actual": 100.0},
            {"product_id": "A", "target_date": 140103, "actual": 110.0},
            {"product_id": "B", "target_date": 140101, "actual": 45.0},
            {"product_id": "B", "target_date": 140102, "actual": 55.0},
            {"product_id": "B", "target_date": 140103, "actual": 65.0},
            {"product_id": "C", "target_date": 140101, "actual": 12.0},
        ]
    )
    return _fake_inputs(v2, legacy, sales)


class TestMatchedPanel(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = build_matched_panel(_tiny_panel())

    def test_join_is_one_to_one_on_the_match_key(self) -> None:
        matched = self.panel.matched
        self.assertEqual(len(matched), 6)
        self.assertEqual(int(matched.duplicated(subset=list(MATCH_KEYS)).sum()), 0)
        self.assertEqual([], validate_matched_panel(self.panel))

    def test_unmatched_rows_are_reported_not_filled(self) -> None:
        # The V2 row without an actual and the legacy-only product stay out of
        # the matched panel and appear in the unmatched tables instead.
        self.assertEqual(len(self.panel.unmatched_v2), 1)
        self.assertEqual(
            self.panel.unmatched_v2.iloc[0]["target_date"], 140104
        )
        self.assertEqual(len(self.panel.unmatched_legacy), 1)
        self.assertEqual(self.panel.unmatched_legacy.iloc[0]["product_id"], "C")
        self.assertEqual(self.panel.coverage_summary["matched_rows"], 6)

    def test_no_zero_fill_anywhere_in_the_matched_panel(self) -> None:
        matched = self.panel.matched
        for column in ("actual", "legacy_prediction", "v2_prediction"):
            self.assertEqual(int(matched[column].isna().sum()), 0)
        # 140104 had no actual, so it must not have been coerced to 0.
        self.assertNotIn(140104, set(matched["target_date"]))

    def test_error_columns_follow_the_stated_conventions(self) -> None:
        row = self.panel.matched.query(
            "product_id == 'A' and target_date == 140101"
        ).iloc[0]
        # Signed error is prediction - actual, so overforecasting is positive.
        self.assertAlmostEqual(row["signed_error_v2"], 100.0 - 90.0)
        self.assertAlmostEqual(row["signed_error_legacy"], 150.0 - 90.0)
        self.assertAlmostEqual(row["absolute_error_v2"], 10.0)
        self.assertAlmostEqual(row["absolute_error_legacy"], 60.0)
        self.assertAlmostEqual(row["absolute_error_reduction"], 50.0)

    def test_actual_disagreement_between_sources_raises(self) -> None:
        data = _tiny_panel()
        data.legacy.loc[0, "actual_legacy_panel"] = 999.0
        with self.assertRaises(AssertionError) as ctx:
            build_matched_panel(data)
        self.assertIn("actual mismatch", str(ctx.exception))

    def test_horizon_groups_partition_1_to_15(self) -> None:
        groups = [horizon_group(h) for h in range(1, 16)]
        self.assertNotIn("other", groups)
        self.assertEqual(groups[0], "h1-3")
        self.assertEqual(groups[-1], "h13-15")

    def test_fully_actualised_pairs_requires_all_fifteen_horizons(self) -> None:
        # The tiny panel has at most 3 horizons per pair, so nothing qualifies.
        self.assertTrue(fully_actualised_pairs(self.panel.matched).empty)


class TestMetricIdentities(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = pd.Series([100.0, 200.0, 50.0, 0.0])
        self.prediction = pd.Series([90.0, 260.0, 55.0, 10.0])

    def test_wmape_matches_the_benchmark_implementation(self) -> None:
        expected = benchmark_wmape(
            self.actual.to_numpy(), self.prediction.to_numpy()
        )
        self.assertAlmostEqual(wmape_pct(self.actual, self.prediction), expected)

    def test_wmape_is_reported_as_a_percent_not_a_ratio(self) -> None:
        # sum|err| = 10 + 60 + 5 + 10 = 85; sum|actual| = 350 -> 24.2857%
        self.assertAlmostEqual(
            wmape_pct(self.actual, self.prediction), 85.0 / 350.0 * 100.0
        )

    def test_wmape_with_a_zero_denominator_is_not_a_number(self) -> None:
        zeros = pd.Series([0.0, 0.0, 0.0])
        value = wmape_pct(zeros, pd.Series([1.0, 2.0, 3.0]))
        self.assertTrue(math.isnan(value))

    def test_negative_actuals_enter_the_denominator_as_magnitudes(self) -> None:
        actual = pd.Series([-100.0, 100.0])
        prediction = pd.Series([0.0, 100.0])
        # sum|actual| = 200 (not 0), sum|err| = 100 -> 50%
        self.assertAlmostEqual(wmape_pct(actual, prediction), 50.0)

    def test_rmse_and_bias_definitions(self) -> None:
        actual = pd.Series([10.0, 20.0])
        prediction = pd.Series([12.0, 26.0])
        self.assertAlmostEqual(rmse(actual, prediction), math.sqrt((4 + 36) / 2))
        self.assertAlmostEqual(signed_bias(actual, prediction), 4.0)

    def test_mean_horizon_mae_weights_horizons_equally(self) -> None:
        frame = pd.DataFrame(
            {
                # Horizon 1 has three rows, horizon 2 has one; equal weighting
                # must give (1 + 100) / 2 = 50.5, not the row mean of 25.75.
                "horizon": [1, 1, 1, 2],
                "absolute_error_v2": [1.0, 1.0, 1.0, 100.0],
            }
        )
        self.assertAlmostEqual(mean_horizon_mae(frame, "absolute_error_v2"), 50.5)

    def test_relative_improvement_identity_and_zero_denominator(self) -> None:
        self.assertAlmostEqual(relative_improvement_pct(40.0, 30.0), 25.0)
        self.assertAlmostEqual(relative_improvement_pct(30.0, 40.0), -100.0 / 3)
        self.assertTrue(math.isnan(relative_improvement_pct(0.0, 5.0)))
        self.assertTrue(math.isnan(relative_improvement_pct(float("nan"), 5.0)))

    def test_bias_is_never_given_a_relative_improvement(self) -> None:
        panel = build_matched_panel(_tiny_panel())
        rows = metric_pair_rows(panel.matched, "overall", "all")
        bias_row = next(r for r in rows if r["metric"] == "signed_bias")
        self.assertTrue(math.isnan(bias_row["relative_improvement_pct"]))
        self.assertIn(bias_row["bias_direction"], {
            "toward_zero", "away_from_zero", "unchanged",
            "toward_zero_sign_flip", "away_from_zero_sign_flip",
        })

    def test_bias_movement_uses_distance_from_zero(self) -> None:
        moved = bias_movement(100.0, -20.0)
        self.assertEqual(moved["bias_abs_change"], -80.0)
        self.assertEqual(moved["bias_direction"], "toward_zero_sign_flip")
        worse = bias_movement(-10.0, -50.0)
        self.assertEqual(worse["bias_direction"], "away_from_zero")

    def test_metric_rows_carry_the_counts_behind_them(self) -> None:
        panel = build_matched_panel(_tiny_panel())
        rows = metric_pair_rows(panel.matched, "overall", "all")
        for row in rows:
            self.assertEqual(row["n"], 6)
            self.assertEqual(row["n_products"], 2)
            self.assertEqual(row["n_origins"], 1)


class TestDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _tiny_panel()
        self.panel = build_matched_panel(self.data)
        self.products = product_metrics(self.panel.matched)

    def test_product_metrics_reconcile_with_the_panel_totals(self) -> None:
        self.assertAlmostEqual(
            self.products["total_absolute_error_reduction"].sum(),
            self.panel.matched["absolute_error_reduction"].sum(),
        )

    def test_product_outcomes_respect_the_tie_band(self) -> None:
        frame = pd.DataFrame(
            {
                "product_id": ["big_win", "flat", "loss", "thin"],
                "n": [10, 10, 10, 1],
                "relative_improvement_pct": [25.0, 0.4, -25.0, 90.0],
            }
        )
        classified, summary = product_outcomes(frame, THRESHOLDS)
        outcomes = dict(zip(classified["product_id"], classified["outcome"]))
        self.assertEqual(outcomes["big_win"], "improved")
        self.assertEqual(outcomes["flat"], "tied")
        self.assertEqual(outcomes["loss"], "regressed")
        # Too few rows to judge, so excluded from the rates entirely.
        self.assertEqual(outcomes["thin"], "insufficient_rows")
        self.assertEqual(summary["n_products_scored"], 3)

    def test_concentration_shares_are_computed_over_gains_only(self) -> None:
        frame = pd.DataFrame(
            {
                "product_id": ["a", "b", "c"],
                "actual_volume": [100.0, 100.0, 100.0],
                "total_absolute_error_reduction": [90.0, 10.0, -40.0],
            }
        )
        result = error_concentration(frame, THRESHOLDS)
        self.assertAlmostEqual(result["total_gain"], 100.0)
        self.assertAlmostEqual(result["total_loss"], 40.0)
        self.assertAlmostEqual(result["net_absolute_error_reduction"], 60.0)
        self.assertAlmostEqual(result["top1_gain_share"], 0.9)
        self.assertIn("one_product_over_25pct_of_gains", result["flags"])

    def test_flat_and_clipped_flags(self) -> None:
        v2 = _v2_frame(
            [
                {"product_id": "F", "quarter": "1401Q1", "forecast_origin": 140101,
                 "target_date": 140101, "horizon": 1, "forecast": 5.0,
                 "raw_forecast": 5.0},
                {"product_id": "F", "quarter": "1401Q1", "forecast_origin": 140101,
                 "target_date": 140102, "horizon": 2, "forecast": 5.0,
                 "raw_forecast": -2.0},
            ]
        )
        v2["actual"] = np.nan
        sales = pd.DataFrame(
            [{"product_id": "F", "target_date": 140012, "actual": 4.0}]
        )
        jobs, summary = forecast_shape_flags(v2, sales, THRESHOLDS)
        self.assertEqual(summary["n_flat_jobs"], 1)
        self.assertEqual(summary["n_clipped_rows"], 1)
        self.assertEqual(summary["n_negative_delivered_rows"], 0)
        self.assertFalse(bool(jobs.iloc[0]["is_extreme"]))

    def test_extreme_flag_uses_pre_origin_history_only(self) -> None:
        v2 = _v2_frame(
            [
                {"product_id": "X", "quarter": "1401Q1", "forecast_origin": 140101,
                 "target_date": 140101, "horizon": 1, "forecast": 1000.0},
                {"product_id": "X", "quarter": "1401Q1", "forecast_origin": 140101,
                 "target_date": 140102, "horizon": 2, "forecast": 900.0},
            ]
        )
        sales = pd.DataFrame(
            [
                {"product_id": "X", "target_date": 140012, "actual": 10.0},
                # Post-origin month must not raise the history ceiling.
                {"product_id": "X", "target_date": 140103, "actual": 5000.0},
            ]
        )
        jobs, summary = forecast_shape_flags(v2, sales, THRESHOLDS)
        self.assertEqual(jobs.iloc[0]["history_max_pre_origin"], 10.0)
        self.assertEqual(summary["n_extreme_jobs"], 1)


class TestConfigContracts(unittest.TestCase):
    def test_match_keys_are_the_documented_four(self) -> None:
        self.assertEqual(
            MATCH_KEYS, ("product_id", "forecast_origin", "target_date", "horizon")
        )

    def test_shifted_origins_cover_the_known_legacy_drift(self) -> None:
        for origin in (140301, 140304, 140306):
            self.assertIn(origin, SHIFTED_ORIGINS)


if __name__ == "__main__":
    unittest.main()
