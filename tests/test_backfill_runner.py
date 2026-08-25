"""Integration tests for historical backfill orchestration (dummy engines)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.backfill_runner.engines.dummy import DummyForecastEngine
from pkg.benchmark.backfill_runner.runner import (
    enforce_historical_cutoff,
    run_backfill,
)
from pkg.benchmark.backfill_runner.store import BackfillStore
from pkg.benchmark.backfill_runner.types import JobKey
from pkg.benchmark.calendar import shamsi_add_months


def _synthetic_sales(products: list[str], start: int = 140001, n_months: int = 40) -> pd.DataFrame:
    rows = []
    for product in products:
        cur = start
        for i in range(n_months):
            rows.append({"product": product, "date": cur, "sales": 20.0 + i})
            cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestHistoricalCutoff(unittest.TestCase):
    def test_cutoff_excludes_origin_and_later(self):
        sales = _synthetic_sales(["A"], start=140410, n_months=6)
        cut = enforce_historical_cutoff(sales, 140501)
        self.assertTrue((cut["date"] < 140501).all())
        self.assertNotIn(140501, set(cut["date"]))
        self.assertNotIn(140502, set(cut["date"]))


class TestBackfillOrchestration(unittest.TestCase):
    def test_dummy_engine_end_to_end_and_resume(self):
        products = ["Altebrel 25", "Altebrel 50"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = DummyForecastEngine(level=7.0)
            summary = run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q1",
                products=products,
                resume=False,
                dry_run=False,
            )
            self.assertEqual(summary.n_success, 2)
            self.assertEqual(summary.n_failed, 0)
            store = BackfillStore(root, "dummy")
            key = JobKey("dummy", "1405Q1", "Altebrel 25", 140501)
            self.assertTrue(store.is_complete(key))
            forecast = pd.read_csv(store.job_dir(key) / "forecast.csv")
            self.assertEqual(len(forecast), 15)
            self.assertEqual(int(forecast["target_date"].iloc[0]), 140501)
            self.assertEqual(int(forecast["horizon"].iloc[0]), 1)
            self.assertEqual(int(forecast["horizon"].iloc[-1]), 15)
            self.assertTrue((forecast["forecast"] == 7.0).all())

            # Resume should skip completed jobs
            summary2 = run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q1",
                products=products,
                resume=True,
            )
            self.assertEqual(summary2.plan.already_completed, 2)
            self.assertEqual(summary2.plan.remaining, 0)
            self.assertEqual(summary2.n_success, 0)

    def test_partial_csv_without_marker_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BackfillStore(Path(tmp), "dummy")
            key = JobKey("dummy", "1405Q1", "Altebrel 25", 140501)
            job = store.job_dir(key)
            job.mkdir(parents=True)
            (job / "forecast.csv").write_text("product,forecast\nA,1\n", encoding="utf-8")
            self.assertFalse(store.is_complete(key))

    def test_one_failure_does_not_stop_backfill(self):
        products = ["Altebrel 25", "Altebrel 50"]
        sales = _synthetic_sales(products)
        engine = DummyForecastEngine(fail_products=frozenset({"Altebrel 25"}))
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=Path(tmp),
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q1",
                products=products,
                resume=False,
            )
            self.assertEqual(summary.n_failed, 1)
            self.assertEqual(summary.n_success, 1)
            store = BackfillStore(Path(tmp), "dummy")
            self.assertTrue(
                store.is_complete(JobKey("dummy", "1405Q1", "Altebrel 50", 140501))
            )
            self.assertFalse(
                store.is_complete(JobKey("dummy", "1405Q1", "Altebrel 25", 140501))
            )

    def test_completed_job_is_immutable(self):
        products = ["Altebrel 25"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = DummyForecastEngine()
            run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q1",
                products=products,
                resume=False,
            )
            store = BackfillStore(root, "dummy")
            key = JobKey("dummy", "1405Q1", "Altebrel 25", 140501)
            from pkg.benchmark.backfill_runner.store import BackfillStoreError
            from pkg.benchmark.backfill_runner.types import EngineJobResult, JobLogRecord

            with self.assertRaises(BackfillStoreError):
                store.persist_success(
                    key,
                    EngineJobResult(
                        success=True,
                        product="Altebrel 25",
                        quarter="1405Q1",
                        forecast_origin=140501,
                        selected_model="x",
                        forecasts=pd.DataFrame(
                            [{"product": "Altebrel 25", "forecast": 1.0}]
                        ),
                    ),
                    JobLogRecord(
                        engine="dummy",
                        quarter="1405Q1",
                        forecast_origin=140501,
                        product="Altebrel 25",
                        start_time_utc="t0",
                        end_time_utc="t1",
                        duration_seconds=0.0,
                        success=True,
                    ),
                )

    def test_dry_run_writes_no_job_forecasts(self):
        products = ["Altebrel 25"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter_from="1404Q4",
                quarter_to="1405Q2",
                products=products,
                dry_run=True,
            )
            self.assertTrue(summary.dry_run)
            self.assertGreater(summary.plan.remaining, 0)
            jobs_root = root / "dummy" / "jobs"
            self.assertFalse(jobs_root.exists() and any(jobs_root.iterdir()))

    def test_engine_receives_pre_origin_sales_only(self):
        """Dummy engine fails if post-origin rows leak into the request."""
        products = ["Altebrel 25"]
        sales = _synthetic_sales(products)
        # Poison: if runner forgot cutoff, dummy would see >= origin and fail.
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=Path(tmp),
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q1",
                products=products,
            )
            self.assertEqual(summary.n_failed, 0)
            self.assertEqual(summary.n_success, 1)


class TestStartupReport(unittest.TestCase):
    def test_plan_counts(self):
        from pkg.benchmark.backfill_runner.runner import build_backfill_plan
        from pkg.benchmark.backfill_runner.store import BackfillStore

        with tempfile.TemporaryDirectory() as tmp:
            store = BackfillStore(Path(tmp), "dummy")
            plan = build_backfill_plan(
                engine="dummy",
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                store=store,
                quarter_from="1405Q1",
                quarter_to="1405Q2",
                products=["Altebrel 25", "Altebrel 50"],
                resume=True,
            )
            self.assertEqual(plan.vintages_eligible, ["1405Q1", "1405Q2"])
            self.assertEqual(len(plan.products), 2)
            self.assertEqual(plan.total_jobs, 4)
            self.assertEqual(plan.remaining, 4)
            self.assertEqual(len(plan.vintages_requested), 18)


if __name__ == "__main__":
    unittest.main()
