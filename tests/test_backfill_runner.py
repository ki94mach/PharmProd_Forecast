"""Integration tests for durable historical backfill orchestration."""
from __future__ import annotations

import json
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
    build_config_payload,
    enforce_historical_cutoff,
    experiment_dir,
    run_backfill,
)
from pkg.benchmark.backfill_runner.state import (
    JOB_FAILED,
    JOB_PENDING,
    JOB_RUNNING,
    JOB_SUCCESS,
    JobIdentity,
    JobStateStore,
    RunLock,
    RunLockError,
    compute_config_hash,
    make_experiment_id,
)
from pkg.benchmark.backfill_runner.store import BackfillStore
from pkg.benchmark.calendar import shamsi_add_months


def _synthetic_sales(products: list[str], start: int = 140001, n_months: int = 40) -> pd.DataFrame:
    rows = []
    for product in products:
        cur = start
        for i in range(n_months):
            rows.append({"product": product, "date": cur, "sales": 20.0 + i})
            cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


def _exp_id(engine: str = "dummy") -> str:
    return make_experiment_id("ts_backfill_1401Q1_1405Q2", "mvp_products", engine)


class TestHistoricalCutoff(unittest.TestCase):
    def test_cutoff_excludes_origin_and_later(self):
        sales = _synthetic_sales(["A"], start=140410, n_months=6)
        cut = enforce_historical_cutoff(sales, 140501)
        self.assertTrue((cut["date"] < 140501).all())
        self.assertNotIn(140501, set(cut["date"]))


class TestBackfillDurableState(unittest.TestCase):
    def test_success_resume_skips_and_force_recomputes(self):
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
                quarter="1405Q1",
                products=products,
                resume=False,
            )
            self.assertEqual(summary.n_success, 2)
            state = JobStateStore(experiment_dir(root, _exp_id()))
            self.assertEqual(state.status_counts(_exp_id())[JOB_SUCCESS], 2)

            summary2 = run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
                resume=True,
            )
            self.assertEqual(summary2.plan.remaining, 0)
            self.assertEqual(summary2.n_success, 0)

            summary3 = run_backfill(
                engine=engine,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                product="Altebrel 25",
                force_job=True,
                resume=True,
            )
            self.assertEqual(summary3.n_success, 1)

    def test_failed_model_and_retry_failed(self):
        products = ["Altebrel 25", "Altebrel 50"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failing = DummyForecastEngine(fail_products=frozenset({"Altebrel 25"}))
            summary = run_backfill(
                engine=failing,
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
            )
            self.assertEqual(summary.n_failed, 1)
            self.assertEqual(summary.n_success, 1)

            summary2 = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
                resume=True,
                retry_failed=False,
            )
            self.assertEqual(summary2.plan.remaining, 0)

            summary3 = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(summary3.n_success, 1)
            state = JobStateStore(experiment_dir(root, _exp_id()))
            self.assertEqual(state.status_counts(_exp_id())[JOB_SUCCESS], 2)
            self.assertEqual(state.status_counts(_exp_id())[JOB_FAILED], 0)

    def test_stale_running_reclaimed_on_resume(self):
        products = ["Altebrel 25"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_id = _exp_id()
            exp = experiment_dir(root, exp_id)
            state = JobStateStore(exp)
            config = build_config_payload(
                engine="dummy",
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
            )
            config_hash = compute_config_hash(config)
            state.upsert_experiment(
                experiment_id=exp_id,
                vintage_manifest="ts_backfill_1401Q1_1405Q2",
                universe_manifest="mvp_products",
                engine_version="dummy",
                config=config,
                config_hash=config_hash,
                git_commit="abc",
            )
            identity = JobIdentity(
                experiment_id=exp_id,
                engine_version="dummy",
                config_hash=config_hash,
                quarter="1405Q1",
                forecast_origin=140501,
                product_id="Altebrel 25",
            )
            state.ensure_job(identity, git_commit="abc")
            state.mark_running(identity, git_commit="abc")
            self.assertEqual(state.get_job(identity.job_id).status, JOB_RUNNING)

            summary = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                experiment_id=exp_id,
                quarter="1405Q1",
                products=products,
                resume=True,
            )
            self.assertEqual(summary.n_reclaimed_stale, 1)
            self.assertEqual(summary.n_success, 1)
            self.assertEqual(
                JobStateStore(exp).get_job(identity.job_id).status, JOB_SUCCESS
            )

    def test_interruption_leaves_running_then_resume_completes(self):
        class ExplodingEngine(DummyForecastEngine):
            def forecast_product(self, request):
                raise RuntimeError("simulated process kill")

        products = ["Altebrel 25"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_backfill(
                engine=ExplodingEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
            )
            self.assertEqual(summary.n_failed, 1)
            exp = experiment_dir(root, _exp_id())
            state = JobStateStore(exp)
            job = state.list_jobs(_exp_id())[0]
            self.assertEqual(job.status, JOB_FAILED)

            # Simulate hard kill: RUNNING never finalized to FAILED
            state.reset_for_force(job.identity, git_commit="kill")
            state.mark_running(job.identity, git_commit="kill")
            self.assertEqual(state.get_job(job.identity.job_id).status, JOB_RUNNING)

            summary2 = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
                resume=True,
            )
            self.assertEqual(summary2.n_reclaimed_stale, 1)
            self.assertEqual(summary2.n_success, 1)

    def test_duplicate_runner_invocation_blocked_by_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = experiment_dir(root, _exp_id())
            lock = RunLock(exp)
            lock.acquire()
            try:
                with self.assertRaises(RunLockError):
                    run_backfill(
                        engine=DummyForecastEngine(),
                        vintage_name="ts_backfill_1401Q1_1405Q2",
                        universe_name="mvp_products",
                        output_root=root,
                        sales=_synthetic_sales(["Altebrel 25"]),
                        quarter="1405Q1",
                        products=["Altebrel 25"],
                        acquire_lock=True,
                    )
            finally:
                lock.release()

    def test_partial_csv_without_marker_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = experiment_dir(root, _exp_id())
            store = BackfillStore(exp, "dummy")
            identity = JobIdentity(
                experiment_id=_exp_id(),
                engine_version="dummy",
                config_hash="x",
                quarter="1405Q1",
                forecast_origin=140501,
                product_id="Altebrel 25",
            )
            job = store.job_dir(identity)
            job.mkdir(parents=True)
            (job / "forecast.csv").write_text("product,forecast\nA,1\n", encoding="utf-8")
            self.assertFalse(store.has_complete_artifacts(identity))

    def test_dry_run_writes_no_forecast_artifacts(self):
        products = ["Altebrel 25"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_backfill(
                engine=DummyForecastEngine(),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=_synthetic_sales(products),
                quarter_from="1405Q1",
                quarter_to="1405Q2",
                products=products,
                dry_run=True,
            )
            self.assertTrue(summary.dry_run)
            self.assertGreater(summary.plan.remaining, 0)
            self.assertEqual(summary.plan.remaining, summary.plan.total_jobs)
            self.assertEqual(summary.workers, 1)
            exp = experiment_dir(root, _exp_id())
            artifacts = exp / "artifacts"
            self.assertFalse(artifacts.exists() and any(artifacts.rglob("forecast.csv")))

    def test_parallel_workers_complete_all_jobs(self):
        products = ["Altebrel 25", "Altebrel 50"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_backfill(
                engine=DummyForecastEngine(level=3.0),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter_from="1405Q1",
                quarter_to="1405Q2",
                products=products,
                workers=2,
            )
            self.assertEqual(summary.n_success, 4)
            self.assertEqual(summary.n_failed, 0)
            self.assertEqual(summary.workers, 2)
            self.assertIsNotNone(summary.runtime_seconds)
            self.assertIn("requested_workers", summary.thread_config)
            exp = experiment_dir(root, _exp_id())
            meta_path = exp / "run_meta.json"
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["requested_workers"], 2)
            self.assertEqual(meta["n_success"], 4)
            state = JobStateStore(exp)
            self.assertEqual(state.status_counts(_exp_id())[JOB_SUCCESS], 4)

    def test_try_claim_job_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = experiment_dir(Path(tmp), _exp_id())
            state = JobStateStore(exp)
            identity = JobIdentity(
                experiment_id=_exp_id(),
                engine_version="dummy",
                config_hash="abc",
                quarter="1405Q1",
                forecast_origin=140501,
                product_id="Altebrel 25",
            )
            state.ensure_job(identity, git_commit="t")
            first = state.try_claim_job(identity, git_commit="t")
            second = state.try_claim_job(identity, git_commit="t")
            self.assertIsNotNone(first)
            self.assertEqual(first.status, JOB_RUNNING)
            self.assertIsNone(second)

    def test_parallel_failure_isolation(self):
        products = ["Altebrel 25", "Altebrel 50"]
        sales = _synthetic_sales(products)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_backfill(
                engine=DummyForecastEngine(fail_products=frozenset({"Altebrel 25"})),
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                output_root=root,
                sales=sales,
                quarter="1405Q1",
                products=products,
                workers=2,
            )
            self.assertEqual(summary.n_failed, 1)
            self.assertEqual(summary.n_success, 1)
            state = JobStateStore(experiment_dir(root, _exp_id()))
            counts = state.status_counts(_exp_id())
            self.assertEqual(counts[JOB_SUCCESS], 1)
            self.assertEqual(counts[JOB_FAILED], 1)


class TestStartupReport(unittest.TestCase):
    def test_plan_counts(self):
        from pkg.benchmark.backfill_runner.runner import build_backfill_plan

        with tempfile.TemporaryDirectory() as tmp:
            exp_id = _exp_id()
            exp = experiment_dir(Path(tmp), exp_id)
            state = JobStateStore(exp)
            config = build_config_payload(
                engine="dummy",
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
            )
            ch = compute_config_hash(config)
            state.upsert_experiment(
                experiment_id=exp_id,
                vintage_manifest="ts_backfill_1401Q1_1405Q2",
                universe_manifest="mvp_products",
                engine_version="dummy",
                config=config,
                config_hash=ch,
                git_commit="t",
            )
            plan = build_backfill_plan(
                engine="dummy",
                vintage_name="ts_backfill_1401Q1_1405Q2",
                universe_name="mvp_products",
                state=state,
                experiment_id=exp_id,
                config_hash=ch,
                git_commit="t",
                quarter_from="1405Q1",
                quarter_to="1405Q2",
                products=["Altebrel 25", "Altebrel 50"],
                resume=True,
            )
            self.assertEqual(plan.vintages_eligible, ["1405Q1", "1405Q2"])
            self.assertEqual(plan.total_jobs, 4)
            self.assertEqual(plan.remaining, 4)


if __name__ == "__main__":
    unittest.main()
