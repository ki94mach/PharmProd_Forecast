"""Tests for portfolio export and SKIPPED state semantics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pkg.benchmark.backfill_runner.state import (
    JOB_SKIPPED,
    JOB_SUCCESS,
    JobIdentity,
    JobStateStore,
    should_run_job,
)
from pkg.ts_v2.config import DEFAULT_CONFIG
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window

REPO = Path(__file__).resolve().parents[2]


class TestPortfolioExport(unittest.TestCase):
    def test_skipped_not_rerun_on_resume(self):
        store = JobStateStore(Path(tempfile.mkdtemp()))
        identity = JobIdentity(
            experiment_id="t",
            engine_version="v2.1",
            config_hash="abcd",
            quarter="1401Q1",
            forecast_origin=140101,
            product_id="ProdX",
        )
        store.ensure_job(identity, git_commit="x")
        store.mark_skipped(identity, reason_code="not_in_allowlist", git_commit="x")
        rec = store.get_job(identity.job_id)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec.status, JOB_SKIPPED)
        self.assertFalse(should_run_job(rec, resume=True, retry_failed=False, force_job=False))
        self.assertTrue(should_run_job(rec, resume=True, retry_failed=False, force_job=True))

    def test_prepare_monthly_series_extends_through_training_end_recigen(self):
        sales_path = (
            REPO
            / "src"
            / "data"
            / "results"
            / "product_universe"
            / "sales_full_portfolio_v1"
            / "sales.parquet"
        )
        if not sales_path.is_file():
            self.skipTest("sales_full_portfolio_v1 missing")
        sales = pd.read_parquet(sales_path)
        product = "Recigen AutoInjector"
        if product not in set(sales["product"].astype(str)):
            self.skipTest("Recigen not in sales")
        origin = 140402
        prepared = prepare_monthly_series(sales, product, origin, config=DEFAULT_CONFIG)
        self.assertGreaterEqual(prepared.n_observations, DEFAULT_CONFIG.min_train_months)
        window = make_forecast_window(origin, config=DEFAULT_CONFIG)
        self.assertEqual(int(prepared.values.index.max()), int(window.training_end))
        self.assertEqual(prepared.missing_month_policy, "zero")

    def test_export_portfolio_success_only_and_seed_mean(self):
        run_dir = REPO / "src" / "data" / "forecast_jobs" / "portfolio_a0_backfill_v2"
        if not (run_dir / "state.sqlite").is_file():
            self.skipTest("portfolio_a0_backfill_v2 not prepared")

        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from export_portfolio_backfill_panel import (  # type: ignore
            assemble_success_panel,
            export_run,
            seed_mean_panel,
        )

        panel = assemble_success_panel(run_dir, engine="a0")
        self.assertFalse(panel.empty)
        self.assertTrue((panel["job_status"] == JOB_SUCCESS).all())
        self.assertNotIn("SKIPPED", set(panel["job_status"].unique()))

        mean = seed_mean_panel(panel)
        self.assertFalse(mean.empty)
        if panel["seed"].notna().any():
            n_jobs = panel.groupby(["product", "forecast_origin"]).ngroups
            n_mean = mean.groupby(["product", "forecast_origin"]).ngroups
            self.assertEqual(n_mean, n_jobs)

        out = Path(tempfile.mkdtemp()) / "export"
        summary = export_run(run_id="portfolio_a0_backfill_v2", engine="a0", out_dir=out)
        self.assertEqual(summary["n_success_jobs"], 636)
        self.assertTrue((out / "export_summary.json").is_file())
        meta = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["status_counts"].get("SUCCESS"), 636)
        self.assertGreaterEqual(meta["status_counts"].get("SKIPPED", 0), 1)


if __name__ == "__main__":
    unittest.main()
