"""Tests for the resumable forecast_jobs CLI (no heavy model fits)."""
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

from pkg.benchmark.backfill_runner.types import EngineJobResult
from pkg.benchmark.calendar import shamsi_add_months
from pkg.forecast_jobs.config import JobConfigError, load_job_config, parse_job_config
from pkg.forecast_jobs.cutoff import apply_cutoff_policy
from pkg.forecast_jobs.plan import build_job_plan, encode_product_key
from pkg.forecast_jobs.runner import run_forecast_jobs, scientific_config_hash
from pkg.forecast_jobs.signals import ShutdownCoordinator
from pkg.forecast_jobs.snapshot import (
    SalesSnapshotError,
    fingerprint_sales_parquet,
    validate_sales_config,
)


def _write_sales_parquet(path: Path, n_months: int = 24) -> None:
    rows = []
    cur = 140201
    for i in range(n_months):
        rows.append({"product": "SKU_A", "date": cur, "sales": float(10 + i)})
        rows.append({"product": "SKU_B", "date": cur, "sales": float(5 + i)})
        cur = shamsi_add_months(cur, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _base_config_dict(sales_path: Path, output_dir: Path, run_id: str = "test_run") -> dict:
    return {
        "schema_version": "1",
        "run": {"run_id": run_id, "output_dir": str(output_dir)},
        "cohort": {"products": ["SKU_A", "SKU_B"]},
        "origins": [140401, 140501],
        "architectures": ["v2.1", "a0", "a1"],
        "seeds": [41, 42],
        "horizon": 15,
        "cutoff_policy": {"name": "production"},
        "sales": {
            "path": str(sales_path),
            "read_only": True,
            "schema": {
                "columns": ["product", "date", "sales"],
                "dtypes": {
                    "product": "string",
                    "date": "int64",
                    "sales": "float64",
                },
            },
            "expected": {},
        },
        "execution": {"resume": True, "workers": 1},
    }


class TestConfigAndSnapshot(unittest.TestCase):
    def test_parse_yaml_roundtrip_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sales_path = tmp_path / "sales.parquet"
            _write_sales_parquet(sales_path)
            cfg_path = tmp_path / "job.yaml"
            out_dir = tmp_path / "out"
            payload = _base_config_dict(sales_path, out_dir)
            # Write YAML via PyYAML
            import yaml

            cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_job_config(cfg_path)
            self.assertEqual(config.schema_version, "1")
            self.assertEqual(config.architectures, ("v2.1", "a0", "a1"))
            self.assertEqual(config.cutoff_policy.name, "production")

            snap = validate_sales_config(config.sales)
            self.assertGreater(snap.n_rows, 0)
            self.assertEqual(len(snap.content_sha256), 64)

            # Pin expected hash and re-validate
            payload["sales"]["expected"] = {
                "content_sha256": snap.content_sha256,
                "n_rows": snap.n_rows,
            }
            config2 = parse_job_config(payload, source_path=cfg_path, run_id_override="r2")
            snap2 = validate_sales_config(config2.sales)
            self.assertEqual(snap2.content_sha256, snap.content_sha256)

            payload["sales"]["expected"]["content_sha256"] = "deadbeef" * 8
            config_bad = parse_job_config(payload, run_id_override="r3")
            with self.assertRaises(SalesSnapshotError):
                validate_sales_config(config_bad.sales)

    def test_rejects_unsupported_architecture(self):
        with self.assertRaises(JobConfigError):
            parse_job_config(
                {
                    "schema_version": "1",
                    "cohort": {"products": ["A"]},
                    "origins": [140401],
                    "architectures": ["a2"],
                    "seeds": [1],
                    "sales": {
                        "path": "missing.parquet",
                        "schema": {"columns": ["product", "date", "sales"]},
                    },
                }
            )


class TestCutoffAndPlan(unittest.TestCase):
    def test_production_cutoff_excludes_partial(self):
        sales = pd.DataFrame(
            {
                "product": ["A"] * 4,
                "date": [140410, 140411, 140412, 140501],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # origin 140501 → last_complete 140411, partial 140412
        out = apply_cutoff_policy(sales, 140501, policy_name="production")
        self.assertEqual(sorted(out["date"].tolist()), [140410, 140411])

    def test_job_grid_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sales_path = tmp_path / "sales.parquet"
            _write_sales_parquet(sales_path)
            config = parse_job_config(
                _base_config_dict(sales_path, tmp_path / "out"),
                run_id_override="grid",
            )
            plan = build_job_plan(
                config, config_hash="abc", state=None, ensure_jobs=False
            )
            # 2 products × 2 origins × (1 v2.1 + 2 seeds*a0 + 2 seeds*a1)
            # = 2*2*(1+2+2) = 20
            self.assertEqual(plan.total_jobs, 20)
            self.assertEqual(len(plan.jobs), 20)
            self.assertEqual(encode_product_key("SKU_A", 41), "SKU_A__seed41")


class TestRunnerResumeAndSignals(unittest.TestCase):
    def test_dry_run_and_execute_with_mock_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sales_path = tmp_path / "sales.parquet"
            _write_sales_parquet(sales_path)
            out_dir = tmp_path / "forecast_jobs"
            config = parse_job_config(
                _base_config_dict(sales_path, out_dir, run_id="mock_run"),
                run_id_override="mock_run",
            )
            # Shrink to one architecture for speed
            from dataclasses import replace

            config = replace(config, architectures=("v2.1",), seeds=())

            dry = run_forecast_jobs(config, dry_run=True, execute=False)
            self.assertTrue(dry.dry_run)
            self.assertTrue((dry.run_path / "input_snapshot.json").is_file())
            self.assertTrue((dry.run_path / "manifest.json").is_file())
            snap = json.loads(
                (dry.run_path / "input_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertIn("content_sha256", snap)

            calls = {"n": 0}

            def fake_executor(**kwargs):
                calls["n"] += 1
                req = kwargs["request"]
                return EngineJobResult(
                    success=True,
                    product=req.product,
                    quarter=req.quarter,
                    forecast_origin=req.forecast_origin,
                    selected_model="mock",
                    forecasts=pd.DataFrame(
                        [
                            {
                                "product": req.product,
                                "horizon": 1,
                                "forecast": 1.0,
                                "raw_forecast": 1.0,
                            }
                        ]
                    ),
                )

            first = run_forecast_jobs(
                config,
                dry_run=False,
                execute=True,
                job_executor=fake_executor,
                install_signals=False,
            )
            self.assertEqual(first.succeeded, 4)  # 2 products × 2 origins
            self.assertEqual(calls["n"], 4)

            second = run_forecast_jobs(
                config,
                dry_run=False,
                execute=True,
                job_executor=fake_executor,
                install_signals=False,
            )
            # Resume skips SUCCESS — no additional executor calls
            self.assertEqual(calls["n"], 4)
            self.assertEqual(second.succeeded, 0)
            self.assertEqual(second.skipped, 4)

    def test_shutdown_coordinator_sets_flag(self):
        coord = ShutdownCoordinator()
        self.assertFalse(coord.should_stop())
        coord.request_stop("test")
        self.assertTrue(coord.should_stop())
        self.assertEqual(coord.reason, "test")

    def test_fingerprint_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
            with self.assertRaises(SalesSnapshotError):
                fingerprint_sales_parquet(path)


if __name__ == "__main__":
    unittest.main()
