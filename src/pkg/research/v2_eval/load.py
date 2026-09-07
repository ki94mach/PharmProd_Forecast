"""Read-only loaders for the completed V2 backfill and the frozen legacy panel.

Nothing here writes to the experiment tree. ``state.sqlite`` is opened through a
read-only URI so a concurrently running backfill cannot be disturbed.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.v2_eval.config import CATEGORY_COLUMNS, V2EvalConfig

JOB_COLUMNS = (
    "job_id",
    "experiment_id",
    "engine_version",
    "config_hash",
    "quarter",
    "forecast_origin",
    "product_id",
    "status",
    "started_at",
    "finished_at",
    "runtime_seconds",
    "attempt_count",
    "error_type",
    "error_message",
    "selected_model",
    "git_commit",
)


@dataclass(frozen=True)
class BackfillInputs:
    """Everything read off disk for one analysis run."""

    v2_forecasts: pd.DataFrame
    jobs: pd.DataFrame
    job_logs: pd.DataFrame
    results: pd.DataFrame
    manifest: dict[str, Any]
    run_meta: Optional[dict[str, Any]]
    stale_failure_files: int
    backtest_files: int
    legacy: pd.DataFrame
    sales: pd.DataFrame
    product_attrs: pd.DataFrame
    vintages: pd.DataFrame
    universe: pd.DataFrame
    input_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    """Stable content hash for a derived frame (column order included)."""
    payload = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    digest = hashlib.sha256(payload)
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()


def load_v2_forecasts(config: V2EvalConfig) -> pd.DataFrame:
    """Concatenate every per-job ``forecast.csv``.

    No aggregated export exists in the backfill runner, so the files are read
    directly. Job slug is retained to tie rows back to logs and artifacts.
    """
    files = sorted(config.forecasts_dir.glob("*/forecast.csv"))
    if not files:
        raise FileNotFoundError(f"No forecast.csv files under {config.forecasts_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["job_slug"] = path.parent.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"product": "product_id"})
    for column in ("forecast_origin", "target_date", "horizon"):
        out[column] = out[column].astype(int)
    for column in ("forecast", "raw_forecast"):
        if column in out.columns:
            out[column] = out[column].astype(float)
    out["product_id"] = out["product_id"].astype(str)
    out["quarter"] = out["quarter"].astype(str)
    return out


def load_jobs(config: V2EvalConfig) -> pd.DataFrame:
    """Job checkpoint rows from ``state.sqlite`` (authoritative job status)."""
    uri = f"file:{config.state_db.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        frame = pd.read_sql_query(
            f"SELECT {', '.join(JOB_COLUMNS)} FROM jobs", conn
        )
    frame["forecast_origin"] = frame["forecast_origin"].astype(int)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["quarter"] = frame["quarter"].astype(str)
    return frame


def load_job_logs(config: V2EvalConfig) -> pd.DataFrame:
    """Per-job ``job_log.json`` records (runtime and error provenance)."""
    rows = []
    for path in sorted(config.logs_dir.glob("*/job_log.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["job_slug"] = path.parent.name
        rows.append(payload)
    return pd.DataFrame(rows)


def load_results(config: V2EvalConfig) -> pd.DataFrame:
    """Per-job ``result.json`` records (selected model and engine extras)."""
    rows = []
    for path in sorted(config.forecasts_dir.glob("*/result.json")):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        extras = payload.pop("extras", {}) or {}
        payload["job_slug"] = path.parent.name
        payload["selected_strategy"] = extras.get("selected_strategy")
        payload["n_training_observations"] = extras.get("n_training_observations")
        rows.append(payload)
    return pd.DataFrame(rows)


def load_manifest(config: V2EvalConfig) -> dict[str, Any]:
    with open(config.manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_meta(config: V2EvalConfig) -> Optional[dict[str, Any]]:
    if not config.run_meta_path.exists():
        return None
    with open(config.run_meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_legacy_panel(config: V2EvalConfig) -> pd.DataFrame:
    """Frozen legacy TS forecasts joined to actuals.

    ``ts_universe.parquet`` holds the delivered output of the original
    :class:`pkg.forecast.SalesForecast` pipeline (quarterly CSV vintages), not a
    re-run and not V3A A0.
    """
    frame = pd.read_parquet(config.legacy_panel)
    keep = {
        "product": "product_id",
        "ts_origin": "forecast_origin",
        "target_date": "target_date",
        "horizon": "horizon",
        "ts_forecast": "legacy_prediction",
        "sales": "actual_legacy_panel",
        "qrt": "legacy_quarter",
        "model": "legacy_model",
    }
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise KeyError(f"legacy panel missing columns {missing}")
    out = frame[list(keep) + [c for c in CATEGORY_COLUMNS if c in frame.columns]]
    out = out.rename(columns=keep)
    for column in ("forecast_origin", "target_date", "horizon"):
        out[column] = out[column].astype(int)
    out["product_id"] = out["product_id"].astype(str)
    return out


def load_sales(config: V2EvalConfig) -> pd.DataFrame:
    """Frozen monthly actual sales (product, Shamsi YYYYMM, sales)."""
    frame = pd.read_parquet(config.raw_sales)
    frame = frame.rename(
        columns={"product": "product_id", "date": "target_date", "sales": "actual"}
    )
    frame["target_date"] = frame["target_date"].astype(int)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["actual"] = frame["actual"].astype(float)
    return frame[["product_id", "target_date", "actual"]]


def load_product_attrs(config: V2EvalConfig) -> pd.DataFrame:
    frame = pd.read_parquet(config.product_attrs)
    frame = frame.rename(columns={"product": "product_id"})
    frame["product_id"] = frame["product_id"].astype(str)
    return frame


def load_vintages(config: V2EvalConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.vintage_manifest)
    frame["forecast_origin"] = frame["forecast_origin"].astype(int)
    return frame


def load_universe(config: V2EvalConfig) -> pd.DataFrame:
    """MVP universe manifest.

    The manifest already carries a numeric ``product_id`` (``Dim.Product.ID_INT``)
    which is *not* the join key used by either engine; it is renamed out of the
    way so ``product_id`` consistently means the English title everywhere.
    """
    frame = pd.read_csv(config.universe_manifest)
    frame = frame.rename(
        columns={"product_id": "product_int_id", "product": "product_id"}
    )
    frame["product_id"] = frame["product_id"].astype(str)
    return frame


def count_stale_failure_files(config: V2EvalConfig) -> int:
    """``failure.json`` files left behind by earlier attempts.

    These are not cleared when a retry succeeds, so they must never be used as
    the failure count; ``state.sqlite`` is the source of truth.
    """
    return sum(1 for _ in config.logs_dir.glob("*/failure.json"))


def count_backtest_files(config: V2EvalConfig) -> int:
    if not config.backtests_dir.exists():
        return 0
    return sum(1 for _ in config.backtests_dir.glob("*/backtest_summary.json"))


def load_all(config: V2EvalConfig) -> BackfillInputs:
    """Read every input once and record content hashes for reproducibility."""
    input_hashes = {
        "manifest.json": sha256_file(config.manifest_path),
        "state.sqlite": sha256_file(config.state_db),
        "ts_universe.parquet": sha256_file(config.legacy_panel),
        "raw/sales.parquet": sha256_file(config.raw_sales),
        "raw/product_attrs.parquet": sha256_file(config.product_attrs),
        "vintages/ts_backfill_1401Q1_1405Q2.csv": sha256_file(config.vintage_manifest),
        "universes/mvp_products.csv": sha256_file(config.universe_manifest),
    }
    v2_forecasts = load_v2_forecasts(config)
    input_hashes["v2_forecasts_concatenated"] = sha256_frame(v2_forecasts)
    return BackfillInputs(
        v2_forecasts=v2_forecasts,
        jobs=load_jobs(config),
        job_logs=load_job_logs(config),
        results=load_results(config),
        manifest=load_manifest(config),
        run_meta=load_run_meta(config),
        stale_failure_files=count_stale_failure_files(config),
        backtest_files=count_backtest_files(config),
        legacy=load_legacy_panel(config),
        sales=load_sales(config),
        product_attrs=load_product_attrs(config),
        vintages=load_vintages(config),
        universe=load_universe(config),
        input_hashes=input_hashes,
    )
