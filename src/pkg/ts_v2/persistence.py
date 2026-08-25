"""Immutable V2 forecast-run persistence.

Layout (separate from V1 ``src/data/results/``)::

    src/data/results_v2/{qrt}/{run_id}/
        forecast.csv
        run_metadata.json
        backtest_scores.csv

Incomplete/checkpoint runs live under::

    src/data/results_v2/{qrt}/.incomplete/{run_id}/

Completed runs are never overwritten or appended to.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG, TSForecastConfig
from pkg.ts_v2.types import EngineResult, ProductFinalForecast

TS_VERSION = "v2"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete"
CHECKPOINT_FILENAME = "checkpoint.json"
COMPLETE_MARKER = ".complete"

FORECAST_CSV_COLUMNS = (
    "run_id",
    "product",
    "product_title",
    "forecast_origin",
    "target_date",
    "horizon",
    "model_or_strategy",
    "raw_forecast",
    "forecast",
)

FORECAST_CSV_NAME = "forecast.csv"
METADATA_JSON_NAME = "run_metadata.json"
BACKTEST_SCORES_CSV_NAME = "backtest_scores.csv"


class RunPersistenceError(Exception):
    """Base error for V2 run storage."""


class RunImmutableError(RunPersistenceError):
    """Attempt to mutate or overwrite a completed run."""


class RunCheckpointError(RunPersistenceError):
    """Invalid checkpoint/resume state."""


def default_results_v2_root() -> Path:
    """Repository ``src/data/results_v2`` (created on demand)."""
    root = Path(__file__).resolve().parents[2] / "data" / "results_v2"
    root.mkdir(parents=True, exist_ok=True)
    return root


def quarter_from_origin(forecast_origin: int) -> str:
    """Shamsi quarter label, e.g. ``140501`` -> ``1405Q1``.

    Canonical definition lives in :func:`pkg.benchmark.calendar.quarter_from_origin`
    (inverse of :func:`pkg.benchmark.calendar.origin_from_quarter`).
    """
    from pkg.benchmark.calendar import quarter_from_origin as _quarter_from_origin

    return _quarter_from_origin(forecast_origin)


def new_run_id(*, created_at: Optional[datetime] = None) -> str:
    """Unique run id: ``YYYYMMDDTHHMMSSZ_<8 hex>``."""
    ts = created_at or datetime.now(timezone.utc)
    stamp = ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def config_to_dict(config: TSForecastConfig) -> dict[str, Any]:
    """JSON-serializable config snapshot."""
    out: dict[str, Any] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, tuple):
            out[f.name] = list(val)
        else:
            out[f.name] = val
    return out


def config_hash(config: TSForecastConfig) -> str:
    """Stable short hash of the full V2 config."""
    payload = json.dumps(config_to_dict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_git_commit() -> Optional[str]:
    """Best-effort ``git rev-parse HEAD`` from repo root."""
    try:
        root = Path(__file__).resolve().parents[3]
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = proc.stdout.strip()
        return commit or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _quarter_dir(base_dir: Path, quarter: str) -> Path:
    return Path(base_dir) / str(quarter)


def incomplete_run_dir(base_dir: Path, quarter: str, run_id: str) -> Path:
    return _quarter_dir(base_dir, quarter) / ".incomplete" / str(run_id)


def complete_run_dir(base_dir: Path, quarter: str, run_id: str) -> Path:
    return _quarter_dir(base_dir, quarter) / str(run_id)


def is_complete_run(run_dir: Path) -> bool:
    """True when ``run_dir`` is a finalized immutable run."""
    path = Path(run_dir)
    if (path / COMPLETE_MARKER).is_file():
        return True
    meta_path = path / METADATA_JSON_NAME
    if meta_path.is_file():
        return _read_json_if_exists(meta_path).get("status") == COMPLETE_STATUS
    return False


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def build_forecast_dataframe(
    engine_result: EngineResult,
    *,
    run_id: str,
    product_titles: Optional[Mapping[str, str]] = None,
    config: Optional[TSForecastConfig] = None,
) -> pd.DataFrame:
    """One row per successful SKU × horizon with explicit origin columns."""
    cfg = config or DEFAULT_CONFIG
    titles = product_titles or {}
    rows: list[dict[str, Any]] = []
    for product, final in sorted(engine_result.final_forecasts.items()):
        _append_product_forecast_rows(
            rows,
            final,
            run_id=str(run_id),
            product_title=str(titles.get(product, product)),
            expected_horizon=int(cfg.forecast_horizon),
        )
    if not rows:
        return pd.DataFrame(columns=list(FORECAST_CSV_COLUMNS))
    return pd.DataFrame(rows, columns=list(FORECAST_CSV_COLUMNS))


def _append_product_forecast_rows(
    rows: list[dict[str, Any]],
    final: ProductFinalForecast,
    *,
    run_id: str,
    product_title: str,
    expected_horizon: int,
) -> None:
    if len(final.horizon_forecasts) != expected_horizon:
        raise RunPersistenceError(
            f"{final.product!r}: expected {expected_horizon} horizon rows, "
            f"got {len(final.horizon_forecasts)}"
        )
    for hf in final.horizon_forecasts:
        rows.append(
            {
                "run_id": run_id,
                "product": str(final.product),
                "product_title": product_title,
                "forecast_origin": int(final.forecast_origin),
                "target_date": int(hf.target_shamsi_yyyymm),
                "horizon": int(hf.horizon),
                "model_or_strategy": str(final.selected_model),
                "raw_forecast": float(hf.raw_forecast),
                "forecast": float(hf.constrained_forecast),
            }
        )


def build_backtest_scores_dataframe(
    engine_result: EngineResult,
    *,
    run_id: str,
) -> pd.DataFrame:
    """Flatten backtest metrics with ``run_id``."""
    if engine_result.backtest is None or engine_result.backtest.metrics is None:
        return pd.DataFrame(columns=["run_id"])
    metrics = engine_result.backtest.metrics.copy()
    if metrics.empty:
        return pd.DataFrame(columns=["run_id"])
    metrics.insert(0, "run_id", str(run_id))
    return metrics


def build_run_metadata(
    *,
    run_id: str,
    forecast_origin: int,
    quarter: str,
    config: TSForecastConfig,
    created_at: str,
    status: str,
    git_commit: Optional[str] = None,
    config_hash_value: Optional[str] = None,
) -> dict[str, Any]:
    """Metadata written to ``run_metadata.json``."""
    return {
        "run_id": str(run_id),
        "ts_version": TS_VERSION,
        "forecast_origin": int(forecast_origin),
        "quarter": str(quarter),
        "created_at": str(created_at),
        "forecast_horizon": int(config.forecast_horizon),
        "selection_metric": str(config.selection_metric),
        "selection_strategy": str(config.selection_strategy),
        "candidate_models": list(config.candidate_models),
        "configuration": config_to_dict(config),
        "git_commit": git_commit,
        "config_hash": config_hash_value or config_hash(config),
        "status": str(status),
    }


@dataclass(frozen=True)
class V2RunCheckpoint:
    """In-progress run; must match ``config_hash`` on resume."""

    run_id: str
    quarter: str
    forecast_origin: int
    config_hash: str
    created_at: str
    run_dir: Path
    config: TSForecastConfig

    @property
    def is_complete(self) -> bool:
        return is_complete_run(self.run_dir)


def begin_v2_run(
    forecast_origin: int,
    config: Optional[TSForecastConfig] = None,
    *,
    base_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    resume: bool = False,
) -> V2RunCheckpoint:
    """Start or resume an incomplete checkpoint run."""
    cfg = config or DEFAULT_CONFIG
    root = Path(base_dir) if base_dir is not None else default_results_v2_root()
    origin = int(forecast_origin)
    qrt = quarter_from_origin(origin)
    rid = run_id or new_run_id(created_at=created_at)
    created = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created_iso = created.strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg_hash = config_hash(cfg)

    complete_dir = complete_run_dir(root, qrt, rid)
    if complete_dir.exists() and is_complete_run(complete_dir):
        raise RunImmutableError(
            f"run {rid!r} already completed at {complete_dir}; cannot restart"
        )

    inc_dir = incomplete_run_dir(root, qrt, rid)
    checkpoint_path = inc_dir / CHECKPOINT_FILENAME

    if resume:
        if not checkpoint_path.is_file():
            raise RunCheckpointError(
                f"cannot resume: missing checkpoint at {checkpoint_path}"
            )
        stored = _read_json_if_exists(checkpoint_path)
        if stored.get("config_hash") != cfg_hash:
            raise RunCheckpointError(
                f"config_hash mismatch for run {rid!r}: "
                f"expected {stored.get('config_hash')!r}, got {cfg_hash!r}"
            )
        if int(stored.get("forecast_origin", -1)) != origin:
            raise RunCheckpointError(
                f"forecast_origin mismatch for run {rid!r}"
            )
        return V2RunCheckpoint(
            run_id=rid,
            quarter=qrt,
            forecast_origin=origin,
            config_hash=cfg_hash,
            created_at=str(stored.get("created_at", created_iso)),
            run_dir=inc_dir,
            config=cfg,
        )

    if inc_dir.exists() and any(inc_dir.iterdir()):
        raise RunCheckpointError(
            f"incomplete run directory already exists: {inc_dir}; use resume=True"
        )
    inc_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "run_id": rid,
        "quarter": qrt,
        "forecast_origin": origin,
        "config_hash": cfg_hash,
        "created_at": created_iso,
        "status": INCOMPLETE_STATUS,
    }
    _write_json(checkpoint_path, checkpoint)
    return V2RunCheckpoint(
        run_id=rid,
        quarter=qrt,
        forecast_origin=origin,
        config_hash=cfg_hash,
        created_at=created_iso,
        run_dir=inc_dir,
        config=cfg,
    )


def write_checkpoint_artifacts(
    checkpoint: V2RunCheckpoint,
    engine_result: EngineResult,
    *,
    product_titles: Optional[Mapping[str, str]] = None,
) -> None:
    """Write CSV/JSON artifacts into the incomplete run directory."""
    if checkpoint.is_complete:
        raise RunImmutableError(f"run {checkpoint.run_id!r} is already complete")

    root = checkpoint.run_dir.parents[2]
    if is_complete_run(complete_run_dir(root, checkpoint.quarter, checkpoint.run_id)):
        raise RunImmutableError(
            f"completed run exists for {checkpoint.run_id!r}; refusing checkpoint write"
        )

    forecast_df = build_forecast_dataframe(
        engine_result,
        run_id=checkpoint.run_id,
        product_titles=product_titles,
        config=checkpoint.config,
    )
    scores_df = build_backtest_scores_dataframe(
        engine_result, run_id=checkpoint.run_id
    )
    metadata = build_run_metadata(
        run_id=checkpoint.run_id,
        forecast_origin=checkpoint.forecast_origin,
        quarter=checkpoint.quarter,
        config=checkpoint.config,
        created_at=checkpoint.created_at,
        status=INCOMPLETE_STATUS,
        git_commit=get_git_commit(),
        config_hash_value=checkpoint.config_hash,
    )

    run_dir = checkpoint.run_dir
    forecast_df.to_csv(run_dir / FORECAST_CSV_NAME, index=False)
    scores_df.to_csv(run_dir / BACKTEST_SCORES_CSV_NAME, index=False)
    _write_json(run_dir / METADATA_JSON_NAME, metadata)


def finalize_v2_run(
    checkpoint: V2RunCheckpoint,
    engine_result: EngineResult,
    *,
    product_titles: Optional[Mapping[str, str]] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Atomically promote an incomplete run to an immutable completed run."""
    root = Path(base_dir) if base_dir is not None else checkpoint.run_dir.parents[2]
    complete_dir = complete_run_dir(root, checkpoint.quarter, checkpoint.run_id)
    if complete_dir.exists():
        if is_complete_run(complete_dir):
            raise RunImmutableError(
                f"run {checkpoint.run_id!r} already finalized at {complete_dir}"
            )
        raise RunCheckpointError(
            f"cannot finalize: non-complete path exists at {complete_dir}"
        )

    write_checkpoint_artifacts(
        checkpoint, engine_result, product_titles=product_titles
    )
    validate_forecast_dataframe(
        pd.read_csv(checkpoint.run_dir / FORECAST_CSV_NAME),
        expected_horizon=int(checkpoint.config.forecast_horizon),
    )

    metadata = build_run_metadata(
        run_id=checkpoint.run_id,
        forecast_origin=checkpoint.forecast_origin,
        quarter=checkpoint.quarter,
        config=checkpoint.config,
        created_at=checkpoint.created_at,
        status=COMPLETE_STATUS,
        git_commit=get_git_commit(),
        config_hash_value=checkpoint.config_hash,
    )
    _write_json(checkpoint.run_dir / METADATA_JSON_NAME, metadata)
    (checkpoint.run_dir / COMPLETE_MARKER).write_text("", encoding="utf-8")

    complete_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(checkpoint.run_dir, complete_dir)
    return complete_dir


def persist_completed_run(
    engine_result: EngineResult,
    forecast_origin: int,
    *,
    config: Optional[TSForecastConfig] = None,
    base_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    product_titles: Optional[Mapping[str, str]] = None,
    created_at: Optional[datetime] = None,
) -> Path:
    """Begin, write, and finalize a V2 run in one step."""
    cfg = config or DEFAULT_CONFIG
    checkpoint = begin_v2_run(
        forecast_origin,
        cfg,
        base_dir=base_dir,
        run_id=run_id,
        created_at=created_at,
    )
    return finalize_v2_run(
        checkpoint,
        engine_result,
        product_titles=product_titles,
        base_dir=base_dir,
    )


def validate_forecast_dataframe(
    forecast_df: pd.DataFrame,
    *,
    expected_horizon: int = 15,
) -> None:
    """Ensure exactly one row per SKU × horizon and explicit origins."""
    required = set(FORECAST_CSV_COLUMNS)
    missing = required - set(forecast_df.columns)
    if missing:
        raise RunPersistenceError(f"forecast.csv missing columns: {sorted(missing)}")

    if forecast_df.empty:
        return

    if forecast_df.duplicated(subset=["run_id", "product", "horizon"]).any():
        raise RunPersistenceError("duplicate SKU × horizon rows in forecast.csv")

    for run_id, run_group in forecast_df.groupby("run_id"):
        origins = run_group["forecast_origin"].unique()
        if len(origins) != 1:
            raise RunPersistenceError(
                f"run {run_id!r}: forecast_origin must be explicit and unique, "
                f"got {origins!r}"
            )
        for product, sku_group in run_group.groupby("product"):
            if len(sku_group) != expected_horizon:
                raise RunPersistenceError(
                    f"run {run_id!r} product {product!r}: expected "
                    f"{expected_horizon} horizons, got {len(sku_group)}"
                )
            horizons = sorted(int(h) for h in sku_group["horizon"].tolist())
            expected = list(range(1, expected_horizon + 1))
            if horizons != expected:
                raise RunPersistenceError(
                    f"run {run_id!r} product {product!r}: horizons {horizons} != "
                    f"{expected}"
                )
