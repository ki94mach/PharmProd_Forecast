"""Versioned YAML/JSON configuration for forecast_jobs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1"})
SUPPORTED_ARCHITECTURES = frozenset({"v2.1", "a0", "a1"})
SUPPORTED_CUTOFF_POLICIES = frozenset({"production", "origin_exclusive"})

DEFAULT_SALES_COLUMNS = ("product", "date", "sales")
DEFAULT_SALES_DTYPES = {
    "product": "string",
    "date": "int64",
    "sales": "float64",
}


class JobConfigError(ValueError):
    """Invalid or incompatible job configuration."""


@dataclass(frozen=True)
class SalesSchemaConfig:
    columns: tuple[str, ...] = DEFAULT_SALES_COLUMNS
    dtypes: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SALES_DTYPES)
    )


@dataclass(frozen=True)
class SalesExpectedSnapshot:
    """Optional pinned fingerprint values from the YAML config."""

    content_sha256: Optional[str] = None
    n_rows: Optional[int] = None
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None


@dataclass(frozen=True)
class SalesConfig:
    path: Path
    schema: SalesSchemaConfig = field(default_factory=SalesSchemaConfig)
    expected: SalesExpectedSnapshot = field(default_factory=SalesExpectedSnapshot)
    read_only: bool = True


@dataclass(frozen=True)
class CohortConfig:
    """Product cohort: universe stem and/or explicit product list."""

    universe: Optional[str] = None
    products: tuple[str, ...] = ()


@dataclass(frozen=True)
class CutoffPolicyConfig:
    name: str = "production"


@dataclass(frozen=True)
class ExecutionConfig:
    resume: bool = True
    retry_failed: bool = False
    force_job: bool = False
    workers: int = 1


@dataclass(frozen=True)
class JobRunConfig:
    """Resolved configuration for one forecast_jobs run."""

    schema_version: str
    run_id: str
    output_dir: Path
    cohort: CohortConfig
    origins: tuple[int, ...]
    architectures: tuple[str, ...]
    seeds: tuple[int, ...]
    sales: SalesConfig
    cutoff_policy: CutoffPolicyConfig
    horizon: int = 15
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    source_path: Optional[Path] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def scientific_payload(self) -> dict[str, Any]:
        """Fields that participate in ``config_hash`` (no host/runtime noise)."""
        return {
            "schema_version": self.schema_version,
            "cohort": {
                "universe": self.cohort.universe,
                "products": list(self.cohort.products),
            },
            "origins": list(self.origins),
            "architectures": list(self.architectures),
            "seeds": list(self.seeds),
            "horizon": int(self.horizon),
            "cutoff_policy": {"name": self.cutoff_policy.name},
            "sales": {
                "path": _normalize_path_str(self.sales.path),
                "schema": {
                    "columns": list(self.sales.schema.columns),
                    "dtypes": dict(self.sales.schema.dtypes),
                },
                "expected": {
                    "content_sha256": self.sales.expected.content_sha256,
                    "n_rows": self.sales.expected.n_rows,
                    "size_bytes": self.sales.expected.size_bytes,
                    "mtime_ns": self.sales.expected.mtime_ns,
                },
                "read_only": bool(self.sales.read_only),
            },
        }


def _normalize_path_str(path: Path) -> str:
    return Path(path).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise JobConfigError(
                "PyYAML is required for .yaml/.yml configs. "
                "Install with: pip install 'pyyaml>=6,<7'"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise JobConfigError(
            f"Unsupported config suffix {suffix!r}; use .yaml, .yml, or .json"
        )
    if not isinstance(data, dict):
        raise JobConfigError("config root must be a mapping")
    return data


def _as_tuple_str(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return tuple(p for p in parts if p)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    raise JobConfigError(f"{field_name} must be a list or comma-separated string")


def _as_tuple_int(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise JobConfigError(f"{field_name} must contain integers") from exc
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise JobConfigError(f"{field_name} must contain integers") from exc
    raise JobConfigError(f"{field_name} must be a list or comma-separated string")


def _resolve_path(raw: Any, *, base_dir: Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        # Prefer repo-root-relative paths (cwd-independent when given absolute later).
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return candidate
    return path.resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _new_run_id() -> str:
    from datetime import datetime, timezone
    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def parse_job_config(
    data: Mapping[str, Any],
    *,
    source_path: Optional[Path] = None,
    run_id_override: Optional[str] = None,
) -> JobRunConfig:
    """Parse and validate a raw config mapping into :class:`JobRunConfig`."""
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise JobConfigError(
            f"unsupported schema_version {schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    run_block = data.get("run") or {}
    if not isinstance(run_block, Mapping):
        raise JobConfigError("run must be a mapping")

    run_id = (
        str(run_id_override).strip()
        if run_id_override
        else str(run_block.get("run_id") or "").strip() or _new_run_id()
    )

    base = source_path.parent if source_path is not None else Path.cwd()
    output_raw = run_block.get("output_dir") or data.get("output_dir")
    if not output_raw:
        output_dir = _repo_root() / "src" / "data" / "forecast_jobs"
    else:
        output_dir = _resolve_path(output_raw, base_dir=base)
        # If path does not exist yet, still resolve relative to repo src/data when possible.
        if not output_dir.exists() and not Path(str(output_raw)).is_absolute():
            repo_candidate = (_repo_root() / str(output_raw)).resolve()
            output_dir = repo_candidate

    cohort_raw = data.get("cohort") or {}
    if not isinstance(cohort_raw, Mapping):
        raise JobConfigError("cohort must be a mapping")
    universe = cohort_raw.get("universe")
    products = _as_tuple_str(cohort_raw.get("products"), field_name="cohort.products")
    if not universe and not products:
        raise JobConfigError("cohort requires universe and/or products")

    origins = _as_tuple_int(data.get("origins"), field_name="origins")
    if not origins:
        raise JobConfigError("origins must be a non-empty list")

    architectures_raw = _as_tuple_str(
        data.get("architectures"), field_name="architectures"
    )
    norm_arch: list[str] = []
    for a in architectures_raw:
        key = str(a).strip().lower()
        if key in {"v2.1", "v21", "v2_1"}:
            key = "v2.1"
        elif key in {"a0", "a1"}:
            pass
        else:
            raise JobConfigError(
                f"unsupported architecture {a!r}; "
                f"allowed: {sorted(SUPPORTED_ARCHITECTURES)}"
            )
        if key not in SUPPORTED_ARCHITECTURES:
            raise JobConfigError(f"unsupported architecture {key!r}")
        if key not in norm_arch:
            norm_arch.append(key)
    architectures = tuple(norm_arch)
    if not architectures:
        raise JobConfigError("architectures must be non-empty")

    seeds = _as_tuple_int(data.get("seeds"), field_name="seeds")
    needs_seeds = any(a in {"a0", "a1"} for a in architectures)
    if needs_seeds and not seeds:
        raise JobConfigError("seeds required when architectures include a0/a1")
    if not needs_seeds:
        seeds = seeds or ()

    sales_raw = data.get("sales")
    if not isinstance(sales_raw, Mapping):
        raise JobConfigError("sales must be a mapping with path/schema")
    sales_path = _resolve_path(sales_raw.get("path"), base_dir=base)
    schema_raw = sales_raw.get("schema") or {}
    if not isinstance(schema_raw, Mapping):
        raise JobConfigError("sales.schema must be a mapping")
    columns = _as_tuple_str(
        schema_raw.get("columns", list(DEFAULT_SALES_COLUMNS)),
        field_name="sales.schema.columns",
    )
    dtypes_raw = schema_raw.get("dtypes") or dict(DEFAULT_SALES_DTYPES)
    if not isinstance(dtypes_raw, Mapping):
        raise JobConfigError("sales.schema.dtypes must be a mapping")
    dtypes = {str(k): str(v) for k, v in dtypes_raw.items()}
    expected_raw = sales_raw.get("expected") or sales_raw.get("snapshot") or {}
    if expected_raw is None:
        expected_raw = {}
    if not isinstance(expected_raw, Mapping):
        raise JobConfigError("sales.expected must be a mapping")
    expected = SalesExpectedSnapshot(
        content_sha256=(
            str(expected_raw["content_sha256"]).lower()
            if expected_raw.get("content_sha256") is not None
            else None
        ),
        n_rows=(
            int(expected_raw["n_rows"])
            if expected_raw.get("n_rows") is not None
            else None
        ),
        size_bytes=(
            int(expected_raw["size_bytes"])
            if expected_raw.get("size_bytes") is not None
            else None
        ),
        mtime_ns=(
            int(expected_raw["mtime_ns"])
            if expected_raw.get("mtime_ns") is not None
            else None
        ),
    )
    sales = SalesConfig(
        path=sales_path,
        schema=SalesSchemaConfig(columns=columns, dtypes=dtypes),
        expected=expected,
        read_only=bool(sales_raw.get("read_only", True)),
    )

    cutoff_raw = data.get("cutoff_policy") or {"name": "production"}
    if isinstance(cutoff_raw, str):
        cutoff_name = cutoff_raw.strip().lower()
    elif isinstance(cutoff_raw, Mapping):
        cutoff_name = str(cutoff_raw.get("name", "production")).strip().lower()
    else:
        raise JobConfigError("cutoff_policy must be a string or mapping")
    if cutoff_name not in SUPPORTED_CUTOFF_POLICIES:
        raise JobConfigError(
            f"unsupported cutoff_policy {cutoff_name!r}; "
            f"allowed: {sorted(SUPPORTED_CUTOFF_POLICIES)}"
        )

    exec_raw = data.get("execution") or {}
    if not isinstance(exec_raw, Mapping):
        raise JobConfigError("execution must be a mapping")
    execution = ExecutionConfig(
        resume=bool(exec_raw.get("resume", True)),
        retry_failed=bool(exec_raw.get("retry_failed", False)),
        force_job=bool(exec_raw.get("force_job", False)),
        workers=max(1, int(exec_raw.get("workers", 1))),
    )

    horizon = int(data.get("horizon", 15))
    if horizon <= 0:
        raise JobConfigError("horizon must be positive")

    return JobRunConfig(
        schema_version=schema_version,
        run_id=run_id,
        output_dir=output_dir,
        cohort=CohortConfig(
            universe=str(universe).strip() if universe else None,
            products=products,
        ),
        origins=origins,
        architectures=architectures,
        seeds=seeds,
        sales=sales,
        cutoff_policy=CutoffPolicyConfig(name=cutoff_name),
        horizon=horizon,
        execution=execution,
        source_path=source_path,
        raw=dict(data),
    )


def load_job_config(
    path: Path | str,
    *,
    run_id_override: Optional[str] = None,
) -> JobRunConfig:
    """Load a versioned YAML/JSON job config from disk."""
    cfg_path = Path(path).resolve()
    if not cfg_path.is_file():
        raise JobConfigError(f"config not found: {cfg_path}")
    data = _load_mapping(cfg_path)
    return parse_job_config(
        data, source_path=cfg_path, run_id_override=run_id_override
    )


def config_to_dict(config: JobRunConfig) -> dict[str, Any]:
    """Serialize a resolved config for manifest / artifact copy."""
    return {
        "schema_version": config.schema_version,
        "run": {
            "run_id": config.run_id,
            "output_dir": _normalize_path_str(config.output_dir),
        },
        "cohort": {
            "universe": config.cohort.universe,
            "products": list(config.cohort.products),
        },
        "origins": list(config.origins),
        "architectures": list(config.architectures),
        "seeds": list(config.seeds),
        "horizon": config.horizon,
        "cutoff_policy": {"name": config.cutoff_policy.name},
        "sales": {
            "path": _normalize_path_str(config.sales.path),
            "read_only": config.sales.read_only,
            "schema": {
                "columns": list(config.sales.schema.columns),
                "dtypes": dict(config.sales.schema.dtypes),
            },
            "expected": {
                "content_sha256": config.sales.expected.content_sha256,
                "n_rows": config.sales.expected.n_rows,
                "size_bytes": config.sales.expected.size_bytes,
                "mtime_ns": config.sales.expected.mtime_ns,
            },
        },
        "execution": {
            "resume": config.execution.resume,
            "retry_failed": config.execution.retry_failed,
            "force_job": config.execution.force_job,
            "workers": config.execution.workers,
        },
    }
