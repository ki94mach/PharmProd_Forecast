"""Paths, keys and diagnostic thresholds for the V2-vs-legacy backfill analysis.

Metric conventions fixed here (and restated in the report):

- ``WMAPE`` is a **percent** (``100 * sum|err| / sum|actual|``), matching
  :func:`pkg.benchmark.evaluate.wmape`. :mod:`pkg.ts_v2.metrics` returns the
  same quantity as a ratio; this analysis always reports percent.
- ``signed error = prediction - actual`` so positive bias means overforecasting.
- Relative error improvement is ``100 * (legacy - v2) / legacy`` and is only
  applied to non-negative error metrics, never to signed bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# One row of the matched panel is uniquely identified by these columns.
MATCH_KEYS: tuple[str, ...] = (
    "product_id",
    "forecast_origin",
    "target_date",
    "horizon",
)

# Horizon groups requested for the breakdown (inclusive bounds).
HORIZON_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("h1-3", 1, 3),
    ("h4-6", 4, 6),
    ("h7-12", 7, 12),
    ("h13-15", 13, 15),
)

# Frozen product category columns carried by the benchmark panel.
CATEGORY_COLUMNS: tuple[str, ...] = ("generic", "Field", "ProductForm", "Provider")

# Legacy vintages whose CSV origin drifted from the canonical quarter origin.
# Legacy 1403Q1 was emitted at 140304 and 1403Q2 at 140306, while V2 uses the
# canonical 140301 / 140304. Matching on forecast_origin keeps the information
# cutoff identical but pairs different quarter labels, so the primary result is
# accompanied by a sensitivity that drops these origins.
SHIFTED_ORIGINS: tuple[int, ...] = (140301, 140304, 140306)

FORECAST_HORIZON = 15


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Explicit rules used to flag suspicious forecasts (never to alter them)."""

    # A job whose 15 horizons collapse to one distinct value.
    flat_distinct_values: int = 1
    # Forecast above this multiple of the SKU's pre-origin maximum history.
    extreme_ratio_vs_history_max: float = 5.0
    # Absolute-error share of a single product that counts as concentrated.
    concentration_top1_share: float = 0.25
    concentration_top5_share: float = 0.50
    # Relative WMAPE change inside this band counts as a tie, not a win/loss.
    product_tie_band_pct: float = 1.0
    # Minimum evaluated rows before a product enters win/tie/loss rates.
    min_rows_per_product: int = 3


@dataclass(frozen=True)
class V2EvalConfig:
    """Resolved input/output locations for one analysis run."""

    repo_root: Path
    src_root: Path
    experiment_dir: Path
    benchmark_root: Path
    vintage_manifest: Path
    universe_manifest: Path
    out_dir: Path
    experiment_id: str = "ts_mvp_backfill_1401Q1_1405Q2"
    engine: str = "v2"
    thresholds: DiagnosticThresholds = field(default_factory=DiagnosticThresholds)

    @property
    def forecasts_dir(self) -> Path:
        return self.experiment_dir / "forecasts"

    @property
    def backtests_dir(self) -> Path:
        return self.experiment_dir / "backtests"

    @property
    def logs_dir(self) -> Path:
        return self.experiment_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.experiment_dir / "manifest.json"

    @property
    def state_db(self) -> Path:
        return self.experiment_dir / "state.sqlite"

    @property
    def run_meta_path(self) -> Path:
        return self.logs_dir / "run_meta.json"

    @property
    def legacy_panel(self) -> Path:
        return self.benchmark_root / "ts_universe.parquet"

    @property
    def raw_sales(self) -> Path:
        return self.benchmark_root / "raw" / "sales.parquet"

    @property
    def product_attrs(self) -> Path:
        return self.benchmark_root / "raw" / "product_attrs.parquet"

    @property
    def charts_dir(self) -> Path:
        return self.out_dir / "charts"


def _src_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_experiment_dir(
    src_root: Path,
    experiment_id: str,
    engine: str,
) -> Path:
    """Locate the backfill experiment directory.

    ``store.default_backfill_root`` writes to ``data/backfills`` but this host's
    completed run lives under ``data/backfill``. Both are accepted; the first
    existing candidate wins.
    """
    candidates = [
        src_root / "data" / "backfill" / experiment_id / engine,
        src_root / "data" / "backfills" / experiment_id / engine,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No backfill experiment directory found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def default_config(
    *,
    experiment_id: str = "ts_mvp_backfill_1401Q1_1405Q2",
    engine: str = "v2",
    out_dir: Path | None = None,
    experiment_dir: Path | None = None,
) -> V2EvalConfig:
    src_root = _src_root()
    repo_root = src_root.parent
    resolved_experiment = (
        Path(experiment_dir)
        if experiment_dir is not None
        else resolve_experiment_dir(src_root, experiment_id, engine)
    )
    resolved_out = (
        Path(out_dir)
        if out_dir is not None
        else src_root / "data" / "results" / "ts_v2_backfill_eval"
    )
    return V2EvalConfig(
        repo_root=repo_root,
        src_root=src_root,
        experiment_dir=resolved_experiment,
        benchmark_root=src_root / "data" / "benchmarks" / "v1",
        vintage_manifest=(
            src_root / "pkg" / "benchmark" / "vintages" / "ts_backfill_1401Q1_1405Q2.csv"
        ),
        universe_manifest=(
            src_root / "pkg" / "benchmark" / "universes" / "mvp_products.csv"
        ),
        out_dir=resolved_out,
        experiment_id=experiment_id,
        engine=engine,
    )
