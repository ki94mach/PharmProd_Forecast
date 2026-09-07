"""V3A neural smoke-test / screening CLI.

Correctness and runtime screening on a small product × origin set before
expensive full architecture screening. Does not select a winner and is not
full historical backfill.

Usage::

    python -m pkg.ts_v3a.screen \\
        --products product1,product2 \\
        --origins 140401,140501 \\
        --architectures a0,a1,a2,a3,a4,a5 \\
        --seeds 41,42,43 \\
        --output data/ts_v3a/screening
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT_CONFIG
from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v3a.architectures import ArchitectureName, coerce_architecture_name
from pkg.ts_v3a.backtest import STATUS_OK, NeuralOuterBacktestResult, run_outer_backtest
from pkg.ts_v3a.config import DEFAULT_CONFIG, NeuralExperimentConfig
from pkg.ts_v3a.model_factory import IMPLEMENTED_ARCHITECTURES
from pkg.ts_v3a.persistence import (
    default_screening_root,
    persist_completed_screening_experiment,
    screening_config_hash,
)

# CLI-only short aliases (library coerce_architecture_name stays strict).
_ARCH_ALIASES: dict[str, ArchitectureName] = {
    "a0": ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM,
    "a1": ArchitectureName.A1_SMALL_RECURSIVE_LSTM,
    "a2": ArchitectureName.A2_MIMO_LSTM,
    "a3": ArchitectureName.A3_STACKED_MIMO_LSTM,
    "a4": ArchitectureName.A4_ENCODER_DECODER_LSTM,
    "a5": ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM,
}

_IMPLEMENTED = {a.value for a in IMPLEMENTED_ARCHITECTURES}


def parse_csv_list(raw: str) -> list[str]:
    """Split a comma-separated CLI list; strip empties."""
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def parse_architecture_aliases(
    raw: str | Sequence[str],
) -> tuple[ArchitectureName, ...]:
    """Resolve ``a0``–``a5`` or full architecture names; reject A6 / unknown."""
    if isinstance(raw, str):
        tokens = parse_csv_list(raw)
    else:
        tokens = [str(t).strip() for t in raw if str(t).strip()]
    if not tokens:
        raise ValueError("architectures list is empty")

    out: list[ArchitectureName] = []
    for token in tokens:
        key = token.lower()
        if key in _ARCH_ALIASES:
            name = _ARCH_ALIASES[key]
        else:
            try:
                name = coerce_architecture_name(token)
            except ValueError as exc:
                known = ", ".join(
                    sorted(_ARCH_ALIASES)
                    + [a.value for a in IMPLEMENTED_ARCHITECTURES]
                )
                raise ValueError(
                    f"Unknown architecture {token!r}; expected one of: {known}"
                ) from exc
        if name not in IMPLEMENTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {name.value!r} is not implemented for screening "
                f"(A0–A5 only)"
            )
        out.append(name)
    return tuple(out)


def parse_origins(raw: str) -> tuple[int, ...]:
    tokens = parse_csv_list(raw)
    if not tokens:
        raise ValueError("--origins is required and must be non-empty")
    origins: list[int] = []
    for tok in tokens:
        try:
            origins.append(int(tok))
        except ValueError as exc:
            raise ValueError(f"Invalid origin {tok!r}; expected Shamsi YYYYMM") from exc
    return tuple(origins)


def parse_seeds(raw: str) -> tuple[int, ...]:
    tokens = parse_csv_list(raw)
    if not tokens:
        raise ValueError("--seeds must be non-empty")
    return tuple(int(t) for t in tokens)


def load_screening_sales(*, sales_parquet: Optional[Path] = None) -> pd.DataFrame:
    """Load monthly sales with V2/V3A columns ``product``, ``date``, ``sales``."""
    if sales_parquet is not None:
        path = Path(sales_parquet)
        if not path.is_file():
            raise FileNotFoundError(f"sales parquet not found: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_numeric(df["date"], errors="coerce").astype(int)
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        df["product"] = df["product"].astype(str)
        return df
    from pkg.benchmark.backfill_runner.runner import load_default_sales

    return load_default_sales()


def filter_products(sales: pd.DataFrame, products: Sequence[str]) -> pd.DataFrame:
    """Keep requested products; raise if any are missing."""
    wanted = [str(p) for p in products]
    if not wanted:
        raise ValueError("--products is required and must be non-empty")
    available = set(sales["product"].astype(str).unique())
    missing = [p for p in wanted if p not in available]
    if missing:
        raise ValueError(
            f"Products not found in sales data: {missing}. "
            f"Available count={len(available)}"
        )
    return sales.loc[sales["product"].astype(str).isin(wanted)].copy()


def build_neural_config(
    *,
    max_epochs: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    seeds: Sequence[int],
    min_successful_seeds: Optional[int] = None,
) -> NeuralExperimentConfig:
    """Base DEFAULT_CONFIG with optional smoke overrides."""
    base = DEFAULT_CONFIG.as_parameters_dict()
    base["random_seeds"] = tuple(int(s) for s in seeds)
    if max_epochs is not None:
        base["max_epochs"] = int(max_epochs)
    if early_stopping_patience is not None:
        base["early_stopping_patience"] = int(early_stopping_patience)
    if min_successful_seeds is not None:
        base["min_successful_seeds"] = int(min_successful_seeds)
    else:
        # Avoid impossible gate when the CLI passes fewer seeds than default 3.
        base["min_successful_seeds"] = min(
            int(DEFAULT_CONFIG.min_successful_seeds), len(tuple(seeds))
        )
    return NeuralExperimentConfig(**base)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pkg.ts_v3a.screen",
        description=(
            "V3A neural smoke-test / screening CLI. Runs a small product×origin "
            "set through A0–A5 under the V2/V3A historical contract. "
            "Does not select a winning architecture."
        ),
    )
    p.add_argument(
        "--products",
        required=True,
        help="Comma-separated product names (exact match on sales.product)",
    )
    p.add_argument(
        "--origins",
        required=True,
        help="Comma-separated Shamsi YYYYMM forecast origins",
    )
    p.add_argument(
        "--architectures",
        default="a0,a1,a2,a3,a4,a5",
        help="Comma-separated aliases a0–a5 or full names (default: all A0–A5)",
    )
    p.add_argument(
        "--seeds",
        default="41,42,43",
        help="Comma-separated random seeds (default: 41,42,43)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Screening root (default: {default_screening_root()})",
    )
    p.add_argument("--experiment-id", default=None, help="Optional fixed experiment id")
    p.add_argument(
        "--min-successful-seeds",
        type=int,
        default=None,
        help="Ensemble gate (default: min(config, n_seeds))",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional smoke override for NeuralExperimentConfig.max_epochs",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Optional smoke override for early_stopping_patience",
    )
    p.add_argument(
        "--sales-parquet",
        type=Path,
        default=None,
        help="Optional sales parquet (default: frozen benchmark sales)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print config_hash; do not train or persist",
    )
    return p


def print_fold_runtime_summary(
    fold_metadata: pd.DataFrame,
    *,
    file: Optional[TextIO] = None,
) -> None:
    """Print one concise row per fold from fold_metadata."""
    out = sys.stdout if file is None else file
    headers = (
        "architecture",
        "product",
        "origin",
        "seed",
        "history_length",
        "train_windows",
        "val_windows",
        "best_epoch",
        "runtime_s",
        "status",
    )
    print("\t".join(headers), file=out)
    if fold_metadata is None or fold_metadata.empty:
        return
    for row in fold_metadata.itertuples(index=False):
        print(
            "\t".join(
                [
                    str(getattr(row, "architecture", "")),
                    str(getattr(row, "product", "")),
                    str(getattr(row, "origin", "")),
                    str(getattr(row, "seed", "")),
                    str(getattr(row, "available_history_length", "")),
                    str(getattr(row, "train_window_count", "")),
                    str(getattr(row, "validation_window_count", "")),
                    str(getattr(row, "best_epoch", "")),
                    f"{float(getattr(row, 'runtime_seconds', 0.0) or 0.0):.3f}",
                    str(getattr(row, "status", "")),
                ]
            ),
            file=out,
        )


def print_completion_summary(
    result: NeuralOuterBacktestResult,
    experiment_dir: Path,
    *,
    file: Optional[TextIO] = None,
) -> None:
    """Short post-run summary: forecasts/actuals/metrics/failures (no winner)."""
    out = sys.stdout if file is None else file
    n_pred = 0 if result.predictions is None else len(result.predictions)
    n_ens_met = (
        0
        if result.ensemble_metrics is None or result.ensemble_metrics.empty
        else len(result.ensemble_metrics)
    )
    failures = result.fold_metadata
    if failures is not None and not failures.empty:
        bad = failures.loc[failures["status"] != STATUS_OK]
    else:
        bad = pd.DataFrame()
    n_fail = len(bad)

    print(file=out)
    print(f"experiment_dir={experiment_dir}", file=out)
    print(f"seed_oof_prediction_rows={n_pred}", file=out)
    print(f"ensemble_metric_rows={n_ens_met}", file=out)
    print(f"failed_folds={n_fail}", file=out)
    if n_fail:
        # Top reasons by count
        reason_col = (
            "unavailable_reason"
            if "unavailable_reason" in bad.columns
            else "error_type"
        )
        if reason_col in bad.columns:
            counts = (
                bad[reason_col]
                .fillna(bad.get("error_type"))
                .astype(str)
                .value_counts()
                .head(5)
            )
            print("top_failure_reasons:", file=out)
            for reason, count in counts.items():
                print(f"  {count}\t{reason}", file=out)
    print(
        "Architecture selection is not performed by this smoke/screening CLI.",
        file=out,
    )


def run_screen(args: argparse.Namespace) -> int:
    """Execute smoke screening (or dry-run). Returns process exit code."""
    products = parse_csv_list(args.products)
    if not products:
        raise ValueError("--products is required and must be non-empty")
    origins = parse_origins(args.origins)
    architectures = parse_architecture_aliases(args.architectures)
    seeds = parse_seeds(args.seeds)
    arch_names = [a.value for a in architectures]

    neural_cfg = build_neural_config(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seeds=seeds,
        min_successful_seeds=args.min_successful_seeds,
    )
    v2_cfg = TSForecastConfig(
        forecast_horizon=int(V2_DEFAULT_CONFIG.forecast_horizon),
        min_train_months=int(V2_DEFAULT_CONFIG.min_train_months),
        activity_start_min_sales=V2_DEFAULT_CONFIG.activity_start_min_sales,
        missing_month_policy=V2_DEFAULT_CONFIG.missing_month_policy,
        nonnegative_forecasts=bool(V2_DEFAULT_CONFIG.nonnegative_forecasts),
    )
    output_root = Path(args.output) if args.output is not None else default_screening_root()
    cfg_hash = screening_config_hash(
        neural_config=neural_cfg,
        v2_config=v2_cfg,
        architectures=arch_names,
        seeds=seeds,
        origins=origins,
        product_universe_id="cli_smoke",
        min_successful_seeds=int(neural_cfg.min_successful_seeds),
    )

    print(f"products={products}")
    print(f"origins={list(origins)}")
    print(f"architectures={arch_names}")
    print(f"seeds={list(seeds)}")
    print(f"min_successful_seeds={neural_cfg.min_successful_seeds}")
    print(f"max_epochs={neural_cfg.max_epochs}")
    print(f"config_hash={cfg_hash}")
    print(f"output={output_root}")

    if args.dry_run:
        print("dry_run=1; skipping train and persistence")
        return 0

    sales = load_screening_sales(sales_parquet=args.sales_parquet)
    sales = filter_products(sales, products)

    result = run_outer_backtest(
        sales,
        products,
        architectures=architectures,
        seeds=seeds,
        config=neural_cfg,
        v2_config=v2_cfg,
        explicit_origins=origins,
        min_successful_seeds=int(neural_cfg.min_successful_seeds),
    )

    print_fold_runtime_summary(result.fold_metadata)

    experiment_dir = persist_completed_screening_experiment(
        result,
        neural_config=neural_cfg,
        v2_config=v2_cfg,
        architectures=arch_names,
        seeds=seeds,
        origins=origins,
        product_universe_id="cli_smoke",
        min_successful_seeds=int(neural_cfg.min_successful_seeds),
        base_dir=output_root,
        experiment_id=args.experiment_id,
    )
    print_completion_summary(result, experiment_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # argparse exits on missing required flags; normalize to return code.
        code = exc.code
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else 2
    try:
        return run_screen(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
