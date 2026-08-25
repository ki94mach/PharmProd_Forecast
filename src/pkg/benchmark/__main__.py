"""``python -m pkg.benchmark`` → help; prefer ``.freeze`` / ``.verify`` / ``.universes``."""
from __future__ import annotations


def main() -> int:
    print(
        "Benchmark v1 commands:\n"
        "  python -m pkg.benchmark.freeze   # build src/data/benchmarks/v1 (needs DB + CSVs)\n"
        "  python -m pkg.benchmark.verify   # offline: reproduce locked WMAPEs\n"
        "  python -m pkg.benchmark.universes validate  # MVP product manifest\n"
        "  python -m pkg.benchmark.universes build --force  # regenerate from freeze\n"
        "  python -m pkg.benchmark.vintages validate  # TS backfill vintage map\n"
        "  python -m pkg.benchmark.vintages build --force\n"
        "  python -m pkg.benchmark.backfill_runner --engine v2 "
        "--vintages ts_backfill_1401Q1_1405Q2 --universe mvp_products --resume\n"
        "  python -m pkg.benchmark.backfill_runner ... --status | --retry-failed | --force-job\n"
        "\n"
        "API:\n"
        "  from pkg.benchmark import backtest, scoreboard\n"
        "  backtest('ts')  # Analysis B PRIMARY matched WMAPE ~43.88\n"
        "  from pkg.benchmark.universes import load_mvp_product_names\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
