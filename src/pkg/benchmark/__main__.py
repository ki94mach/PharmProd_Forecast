"""``python -m pkg.benchmark`` → help; prefer ``.freeze`` / ``.verify`` / ``.universes``."""
from __future__ import annotations


def main() -> int:
    print(
        "Benchmark v1 commands:\n"
        "  python -m pkg.benchmark.freeze   # build src/data/benchmarks/v1 (needs DB + CSVs)\n"
        "  python -m pkg.benchmark.verify   # offline: reproduce locked WMAPEs\n"
        "  python -m pkg.benchmark.universes validate  # MVP product manifest\n"
        "  python -m pkg.benchmark.universes build --force  # regenerate from freeze\n"
        "\n"
        "API:\n"
        "  from pkg.benchmark import backtest, scoreboard\n"
        "  backtest('ts')  # Analysis B PRIMARY matched WMAPE ~43.88\n"
        "  from pkg.benchmark.universes import load_mvp_product_names\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
