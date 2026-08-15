"""``python -m pkg.benchmark`` → help; prefer ``.freeze`` / ``.verify``."""
from __future__ import annotations


def main() -> int:
    print(
        "Benchmark v1 commands:\n"
        "  python -m pkg.benchmark.freeze   # build src/data/benchmarks/v1 (needs DB + CSVs)\n"
        "  python -m pkg.benchmark.verify   # offline: reproduce locked WMAPEs\n"
        "\n"
        "API:\n"
        "  from pkg.benchmark import backtest, scoreboard\n"
        "  backtest('ts')  # Analysis B PRIMARY matched WMAPE ~43.88\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
