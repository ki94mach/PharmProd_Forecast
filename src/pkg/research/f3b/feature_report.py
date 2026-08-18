"""Write docs/f3b_price_feature_audit.md from F3B Step 2 feature-audit CSVs."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.f3b.config import docs_dir, f3b_feature_audit_dir
from pkg.research.harness.report import md_table, read_csv, repo_relative


def _s(df: Optional[pd.DataFrame], key: str, default: str = "n/a") -> str:
    if df is None or df.empty or key not in df.columns:
        return default
    v = df.iloc[0][key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    if isinstance(v, float) and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
    return str(v)


def write_price_feature_audit(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else report.get("out_dir") or f3b_feature_audit_dir()
    path = path or (docs_dir() / "f3b_price_feature_audit.md")
    overall = report.get("overall")
    if overall is None:
        overall = read_csv(out_dir, "coverage_overall.csv")
    dist = report.get("distributions")
    if dist is None:
        dist = read_csv(out_dir, "distributions.csv")
    by_o = report.get("by_origin")
    if by_o is None:
        by_o = read_csv(out_dir, "coverage_by_origin.csv")
    by_p = report.get("by_product")
    if by_p is None:
        by_p = read_csv(out_dir, "coverage_by_product.csv")
    temporal = report.get("temporal")
    if temporal is None:
        temporal = read_csv(out_dir, "temporal_variation.csv")
    extremes = report.get("extremes")
    if extremes is None:
        extremes = read_csv(out_dir, "extreme_changes.csv")
    n_vary = report.get("n_products_varying_across_origins")
    if n_vary is None and temporal is not None and not temporal.empty:
        n_vary = int(temporal["price_state_changes_across_primary_origins"].sum())

    n_prod = _s(overall, "n_products")
    sections = [
        "# F3B Step 2 — Point-in-time price feature audit\n",
        f"**Date:** {date.today().isoformat()}  \n",
        f"**Audit artifacts:** `{repo_relative(Path(out_dir))}`  \n",
        "**Frozen source:** `src/data/results/f3b/source/price_history.parquet`\n",
        "\nNo XGBoost, no WMAPE, no transformation tuning. Features are predefined.\n",
        "\n## Forecast-time information rule\n\n",
        "Benchmark origins are Shamsi YYYYMM integers (e.g. `140404`) and mean the "
        "forecast-creation **month**, treated as the start of that month.\n\n",
        "A price observation is visible at origin `O` iff `effective_month < O` "
        "(same strictly-before convention as F3A `sales.date < origin` and F2 "
        "`target_date < origin`). A price change in the origin month itself is "
        "**not** visible. A change after origin must never influence that historical "
        "forecast, even when the target month is later.\n\n",
        "Example: origin `140404` may use a `140403` price and must not use `140405` "
        "or `140406`.\n",
        "\n## Limitation\n\n",
        "These features describe **official consumer-price state known by forecast origin**. "
        "They do **not** describe future planned price changes. They cannot directly "
        "predict demand effects of a future increase unless that future effective price "
        "was historically archived as known before origin. We do not have that evidence "
        "in the Triple Price freeze.\n",
        "\n## Scored features (consumer price only)\n\n",
        "Exactly three scored columns; distributor and pharmacy prices stay in the "
        "frozen source and are not attached.\n\n",
        "- `log_consumer_price_asof_origin` = `log1p(consumer_price_asof_origin)` "
        "(raw price is diagnostic only; log chosen ex ante for scale).\n",
        "- `last_consumer_price_change_pct` = `(current - previous) / previous` on the "
        "two most recent **distinct** consumer-price states before origin. "
        "Consecutive identical observations are not a change. Missing = NaN, not 0.\n",
        "- `months_since_last_consumer_price_change` via `shamsi_month_diff` "
        "(never YYYYMM integer subtraction). Missing = NaN, not 0.\n",
        "\n## PRIMARY coverage\n\n",
        md_table(overall) if overall is not None else "_missing_\n",
        f"- **n_rows:** {_s(overall, 'n_rows')}\n",
        f"- **n_products:** {n_prod}\n",
        f"- **current_price_coverage_pct:** {_s(overall, 'current_price_coverage_pct')}\n",
        f"- **last_change_coverage_pct:** {_s(overall, 'last_change_coverage_pct')}\n",
        f"- **months_since_change_coverage_pct:** {_s(overall, 'months_since_change_coverage_pct')}\n",
        "\n## Distributions (finite values)\n\n",
        md_table(dist) if dist is not None else "_missing_\n",
        "\nExtreme `|last_consumer_price_change_pct|` values are **flagged, not clipped** "
        "(one row per product × origin).\n\n",
        md_table(extremes, max_rows=25) if extremes is not None else "_none_\n",
        "\n## Coverage by PRIMARY origin\n\n",
        md_table(by_o) if by_o is not None else "_missing_\n",
        "\n## Coverage by product\n\n",
        md_table(by_p, max_rows=60) if by_p is not None else "_missing_\n",
        "\n## Temporal variation\n\n",
        "Distinguish a true time-varying feature from almost-static product metadata. "
        "`n_distinct_price_states` counts distinct positive consumer prices in the "
        "frozen history. `n_distinct_price_states_across_primary_origins` counts "
        "distinct `consumer_price_asof_origin` values on the five PRIMARY origins.\n\n",
        f"**MVP products whose as-of price actually changes across PRIMARY origins:** "
        f"{n_vary if n_vary is not None else 'n/a'} of {n_prod}.\n\n",
        md_table(temporal, max_rows=60) if temporal is not None else "_missing_\n",
        "\n## What was not done\n\n",
        "- No XGBoost, no WMAPE, no `FamilySession` / `evaluate_f3b`.\n",
        "- No clip, inflation adjustment, distributor/pharmacy scores, or "
        "price × horizon interactions.\n",
        "- Transformations were not chosen by PRIMARY performance (log is ex ante).\n",
        "- Step 3 was not started. Frozen v1 panels and Step 1 source parquet were not written.\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
