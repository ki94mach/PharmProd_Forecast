"""Write docs/f3b_price_source_audit.md from F3B Step 1 source artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pkg.research.f3b.config import docs_dir, f3b_source_dir
from pkg.research.harness.report import md_table, read_csv, repo_relative


def _s(summary: Optional[pd.DataFrame], key: str, default: str = "n/a") -> str:
    if summary is None or summary.empty or key not in summary.columns:
        return default
    v = summary.iloc[0][key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    if isinstance(v, float) and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_price_source_audit(
    report: dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else report.get("out_dir") or f3b_source_dir()
    path = path or (docs_dir() / "f3b_price_source_audit.md")
    summary = report.get("summary")
    if summary is None:
        summary = read_csv(out_dir, "source_summary.csv")
    mvp = report.get("mvp")
    if mvp is None:
        mvp = read_csv(out_dir, "mvp_product_coverage.csv")
    origins = report.get("origins")
    if origins is None:
        origins = read_csv(out_dir, "origin_coverage.csv")
    unmatched = report.get("unmatched")
    if unmatched is None:
        unmatched = read_csv(out_dir, "unmatched_products.csv")
    ambiguous = report.get("ambiguous")
    if ambiguous is None:
        ambiguous = read_csv(out_dir, "ambiguous_products.csv")
    collapsed = report.get("collapsed")
    if collapsed is None:
        collapsed = read_csv(out_dir, "duplicate_collapsed.csv")
    pack_qty = report.get("pack_qty_conflicts")
    if pack_qty is None:
        pack_qty = read_csv(out_dir, "pack_quantity_only_conflicts.csv")
    conflicts = report.get("conflicts")
    if conflicts is None:
        conflicts = read_csv(out_dir, "conflicting_prices.csv")
    mapping = report.get("mapping")
    if mapping is None:
        mapping = read_csv(out_dir, "product_name_map.csv")

    sections = [
        "# F3B Step 1 — Price history source audit\n",
        f"**Date:** {date.today().isoformat()}  \n",
        f"**Frozen source artifacts:** `{repo_relative(Path(out_dir))}`\n",
        "\nThis step freezes a product-level Shamsi price history for later "
        "point-in-time features. It does **not** train XGBoost, compute WMAPE, "
        "or choose price transformations.\n",
        "\n## Source definition\n\n",
        "- Workbook: `src/data/external/f3b_price/فرم قیمت سه گانهsc-fr-008 (2).xlsx`\n",
        "- History sheet **by name**: `جدول تغییر قیمت ها` (Excel table `Table10`, not sheet index).\n",
        "- Replacement map: `src/data/external/f3b_price/Map Product-Delivery dis.xlsx` sheet `map` "
        "(exact normalized names only; no fuzzy match).\n",
        "- Dim join: `[Iris_DW].[Dim].[Product].[Title]` after the map; canonical SKU is `ProductTitleEN`.\n",
        "- Dates: Shamsi `YYYY/MM/DD` → `effective_date` (YYYYMMDD) and `effective_month` (YYYYMM) "
        "via `parse_shamsi_ymd`. Placeholders such as `0000/00/00` are rejected. "
        "No YYYYMM integer subtraction; no Gregorian `to_datetime` on these strings.\n",
        "- Modeling history requires MATCHED Dim.Product, a valid effective date, and all three "
        "prices present and strictly positive. Missing prices are not imputed or forward-filled.\n",
        "- Product/date groups: identical prices+pack collapse to one row; "
        "identical prices with differing ``pack_quantity`` also collapse to one canonical row "
        "(original rows kept in `pack_quantity_only_conflicts.csv`); "
        "true price conflicts (any of distributor/pharmacy/consumer differ) are excluded "
        "entirely. Pack quantity alone is not a price conflict.\n",
        "\n## Counts\n\n",
        md_table(summary) if summary is not None else "_missing source_summary.csv_\n",
        "\n## Audit answers\n\n",
        f"1. **How many raw price-history rows were found?** {_s(summary, 'n_raw_rows')}\n",
        f"2. **How many unique Persian product names?** {_s(summary, 'n_unique_source_names')}\n",
        f"3. **How many were changed by the explicit mapping workbook?** {_s(summary, 'n_names_changed_by_map')}\n",
        f"4. **What percentage map exactly to Dim.Product?** {_s(summary, 'match_pct_dim_product')}% "
        f"({_s(summary, 'n_source_names_matched')} of {_s(summary, 'n_unique_source_names')} source names).\n",
        f"5. **How many are unmatched?** {_s(summary, 'n_source_names_unmatched')}\n",
        f"6. **How many are ambiguous?** {_s(summary, 'n_source_names_ambiguous')} "
        "(not resolved; none of these enter `price_history.parquet`).\n",
        f"7. **How many valid dated price observations remain?** {_s(summary, 'n_valid_dated_observations')}\n",
        f"8. **Date range of valid observations?** Shamsi day {_s(summary, 'valid_date_min')}–"
        f"{_s(summary, 'valid_date_max')} "
        f"(months {_s(summary, 'valid_month_min')}–{_s(summary, 'valid_month_max')}).\n",
        f"9. **How many identical-price duplicate product/date groups?** "
        f"{_s(summary, 'n_identical_price_duplicate_groups')} groups "
        f"({_s(summary, 'n_rows_collapsed_as_duplicates')} rows collapsed; "
        "same distributor/pharmacy/consumer and same pack quantity).\n",
        f"10. **How many pack-quantity-only conflict groups?** "
        f"{_s(summary, 'n_pack_quantity_only_conflict_groups')} "
        "(prices identical; pack quantity differs; collapsed into history; "
        "original rows preserved in `pack_quantity_only_conflicts.csv`).\n",
        f"11. **How many true price-conflict groups?** "
        f"{_s(summary, 'n_true_price_conflict_groups')} "
        "(at least one of distributor/pharmacy/consumer differs; "
        "excluded entirely from `price_history.parquet`; no silent pick).\n",
        f"12. **Coverage for the 55 MVP products?** {_s(summary, 'n_mvp_with_valid_price_history')} of "
        f"{_s(summary, 'n_mvp_products')} "
        f"({_s(summary, 'mvp_coverage_pct')}%) have at least one valid observation.\n",
        "13. **Per MVP product** first/last date, observation count, and whether history exists: "
        "see the table below and `mvp_product_coverage.csv`.\n",
        "\n## Explicit replacement map\n\n",
        md_table(mapping, max_rows=20) if mapping is not None else "_missing_\n",
        "\n## Unmatched source names\n\n",
        md_table(unmatched, max_rows=40) if unmatched is not None else "_none_\n",
        "\n## Ambiguous Dim.Product titles\n\n",
        md_table(ambiguous, max_rows=40) if ambiguous is not None else "_none_\n",
        "\n## Collapsed identical duplicates\n\n",
        md_table(collapsed, max_rows=20) if collapsed is not None else "_none_\n",
        "\n## Pack-quantity-only conflicts (kept in history)\n\n",
        "Prices match; ``pack_quantity`` differs. Canonical observation is the first "
        "Excel row in the group. Original rows are preserved here.\n\n",
        md_table(pack_qty, max_rows=20) if pack_qty is not None else "_none_\n",
        "\n## True price conflicts (excluded from history)\n\n",
        "At least one of distributor, pharmacy, or consumer price differs. "
        "Original rows are preserved here; none enter `price_history.parquet`.\n\n",
        md_table(conflicts, max_rows=20) if conflicts is not None else "_none_\n",
        "\n## MVP product coverage\n\n",
        md_table(mvp, max_rows=60) if mvp is not None else "_missing_\n",
        "\n## PRIMARY origin coverage (no features)\n\n",
        "Count of MVP products with at least one valid observation whose "
        "`effective_month` is strictly before the origin. This is coverage only.\n\n",
        md_table(origins) if origins is not None else "_missing_\n",
        "\n## What was not done\n\n",
        "- No XGBoost training, no WMAPE, no residual backtest, no `FamilySession`.\n",
        "- No price-change thresholds, no Step 2 features, no `evaluate_f3b`.\n",
        "- Frozen benchmark v1 panels / `raw/sales.parquet` were not written.\n",
        "- F0 / F1 / F2 / F3A artifacts were not overwritten.\n",
        "- Later F3B experiments must read `src/data/results/f3b/source/` only "
        "(no live SQL, no live Excel).\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sections), encoding="utf-8")
    return path
