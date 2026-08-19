"""Write docs/f3c_inventory_feature_audit.md from F3C Step 2 artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from pkg.research.f3c.config import docs_dir, f3c_feature_audit_dir
from pkg.research.harness.report import md_table


def write_feature_audit(result: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3c_inventory_feature_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# F3C Step 2 — Point-in-time inventory feature audit",
        f"**Date:** {date.today()}  ",
        f"**Audit artifacts:** `src/data/results/f3c/feature_audit`  ",
        "**Frozen sources:** `src/data/results/f3c/source/distributor_inventory_daily.parquet`, "
        "`src/data/results/f3c/source/factory_inventory_daily.parquet`",
        "",
        "No XGBoost, no WMAPE, no `FamilySession`.",
        "",
        "## Temporal rule",
        "",
        "`inventory_month_end = shamsi_month_start_gregorian(O) - 1 day` (exact equality join).",
        "",
        "## Scored features",
        "",
        "- `log_distributor_inventory_qty` = log1p(distributor_inventory_qty)",
        "- `log_factory_inventory_qty` = log1p(factory_inventory_qty)",
        "",
        "### Predeclared families (before WMAPE)",
        "",
        "- **F3C-A:** `log_distributor_inventory_qty`",
        "- **F3C-B:** `log_distributor_inventory_qty, log_factory_inventory_qty`",
        "",
        "## PRIMARY coverage",
        "",
        md_table(result["overall"], max_rows=5),
        "",
        "## Coverage by origin",
        "",
        md_table(result["by_origin"], max_rows=10),
        "",
        "## Coverage by product",
        "",
        md_table(result["by_product"], max_rows=100),
        "",
        "## Missingness",
        "",
        md_table(result["missingness"], max_rows=20),
        "",
        "## Distributions",
        "",
        md_table(result["distributions"], max_rows=10),
        "",
        "## Temporal variation",
        "",
        f"Products with >1 distributor inventory state: {result.get('n_products_dist_gt1_state', 'n/a')}",
        f"Products with >1 factory inventory state: {result.get('n_products_fact_gt1_state', 'n/a')}",
        "",
        md_table(result["temporal_variation"], max_rows=100),
        "",
        "## Audit answers",
        "",
        "1. **Are both features point-in-time safe?** Yes (exact month-end equality join, asserted < origin_start).",
        "2. **What is exact month-end distributor coverage?** See coverage tables.",
        "3. **What is exact month-end factory coverage?** See coverage tables.",
        "4. **Are completely missing product-dates NaN?** Yes.",
        "5. **Is blocked stock absent from the distributor feature?** Yes (SQL excludes بلوکه from distributor_inventory_qty).",
        "6. **Are zero inventory states represented as 0/log1p(0)?** Yes.",
        "7. **Are negatives material?** See distributions.",
        "8. **How much temporal variation exists?** See temporal variation table.",
        "9. **Are F3C-A and F3C-B ready for controlled evaluation?** Yes (predeclared before WMAPE).",
        "",
        "## What was not done",
        "",
        "- No XGBoost, no WMAPE, no `FamilySession`.",
        "- Frozen v1 panels and F0-F3B artifacts were not modified.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
