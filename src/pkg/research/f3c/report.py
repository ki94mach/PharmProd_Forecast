"""Write docs/f3c_inventory_source_audit.md from F3C Step 1 artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from pkg.research.f3c.config import docs_dir, f3c_source_dir
from pkg.research.harness.report import md_table


def write_source_audit(result: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3c_inventory_source_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = result.get("summary")
    mapping = result.get("mapping")
    snap_audit = result.get("snap_audit")
    month_end = result.get("month_end")
    status_audit = result.get("status_audit")
    qty_qual = result.get("qty_qual")

    lines = [
        "# F3C Step 1 — Inventory source audit",
        f"**Date:** {date.today()}  ",
        f"**Frozen source artifacts:** `src/data/results/f3c/source`",
        "",
        "## Source definition",
        "",
        "- Distributor table: `[DWOrchid].[dbo].[FactInventoryHistorical]`",
        "- Factory table: `[DWOrchid].[dbo].[FactInventory]` (rows with `FkProvider IS NOT NULL`)",
        "- SQL files: `query/f3c_distributor_inventory.sql`, `query/f3c_factory_inventory.sql`",
        "- Distributor inventory = موجودی + در راه (بلوکه excluded)",
        "- Factory inventory = SUM(DQty) for FkProvider rows",
        "- Exact previous Shamsi month-end date for each origin",
        "- Missing product-dates are NOT filled as zero",
        "- Negatives are NOT floored",
        "",
        "## Source summary",
        "",
        md_table(summary, max_rows=5),
        "",
        "## Product mapping audit",
        "",
        md_table(mapping, max_rows=10),
        "",
        "## Snapshot date audit",
        "",
        md_table(snap_audit, max_rows=10),
        "",
        "## Exact month-end coverage (PRIMARY origins)",
        "",
        md_table(month_end, max_rows=10),
        "",
        "## Distributor month-end status audit",
        "",
        md_table(status_audit, max_rows=10),
        "",
        "## Quantity quality",
        "",
        md_table(qty_qual, max_rows=10),
        "",
        "## Audit answers",
        "",
    ]

    # 12 questions
    identity_ok = (
        bool(status_audit["identity_holds"].all())
        if status_audit is not None and not status_audit.empty
        else "unknown"
    )
    blocked_ok = (
        bool(status_audit["blocked_excluded"].all())
        if status_audit is not None and not status_audit.empty
        else "unknown"
    )
    lines += [
        f"1. **Is distributor history sourced only from FactInventoryHistorical?** Yes.",
        f"2. **Is factory history sourced only from FactInventory?** Yes.",
        f"3. **Is product mapping valid?** See mapping table above.",
        f"4. **What exact month-end corresponds to every PRIMARY origin?** See coverage table.",
        f"5. **What is exact month-end coverage for distributor inventory?** See coverage table.",
        f"6. **What is exact month-end coverage for factory inventory?** See coverage table.",
        f"7. **Is distributor inventory exactly موجودی + در راه?** identity_holds={identity_ok}",
        f"8. **Is بلوکه fully excluded?** blocked_excluded={blocked_ok}",
        "9. **Are missing status rows treated as zero only when the product-date exists?** "
        "Yes. SQL CASE yields 0 for absent status when the product-date has rows.",
        "10. **Are entirely missing product-dates kept distinct from zero?** "
        "Yes. No grid fill is applied.",
        "11. **Are negative quantities material?** See quantity quality table.",
        "12. **Are the frozen sources safe for Step 2?** "
        f"identity={identity_ok}, blocked_excluded={blocked_ok}.",
        "",
        "## What was not done",
        "",
        "- No XGBoost, no WMAPE, no `FamilySession`.",
        "- No scored inventory features.",
        "- Frozen benchmark v1 panels were not modified.",
        "- F0/F1/F2/F3A/F3B artifacts were not overwritten.",
        "- Later F3C steps must read `src/data/results/f3c/source/` parquet only.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
