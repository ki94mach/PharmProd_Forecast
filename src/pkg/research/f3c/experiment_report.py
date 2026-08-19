"""Write docs/f3c_inventory.md from F3C Step 3 CSV artifacts."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.research.f3c.config import docs_dir, f3c_output_dir
from pkg.research.harness.report import md_table

VERDICT_TEXT = {
    "A": (
        "A — distributor inventory robustly useful for both TS and Human. "
        "Retain as promising research evidence requiring future/shadow-origin confirmation."
    ),
    "B": (
        "B — distributor + factory robustly useful. "
        "Retain as promising research evidence requiring future/shadow-origin confirmation."
    ),
    "C": (
        "C — anchor-specific usefulness. "
        "Retain for the improving anchor; investigate the non-improving one."
    ),
    "D": (
        "D — weak/non-robust signal; descriptive only. "
        "Do not automatically retain as scored feature."
    ),
    "E": (
        "E — current inventory representation fails. "
        "Do not tune month-end definition, status composition, transforms, or hyperparameters."
    ),
}


def _row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df.loc[df["experiment"] == name]
    return sub.iloc[0] if len(sub) else None


def _yes_no_improve(r: Optional[pd.Series]) -> str:
    if r is None:
        return "not run"
    rel = float(r["rel_wmape_vs_control_pct"])
    if rel > 0:
        return f"yes ({rel:+.2f}% relative WMAPE vs {r['control']})"
    return f"no ({rel:+.2f}% relative WMAPE vs {r['control']})"


def write_f3c_results(report: dict, *, path: Optional[Path] = None) -> Path:
    out = path or (docs_dir() / "f3c_inventory.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    overall = report.get("overall")
    verdict = report.get("verdict", "unknown")
    gates = report.get("gates")

    i1t = _row(overall, "I1_TS_DISTRIBUTOR")
    i1h = _row(overall, "I1_HUMAN_DISTRIBUTOR")
    i2t = _row(overall, "I2_TS_DISTRIBUTOR_FACTORY")
    i2h = _row(overall, "I2_HUMAN_DISTRIBUTOR_FACTORY")

    lines = [
        "# F3C — Point-in-time month-end inventory",
        f"**Date:** {date.today()}  ",
        f"**CSV artifacts:** `src/data/results/f3c`",
        "",
        "F3C is a hypothesis evaluated on an already reused research test panel. "
        "Results are useful for research direction but should not be treated as "
        "unbiased production estimates.",
        "",
        "## Business contract",
        "",
        "**Distributor inventory:**",
        "- Source: `[DWOrchid].[dbo].[FactInventoryHistorical]`",
        "- Exact previous Shamsi month-end",
        "- موجودی + در راه",
        "- بلوکه excluded",
        "",
        "**Factory inventory:**",
        "- Source: `[DWOrchid].[dbo].[FactInventory]`",
        "- Exact previous Shamsi month-end",
        "- SUM(DQty) for FkProvider rows",
        "- No reserved/quarantine/delivery subtraction",
        "",
        "## Scored features",
        "",
        "- `log_distributor_inventory_qty` = log1p(distributor_inventory_qty)",
        "- `log_factory_inventory_qty` = log1p(factory_inventory_qty)",
        "",
        "## F0 reproduction",
        "",
        md_table(gates, max_rows=5) if gates is not None else "_No gates._",
        "",
        "## Overall results",
        "",
        md_table(overall, max_rows=10, cols=[
            "experiment", "anchor", "control", "wmape", "wmape_control",
            "rel_wmape_vs_control_pct", "rmse", "mae", "bias", "n",
            "origins_improved", "origins_total", "product_win_rate",
            "median_product_improvement_pct",
        ]) if overall is not None else "_No results._",
        "",
        "## Verdict questions",
        "",
        f"1. **Did canonical F0 reproduction pass?** "
        f"{'yes' if gates is not None and bool(gates['ok'].all()) else 'no'}",
        f"2. **Does distributor month-end inventory improve TS?** {_yes_no_improve(i1t)}",
        f"3. **Does distributor month-end inventory improve Human?** {_yes_no_improve(i1h)}",
        f"4. **Relative WMAPE improvement:** TS={float(i1t['rel_wmape_vs_control_pct']):.2f}%, "
        f"Human={float(i1h['rel_wmape_vs_control_pct']):.2f}%" if i1t is not None and i1h is not None else "",
        f"5. **Origins improved:** TS={int(i1t['origins_improved'])}/{int(i1t['origins_total'])}, "
        f"Human={int(i1h['origins_improved'])}/{int(i1h['origins_total'])}" if i1t is not None and i1h is not None else "",
        f"6. **Products improved:** TS win_rate={float(i1t['product_win_rate']):.2f}, "
        f"Human win_rate={float(i1h['product_win_rate']):.2f}" if i1t is not None and i1h is not None else "",
        "7. **Gains/losses:** See error_concentration.csv and high_volume_watchlist.csv.",
        "8. **High-volume products:** See watchlist.",
        f"9. **Does factory add incremental value?** TS: {_yes_no_improve(i2t)}, "
        f"Human: {_yes_no_improve(i2h)}",
        "10. **Inventory state concentration:** See inventory_regime_analysis.csv.",
        "11. **Feature usage:** See feature_importance.csv (diagnostic only, not promotion evidence).",
        f"12. **Should F3C inventory be retained?** Verdict: **{verdict}**",
        "",
        f"## Verdict: {verdict}",
        "",
        VERDICT_TEXT.get(verdict, f"Unknown verdict: {verdict}"),
        "",
        "## Research limitation",
        "",
        "The five PRIMARY origins have already been repeatedly used for previous "
        "feature-family research. Therefore any positive F3C result must be described as "
        "**promising research evidence requiring future/shadow-origin confirmation**, "
        "not unbiased production performance.",
        "",
        "## What was not done",
        "",
        "- No factory-only, on-hand-only, in-transit-only, blocked, price, F3A, interactions.",
        "- No hyperparameter tuning, early-stopping redesign, or feature subset search.",
        "- No SHAP analysis.",
        "- F0/F1/F2/F3A/F3B artifacts were not modified.",
        "- Frozen benchmark v1 panels were not changed.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_gate_failure(msg: str, *, out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else f3c_output_dir()
    path = out_dir / "gate_failure.txt"
    path.write_text(msg, encoding="utf-8")
    return path
