"""Generate docs/m2_residual_learner_benchmark.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pkg.research.model_benchmark.config import (
    EXPECTED_MATCHED_N,
    EXPECTED_MATCHED_ORIGINS,
    M1R_F0_WMAPE_REF,
    docs_path,
)


def _pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x * 100:.2f}%"


def write_m2_report(
    *,
    out_dir: Path,
    env: dict[str, Any],
    repeat_summary: dict[str, Any],
    f0_ts_primary_wmape: float,
    f0_human_primary_wmape: float,
    ts_suite_meta: dict[str, Any],
    human_suite_meta: dict[str, Any],
    ts_overall: pd.DataFrame,
    human_overall: pd.DataFrame,
    ts_primary: pd.DataFrame,
    matched_primary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    verdicts: dict[str, dict[str, str]],
    overall_conclusion: str,
    tree_repeat: pd.DataFrame,
) -> Path:
    path = docs_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_No rows._"
        try:
            return df.to_markdown(index=False, floatfmt=".4f")
        except ImportError:
            return "```\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n```"

    ts_w = ts_overall.loc[ts_overall["model"] == "xgboost", "wmape"]
    hu_w = human_overall.loc[human_overall["model"] == "xgboost", "wmape"]

    text = f"""# M2 — Residual Learner Benchmark

Fixed-configuration model-class comparison: XGBoost F0 (reference), Ridge, ElasticNet,
CatBoost, LightGBM. **No hyperparameter tuning.**

**PRIMARY and matched PRIMARY slices are exploratory diagnostics**, repeatedly inspected
in prior F3/M1 research — not unbiased holdout performance. Any promoted learner requires
future shadow-origin confirmation before production replacement.

Human history is much shorter than TS; interpret Human results cautiously.

## Reproducibility

1. **Deterministic F0 reproduction passed?** {"Yes" if repeat_summary.get("max_abs_prediction_diff", 1) <= 1e-9 else "No"}
2. Max repeated XGBoost prediction diff: `{repeat_summary.get("max_abs_prediction_diff", "n/a")}`
3. TS PRIMARY XGBoost F0 WMAPE: `{f0_ts_primary_wmape:.4f}` (M1R ref `{M1R_F0_WMAPE_REF:.4f}`, not forced)
4. Human PRIMARY XGBoost F0 WMAPE: `{f0_human_primary_wmape:.4f}`
5. Environment: `{json.dumps(env)}`

## Origins and scale

6. **TS historical origins:** `{ts_suite_meta.get("origins", [])}` (first `{ts_suite_meta.get("first_origin")}`, last `{ts_suite_meta.get("last_origin")}`)
7. **Human origins:** `{human_suite_meta.get("origins", [])}`
8. TS rows/products/origins: n=`{ts_suite_meta.get("n")}`, products=`{ts_suite_meta.get("n_products")}`, origins=`{ts_suite_meta.get("n_origins")}`
9. Human rows/products/origins: n=`{human_suite_meta.get("n")}`, products=`{human_suite_meta.get("n_products")}`, origins=`{human_suite_meta.get("n_origins")}`

## TS results (M2A broad history)

10. TS XGBoost F0 WMAPE: `{float(ts_w.iloc[0]) if len(ts_w) else float("nan"):.4f}`

11. Alternative TS WMAPEs:

{_table(ts_overall.loc[ts_overall["model"] != "xgboost", ["model", "wmape", "relative_wmape_vs_xgb_pct", "origins_improved_vs_xgb", "product_win_rate"]])}

12. Models beating XGBoost on aggregate TS WMAPE: `{", ".join(ts_overall.loc[(ts_overall["model"] != "xgboost") & (ts_overall["relative_wmape_vs_xgb_pct"] > 0), "model"].tolist()) or "none"}`

13. Models beating XGBoost on majority of TS origins: `{", ".join(ts_overall.loc[(ts_overall["model"] != "xgboost") & (ts_overall["origins_improved_vs_xgb"] > ts_overall["origins_total"] / 2), "model"].tolist()) or "none"}`

14. Product win rates (aggregate): see `by_product.csv` and `ts_overall.csv`.

15. Error concentration / high-volume watchlist: see `error_concentration.csv`, `high_volume_watchlist.csv`.

16. By horizon: see `by_horizon.csv`.

## Human results (M2B)

17. Human XGBoost F0 WMAPE: `{float(hu_w.iloc[0]) if len(hu_w) else float("nan"):.4f}`

18. Human alternative WMAPEs:

{_table(human_overall.loc[human_overall["model"] != "xgboost", ["model", "wmape", "relative_wmape_vs_xgb_pct"]])}

19. Human results stable enough? {"Limited sample — interpret cautiously." if human_suite_meta.get("n_origins", 0) <= 5 else "Moderate origin count — still shorter than TS."}

## Matched PRIMARY (n={EXPECTED_MATCHED_N}, origins={EXPECTED_MATCHED_ORIGINS})

20. Best anchor × learner on matched PRIMARY:

{_table(matched_primary.sort_values("wmape"))}

## Verdicts

| Anchor | Model | Verdict |
|--------|-------|---------|
"""
    for anchor, models in verdicts.items():
        for model, v in sorted(models.items()):
            text += f"| {anchor} | {model} | {v} |\n"

    text += f"""
**Overall M2 conclusion:** `{overall_conclusion}`

- CASE A: one alternative clearly beats XGBoost → promote for dedicated tuning
- CASE B: competitive but no clear winner → XGBoost remains default
- CASE C: XGBoost clearly strongest → stop model-class search for now

## Diagnostic questions

21. Does any alternative robustly beat XGBoost? See verdicts above.
22. Ridge/ElasticNet surprisingly competitive? Compare linear model WMAPEs in overall tables.
23. CatBoost native categoricals help? Compare CatBoost vs XGBoost breadth metrics.
24. LightGBM vs XGBoost? Compare `lightgbm` row in overall tables.
25. Next tuning candidate? First model with `BEATS_XGBOOST` verdict, else none yet.

## Tree repeatability (2× run)

{_table(tree_repeat)}

## TS PRIMARY diagnostic slice

{_table(ts_primary)}

## Artifacts

All outputs under `{out_dir}`.

**STOP — no automatic tuning, ensembles, or F3 features after M2.**
"""
    path.write_text(text, encoding="utf-8")
    return path
