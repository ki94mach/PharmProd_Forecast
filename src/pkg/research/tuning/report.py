"""Write docs/m1_optuna_tuning.md from M1 evaluation artifacts."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pkg.research.tuning.config import (
    BASELINE_HUMAN_WMAPE_REF,
    BASELINE_TS_WMAPE_REF,
    INNER_EARLY_STOPPING_ROUNDS,
    INNER_EVAL_METRIC,
    INNER_MIN_BUDGET_VINTAGES,
    INNER_MIN_HISTORY_MONTHS,
    INNER_MIN_TRAIN_ROWS,
    MIN_INNER_FOLDS,
    N_TRIALS,
    OPTUNA_SEED,
    PRE_PRIMARY_CUTOFF,
    docs_dir,
    m1_output_dir,
)


def _load(out_dir: Path, name: str) -> pd.DataFrame:
    p = out_dir / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_json(out_dir: Path, name: str) -> dict:
    p = out_dir / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.2f}%"


def _md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "_no data_"
    df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if np.isfinite(v) else "n/a")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _get_overall_row(overall: pd.DataFrame, anchor: str) -> Optional[pd.Series]:
    if overall.empty:
        return None
    sub = overall.loc[overall["anchor"] == anchor]
    return sub.iloc[0] if len(sub) else None


def write_m1_report(
    eval_result: Optional[dict[str, Any]] = None,
    optuna_results: Optional[dict[str, Any]] = None,
    *,
    out_dir: Optional[Path] = None,
    docs_output: Optional[Path] = None,
) -> Path:
    """Write docs/m1_optuna_tuning.md and return the path."""
    out_dir = out_dir or m1_output_dir()
    docs_output = docs_output or (docs_dir() / "m1_optuna_tuning.md")

    overall = _load(out_dir, "overall.csv")
    by_origin = _load(out_dir, "by_origin.csv")
    by_horizon = _load(out_dir, "by_horizon.csv")
    by_product = _load(out_dir, "by_product.csv")
    conc_df = _load(out_dir, "error_concentration.csv")
    watch_df = _load(out_dir, "high_volume_watchlist.csv")
    importance = _load(out_dir, "optuna_parameter_importance.csv")

    ts_params = _load_json(out_dir, "ts_best_params.json")
    human_params = _load_json(out_dir, "human_best_params.json")

    verdicts: dict[str, str] = {}
    if eval_result and "verdicts" in eval_result:
        verdicts = eval_result["verdicts"]
    else:
        for anchor in ("ts", "human"):
            r = _get_overall_row(overall, anchor)
            if r is not None and "verdict" in r:
                verdicts[anchor] = str(r["verdict"])

    ts_row = _get_overall_row(overall, "ts")
    human_row = _get_overall_row(overall, "human")

    def _gate_status(row):
        if row is None:
            return "NOT RUN"
        return f"PASSED (baseline WMAPE {_fmt(row.get('wmape_baseline', float('nan')))})"

    def _verdict_label(v: str) -> str:
        labels = {
            "PROMOTE": "**PROMOTE**",
            "WEAK_NEEDS_CONFIRMATION": "**WEAK / NEEDS CONFIRMATION**",
            "REJECT": "**REJECT**",
        }
        return labels.get(v, v or "n/a")

    # F0 canonical params for comparison
    f0_defaults = dict(
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        n_estimators=200, min_child_weight=1, reg_alpha=0, reg_lambda=1, gamma=0,
    )

    lines: list[str] = [
        "# M1 Optuna Hyperparameter Optimization Report",
        f"**Date:** {date.today()}",
        "**Experiment:** M1A (TS + XGB) and M1B (Human + XGB) hyperparameter tuning via Optuna.",
        "",
        "> Hyperparameters tuned ONLY on pre-PRIMARY origins. PRIMARY panel is evaluation only.",
        "> No F3A–F3E features. No SQL. No post-hoc retuning after observing PRIMARY.",
        "",
        "---",
        "",
        "## Design Summary",
        "",
        f"- Pre-PRIMARY cutoff: `{PRE_PRIMARY_CUTOFF}` (all tuning uses origins < this)",
        f"- Optuna trials per anchor: {N_TRIALS}",
        f"- Sampler: TPESampler(seed={OPTUNA_SEED}), direction=minimize, no pruner",
        f"- Inner fit: n_estimators={3000}, early_stopping_rounds={INNER_EARLY_STOPPING_ROUNDS}, eval_metric={INNER_EVAL_METRIC!r}",
        f"- Min inner folds required: {MIN_INNER_FOLDS}",
        f"- Inner eligibility: train rows >= {INNER_MIN_TRAIN_ROWS}, unique months >= {INNER_MIN_HISTORY_MONTHS}; Human also: vintages >= {INNER_MIN_BUDGET_VINTAGES}",
        f"- PRIMARY panel: n=1877, 5 origins (140404, 140407, 140410, 140501, 140504)",
        "",
        "---",
        "",
        "## Q1. Did canonical F0 reproduction pass?",
        "",
        f"- TS: {_gate_status(ts_row)} (current-env reference {BASELINE_TS_WMAPE_REF}, tolerance ±0.10)",
        f"- Human: {_gate_status(human_row)} (current-env reference {BASELINE_HUMAN_WMAPE_REF}, tolerance ±0.10)",
        "",
    ]

    # Q2–Q4 Inner origins
    lines += [
        "## Q2. Pre-PRIMARY origins used for TS tuning?",
        "",
        f"- Inner origins: {ts_params.get('inner_origins', 'n/a')}",
        f"- n_inner_folds: {ts_params.get('inner_origins', [])!r}",
        "",
        "## Q3. Pre-PRIMARY origins used for Human tuning?",
        "",
        f"- Inner origins: {human_params.get('inner_origins', 'n/a')}",
        "",
        "## Q4. Inner folds available per anchor?",
        "",
        f"- TS: {len(ts_params.get('inner_origins', []))} folds",
        f"- Human: {len(human_params.get('inner_origins', []))} folds",
        "",
    ]

    # Q5–Q7 Best params
    def _param_table(params: dict, label: str) -> list[str]:
        search_keys = [
            "max_depth", "min_child_weight", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
        ]
        rows_out = [
            f"| {k} | {_fmt(params.get(k, float('nan')), 6)} | {_fmt(f0_defaults.get(k, float('nan')), 6)} |"
            for k in search_keys
        ]
        return [
            f"### {label}",
            "",
            "| parameter | tuned | F0 default |",
            "| --- | --- | --- |",
        ] + rows_out + [
            "",
            f"- frozen_n_estimators: {params.get('frozen_n_estimators', 'n/a')}  "
            f"(F0 default: 200)",
            f"- best_inner_pooled_wmape: {_fmt(params.get('best_inner_pooled_wmape', float('nan')))}",
        ]

    lines += ["## Q5–Q7. Best hyperparameters and frozen n_estimators", ""]
    lines += _param_table(ts_params, "M1A — TS")
    lines += [""]
    lines += _param_table(human_params, "M1B — Human")
    lines += [""]

    # Q8 Best inner pooled WMAPE
    lines += [
        "## Q8. Best pre-PRIMARY pooled WMAPE?",
        "",
        f"- TS: {_fmt(ts_params.get('best_inner_pooled_wmape', float('nan')))}",
        f"- Human: {_fmt(human_params.get('best_inner_pooled_wmape', float('nan')))}",
        "",
    ]

    # Q9–Q11 PRIMARY results
    def _overall_block(row, anchor_label: str) -> list[str]:
        if row is None:
            return [f"_{anchor_label}: not evaluated_", ""]
        return [
            f"- baseline WMAPE: {_fmt(row.get('wmape_baseline', float('nan')))}",
            f"- tuned WMAPE:    {_fmt(row.get('wmape_tuned', float('nan')))}",
            f"- relative improvement: {_pct(row.get('rel_wmape_improvement_pct', float('nan')))}",
            f"- RMSE baseline/tuned: {_fmt(row.get('rmse_baseline', float('nan')))} / {_fmt(row.get('rmse_tuned', float('nan')))}",
            f"- MAE baseline/tuned:  {_fmt(row.get('mae_baseline', float('nan')))} / {_fmt(row.get('mae_tuned', float('nan')))}",
            f"- Bias baseline/tuned: {_fmt(row.get('bias_baseline', float('nan')))} / {_fmt(row.get('bias_tuned', float('nan')))} (delta {_fmt(row.get('bias_delta', float('nan')))})",
            f"- n: {row.get('n', 'n/a')}",
            "",
        ]

    lines += [
        "## Q9–Q11. PRIMARY WMAPE results and relative improvement",
        "",
        "### M1A — TS",
        "",
    ]
    lines += _overall_block(ts_row, "M1A TS")
    lines += ["### M1B — Human", ""]
    lines += _overall_block(human_row, "M1B Human")

    # Q12 Origins improved
    lines += [
        "## Q12. How many PRIMARY origins improve?",
        "",
    ]
    for anchor in ("ts", "human"):
        row = _get_overall_row(overall, anchor)
        if row is not None:
            oi = row.get("origins_improved", "n/a")
            ot = row.get("origins_total", 5)
            lines.append(f"- {anchor.upper()}: {oi} / {ot} origins improved")
    lines += [""]

    # Q13 Product win rate
    lines += [
        "## Q13. Percentage of products that improve?",
        "",
    ]
    for anchor in ("ts", "human"):
        row = _get_overall_row(overall, anchor)
        if row is not None:
            wr = row.get("product_product_win_rate", row.get("product_win_rate", float("nan")))
            med = row.get("product_median_product_improvement_pct", row.get("median_product_improvement_pct", float("nan")))
            lines.append(f"- {anchor.upper()}: win_rate={_pct(wr)}  median_product_improvement={_pct(med)}")
    lines += [""]

    # Q14 Bias
    lines += [
        "## Q14. Does tuning improve or worsen bias?",
        "",
    ]
    for anchor in ("ts", "human"):
        row = _get_overall_row(overall, anchor)
        if row is not None:
            lines.append(
                f"- {anchor.upper()}: bias_baseline={_fmt(row.get('bias_baseline', float('nan')))} "
                f"→ bias_tuned={_fmt(row.get('bias_tuned', float('nan')))} "
                f"(delta {_fmt(row.get('bias_delta', float('nan')))})"
            )
    lines += [""]

    # Q15 Error concentration
    lines += [
        "## Q15. Are gains/losses concentrated in high-volume products?",
        "",
        "### Error Concentration",
        "",
        _md_table(conc_df),
        "",
        "### High-Volume Watchlist",
        "",
        _md_table(watch_df),
        "",
    ]

    # Q16 By horizon
    lines += [
        "## Q16. Performance across forecast horizons",
        "",
        _md_table(by_horizon),
        "",
    ]

    # Q17 Params vs F0
    lines += [
        "## Q17. Are selected parameters materially different from F0?",
        "",
        "F0 canonical defaults: max_depth=4, learning_rate=0.05, subsample=0.8, "
        "colsample_bytree=0.8, n_estimators=200, min_child_weight=1, reg_alpha=0, "
        "reg_lambda=1, gamma=0.",
        "",
        "Tuned values shown in Q5–Q7 tables above.",
        "",
    ]

    # Q18–Q19 Recommendations
    ts_verdict = verdicts.get("ts", "n/a")
    human_verdict = verdicts.get("human", "n/a")

    lines += [
        "## Q18. Should tuned TS replace untuned TS?",
        "",
        f"Verdict M1A: {_verdict_label(ts_verdict)}",
        "",
        "## Q19. Should tuned Human replace untuned Human?",
        "",
        f"Verdict M1B: {_verdict_label(human_verdict)}",
        "",
    ]

    # Full tables
    lines += [
        "---",
        "",
        "## Overall Metrics",
        "",
        _md_table(overall),
        "",
        "## By Origin",
        "",
        _md_table(by_origin),
        "",
        "## By Product (top 50 by actual volume)",
        "",
        _md_table(
            by_product.sort_values("actual_volume", ascending=False).head(50)
            if not by_product.empty and "actual_volume" in by_product.columns
            else by_product
        ),
        "",
        "## Optuna Parameter Importance (descriptive only)",
        "",
        _md_table(importance),
        "",
        "---",
        "",
        "## Verdicts",
        "",
        "Both verdicts are pre-specified from inner-validation logic. "
        "PRIMARY outcomes did NOT influence parameter selection.",
        "",
        f"**M1A — TS tuning:** {_verdict_label(ts_verdict)}",
        "",
        f"**M1B — Human tuning:** {_verdict_label(human_verdict)}",
        "",
        "> Even a PROMOTE verdict represents *research evidence requiring future/shadow-origin "
        "confirmation*, not unbiased production performance. The five PRIMARY origins have "
        "been used repeatedly in prior feature research.",
        "",
        "---",
        "_Generated by pkg.research.evaluate_m1_",
    ]

    text = "\n".join(lines) + "\n"
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(text, encoding="utf-8")
    print(f"Report written to {docs_output}")
    return docs_output
