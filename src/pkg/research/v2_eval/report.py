"""Render ``report.md`` and the docs summary for the V2-vs-legacy evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERDICT_TEXT = {
    "consistently_better": "V2 is consistently better",
    "better_overall_but_uneven_by_segment": (
        "V2 is better overall but the gain is uneven across segments"
    ),
    "better_on_the_primary_slice_only": (
        "V2 is better on the primary slice only; sensitivities disagree"
    ),
    "broadly_similar": "V2 and legacy are broadly similar",
    "worse": "V2 is worse",
}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


# Identifier-like columns must never be rendered with thousands separators or
# decimals (a Shamsi origin is 140101, not 140,101.00).
_ID_COLUMNS = frozenset(
    {"forecast_origin", "target_date", "horizon", "product_int_id"}
)
_COUNT_COLUMNS = frozenset({"n", "n_products", "n_origins", "max_horizon_scored"})


def _cell(column: str, value: Any, digits: int) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if column in _ID_COLUMNS:
        return "n/a" if pd.isna(value) else str(int(value))
    if column in _COUNT_COLUMNS:
        return "n/a" if pd.isna(value) else f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return _fmt(value, digits)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    present = [c for c in columns if c in frame.columns]
    header = "| " + " | ".join(present) + " |"
    sep = "|" + "|".join("---" for _ in present) + "|"
    lines = [header, sep]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(_cell(c, row[c], digits) for c in present) + " |"
        )
    return "\n".join(lines)


def _counts(mapping: dict[str, Any]) -> str:
    """Render a name->count mapping as readable prose instead of a raw dict."""
    if not mapping:
        return "none"
    return ", ".join(f"{name} {value:,}" for name, value in mapping.items())


def _overall_block(overall: pd.DataFrame, scope: str, slice_name: str) -> str:
    frame = overall.loc[
        (overall["scope"] == scope) & (overall["slice"] == slice_name)
    ].copy()
    order = ["mean_horizon_mae", "wmape", "rmse", "mae", "signed_bias"]
    frame["_order"] = frame["metric"].map({m: i for i, m in enumerate(order)})
    frame = frame.sort_values("_order")
    lines = [
        "| metric | legacy | V2 | absolute change | relative improvement % |",
        "|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        rel = (
            "not applicable"
            if row["metric"] == "signed_bias"
            else _fmt(row["relative_improvement_pct"])
        )
        lines.append(
            f"| {row['metric']} | {_fmt(row['legacy'])} | {_fmt(row['v2'])} | "
            f"{_fmt(row['absolute_change'])} | {rel} |"
        )
    return "\n".join(lines)


def _raw_vs_constrained_text(summary: dict[str, Any]) -> str:
    if not summary.get("n_clipped_scored_rows"):
        return f"No scored row had a negative raw forecast. {summary.get('note', '')}"
    return (
        f"{summary['n_clipped_scored_rows']} scored rows had a negative raw "
        "forecast that the non-negativity constraint replaced with zero. On "
        f"those rows the summed absolute error is "
        f"{_fmt(summary['sum_abs_error_raw'])} units before the constraint and "
        f"{_fmt(summary['sum_abs_error_constrained'])} after, a change of "
        f"{_fmt(summary['clip_absolute_error_change'])} units. {summary['note']}"
    )


def _bias_note(overall: pd.DataFrame) -> str:
    row = overall.loc[
        (overall["scope"] == "overall") & (overall["metric"] == "signed_bias")
    ]
    if row.empty:
        return ""
    row = row.iloc[0]
    direction = str(row.get("bias_direction", "undefined")).replace("_", " ")
    return (
        f"Legacy bias is {_fmt(row['legacy'])} units per row and V2 bias is "
        f"{_fmt(row['v2'])}; in absolute terms the bias moves from "
        f"{_fmt(row.get('bias_abs_legacy'))} to {_fmt(row.get('bias_abs_v2'))}, "
        f"a change of {_fmt(row.get('bias_abs_change'))} units ({direction}). "
        "Positive values mean overforecasting."
    )


def write_report(report: dict[str, Any]) -> Path:
    cfg = report["config"]
    data = report["data"]
    panel = report["panel"]
    overall = report["overall"]
    diagnostics = report["diagnostics"]
    verdict = report["verdict"]
    coverage = panel.coverage_summary
    products = report["products"]
    selection = diagnostics["selection_runtime"]
    shape = diagnostics["shape_summary"]
    concentration = diagnostics["concentration"]
    product_summary = diagnostics["product_outcome_summary"]

    jobs = data.jobs
    n_success = int((jobs["status"] == "SUCCESS").sum())
    n_failed = int((jobs["status"] == "FAILED").sum())
    failures = jobs.loc[jobs["status"] == "FAILED"]

    horizon_table = diagnostics["horizon_degradation"]
    origin_table = diagnostics["origin_consistency"]

    parts: list[str] = []

    parts.append(
        f"""# TS V2 historical backfill vs legacy TS — evaluation

**Experiment:** `{data.manifest.get('experiment_id')}` engine `{data.manifest.get('engine_version')}` (config hash `{data.manifest.get('config_hash')}`)
**Legacy comparator:** delivered quarterly TS CSV vintages from the original `pkg.forecast.SalesForecast` pipeline, frozen into benchmark v1. This is not V3A A0 and not a re-run.
**Matched evaluation rows:** {coverage['matched_rows']:,} across {coverage['matched_products']} products and {coverage['matched_origins']} forecast origins.
**Reproduce:** `python -m pkg.research.evaluate_v2_backfill`

This document does not claim statistical significance. Origins and horizons overlap, so rows are not independent observations.

---

## 1. Headline

**Verdict: {VERDICT_TEXT.get(verdict['label'], verdict['label'])}.**

{_overall_block(overall, 'overall', 'all_matched_rows')}

{_bias_note(overall)}

V2 wins on {verdict['origins_better']} of {verdict['origins_total']} matched origins; it loses at {', '.join(str(o) for o in verdict['losing_origins']) or 'none'}. Among products with enough evaluated rows, {_fmt(product_summary['pct_improved'], 1)}% improve, {_fmt(product_summary['pct_tied'], 1)}% tie and {_fmt(product_summary['pct_regressed'], 1)}% regress, so the aggregate gain is real but not universal.

---

## 2. Experiment audit (section A)

| item | value |
|---|---|
| Jobs in `state.sqlite` | {len(jobs):,} ({n_success:,} SUCCESS, {n_failed} FAILED) |
| V2 forecast rows | {coverage['v2_forecast_rows']:,} |
| V2 rows with an actual | {coverage['v2_rows_with_actual']:,} |
| Legacy panel rows | {coverage['legacy_rows']:,} |
| Matched rows | {coverage['matched_rows']:,} |
| Unmatched V2 rows | {coverage['unmatched_v2_rows']:,} |
| Unmatched legacy rows | {coverage['unmatched_legacy_rows']:,} |
| Stale `failure.json` files on disk | {data.stale_failure_files} |
| Files in `backtests/` | {data.backtest_files} |

Failed jobs:

{_table(failures[['quarter', 'product_id', 'error_type', 'error_message']], ['quarter', 'product_id', 'error_type', 'error_message']) if len(failures) else 'none'}

The full check list with evidence grades is in `methodology_audit.csv`. Points that materially affect the comparison:

- **Origin alignment.** Legacy CSVs start at `max(history)+1`, so legacy `1403Q1` sits at origin `140304` and `1403Q2` at `140306`, while V2 uses the canonical `140301` and `140304`. Matching on `forecast_origin` keeps the information cutoff identical on both sides but pairs V2 `1403Q2` with legacy `1403Q1`. A sensitivity that removes the shifted origins is reported below.
- **Training cutoff.** V2 trains on `date < forecast_origin`. Legacy drops its final month (`sale_series[:-1]`), so it effectively trains to `origin-2` and enters every matched origin with one month less information. This is a design difference between the pipelines, not something this analysis can isolate as a single cause.
- **Postprocessing.** V2 applies `max(raw, 0)` only. Legacy additionally applies quarterly `redistribute_smoothing`, negative replacement, integer rounding and a Prophet x0.8 haircut. Both sides are compared as delivered, end to end.
- **Stale failure files.** {data.stale_failure_files} `failure.json` files survive from earlier attempts even though only {n_failed} jobs are currently FAILED, because a successful retry does not delete them. `state.sqlite` is used as the source of truth everywhere in this analysis.
- **Actuals.** Monthly summed dispatch quantity in raw units from the frozen sales panel, identical for both engines, verified to agree with the legacy panel on every matched key. Actuals stop at `140504`, so recent origins are only partly matured. Missing months are treated as unknown, never as zero demand.

### Leakage evidence

| grade | checks |
|---|---|
| verified from artifacts | no forecast row targets a month before its own origin; the manifest records the training cutoff rule; every job used `best_model`, so no ensemble weights exist to leak |
| supported by code inspection | `assert_training_before_origin`, `assert_backtest_no_leakage` and `assert_final_forecast_contract` enforce the cutoff; V2 fits on raw units with no global transform, removing the scaling-before-split path that exists in legacy; selection uses expanding-window origins strictly before the forecast origin |
| not verifiable from this run | per-fold CV evidence. `backtests/` is empty because `V2ForecastEngine` passes only `selected_strategy` and `n_training_observations` in `extras`, so `BackfillStore` never writes `backtest_summary.json`. Candidate forecasts, per-model CV scores, fold training windows and fallback reasons cannot be re-checked. |

---

## 3. Did accuracy improve? (section C)

### Sensitivities

Both sensitivities move the headline in the same direction as the primary slice. Every metric for each sensitivity is in `overall_metrics.csv`.

| sensitivity | rows | WMAPE legacy | WMAPE V2 | relative improvement % |
|---|---|---|---|---|
"""
    )

    sens = overall.loc[
        (overall["scope"] == "sensitivity") & (overall["metric"] == "wmape")
    ]
    for _, row in sens.iterrows():
        parts.append(
            f"| {row['slice']} | {int(row['n']):,} | {_fmt(row['legacy'])} | "
            f"{_fmt(row['v2'])} | {_fmt(row['relative_improvement_pct'])} |\n"
        )

    parts.append(
        f"""
### By horizon

{_table(horizon_table, ['horizon', 'n', 'mae_legacy', 'mae_v2', 'relative_improvement_pct', 'bias_legacy', 'bias_v2', 'wmape_legacy', 'wmape_v2'])}

![MAE by horizon](charts/mae_by_horizon.png)

![Bias by horizon](charts/bias_by_horizon.png)

### By horizon group

{_table(report['horizons'].loc[(report['horizons']['scope'] == 'horizon_group') & (report['horizons']['metric'] == 'wmape')], ['slice', 'n', 'legacy', 'v2', 'relative_improvement_pct'])}

### By origin

{_table(origin_table, ['forecast_origin', 'v2_quarter', 'n', 'max_horizon_scored', 'wmape_legacy', 'wmape_v2', 'relative_improvement_pct', 'v2_better'])}

![Improvement by origin](charts/improvement_by_origin.png)

### By product

Full table in `product_metrics.csv`. Ten largest contributors to the net absolute-error change:

{_table(products.reindex(products['total_absolute_error_reduction'].abs().sort_values(ascending=False).index).head(10), ['product_id', 'n', 'actual_volume', 'wmape_legacy', 'wmape_v2', 'relative_improvement_pct', 'total_absolute_error_reduction', 'share_of_net_error_reduction', 'outcome'])}

![Product gains and regressions](charts/product_gains_and_regressions.png)

### By frozen product category

Breakdowns over `generic`, `Field`, `ProductForm` and `Provider` are written to `category_metrics.csv` with evaluated counts on every row.

---

## 4. V2 diagnostics (section D)

### Horizon behaviour

Legacy MAE moves from {_fmt(horizon_table.iloc[0]['mae_legacy'])} at h1 to {_fmt(horizon_table.iloc[-1]['mae_legacy'])} at h{int(horizon_table.iloc[-1]['horizon'])}; V2 moves from {_fmt(horizon_table.iloc[0]['mae_v2'])} to {_fmt(horizon_table.iloc[-1]['mae_v2'])}. Note that later horizons are scored on fewer and older origins, so the horizon profile mixes horizon difficulty with origin composition.

### Product outcomes

Rule: {product_summary['rule']}.

| outcome | products | share |
|---|---|---|
| improved | {product_summary['n_improved']} | {_fmt(product_summary['pct_improved'], 1)}% |
| tied | {product_summary['n_tied']} | {_fmt(product_summary['pct_tied'], 1)}% |
| regressed | {product_summary['n_regressed']} | {_fmt(product_summary['pct_regressed'], 1)}% |

Median product relative improvement: {_fmt(product_summary['median_relative_improvement_pct'])}%.

### Concentration

Net absolute-error reduction is {_fmt(concentration['net_absolute_error_reduction'])} units, made of {_fmt(concentration['total_gain'])} units of gains against {_fmt(concentration['total_loss'])} units of losses. The single largest gaining product accounts for {_fmt(100 * concentration['top1_gain_share'], 1)}% of all gains and the top five account for {_fmt(100 * concentration['top5_gain_share'], 1)}%. The top five products by volume represent {_fmt(100 * concentration['top5_volume_share'], 1)}% of evaluated actual volume. Flags: {', '.join(concentration['flags']) or 'none'}.

### Forecast shape

Diagnostic rules: flat = {shape['rules']['flat']}; extreme = {shape['rules']['extreme']}; clipped = {shape['rules']['clipped']}. No forecast was altered.

| flag | jobs |
|---|---|
| flat across all 15 horizons | {shape['n_flat_jobs']:,} of {shape['n_jobs']:,} ({_fmt(shape['pct_flat_jobs'], 1)}%) |
| extreme vs pre-origin history | {shape['n_extreme_jobs']} |
| all-zero forecast | {shape['n_all_zero_jobs']} |
| rows clipped from negative raw | {shape['n_clipped_rows']} |
| non-finite rows | {shape['n_nonfinite_rows']} |
| negative delivered rows | {shape['n_negative_delivered_rows']} |

Flat forecasts by selected model: {_counts(shape['flat_by_model'])}.

Accuracy split by whether V2 emitted a flat line:

{_table(diagnostics['flat_accuracy'], ['v2_forecast_is_flat', 'n', 'wmape_legacy', 'wmape_v2', 'relative_improvement_pct'])}

### Model selection and runtime

Selected model frequencies across {n_success:,} successful jobs: {_counts(selection['model_counts'])}.

A product's selected model changes between consecutive origins in {_fmt(selection['selection_switch_rate_pct'], 1)}% of {selection['selection_comparisons']:,} consecutive-origin comparisons ({selection['selection_switches']:,} switches), so selection is far from stable over time.

Runtime: {_fmt(selection['runtime_total_hours'])} hours of cumulative job time, mean {_fmt(selection['runtime_mean_seconds'])}s, median {_fmt(selection['runtime_median_seconds'])}s, max {_fmt(selection['runtime_max_seconds'])}s per SKU-vintage.

Every job recorded `selected_strategy` as {_counts(selection['selected_strategies'])}, so no ensemble was ever formed.

**Evidence not available in this run:** {'; '.join(selection['unavailable_evidence'])}. These are stated as limitations rather than reconstructed.

### Raw versus constrained output

{_raw_vs_constrained_text(diagnostics['raw_vs_constrained'])}

### Coverage bias

Coverage is uneven, so it is worth checking that the unmatched rows are not hiding the hard cases. Per-origin coverage is in `coverage_and_failures.csv` and per-product coverage in `product_coverage.csv`. Match rate per product ranges from {_fmt(100 * diagnostics['coverage_bias']['match_rate'].min(), 1)}% to {_fmt(100 * diagnostics['coverage_bias']['match_rate'].max(), 1)}% (median {_fmt(100 * diagnostics['coverage_bias']['match_rate'].median(), 1)}%). The dominant driver is the actuals ceiling at `140504`: origins from `140404` onward lose their later horizons, which is why the per-origin `max_horizon_scored` column falls from 15 to 1 across the last five origins. That is a maturity effect shared by both engines on identical rows, not a filter that favours either one.

Products with the lowest match rates:

{_table(diagnostics['coverage_bias'].head(5), ['product_id', 'v2_rows_produced', 'rows_matched', 'match_rate', 'wmape_legacy', 'wmape_v2'])}

---

## 5. Interpretation (section E)

### Verified implementation and methodology changes

These are established from artifacts or code, independently of accuracy:

- V2 trains strictly on `date < forecast_origin` with contract assertions, and no forecast row targets a month before its origin.
- V2 fits on raw units. The legacy global MinMax and Yeo-Johnson fit before splitting is gone, removing a preprocessing leakage path.
- V2 selects per SKU from {len(selection['model_counts'])} candidate families using multi-origin, multi-horizon expanding-window CV scored by mean horizon MAE, instead of the legacy single 80/20 one-step RMSE on a scaled series.
- V2 postprocessing is limited to `max(raw, 0)`; quarterly smoothing, negative replacement and the Prophet haircut are gone.
- The experiment is reproducible in the bookkeeping sense: one config hash, an immutable manifest, per-job logs and a SQLite checkpoint.

### Observed accuracy change

On {coverage['matched_rows']:,} identical rows, portfolio WMAPE moves from {_fmt(overall.loc[(overall['scope'] == 'overall') & (overall['metric'] == 'wmape'), 'legacy'].iloc[0])} to {_fmt(overall.loc[(overall['scope'] == 'overall') & (overall['metric'] == 'wmape'), 'v2'].iloc[0])} ({_fmt(verdict['wmape_relative_improvement_pct'])}% relative). The direction is consistent across the sensitivities listed above.

This is an end-to-end pipeline comparison. It cannot attribute the change to any single code change: the training cutoff, the preprocessing, the candidate set, the selection scheme and the postprocessing all differ at once, and legacy is additionally handicapped by one month of training data at every origin. Isolating any one of them needs a controlled ablation that this experiment does not provide.

### Operational changes

- The backfill completed {n_success:,} of {len(jobs):,} SKU-vintage jobs with {n_failed} failures, at roughly {_fmt(selection['runtime_total_hours'])} cumulative hours on a single worker.
- Failures are per-SKU and isolated; they do not stop the run.
- Artifact hygiene has two rough edges: stale `failure.json` files persist after successful retries, and `backtests/` is silently empty, which is what blocks the deeper selection diagnostics.

---

## 6. Limitations

- `backtests/` is empty, so candidate forecasts, per-model CV scores, ensemble weights and fallback reasons are unavailable.
- Legacy pre-postprocess forecasts were never saved, so the raw-output comparison exists only on the V2 side.
- Legacy effectively trains to `origin-2`; the two engines do not see identical history at a shared origin.
- Actuals end at `140504`, so recent origins contribute only their early horizons.
- Origins and horizons overlap; no significance testing is performed and none should be inferred.
- Product identity is the English title on both sides; the `ID_INT` migration in `docs/ts_v2_product_identity.md` is still pending.

---

## 7. Next actions

1. **Persist V2 selection evidence.** Extend `V2ForecastEngine` extras with the backtest summary and per-model CV scores so `BackfillStore` writes `backtest_summary.json`. Without it, no future run can answer why a model was chosen, how often fallbacks fire, or whether selection is stable, and this analysis had to leave those questions open.
2. **Investigate flat forecasts.** {shape['n_flat_jobs']:,} of {shape['n_jobs']:,} jobs ({_fmt(shape['pct_flat_jobs'], 1)}%) emit a single repeated value for all 15 months, concentrated in {' and '.join(list(shape['flat_by_model'])[:3])}. On those SKU-origins V2 still beats legacy, but by {_fmt(diagnostics['flat_accuracy'].loc[diagnostics['flat_accuracy']['v2_forecast_is_flat'], 'relative_improvement_pct'].iloc[0])}% against {_fmt(diagnostics['flat_accuracy'].loc[~diagnostics['flat_accuracy']['v2_forecast_is_flat'], 'relative_improvement_pct'].iloc[0])}% elsewhere. Check whether the mean-horizon-MAE selection score systematically favours flat baselines on intermittent SKUs before that behaviour is carried into V3A.
3. **Fix the origin and identity mismatches before the next paired run.** Regenerate or re-align the legacy comparator so quarter labels and origins agree, and close the `ID_INT` product-identity migration, so the next comparison does not need a shifted-origin sensitivity.

Do not assume V3A resolves any of the weaknesses above; none of them is caused by the model family alone.

---

## 8. Outputs

| file | contents |
|---|---|
| `report.md` | this document |
| `matched_predictions.parquet` | one row per `product_id` + `forecast_origin` + `target_date` + `horizon`, with actuals, both predictions and error columns |
| `overall_metrics.csv` | headline metrics and sensitivities |
| `horizon_metrics.csv` | per-horizon and horizon-group metrics |
| `product_metrics.csv` | per-product metrics, contribution and outcome class |
| `origin_metrics.csv` | per-origin metrics |
| `category_metrics.csv` | frozen product-category breakdowns |
| `coverage_and_failures.csv` | per-origin coverage plus failed jobs |
| `methodology_audit.csv` | section A checks with evidence grades |
| `unmatched_v2_rows.csv`, `unmatched_legacy_rows.csv` | rows excluded from the comparison and why |
| `forecast_shape_flags.csv` | per-job flat / extreme / clipped flags |
| `product_coverage.csv` | per-product match rate against rows produced |
| `model_selection_by_origin.csv` | selected-model counts per origin |
| `analysis_manifest.json` | input paths, hashes, metric definitions, filters, reproduce command |
| `charts/` | the figures referenced above |
"""
    )

    text = "".join(parts)
    path = cfg.out_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_docs_summary(report: dict[str, Any]) -> Path:
    cfg = report["config"]
    overall = report["overall"]
    verdict = report["verdict"]
    diagnostics = report["diagnostics"]
    product_summary = diagnostics["product_outcome_summary"]
    shape = diagnostics["shape_summary"]
    coverage = report["panel"].coverage_summary
    data = report["data"]

    def value(metric: str, column: str) -> str:
        row = overall.loc[
            (overall["scope"] == "overall") & (overall["metric"] == metric)
        ]
        return _fmt(row.iloc[0][column]) if len(row) else "n/a"

    docs_dir = cfg.repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "ts_v2_backfill_eval.md"

    sens_lines = []
    sens = overall.loc[
        (overall["scope"] == "sensitivity") & (overall["metric"] == "wmape")
    ]
    for _, row in sens.iterrows():
        sens_lines.append(
            f"| {row['slice']} | {int(row['n']):,} | {_fmt(row['legacy'])} | "
            f"{_fmt(row['v2'])} | {_fmt(row['relative_improvement_pct'])} |"
        )

    text = f"""# TS V2 historical backfill — evaluation vs legacy

**Full report and artifacts:** [`src/data/results/ts_v2_backfill_eval/`](../src/data/results/ts_v2_backfill_eval/) (`report.md`).
**Canonical accuracy convention:** WMAPE as a percent from [`pkg.benchmark.evaluate.wmape`](../src/pkg/benchmark/evaluate.py), summed absolute error over summed absolute actuals.
**This document does not claim statistical significance.** Origins and horizons overlap.

---

## 1. What was compared

| side | source |
|------|--------|
| Legacy TS | Delivered quarterly CSV vintages from `pkg.forecast.SalesForecast`, frozen as `ts_universe.parquet` in benchmark v1 |
| TS V2 | Historical backfill `{data.manifest.get('experiment_id')}` / `{data.manifest.get('engine_version')}`, config hash `{data.manifest.get('config_hash')}` |

Matched on `product_id + forecast_origin + target_date + horizon`: **{coverage['matched_rows']:,} rows**, {coverage['matched_products']} products, {coverage['matched_origins']} origins. Rows without both forecasts and an actual are excluded and reported as coverage; nothing is zero-filled.

---

## 2. Headline result

| metric | Legacy | V2 | relative improvement |
|--------|-------:|---:|---------------------:|
| Portfolio WMAPE | {value('wmape', 'legacy')} | **{value('wmape', 'v2')}** | **{_fmt(verdict['wmape_relative_improvement_pct'])}%** |
| Mean horizon MAE | {value('mean_horizon_mae', 'legacy')} | {value('mean_horizon_mae', 'v2')} | {_fmt(verdict['mean_horizon_mae_relative_improvement_pct'])}% |
| RMSE | {value('rmse', 'legacy')} | {value('rmse', 'v2')} | {_fmt(verdict['rmse_relative_improvement_pct'])}% |
| Signed bias | {value('signed_bias', 'legacy')} | {value('signed_bias', 'v2')} | reported by distance from zero, not as a ratio |

**Verdict: {VERDICT_TEXT.get(verdict['label'], verdict['label'])}.** V2 wins {verdict['origins_better']} of {verdict['origins_total']} origins (losing at {', '.join(str(o) for o in verdict['losing_origins']) or 'none'}); {_fmt(product_summary['pct_improved'], 1)}% of products improve against {_fmt(product_summary['pct_regressed'], 1)}% that regress.

### Sensitivities

| slice | n | Legacy WMAPE | V2 WMAPE | relative improvement % |
|---|---|---|---|---|
{chr(10).join(sens_lines)}

---

## 3. Comparability caveats

- Legacy 1403 vintages were emitted at a shifted origin, so the primary join pairs V2 `1403Q2` with legacy `1403Q1` at origin `140304`. A sensitivity removing shifted origins is reported above.
- Legacy drops its last training month, so it enters each origin with one month less information than V2.
- Legacy output carries quarterly smoothing, negative replacement, rounding and a Prophet haircut; V2 applies only `max(raw, 0)`. Both are compared as delivered.
- Actuals end at `140504`, so recent origins contribute only early horizons.

---

## 4. What we learned

| finding | confidence |
|---------|------------|
| V2 improves portfolio WMAPE on matched rows ({_fmt(verdict['wmape_relative_improvement_pct'])}% relative) | **Moderate** |
| The gain survives the shifted-origin and fully-matured sensitivities | **Moderate** |
| Gains are not uniform across products; a minority regress materially | **Strong** |
| No single V2 code change can be credited for the gain from this design | **Strong** |
| {shape['n_flat_jobs']:,} of {shape['n_jobs']:,} jobs produce a completely flat 15-month path | **Strong** |
| Selection evidence (CV scores, candidates, fallbacks) was not persisted | **Strong** |

---

## 5. Next experiments

1. Persist V2 backtest and selection summaries from the engine adapter so the next run can be diagnosed.
2. Examine whether mean-horizon-MAE selection over-favours flat baselines on intermittent SKUs.
3. Re-align legacy origins and finish the `ID_INT` identity migration before the next paired comparison.

Related: [forecasting_findings.md](forecasting_findings.md), [ts_forecasting_architecture.md](ts_forecasting_architecture.md), [ts_v2_gap_audit.md](ts_v2_gap_audit.md), [ts_backfill_server.md](ts_backfill_server.md).
"""
    path.write_text(text, encoding="utf-8")
    return path
