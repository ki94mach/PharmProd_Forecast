"""Write V3A architecture-validation reports from screening artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pkg.ts_v3a.analysis.metrics import (
    architecture_summary,
    paired_architecture_comparisons,
    rebuild_ensemble,
    slice_summaries,
)
from pkg.ts_v3a.analysis.recigen import diagnose_recigen
from pkg.ts_v3a.persistence import load_screening_experiment

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS = REPO_ROOT / "src" / "data" / "results" / "ts_v3a_architecture_validation"
DEFAULT_DOCS = REPO_ROOT / "docs" / "ts_v3a_architecture_validation.md"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids optional pandas ``tabulate`` dependency)."""
    if df is None or df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        cells: list[str] = []
        for v in row:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_validation_report(
    *,
    experiment_id: str,
    base_dir: Path,
    results_dir: Path = DEFAULT_RESULTS,
    docs_path: Path = DEFAULT_DOCS,
    panel_csv: Optional[Path] = None,
    origins: Optional[list[int]] = None,
    recigen_smoke_id: Optional[str] = None,
    recommendation: Optional[str] = None,
) -> Path:
    """Load experiment, write CSV tables + report.md + docs summary."""
    loaded = load_screening_experiment(experiment_id, base_dir=base_dir)
    ens = rebuild_ensemble(loaded)
    summary = architecture_summary(loaded, ensemble=ens)
    paired = paired_architecture_comparisons(ens)
    slices = slice_summaries(loaded, ens)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "architecture_summary.csv", index=False)
    paired.to_csv(results_dir / "paired_comparisons.csv", index=False)
    ens.to_parquet(results_dir / "ensemble_predictions.parquet", index=False)
    for name, df in slices.items():
        if df is not None and not df.empty:
            df.to_csv(results_dir / f"slice_{name}.csv", index=False)

    fold = loaded.fold_metadata
    fold.to_parquet(results_dir / "fold_metadata.parquet", index=False)

    recigen_conclusion = ""
    refined = results_dir / "recigen" / "recigen_conclusion.txt"
    if refined.is_file():
        recigen_conclusion = refined.read_text(encoding="utf-8").strip()
    if recigen_smoke_id and not recigen_conclusion:
        smoke = load_screening_experiment(recigen_smoke_id, base_dir=base_dir)
        rec = diagnose_recigen(smoke)
        for key in ("detail", "by_horizon", "fold_summary", "scaler"):
            df = rec.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(results_dir / f"recigen_{key}.csv", index=False)
        recigen_conclusion = str(rec.get("conclusion", ""))
    elif recigen_smoke_id:
        # Still export tables even when conclusion file already exists.
        smoke = load_screening_experiment(recigen_smoke_id, base_dir=base_dir)
        rec = diagnose_recigen(smoke)
        for key in ("detail", "by_horizon", "fold_summary", "scaler"):
            df = rec.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(results_dir / f"recigen_{key}.csv", index=False)

    products = sorted(ens["product"].unique().tolist()) if not ens.empty else []
    if origins is None:
        origins = list(loaded.manifest.get("origins") or [])
    n_arch = len(loaded.manifest.get("architectures") or [])
    n_seeds = len(loaded.manifest.get("seeds") or [])
    planned = len(products) * len(origins) * n_arch * n_seeds

    if recommendation is None:
        recommendation = _default_recommendation(summary, paired)

    report_lines = [
        "# V3A architecture validation report",
        "",
        f"- Experiment: `{experiment_id}`",
        f"- Config hash: `{loaded.manifest.get('config_hash')}`",
        f"- Products ({len(products)}): {', '.join(products)}",
        f"- Origins: {origins}",
        f"- Planned folds: {planned} (products × origins × arch × seeds)",
        f"- Observed folds: {len(fold)} (ok={(fold['status']=='ok').sum() if not fold.empty else 0})",
        f"- Artifacts: `src/data/ts_v3a/screening/{experiment_id}/`",
        f"- Analysis tables: `{results_dir.as_posix()}`",
        "",
        "## Architecture summary",
        "",
        _df_to_markdown(summary) if not summary.empty else "_no ensemble rows_",
        "",
        "## Paired comparisons (identical SKU×origin×horizon)",
        "",
        _df_to_markdown(paired) if not paired.empty else "_none_",
        "",
        "## Recigen diagnosis (smoke)",
        "",
        recigen_conclusion or "_not run_",
        "",
        "## Recommendation (report-only; defaults unchanged)",
        "",
        recommendation,
        "",
        "## Unresolved issues",
        "",
        "- Full `1401Q1→1405Q3` backfill not run.",
        "- A6 not implemented.",
        "- Smoke Recigen scaler stats were rebuilt offline (not in original fold_metadata).",
        "- Origin coverage for short A0 tiers depends on panel×origin history; see slice_a0_by_tier.csv.",
        "",
    ]
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        "\n".join(
            [
                "# V3A architecture validation (summary)",
                "",
                f"Full report: [`src/data/results/ts_v3a_architecture_validation/report.md`](../src/data/results/ts_v3a_architecture_validation/report.md).",
                "",
                f"Experiment `{experiment_id}` · {len(products)} MVP products · origins {origins}.",
                "",
                "## Recommendation",
                "",
                recommendation,
                "",
                "## Recigen",
                "",
                recigen_conclusion or "_see full report_",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if panel_csv is not None and panel_csv.is_file():
        # copy reference
        pd.read_csv(panel_csv).to_csv(results_dir / "validation_products.csv", index=False)
    return report_path


def _default_recommendation(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    if summary.empty:
        return (
            "Insufficient ensemble metrics to recommend. Re-run analysis after "
            "screening completes."
        )
    top = summary.sort_values("mean_horizon_MAE").head(3)
    lines = [
        "Candidates for a future full V3A backfill (not applied to defaults):",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"- **{r['arch']}** (`{r['architecture']}`): mean_horizon_MAE={r['mean_horizon_MAE']:.1f}, "
            f"portfolio_WMAPE={r['portfolio_wmape_pct']:.1f}%, mean_rank={r.get('mean_rank', float('nan'))}, "
            f"mae_wins={r.get('mae_wins', 0)}, mean_fold_s={r.get('mean_runtime_s', float('nan')):.1f}"
        )
    # A0 vs A1 note
    p = paired.loc[(paired["left"] == "A0") & (paired["right"] == "A1")]
    if not p.empty:
        d = float(p.iloc[0]["mae_diff_left_minus_right"])
        lines.append(
            f"A0 vs A1 paired MAE diff (A0−A1)={d:.2f} on {int(p.iloc[0]['n'])} cells "
            f"(negative means A0 better)."
        )
    p24 = paired.loc[(paired["left"] == "A2") & (paired["right"] == "A4")]
    if not p24.empty:
        d = float(p24.iloc[0]["mae_diff_left_minus_right"])
        lines.append(
            f"A2 vs A4 paired MAE diff (A2−A4)={d:.2f}; positive means A2 worse than A4 "
            "(A4 preferred among non-recursive if positive)."
        )
    return "\n".join(lines)
