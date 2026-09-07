"""Charts for the V2-vs-legacy backfill report.

Matplotlib only, Agg backend, PNG output under ``{out_dir}/charts``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pkg.research.v2_eval.config import V2EvalConfig  # noqa: E402

LEGACY_LABEL = "Legacy TS"
V2_LABEL = "TS V2"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def mae_by_horizon(horizon_table: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        horizon_table["horizon"], horizon_table["mae_legacy"], marker="o",
        label=LEGACY_LABEL,
    )
    ax.plot(
        horizon_table["horizon"], horizon_table["mae_v2"], marker="s", label=V2_LABEL
    )
    ax.set_xlabel("Horizon (months ahead)")
    ax.set_ylabel("MAE (units)")
    ax.set_title("MAE by horizon on matched rows")
    ax.set_xticks(horizon_table["horizon"].tolist())
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / "mae_by_horizon.png")


def bias_by_horizon(horizon_table: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        horizon_table["horizon"], horizon_table["bias_legacy"], marker="o",
        label=LEGACY_LABEL,
    )
    ax.plot(
        horizon_table["horizon"], horizon_table["bias_v2"], marker="s", label=V2_LABEL
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Horizon (months ahead)")
    ax.set_ylabel("Signed bias, prediction - actual (units)")
    ax.set_title("Bias by horizon (positive = overforecast)")
    ax.set_xticks(horizon_table["horizon"].tolist())
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out_dir / "bias_by_horizon.png")


def improvement_by_origin(origin_table: pd.DataFrame, out_dir: Path) -> Path:
    frame = origin_table.sort_values("forecast_origin")
    colors = ["tab:green" if v > 0 else "tab:red" for v in frame["relative_improvement_pct"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        frame["forecast_origin"].astype(str),
        frame["relative_improvement_pct"],
        color=colors,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Forecast origin (Shamsi YYYYMM)")
    ax.set_ylabel("Relative WMAPE improvement (%)")
    ax.set_title("V2 vs legacy WMAPE improvement by origin (positive = V2 better)")
    ax.tick_params(axis="x", rotation=60)
    for x, (value, n) in enumerate(zip(frame["relative_improvement_pct"], frame["n"])):
        ax.annotate(
            f"n={n}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 4 if value >= 0 else -12),
            ha="center",
            fontsize=7,
        )
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir / "improvement_by_origin.png")


def product_gains_and_regressions(
    product_table: pd.DataFrame,
    out_dir: Path,
    top_n: int = 10,
) -> Path:
    ranked = product_table.sort_values("total_absolute_error_reduction")
    worst = ranked.head(top_n)
    best = ranked.tail(top_n)
    frame = pd.concat([worst, best]).drop_duplicates(subset=["product_id"])
    frame = frame.sort_values("total_absolute_error_reduction")
    colors = [
        "tab:green" if v > 0 else "tab:red"
        for v in frame["total_absolute_error_reduction"]
    ]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(frame) + 2)))
    ax.barh(frame["product_id"], frame["total_absolute_error_reduction"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Total absolute error reduction, legacy - V2 (units)")
    ax.set_title(f"Largest product gains and regressions (top {top_n} each way)")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_dir / "product_gains_and_regressions.png")


def _month_index(shamsi: pd.Series) -> pd.Series:
    """Sequential month number so gaps and spacing are drawn to scale."""
    values = shamsi.astype(int)
    return values // 100 * 12 + (values % 100 - 1)


def case_study(
    matched: pd.DataFrame,
    sales: pd.DataFrame,
    product_id: str,
    out_dir: Path,
    tag: str,
    history_lead_months: int = 18,
) -> Path:
    """Actuals versus forecasts for one SKU, each origin drawn separately."""
    rows = matched.loc[matched["product_id"] == product_id].copy()
    history = sales.loc[sales["product_id"] == product_id].sort_values("target_date").copy()
    rows["x"] = _month_index(rows["target_date"])
    history["x"] = _month_index(history["target_date"])

    # Show enough history for context without shrinking the forecast window.
    start = int(rows["x"].min()) - history_lead_months
    history = history.loc[history["x"] >= start]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        history["x"], history["actual"], color="black", linewidth=2, label="Actual"
    )
    origins = sorted(rows["forecast_origin"].unique())
    cmap = plt.get_cmap("tab20")
    for index, origin in enumerate(origins):
        group = rows.loc[rows["forecast_origin"] == origin].sort_values("x")
        color = cmap(index % 20)
        ax.plot(
            group["x"], group["legacy_prediction"], linestyle="--", marker="o",
            markersize=3, color=color, alpha=0.6,
            label=LEGACY_LABEL if index == 0 else None,
        )
        ax.plot(
            group["x"], group["v2_prediction"], linestyle="-", marker="s",
            markersize=3, color=color,
            label=V2_LABEL if index == 0 else None,
        )
        ax.axvline(_month_index(pd.Series([origin])).iloc[0], color=color,
                   alpha=0.25, linewidth=1)

    all_x = pd.concat([history["x"], rows["x"]])
    all_dates = pd.concat(
        [history["target_date"], rows["target_date"]]
    ).astype(int)
    lookup = dict(zip(all_x, all_dates))
    ticks = sorted(lookup)[::3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(lookup[t]) for t in ticks], rotation=60, fontsize=8)

    ax.set_title(
        f"{product_id} — actuals vs forecasts ({tag}); one colour per origin, "
        "vertical line marks each origin, dashed = legacy, solid = V2"
    )
    ax.set_xlabel("Target month (Shamsi YYYYMM)")
    ax.set_ylabel("Units")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    safe = product_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return _save(fig, out_dir / f"case_{tag}_{safe}.png")


def build_charts(
    config: V2EvalConfig,
    matched: pd.DataFrame,
    sales: pd.DataFrame,
    horizon_table: pd.DataFrame,
    origin_table: pd.DataFrame,
    product_table: pd.DataFrame,
    n_cases: int = 2,
) -> dict[str, Any]:
    out_dir = config.charts_dir
    paths = {
        "mae_by_horizon": mae_by_horizon(horizon_table, out_dir),
        "bias_by_horizon": bias_by_horizon(horizon_table, out_dir),
        "improvement_by_origin": improvement_by_origin(origin_table, out_dir),
        "product_gains_and_regressions": product_gains_and_regressions(
            product_table, out_dir
        ),
    }
    ranked = product_table.sort_values("total_absolute_error_reduction")
    improved = ranked.tail(n_cases)["product_id"].tolist()
    regressed = ranked.head(n_cases)["product_id"].tolist()
    cases = []
    for product in improved:
        cases.append(case_study(matched, sales, product, out_dir, "improved"))
    for product in regressed:
        cases.append(case_study(matched, sales, product, out_dir, "regressed"))
    paths["case_studies"] = cases
    return paths
