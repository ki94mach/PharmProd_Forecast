"""Metric helpers for V3A screening analysis (bracket column access only)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import wmape as portfolio_wmape_pct
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.backtest import STATUS_OK
from pkg.ts_v3a.metrics import build_seed_ensemble_predictions
from pkg.ts_v3a.models.a0_legacy_adaptive_recursive_lstm import resolve_legacy_architecture
from pkg.ts_v3a.persistence import LoadedScreeningExperiment

SHORT_ARCH = {
    ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value: "A0",
    ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value: "A1",
    ArchitectureName.A2_MIMO_LSTM.value: "A2",
    ArchitectureName.A3_STACKED_MIMO_LSTM.value: "A3",
    ArchitectureName.A4_ENCODER_DECODER_LSTM.value: "A4",
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM.value: "A5",
}

PAIRED_COMPARISONS = (
    ("A0", "A1"),
    ("A1", "A2"),
    ("A2", "A3"),
    ("A2", "A4"),
    ("A2", "A5"),
)


def history_bucket(n: int) -> str:
    n = int(n)
    if n > 36:
        return ">36"
    if n > 24:
        return "25-36"
    if n > 12:
        return "13-24"
    if n > 6:
        return "7-12"
    return "<=6"


def a0_tier_label(history_length: int) -> str:
    spec = resolve_legacy_architecture(int(history_length))
    return f"l1={spec.l1}_l2={spec.l2}_lb={spec.lookback}"


def portfolio_wmape(actual: pd.Series, prediction: pd.Series) -> float:
    """True portfolio WMAPE in percent: sum(|e|)/sum(|actual|)*100."""
    return float(portfolio_wmape_pct(actual, prediction))


def _short(arch: str) -> str:
    return SHORT_ARCH.get(str(arch), str(arch))


def rebuild_ensemble(
    loaded: LoadedScreeningExperiment,
    *,
    min_successful_seeds: Optional[int] = None,
) -> pd.DataFrame:
    """Rebuild seed-ensemble OOF from persisted seed rows + fold metadata."""
    seeds = loaded.manifest.get("seeds") or [41, 42, 43]
    min_ok = min_successful_seeds
    if min_ok is None:
        min_ok = (
            loaded.manifest.get("eligibility_configuration", {}) or {}
        ).get("min_successful_seeds", 3)
    ens, _status = build_seed_ensemble_predictions(
        loaded.oof_predictions,
        loaded.fold_metadata,
        seeds=seeds,
        min_successful_seeds=int(min_ok),
        status_ok=STATUS_OK,
    )
    if ens.empty:
        return ens
    out = ens.copy()
    out["arch"] = out["architecture"].map(_short)
    out["abs_error"] = (out["prediction"] - out["actual"]).abs()
    out["signed_error"] = out["prediction"] - out["actual"]
    return out


def architecture_summary(
    loaded: LoadedScreeningExperiment,
    *,
    ensemble: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-architecture headline metrics from ensemble OOF + fold metadata."""
    ens = ensemble if ensemble is not None else rebuild_ensemble(loaded)
    fold = loaded.fold_metadata
    seed_m = loaded.seed_metrics
    rows: list[dict] = []
    if ens.empty:
        return pd.DataFrame()

    for architecture, g in ens.groupby("architecture", sort=True):
        arch = str(architecture)
        short = _short(arch)
        actual = g["actual"]
        pred = g["prediction"]
        err = pred - actual
        # Equal-weight mean of per-horizon MAE (matches screening primary metric).
        by_h = g.groupby("horizon", sort=True).apply(
            lambda x: float(np.mean(np.abs(x["prediction"] - x["actual"]))),
            include_groups=False,
        )
        mean_h_mae = float(by_h.mean()) if len(by_h) else float("nan")
        sku_wmape = []
        for _, pg in g.groupby("product"):
            denom = float(np.abs(pg["actual"]).sum())
            if denom > 0:
                sku_wmape.append(float(np.abs(pg["prediction"] - pg["actual"]).sum()) / denom)

        fold_a = fold.loc[fold["architecture"] == arch]
        ok = fold_a.loc[fold_a["status"] == STATUS_OK]
        seed_a = seed_m.loc[seed_m["architecture"] == arch] if seed_m is not None and not seed_m.empty else pd.DataFrame()
        seed_std = (
            float(seed_a.groupby("product")["mean_horizon_MAE"].std().mean())
            if not seed_a.empty and "mean_horizon_MAE" in seed_a.columns
            else float("nan")
        )

        row = {
            "arch": short,
            "architecture": arch,
            "mean_horizon_MAE": mean_h_mae,
            "portfolio_wmape_pct": portfolio_wmape(actual, pred),
            "mean_sku_wmape": float(np.mean(sku_wmape)) if sku_wmape else float("nan"),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "bias": float(np.mean(err)),
            "n_predictions": int(len(g)),
            "n_products": int(g["product"].nunique()),
            "n_origins": int(g["origin"].nunique()),
            "folds_ok": int(len(ok)),
            "folds_total": int(len(fold_a)),
            "unavailable_folds": int((fold_a["status"] != STATUS_OK).sum()),
            "runtime_hours": float(fold_a["runtime_seconds"].fillna(0).sum()) / 3600.0,
            "mean_runtime_s": float(ok["runtime_seconds"].mean()) if len(ok) else float("nan"),
            "median_parameter_count": float(ok["parameter_count"].median()) if len(ok) else float("nan"),
            "seed_mae_std_mean": seed_std,
        }
        for h, v in by_h.items():
            row[f"mae_h{int(h)}"] = float(v)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Ranks / wins on mean_horizon_MAE across products (from architecture_metrics if present).
    am = loaded.architecture_metrics
    if am is not None and not am.empty and "mean_horizon_MAE" in am.columns:
        tmp = am.copy()
        tmp["arch"] = tmp["architecture"].map(_short)
        tmp["rank"] = tmp.groupby("product")["mean_horizon_MAE"].rank(method="min")
        ranks = (
            tmp.groupby("arch", sort=False)
            .agg(
                mean_rank=("rank", "mean"),
                mae_wins=("rank", lambda s: int((s == 1).sum())),
            )
            .reset_index()
        )
        summary = summary.merge(ranks, on="arch", how="left")
    else:
        summary["mean_rank"] = np.nan
        summary["mae_wins"] = 0
    return summary.sort_values("mean_horizon_MAE").reset_index(drop=True)


def paired_architecture_comparisons(
    ensemble: pd.DataFrame,
    pairs: Sequence[tuple[str, str]] = PAIRED_COMPARISONS,
) -> pd.DataFrame:
    """Paired MAE on identical product × origin × horizon observations."""
    if ensemble is None or ensemble.empty:
        return pd.DataFrame()
    work = ensemble.copy()
    if "arch" not in work.columns:
        work["arch"] = work["architecture"].map(_short)
    keys = ["product", "origin", "horizon"]
    rows: list[dict] = []
    for left, right in pairs:
        a = work.loc[work["arch"] == left, keys + ["abs_error", "actual", "prediction"]].rename(
            columns={
                "abs_error": "ae_left",
                "prediction": "pred_left",
            }
        )
        b = work.loc[work["arch"] == right, keys + ["abs_error", "prediction"]].rename(
            columns={
                "abs_error": "ae_right",
                "prediction": "pred_right",
            }
        )
        m = a.merge(b, on=keys, how="inner")
        if m.empty:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": 0,
                    "mae_left": np.nan,
                    "mae_right": np.nan,
                    "mae_diff_left_minus_right": np.nan,
                    "left_wins": 0,
                    "right_wins": 0,
                    "ties": 0,
                }
            )
            continue
        diff = m["ae_left"] - m["ae_right"]
        rows.append(
            {
                "left": left,
                "right": right,
                "n": int(len(m)),
                "mae_left": float(m["ae_left"].mean()),
                "mae_right": float(m["ae_right"].mean()),
                "mae_diff_left_minus_right": float(diff.mean()),
                "left_wins": int((diff < 0).sum()),
                "right_wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def slice_summaries(
    loaded: LoadedScreeningExperiment,
    ensemble: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Architecture metrics sliced by history / windows / intermittency / A0 tier."""
    fold = loaded.fold_metadata.copy()
    fold["arch"] = fold["architecture"].map(_short)
    fold["history_bucket"] = fold["available_history_length"].map(history_bucket)
    fold["a0_tier"] = fold["available_history_length"].map(a0_tier_label)

    # Attach mean train windows per product×origin (from any ok fold).
    win = (
        fold.loc[fold["status"] == STATUS_OK]
        .groupby(["product", "origin"], as_index=False)
        .agg(
            train_window_count=("train_window_count", "median"),
            available_history_length=("available_history_length", "median"),
            history_bucket=("history_bucket", "first"),
            a0_tier=("a0_tier", "first"),
        )
    )

    ens = ensemble.merge(win, on=["product", "origin"], how="left")
    out: dict[str, pd.DataFrame] = {}

    def _slice(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty or key not in df.columns:
            return pd.DataFrame()
        rows = []
        for (arch, bucket), g in df.groupby(["arch", key], sort=True):
            rows.append(
                {
                    "arch": arch,
                    key: bucket,
                    "mean_horizon_MAE": float(
                        g.groupby("horizon")
                        .apply(
                            lambda x: float(np.mean(np.abs(x["prediction"] - x["actual"]))),
                            include_groups=False,
                        )
                        .mean()
                    ),
                    "portfolio_wmape_pct": portfolio_wmape(g["actual"], g["prediction"]),
                    "n": int(len(g)),
                }
            )
        return pd.DataFrame(rows)

    out["by_history_bucket"] = _slice(ens, "history_bucket")
    # Window count buckets
    ens["window_bucket"] = pd.cut(
        ens["train_window_count"],
        bins=[-np.inf, 20, 50, 100, np.inf],
        labels=["<=20", "21-50", "51-100", ">100"],
    ).astype(str)
    out["by_train_windows"] = _slice(ens, "window_bucket")

    # Zero fraction from actuals in OOF window (proxy intermittency on eval months)
    z = (
        ens.groupby(["product", "origin"], as_index=False)
        .agg(
            zero_fraction=("actual", lambda s: float((np.asarray(s) == 0).mean())),
        )
    )
    ens2 = ens.merge(z, on=["product", "origin"], how="left")
    ens2["zero_bucket"] = pd.cut(
        ens2["zero_fraction"],
        bins=[-0.01, 0.05, 0.2, 0.5, 1.01],
        labels=["<=5%", "5-20%", "20-50%", ">50%"],
    ).astype(str)
    out["by_zero_fraction"] = _slice(ens2, "zero_bucket")

    a0_ens = ens.loc[ens["arch"] == "A0"]
    out["a0_by_tier"] = _slice(a0_ens, "a0_tier")
    return out
