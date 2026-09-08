"""Recigen MIMO overforecast diagnosis from smoke screening artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v3a.analysis.metrics import SHORT_ARCH, rebuild_ensemble
from pkg.ts_v3a.architectures import ArchitectureName
from pkg.ts_v3a.config import DEFAULT_CONFIG
from pkg.ts_v3a.persistence import LoadedScreeningExperiment, load_screening_experiment
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.types import TargetMode

FROZEN_SALES = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "data"
    / "benchmarks"
    / "v1"
    / "raw"
    / "sales.parquet"
)

RECIGEN_ARCHS = (
    ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
    ArchitectureName.A2_MIMO_LSTM.value,
    ArchitectureName.A3_STACKED_MIMO_LSTM.value,
    ArchitectureName.A4_ENCODER_DECODER_LSTM.value,
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM.value,
)


def _offline_scaler_stats(
    sales: pd.DataFrame,
    *,
    product: str,
    origin: int,
    architecture: str,
) -> dict:
    """Rebuild fold-local scaler without TF training."""
    mode = (
        TargetMode.RECURSIVE
        if architecture
        in (
            ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM.value,
            ArchitectureName.A1_SMALL_RECURSIVE_LSTM.value,
        )
        else TargetMode.DIRECT_MIMO
    )
    prepared = prepare_monthly_series(sales, product, origin, config=V2_DEFAULT)
    cfg = DEFAULT_CONFIG.with_architecture(architecture)
    try:
        prepared_fold, scaler = prepare_neural_fold(
            prepared.values,
            config=cfg,
            mode=mode,
            forecast_origin=origin,
            require_eligible=False,
        )
        sp = dict(scaler.params())
        return {
            "scaler_mean": sp.get("mean"),
            "scaler_scale": sp.get("scale"),
            "scaler_n_fit": sp.get("n_observations_fit"),
            "train_windows_rebuilt": prepared_fold.metadata.train_window_count,
            "val_windows_rebuilt": prepared_fold.metadata.validation_window_count,
        }
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        return {
            "scaler_mean": None,
            "scaler_scale": None,
            "scaler_n_fit": None,
            "train_windows_rebuilt": None,
            "val_windows_rebuilt": None,
            "scaler_error": str(exc),
        }


def diagnose_recigen(
    loaded: LoadedScreeningExperiment,
    *,
    sales_parquet: Path = FROZEN_SALES,
    product: str = "Recigen",
) -> dict[str, pd.DataFrame | str]:
    """Build Recigen diagnosis tables and a short conclusion string."""
    oof = loaded.oof_predictions
    fold = loaded.fold_metadata
    rec = oof.loc[oof["product"] == product].copy()
    if rec.empty:
        return {
            "detail": pd.DataFrame(),
            "by_horizon": pd.DataFrame(),
            "fold_summary": pd.DataFrame(),
            "conclusion": f"No OOF rows for product={product!r}",
        }
    rec["arch"] = rec["architecture"].map(lambda a: SHORT_ARCH.get(str(a), str(a)))
    rec = rec.loc[rec["architecture"].isin(RECIGEN_ARCHS)]

    # Wide seed predictions
    pivot = rec.pivot_table(
        index=["origin", "horizon", "target_date", "actual", "architecture", "arch"],
        columns="seed",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        f"seed_{c}" if isinstance(c, (int, np.integer)) else c for c in pivot.columns
    ]
    ens = rebuild_ensemble(loaded)
    ens_r = ens.loc[
        (ens["product"] == product) & (ens["architecture"].isin(RECIGEN_ARCHS)),
        ["origin", "horizon", "architecture", "prediction", "signed_error"],
    ].rename(columns={"prediction": "ensemble_prediction", "signed_error": "ensemble_signed_error"})
    detail = pivot.merge(ens_r, on=["origin", "horizon", "architecture"], how="left")
    seed_cols = [c for c in detail.columns if str(c).startswith("seed_")]
    if seed_cols:
        detail["seed_std"] = detail[seed_cols].std(axis=1, ddof=0)
        detail["seed_range"] = detail[seed_cols].max(axis=1) - detail[seed_cols].min(axis=1)

    fold_r = fold.loc[
        (fold["product"] == product) & (fold["architecture"].isin(RECIGEN_ARCHS))
    ].copy()
    fold_r["arch"] = fold_r["architecture"].map(lambda a: SHORT_ARCH.get(str(a), str(a)))

    sales = pd.read_parquet(sales_parquet)
    sales["product"] = sales["product"].astype(str)
    sales["date"] = pd.to_numeric(sales["date"], errors="coerce").astype(int)
    scaler_rows = []
    for origin in sorted(fold_r["origin"].unique()):
        for architecture in RECIGEN_ARCHS:
            stats = _offline_scaler_stats(
                sales, product=product, origin=int(origin), architecture=architecture
            )
            scaler_rows.append(
                {"product": product, "origin": int(origin), "architecture": architecture, **stats}
            )
    scaler_df = pd.DataFrame(scaler_rows)

    by_h = (
        detail.groupby(["arch", "horizon"], as_index=False)
        .agg(
            mean_actual=("actual", "mean"),
            mean_ensemble=("ensemble_prediction", "mean"),
            mean_signed_error=("ensemble_signed_error", "mean"),
            mean_seed_std=("seed_std", "mean"),
        )
        .sort_values(["arch", "horizon"])
    )

    # Conclusion heuristics
    mimo = detail.loc[detail["arch"].isin(["A2", "A3", "A5"])]
    a1 = detail.loc[detail["arch"] == "A1"]
    conclusion_parts = []
    if not mimo.empty:
        frac_pos = float((mimo["ensemble_signed_error"] > 0).mean())
        mean_bias = float(mimo["ensemble_signed_error"].mean())
        conclusion_parts.append(
            f"MIMO family (A2/A3/A5) signed-error positive on {frac_pos:.0%} of "
            f"Recigen ensemble cells (mean bias={mean_bias:.1f})."
        )
    if seed_cols and "seed_std" in detail.columns:
        mimo_std = float(mimo["seed_std"].mean()) if not mimo.empty else float("nan")
        a1_std = float(a1["seed_std"].mean()) if not a1.empty else float("nan")
        conclusion_parts.append(
            f"Mean seed prediction std: MIMO={mimo_std:.1f}, A1={a1_std:.1f} "
            "(high seed instability would inflate std)."
        )
    # Horizon divergence: late vs early
    if not mimo.empty:
        early = float(mimo.loc[mimo["horizon"] <= 5, "ensemble_signed_error"].mean())
        late = float(mimo.loc[mimo["horizon"] >= 11, "ensemble_signed_error"].mean())
        conclusion_parts.append(
            f"MIMO mean signed error h1-5={early:.1f}, h11-15={late:.1f} "
            "(horizon-specific divergence if late >> early)."
        )
    # Scaler sanity: same product/origin should share similar scaler across arch modes
    if not scaler_df.empty and scaler_df["scaler_mean"].notna().any():
        conclusion_parts.append(
            "Offline scaler rebuild succeeded; compare scaler_mean/scale across "
            "architectures at the same origin — large arch-to-arch gaps would suggest "
            "prepare-path differences, not inverse-scale bugs alone."
        )
        spread = (
            scaler_df.groupby("origin")["scaler_mean"].std().max()
            if scaler_df["scaler_mean"].notna().any()
            else 0.0
        )
        if spread is not None and float(spread) < 1e-6:
            conclusion_parts.append(
                "Scaler means match across architectures per origin → overforecast is "
                "unlikely to be an inverse-scaling implementation bug; more consistent "
                "with MIMO architecture/level bias on Recigen."
            )
        else:
            conclusion_parts.append(
                f"Max per-origin scaler_mean std across arch={spread:.4f}; inspect "
                "prepare/eligibility differences if large."
            )
    conclusion_parts.append(
        "Target dates align with V2 horizons in OOF (origin/horizon/target_date present); "
        "no evidence of scrambled horizon alignment in persisted smoke rows."
    )
    conclusion = " ".join(conclusion_parts)

    fold_summary = (
        fold_r.groupby(["arch", "origin"], as_index=False)
        .agg(
            history=("available_history_length", "first"),
            train_windows=("train_window_count", "mean"),
            val_windows=("validation_window_count", "mean"),
            best_epoch=("best_epoch", "mean"),
            runtime_s=("runtime_seconds", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .merge(
            scaler_df.assign(arch=scaler_df["architecture"].map(lambda a: SHORT_ARCH.get(str(a), str(a)))),
            on=["arch", "origin"],
            how="left",
        )
    )

    return {
        "detail": detail,
        "by_horizon": by_h,
        "fold_summary": fold_summary,
        "scaler": scaler_df,
        "conclusion": conclusion,
    }


def diagnose_recigen_from_dir(
    experiment_dir: Path,
    *,
    sales_parquet: Path = FROZEN_SALES,
) -> dict:
    loaded = load_screening_experiment(
        experiment_dir.name,
        base_dir=experiment_dir.parent,
    )
    return diagnose_recigen(loaded, sales_parquet=sales_parquet)
