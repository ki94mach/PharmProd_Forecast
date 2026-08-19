"""Point-in-time patient-consumption profile features (F3D).

Source: frozen ``src/data/results/f3d/source/product_profile.parquet``.
No live SQL is used here.

These are static product attributes.  The same profile is attached to every
forecast origin, target month, and horizon for a given product.  Nothing is
derived from future sales.  The features cannot explain temporary events; they
allow the shared model to learn different error behaviour across product types.

Assumption: ``Dim.Product`` is a current snapshot.  No historical
reconstruction is attempted because the source does not contain dated rows.

Scored features
---------------
is_continuous_consumption
    ``PatientConsumeType == "Continuous"`` → 1.0
    ``PatientConsumeType == "SinglePeriod"`` → 0.0
    missing or unexpected type → NaN (never imputed)

log_patient_annual_consumption
    ``np.log1p(patient_annual_consumption)`` for finite, non-negative values.
    NaN when PatientConsumePerPeriod is missing, unexpected type, or negative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCORED_FEATURES: tuple[str, ...] = (
    "is_continuous_consumption",
    "log_patient_annual_consumption",
)
FEATURE_NAMES: tuple[str, ...] = SCORED_FEATURES

INTERMEDIATE_NAMES: tuple[str, ...] = ("patient_annual_consumption",)

DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "PatientConsumeType",
    "PatientConsumePerPeriod",
    "patient_annual_consumption",
    "is_continuous_consumption",
    "log_patient_annual_consumption",
)

KNOWN_TYPES: frozenset[str] = frozenset({"Continuous", "SinglePeriod"})


def _compute_annual(
    ptype: Optional[str], pperiod: Optional[float]
) -> float:
    """Return annualised consumption or NaN."""
    if ptype is None or (isinstance(ptype, float) and np.isnan(ptype)):
        return np.nan
    if pperiod is None or (isinstance(pperiod, float) and np.isnan(pperiod)):
        return np.nan
    val = float(pperiod)
    if ptype == "Continuous":
        return val * 12.0
    if ptype == "SinglePeriod":
        return val
    # unexpected type
    return np.nan


def _compute_indicator(ptype: Optional[str]) -> float:
    """Return 1.0 / 0.0 / NaN indicator for Continuous vs SinglePeriod."""
    if ptype is None or (isinstance(ptype, float) and np.isnan(ptype)):
        return np.nan
    if ptype == "Continuous":
        return 1.0
    if ptype == "SinglePeriod":
        return 0.0
    return np.nan


def load_frozen_profile(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the frozen product profile parquet produced by ``prepare_f3d``."""
    if path is None:
        from pkg.research.f3d.config import f3d_source_dir
        path = f3d_source_dir() / "product_profile.parquet"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen F3D product profile missing: {path}. "
            "Run: python -m pkg.research.prepare_f3d"
        )
    df = pd.read_parquet(path)
    df["product"] = df["product"].astype(str)
    return df


def add_patient_consumption_features(
    panel: pd.DataFrame,
    profile: Optional[pd.DataFrame] = None,
    *,
    negative_report: Optional[list] = None,
) -> pd.DataFrame:
    """Attach F3D scored features to *panel* in-place (copy returned).

    Parameters
    ----------
    panel:
        DataFrame with at least a ``product`` column.
    profile:
        Pre-loaded frozen profile.  Loaded from disk if None.
    negative_report:
        If a list is provided, any products with negative
        ``PatientConsumePerPeriod`` are appended as dicts.
    """
    if profile is None:
        profile = load_frozen_profile()

    prof = profile[["product", "PatientConsumeType", "PatientConsumePerPeriod"]].copy()
    prof["product"] = prof["product"].astype(str)
    prof = prof.drop_duplicates("product")

    out = panel.copy()
    out["product"] = out["product"].astype(str)
    out = out.merge(prof, on="product", how="left")

    ptypes = out["PatientConsumeType"].to_numpy(dtype=object)
    pperiods = pd.to_numeric(out["PatientConsumePerPeriod"], errors="coerce").to_numpy(dtype=float)

    annual = np.array([_compute_annual(t, p) for t, p in zip(ptypes, pperiods)])

    # Negative raw values → NaN annual + report
    neg_mask = np.isfinite(pperiods) & (pperiods < 0)
    if neg_mask.any():
        annual[neg_mask] = np.nan
        if negative_report is not None:
            for idx in np.where(neg_mask)[0]:
                negative_report.append(
                    {
                        "product": str(out["product"].iloc[idx]),
                        "PatientConsumeType": str(ptypes[idx]),
                        "PatientConsumePerPeriod": float(pperiods[idx]),
                    }
                )

    log_annual = np.where(
        np.isfinite(annual) & (annual >= 0),
        np.log1p(annual),
        np.nan,
    )

    indicator = np.array([_compute_indicator(t) for t in ptypes])

    out["patient_annual_consumption"] = annual
    out["log_patient_annual_consumption"] = log_annual
    out["is_continuous_consumption"] = indicator

    return out
