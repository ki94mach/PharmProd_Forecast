"""Section 1: F0_CONTROL equivalence audit."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pkg.benchmark.dataset import BenchmarkDataset
from pkg.benchmark.evaluate import wmape
from pkg.research.audit.common import (
    align_predictions,
    audit_output_dir,
    run_f0_control_backtest,
    run_frozen_backtest,
    save_csv,
)

TOLS = (1e-6, 1e-3, 1.0)
MATERIAL_TOL = 1e-3
MATERIAL_WMAPE_DIFF = 0.01


def run_f0_control(
    ds: BenchmarkDataset,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    """Compare F0_CONTROL (research adapter) vs frozen ts_xgb / human_xgb."""
    out_dir = out_dir or audit_output_dir()
    rows = []
    passed = True
    details: dict[str, pd.DataFrame] = {}

    for anchor in ("ts", "human"):
        frozen = run_frozen_backtest(ds, anchor)
        control = run_f0_control_backtest(ds, anchor)

        frozen_preds = frozen.predictions.copy()
        frozen_preds["actual"] = frozen_preds["sales"].astype(float)
        frozen_preds["test_origin"] = frozen_preds["test_origin"].astype(int)

        control_preds = control.predictions.copy()
        control_preds["test_origin"] = control_preds["test_origin"].astype(int)

        merged = align_predictions(frozen_preds, control_preds)
        details[anchor] = merged

        max_diff = float(merged["abs_diff"].max())
        mean_diff = float(merged["abs_diff"].mean())
        wmape_frozen = wmape(merged["actual"], merged["pred_frozen"])
        wmape_control = wmape(merged["actual"], merged["pred_control"])
        wmape_diff = wmape_control - wmape_frozen
        rmse_frozen = float(np.sqrt(((merged["actual"] - merged["pred_frozen"]) ** 2).mean()))
        rmse_control = float(np.sqrt(((merged["actual"] - merged["pred_control"]) ** 2).mean()))
        rmse_diff = rmse_control - rmse_frozen

        tol_counts = {}
        for tol in TOLS:
            tol_counts[f"rows_gt_{tol}"] = int((merged["abs_diff"] > tol).sum())

        gate_ok = max_diff <= MATERIAL_TOL and abs(wmape_diff) <= MATERIAL_WMAPE_DIFF
        if not gate_ok:
            passed = False

        rows.append(
            {
                "anchor": anchor,
                "n_rows": len(merged),
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "wmape_frozen": wmape_frozen,
                "wmape_control": wmape_control,
                "wmape_diff": wmape_diff,
                "rmse_frozen": rmse_frozen,
                "rmse_control": rmse_control,
                "rmse_diff": rmse_diff,
                **tol_counts,
                "gate_passed": gate_ok,
            }
        )

        save_csv(merged, out_dir, f"f0_control_diff_{anchor}.csv")

    summary = pd.DataFrame(rows)
    save_csv(summary, out_dir, "f0_control_summary.csv")

    return {
        "summary": summary,
        "passed": passed,
        "details": details,
    }
