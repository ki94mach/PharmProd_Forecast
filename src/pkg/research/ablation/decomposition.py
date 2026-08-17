"""Absolute-error decomposition vs F0 for ablation experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pkg.benchmark.evaluate import BacktestResult, wmape
from pkg.research.f2.config import HIGH_VOLUME_WATCHLIST
from pkg.research.harness.metrics import error_concentration as _error_concentration
from pkg.research.harness.metrics import merge_ae as _merge_ae


def decompose_vs_f0(
    f0: BacktestResult,
    cand: BacktestResult,
    experiment: str,
    anchor: str,
    out_dir: Path,
) -> dict:
    m = _merge_ae(f0, cand)
    conc = _error_concentration(m, experiment, anchor)
    by_p = conc["by_product"].copy()
    by_p["experiment"] = experiment
    by_p["anchor"] = anchor
    by_p.to_csv(out_dir / f"by_product_{experiment}_{anchor}.csv", index=False)

    top_rows = []
    for _, row in conc["top10_det"].iterrows():
        top_rows.append(
            {
                "experiment": experiment,
                "anchor": anchor,
                "direction": "deterioration",
                **row.to_dict(),
            }
        )
    for _, row in conc["top5_imp"].iterrows():
        top_rows.append(
            {
                "experiment": experiment,
                "anchor": anchor,
                "direction": "improvement",
                **row.to_dict(),
            }
        )
    for sku in HIGH_VOLUME_WATCHLIST:
        sub = m.loc[m["product"] == sku]
        if sub.empty:
            continue
        top_rows.append(
            {
                "experiment": experiment,
                "anchor": anchor,
                "direction": "watchlist",
                "product": sku,
                "delta_ae": float(sub["delta_ae"].sum()),
                "actual_volume": float(np.abs(sub["actual"]).sum()),
                "n": len(sub),
                "wmape_f0": wmape(sub["actual"], sub["pred_f0"]),
                "wmape_cand": wmape(sub["actual"], sub["pred_cand"]),
            }
        )
    return {
        "summary": {
            "experiment": experiment,
            "anchor": anchor,
            "net_delta_ae": conc["net_delta_ae"],
            "total_deterioration": conc["total_deterioration"],
            "total_improvement": conc["total_improvement"],
            "top1_deterioration_share": conc["top1_deterioration_share"],
            "top5_deterioration_share": conc["top5_deterioration_share"],
            "top10_deterioration_share": conc["top10_deterioration_share"],
            "top5_improvement_share": conc["top5_improvement_share"],
            "flags": ";".join(conc["flags"]),
        },
        "top_rows": top_rows,
        "merged": m,
    }
