"""Launch expanded V3A architecture-validation screening (recoverable).

Checkpoints after each SKU under src/data/ts_v3a/screening/.incomplete/.
If that incomplete dir already exists, the CLI auto-resumes and skips
finished products. Pass --resume explicitly only when forcing resume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pkg.ts_v3a.screen import main

PANEL = ROOT / "src/pkg/benchmark/universes/v3a_architecture_validation_products.csv"
products = ",".join(pd.read_csv(PANEL)["product"].astype(str).tolist())
origins = "140104,140201,140301,140402"
experiment_id = "20260908T103000Z_archval"
output = ROOT / "src/data/ts_v3a/screening"
incomplete_ckpt = output / ".incomplete" / experiment_id / "checkpoint.json"

argv = [
    "--products",
    products,
    "--origins",
    origins,
    "--architectures",
    "a0,a1,a2,a3,a4,a5",
    "--seeds",
    "41,42,43",
    "--output",
    str(output),
    "--experiment-id",
    experiment_id,
]
if incomplete_ckpt.is_file():
    argv.append("--resume")

raise SystemExit(main(argv))
