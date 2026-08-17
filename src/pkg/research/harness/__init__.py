"""Shared Template Method for frozen-benchmark feature-family experiments."""
from pkg.research.harness.dataset import copy_dataset, enrich_dataset, resolve_origin_col
from pkg.research.harness.gates import (
    assert_freeze_unchanged,
    assert_wmape_gate,
    confirm_canonical_f0,
    freeze_checksums,
    wmape_gate_row,
)
from pkg.research.harness.metrics import (
    ROW_KEYS,
    assert_same_eval_rows,
    error_concentration,
    rel_wmape,
)
from pkg.research.harness.residual import make_residual_model
from pkg.research.harness.run import FamilySession, run_family
from pkg.research.harness.spec import ExperimentSpec, FamilyConfig

__all__ = [
    "ExperimentSpec",
    "FamilyConfig",
    "FamilySession",
    "ROW_KEYS",
    "assert_freeze_unchanged",
    "assert_same_eval_rows",
    "assert_wmape_gate",
    "confirm_canonical_f0",
    "copy_dataset",
    "enrich_dataset",
    "error_concentration",
    "freeze_checksums",
    "make_residual_model",
    "rel_wmape",
    "resolve_origin_col",
    "run_family",
    "wmape_gate_row",
]
