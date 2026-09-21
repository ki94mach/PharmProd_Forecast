"""Train-only categorical encoders with deterministic unknown handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

UNKNOWN_TOKEN = "__UNKNOWN__"
CATEGORICAL_RAW_COLS: tuple[str, ...] = ("model", "Field", "ProductForm", "Provider")
CATEGORICAL_ENC_COLS: tuple[str, ...] = (
    "model_enc",
    "field_enc",
    "form_enc",
    "provider_enc",
)


class TrainOnlyCategoricalEncoder:
    """Fit category→int maps on training data only; map unseen to UNKNOWN code."""

    def __init__(self) -> None:
        self.maps_: dict[str, dict[str, int]] = {}
        self.unknown_code_: dict[str, int] = {}

    def fit(self, train: pd.DataFrame, raw_cols: tuple[str, ...] = CATEGORICAL_RAW_COLS) -> TrainOnlyCategoricalEncoder:
        for raw, enc in zip(raw_cols, CATEGORICAL_ENC_COLS):
            if raw not in train.columns:
                # Allow pre-encoded panels: fall back to empty map + unknown=0
                self.maps_[enc] = {}
                self.unknown_code_[enc] = 0
                continue
            vals = (
                train[raw]
                .fillna(UNKNOWN_TOKEN)
                .astype(str)
                .replace({"nan": UNKNOWN_TOKEN, "None": UNKNOWN_TOKEN})
            )
            categories = sorted(set(vals.tolist()) | {UNKNOWN_TOKEN})
            mapping = {c: i for i, c in enumerate(categories)}
            self.maps_[enc] = mapping
            self.unknown_code_[enc] = int(mapping[UNKNOWN_TOKEN])
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        raw_cols: tuple[str, ...] = CATEGORICAL_RAW_COLS,
    ) -> pd.DataFrame:
        out = frame.copy()
        for raw, enc in zip(raw_cols, CATEGORICAL_ENC_COLS):
            unk = int(self.unknown_code_.get(enc, 0))
            mapping = self.maps_.get(enc, {})
            if raw in out.columns:
                vals = (
                    out[raw]
                    .fillna(UNKNOWN_TOKEN)
                    .astype(str)
                    .replace({"nan": UNKNOWN_TOKEN, "None": UNKNOWN_TOKEN})
                )
                out[enc] = vals.map(lambda v: mapping.get(v, unk)).astype(np.int32)
            elif enc in out.columns:
                # Already encoded — leave as-is (smoke paths without attrs)
                out[enc] = out[enc].fillna(unk).astype(np.int32)
            else:
                out[enc] = unk
        return out

    def to_state(self) -> dict[str, Any]:
        return {
            "maps": self.maps_,
            "unknown_code": self.unknown_code_,
            "unknown_token": UNKNOWN_TOKEN,
            "fit_on": "training_rows_only",
        }
