"""Deterministic Persian text normalization for exact F3B name matching."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

# Arabic / presentation forms → Persian letters. Do not stem or drop dosages.
_YEH = {
    "\u064a": "\u06cc",  # ي → ی
    "\u0649": "\u06cc",  # ى → ی
    "\u06d0": "\u06cc",
    "\u06d2": "\u06cc",
}
_KAF = {
    "\u0643": "\u06a9",  # ك → ک
    "\u06aa": "\u06a9",
}

_ZERO_WIDTH = frozenset(
    {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2060",
        "\ufeff",
    }
)


def normalize_fa(value: Any) -> str:
    """Strip, collapse whitespace, NFKC, Yeh/Kaf, drop zero-width chars.

    Does not remove dosage numbers, units, or formulation terms.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    out: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH or unicodedata.category(ch) == "Cf":
            continue
        if ch in _YEH:
            out.append(_YEH[ch])
            continue
        if ch in _KAF:
            out.append(_KAF[ch])
            continue
        out.append(ch)
    collapsed = re.sub(r"\s+", " ", "".join(out), flags=re.UNICODE)
    return collapsed.strip()
