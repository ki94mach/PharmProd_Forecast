"""Excel unit status from Dim.Product OrchidBoxQuantity."""
from __future__ import annotations

import pandas as pd

# Dim.Product.OrchidBoxQuantity stores the unit label itself (validated in DW).
_PACK = "بسته"
_UNIT = "عدد"
_ALLOWED = {_PACK, _UNIT}


def status_from_orchid_box_quantity(orchid_box_quantity) -> str:
    """Return Excel status from Dim.Product OrchidBoxQuantity.

    Warehouse values are already ``بسته`` or ``عدد``. Missing / unexpected
    values fall back to ``عدد``.
    """
    if orchid_box_quantity is None or pd.isna(orchid_box_quantity):
        return _UNIT
    text = str(orchid_box_quantity).strip()
    if text in _ALLOWED:
        return text
    return _UNIT
