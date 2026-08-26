"""Status from Dim.Product OrchidBoxQuantity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.product_status import status_from_orchid_box_quantity


class TestStatusFromOrchidBoxQuantity(unittest.TestCase):
    def test_pass_through_pack_and_unit(self):
        self.assertEqual(status_from_orchid_box_quantity("بسته"), "بسته")
        self.assertEqual(status_from_orchid_box_quantity("عدد"), "عدد")

    def test_strips_whitespace(self):
        self.assertEqual(status_from_orchid_box_quantity("  بسته  "), "بسته")

    def test_missing_or_unknown_is_unit(self):
        self.assertEqual(status_from_orchid_box_quantity(None), "عدد")
        self.assertEqual(status_from_orchid_box_quantity(pd.NA), "عدد")
        self.assertEqual(status_from_orchid_box_quantity(1), "عدد")
        self.assertEqual(status_from_orchid_box_quantity("other"), "عدد")


if __name__ == "__main__":
    unittest.main()
