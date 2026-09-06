"""Insufficient-history handling for V3A window builders."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.types import TargetMode
from pkg.ts_v3a.windows import (
    InsufficientHistoryError,
    build_mimo_windows,
    build_recursive_windows,
    min_history_length,
)


class TestInsufficientHistory(unittest.TestCase):
    def test_recursive_too_short_raises(self):
        history = np.arange(12, dtype=float)  # need lookback+1 = 13
        with self.assertRaises(InsufficientHistoryError) as ctx:
            build_recursive_windows(history, lookback=12)
        err = ctx.exception
        self.assertEqual(err.mode, TargetMode.RECURSIVE)
        self.assertEqual(err.n_history, 12)
        self.assertEqual(err.n_samples, 0)

    def test_mimo_too_short_raises(self):
        history = np.arange(26, dtype=float)  # need 27
        with self.assertRaises(InsufficientHistoryError) as ctx:
            build_mimo_windows(history, lookback=12, horizon=15)
        err = ctx.exception
        self.assertEqual(err.mode, TargetMode.DIRECT_MIMO)
        self.assertEqual(err.lookback, 12)
        self.assertEqual(err.horizon, 15)

    def test_require_samples_false_returns_empty(self):
        history = np.arange(10, dtype=float)
        ds = build_recursive_windows(history, lookback=12, require_samples=False)
        self.assertEqual(ds.n_samples, 0)
        self.assertEqual(ds.X.shape, (0, 12, 1))
        self.assertEqual(ds.y.shape, (0, 1))

        ds2 = build_mimo_windows(
            history, lookback=12, horizon=15, require_samples=False
        )
        self.assertEqual(ds2.n_samples, 0)
        self.assertEqual(ds2.X.shape, (0, 12, 1))
        self.assertEqual(ds2.y.shape, (0, 15))

    def test_min_history_length_helpers(self):
        self.assertEqual(min_history_length(12, 1, TargetMode.RECURSIVE), 13)
        self.assertEqual(min_history_length(12, 15, TargetMode.DIRECT_MIMO), 27)


if __name__ == "__main__":
    unittest.main()
