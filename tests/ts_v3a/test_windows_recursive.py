"""Recursive window shape and count tests for V3A."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.windows import build_recursive_windows, expected_n_samples
from pkg.ts_v3a.types import TargetMode


class TestRecursiveWindows(unittest.TestCase):
    def test_recursive_shapes_lookback_12(self):
        history = np.arange(36, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        self.assertEqual(ds.X.shape, (24, 12, 1))
        self.assertEqual(ds.y.shape, (24, 1))
        self.assertEqual(ds.mode, TargetMode.RECURSIVE)
        self.assertEqual(ds.horizon, 1)
        self.assertEqual(ds.n_samples, 24)

    def test_recursive_sample_alignment(self):
        history = np.arange(20, dtype=float)
        ds = build_recursive_windows(history, lookback=12)
        # First sample: X = 0..11, y = 12
        np.testing.assert_array_equal(ds.X[0, :, 0], history[0:12])
        self.assertEqual(ds.y[0, 0], history[12])
        self.assertEqual(ds.end_indices[0], 11)
        # Last sample: X = 7..18, y = 19
        np.testing.assert_array_equal(ds.X[-1, :, 0], history[7:19])
        self.assertEqual(ds.y[-1, 0], history[19])
        self.assertEqual(ds.end_indices[-1], 18)

    def test_sample_counts_for_example_histories(self):
        for n, expected in ((36, 24), (48, 36), (60, 48)):
            self.assertEqual(
                expected_n_samples(n, 12, 1, TargetMode.RECURSIVE), expected
            )
            ds = build_recursive_windows(np.arange(n, dtype=float), lookback=12)
            self.assertEqual(ds.n_samples, expected)


if __name__ == "__main__":
    unittest.main()
