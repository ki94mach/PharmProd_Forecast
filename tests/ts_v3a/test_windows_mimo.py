"""MIMO window shapes and exact 15-output target alignment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.types import TargetMode
from pkg.ts_v3a.windows import build_mimo_windows, expected_n_samples


class TestMimoWindows(unittest.TestCase):
    def test_mimo_shapes_lookback_12_horizon_15(self):
        history = np.arange(36, dtype=float)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        self.assertEqual(ds.X.shape, (10, 12, 1))
        self.assertEqual(ds.y.shape, (10, 15))
        self.assertEqual(ds.mode, TargetMode.DIRECT_MIMO)
        self.assertEqual(ds.horizon, 15)

    def test_exact_15_output_target_alignment(self):
        history = np.arange(40, dtype=float)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        for i, t in enumerate(ds.end_indices):
            # y must be exactly t+1 ... t+15
            expected = history[t + 1 : t + 16]
            self.assertEqual(len(expected), 15)
            np.testing.assert_array_equal(ds.y[i], expected)
            np.testing.assert_array_equal(ds.X[i, :, 0], history[t - 11 : t + 1])
            self.assertLess(t + 15, len(history))

    def test_sample_counts_for_example_histories(self):
        for n, expected in ((36, 10), (48, 22), (60, 34)):
            self.assertEqual(
                expected_n_samples(n, 12, 15, TargetMode.DIRECT_MIMO), expected
            )
            ds = build_mimo_windows(np.arange(n, dtype=float), lookback=12, horizon=15)
            self.assertEqual(ds.n_samples, expected)


if __name__ == "__main__":
    unittest.main()
