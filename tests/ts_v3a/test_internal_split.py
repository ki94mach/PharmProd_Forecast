"""Internal validation entirely before forecast origin; outer horizon untouched."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.dates import make_screening_forecast_window, target_month
from pkg.ts_v3a.split import assert_split_before_origin, chronological_train_val_split
from pkg.ts_v3a.windows import build_mimo_windows, build_recursive_windows


def _shamsi_months(start: int, n: int) -> list[int]:
    return [target_month(start, i + 1) for i in range(n)]


class TestInternalSplit(unittest.TestCase):
    def test_validation_entirely_before_forecast_origin(self):
        origin = 140501
        # MIMO / screening contract: history through origin - 1.
        window = make_screening_forecast_window(origin)
        months = _shamsi_months(140201, 36)
        self.assertEqual(months[-1], window.training_end)
        history = pd.Series(np.arange(36, dtype=float), index=months)
        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)
        self.assertGreater(split.n_validation, 0)
        assert_split_before_origin(
            split, n_history=36, forecast_origin=origin
        )
        for end_idx in split.validation.end_indices:
            last_target_idx = end_idx + 15
            self.assertLess(last_target_idx, 36)
        for end_date in split.validation.end_dates:
            self.assertIsNotNone(end_date)
            self.assertLess(end_date, origin)

    def test_outer_test_horizon_untouched_by_early_stopping(self):
        origin = 140501
        window = make_screening_forecast_window(origin)
        months = _shamsi_months(140201, 36)
        history = pd.Series(np.arange(36, dtype=float), index=months)
        outer_dates = set(window.target_dates)
        self.assertEqual(len(outer_dates), 15)

        ds = build_mimo_windows(history, lookback=12, horizon=15)
        split = chronological_train_val_split(ds, validation_fraction=0.2)

        used_end_dates = set(split.train.end_dates)
        if split.validation is not None:
            used_end_dates |= set(split.validation.end_dates)

        # No window end_date may be an outer target month; all ends are < origin.
        self.assertTrue(used_end_dates.isdisjoint(outer_dates))
        for d in used_end_dates:
            self.assertLess(d, origin)

        # Targets themselves are months after end_date; last target month for
        # each sample is still before origin on a contiguous pre-origin grid.
        for part in (split.train, split.validation):
            if part is None:
                continue
            for end_idx, end_date in zip(part.end_indices, part.end_dates):
                for h in range(1, 16):
                    target_idx = end_idx + h
                    self.assertLess(target_idx, 36)
                    target_date = months[target_idx]
                    self.assertNotIn(target_date, outer_dates)
                    self.assertLess(target_date, origin)

    def test_recursive_split_also_before_origin(self):
        origin = 140501
        months = _shamsi_months(140201, 36)
        history = pd.Series(np.arange(36, dtype=float), index=months)
        ds = build_recursive_windows(history, lookback=12)
        split = chronological_train_val_split(ds, validation_fraction=0.2)
        assert_split_before_origin(split, n_history=36, forecast_origin=origin)
        self.assertGreater(split.n_train, 0)
        self.assertGreater(split.n_validation, 0)


if __name__ == "__main__":
    unittest.main()
