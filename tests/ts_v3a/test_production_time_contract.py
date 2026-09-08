"""A0/A1 production time contract: leakage cut and 16→15 horizon mapping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import shamsi_add_months
from pkg.ts_v2.dates import (
    make_forecast_window,
    make_screening_forecast_window,
)
from pkg.ts_v3a.architectures import (
    ArchitectureName,
    uses_production_time_contract,
)
from pkg.ts_v3a.models.a1_small_recursive_lstm import SmallRecursiveLSTM
from pkg.ts_v3a.scaling import FoldScaler


class TestArchitectureTimeContractRouting(unittest.TestCase):
    def test_a0_a1_use_production_a2_a5_screening(self):
        self.assertTrue(
            uses_production_time_contract(
                ArchitectureName.A0_LEGACY_ADAPTIVE_RECURSIVE_LSTM
            )
        )
        self.assertTrue(
            uses_production_time_contract(ArchitectureName.A1_SMALL_RECURSIVE_LSTM)
        )
        self.assertFalse(uses_production_time_contract(ArchitectureName.A2_MIMO_LSTM))
        self.assertFalse(
            uses_production_time_contract(ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM)
        )


class TestA1ProductionPredictBridge(unittest.TestCase):
    def test_predict_rolls_sixteen_discards_bridge(self):
        window = make_forecast_window(140501)
        self.assertEqual(len(window.internal_target_dates), 16)
        self.assertEqual(len(window.target_dates), 15)

        model = SmallRecursiveLSTM()
        # Bypass fit: inject scaler + fake trainer that returns ascending preds.
        scaler = FoldScaler()
        lookback = np.arange(12, dtype=float)
        scaler.fit_on_series(lookback)

        fake_keras = MagicMock()
        # Each predict call returns next integer in scaled space.
        counter = {"n": 0}

        def _predict(x, verbose=0):  # noqa: ANN001
            counter["n"] += 1
            return np.array([[float(counter["n"])]], dtype=float)

        fake_keras.predict.side_effect = _predict
        trainer = MagicMock()
        trainer.model_ = fake_keras

        model._trainer = trainer
        model._scaler = scaler
        model._last_lookback_raw = lookback.copy()
        model.metadata_ = None

        outcome = model.predict(window)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        self.assertEqual(outcome.horizons, window.horizons)
        # Internal steps 1..16 in scaled space; bridge (1) discarded → delivered 2..16
        self.assertEqual(counter["n"], 16)
        delivered_scaled = np.arange(2, 17, dtype=float)
        expected = scaler.inverse_transform_y(delivered_scaled).reshape(-1)
        np.testing.assert_allclose(
            np.asarray(outcome.predictions, dtype=float), expected, rtol=1e-6
        )

    def test_screening_window_still_fifteen_steps(self):
        window = make_screening_forecast_window(140501)
        model = SmallRecursiveLSTM()
        scaler = FoldScaler()
        lookback = np.arange(12, dtype=float)
        scaler.fit_on_series(lookback)
        fake_keras = MagicMock()
        counter = {"n": 0}

        def _predict(x, verbose=0):  # noqa: ANN001
            counter["n"] += 1
            return np.array([[float(counter["n"])]], dtype=float)

        fake_keras.predict.side_effect = _predict
        trainer = MagicMock()
        trainer.model_ = fake_keras
        model._trainer = trainer
        model._scaler = scaler
        model._last_lookback_raw = lookback.copy()
        model.metadata_ = None

        outcome = model.predict(window)
        self.assertEqual(counter["n"], 15)
        self.assertEqual(len(outcome.predictions), 15)


class TestProductionTrainingCutMonths(unittest.TestCase):
    def test_prepared_production_history_ends_at_last_complete(self):
        from pkg.ts_v2.data import prepare_monthly_series

        origin = 140501
        start = shamsi_add_months(origin, -24)
        rows = []
        cur = start
        while cur <= origin:
            rows.append({"product": "P", "date": cur, "sales": 10.0})
            cur = shamsi_add_months(cur, 1)
        sales = pd.DataFrame(rows)
        window = make_forecast_window(origin)
        prepared = prepare_monthly_series(sales, "P", window)
        self.assertEqual(prepared.last_training_month, 140411)
        self.assertNotIn(140412, prepared.dates)
        self.assertNotIn(140501, prepared.dates)
        self.assertEqual(max(prepared.dates), 140411)


if __name__ == "__main__":
    unittest.main()
