"""A2–A5 production alignment: O-2 cutoff, bridge gap, h1=origin, fold leakage."""
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
from pkg.ts_v2.config import DEFAULT_CONFIG as V2_DEFAULT
from pkg.ts_v2.data import prepare_monthly_series
from pkg.ts_v2.dates import make_forecast_window
from pkg.ts_v3a.architectures import ArchitectureName, uses_production_time_contract
from pkg.ts_v3a.backtest import _window_for_architecture
from pkg.ts_v3a.models.a2_mimo_lstm import MimoLSTM
from pkg.ts_v3a.models.a3_stacked_mimo_lstm import StackedMimoLSTM
from pkg.ts_v3a.models.a4_encoder_decoder_lstm import EncoderDecoderLSTM
from pkg.ts_v3a.models.a5_bidirectional_mimo_lstm import BidirectionalMimoLSTM
from pkg.ts_v3a.prepare import prepare_neural_fold
from pkg.ts_v3a.scaling import FoldScaler
from pkg.ts_v3a.types import TargetMode

_MIMO_ARCHES = (
    ArchitectureName.A2_MIMO_LSTM,
    ArchitectureName.A3_STACKED_MIMO_LSTM,
    ArchitectureName.A4_ENCODER_DECODER_LSTM,
    ArchitectureName.A5_BIDIRECTIONAL_MIMO_LSTM,
)

_ORIGIN = 140501
_LAST_COMPLETE = 140411  # origin - 2
_BRIDGE = 140412  # origin - 1


def _synthetic_sales(origin: int = _ORIGIN, n_months: int = 36) -> pd.DataFrame:
    start = shamsi_add_months(origin, -n_months)
    rows = []
    cur = start
    while cur <= origin:
        rows.append({"product": "P", "date": cur, "sales": 10.0 + (cur % 7)})
        cur = shamsi_add_months(cur, 1)
    return pd.DataFrame(rows)


class TestA2A5UseProductionContract(unittest.TestCase):
    def test_all_mimo_arches_on_production(self):
        for arch in _MIMO_ARCHES:
            self.assertTrue(uses_production_time_contract(arch), msg=arch.value)

    def test_window_routing_matches_make_forecast_window(self):
        for arch in _MIMO_ARCHES:
            win = _window_for_architecture(_ORIGIN, arch, v2_config=V2_DEFAULT)
            prod = make_forecast_window(_ORIGIN, config=V2_DEFAULT)
            self.assertEqual(win.target_dates, prod.target_dates)
            self.assertEqual(win.internal_target_dates, prod.internal_target_dates)
            self.assertEqual(win.training_end, prod.training_end)
            self.assertEqual(win.current_partial_month, prod.current_partial_month)


class TestProductionWindowSixteenFifteen(unittest.TestCase):
    def test_bridge_and_delivered_calendar(self):
        win = make_forecast_window(_ORIGIN)
        self.assertEqual(len(win.internal_target_dates), 16)
        self.assertEqual(len(win.target_dates), 15)
        self.assertEqual(win.target_dates[0], _ORIGIN)
        self.assertEqual(win.horizons[0], 1)
        self.assertEqual(win.internal_target_dates[0], _BRIDGE)
        self.assertNotIn(_BRIDGE, win.target_dates)
        self.assertEqual(win.training_end, _LAST_COMPLETE)
        self.assertEqual(win.current_partial_month, _BRIDGE)


class TestMimoDense15EquivalentOffset(unittest.TestCase):
    """MIMO keeps Dense(15) on delivered dates; bridge is calendar gap only."""

    def _inject_predict(self, model, n_out: int = 15):
        scaler = FoldScaler()
        lookback = np.arange(12, dtype=float)
        scaler.fit_on_series(lookback)
        fake_keras = MagicMock()
        fake_keras.predict.return_value = np.arange(1, n_out + 1, dtype=float).reshape(
            1, n_out
        )
        trainer = MagicMock()
        trainer.model_ = fake_keras
        model._trainer = trainer
        model._scaler = scaler
        model._last_lookback_raw = lookback.copy()
        model.metadata_ = None
        return scaler

    def test_a2_predict_fifteen_aligned_to_delivered(self):
        window = make_forecast_window(_ORIGIN)
        model = MimoLSTM()
        scaler = self._inject_predict(model, 15)
        outcome = model.predict(window)
        self.assertEqual(len(outcome.predictions), 15)
        self.assertEqual(outcome.target_dates, window.target_dates)
        self.assertEqual(outcome.target_dates[0], _ORIGIN)
        self.assertEqual(outcome.horizons[0], 1)
        expected = scaler.inverse_transform_y(
            np.arange(1, 16, dtype=float)
        ).reshape(-1)
        np.testing.assert_allclose(
            np.asarray(outcome.predictions, dtype=float), expected, rtol=1e-6
        )

    def test_a3_a4_a5_predict_length_and_h1(self):
        window = make_forecast_window(_ORIGIN)
        for cls in (StackedMimoLSTM, EncoderDecoderLSTM, BidirectionalMimoLSTM):
            model = cls()
            self._inject_predict(model, 15)
            outcome = model.predict(window)
            self.assertEqual(len(outcome.predictions), 15, msg=cls.__name__)
            self.assertEqual(outcome.target_dates[0], _ORIGIN, msg=cls.__name__)


class TestCutoffLeakageAndValidationFolds(unittest.TestCase):
    def test_prepared_history_ends_at_origin_minus_2(self):
        sales = _synthetic_sales()
        window = make_forecast_window(_ORIGIN)
        prepared = prepare_monthly_series(sales, "P", window)
        self.assertEqual(prepared.last_training_month, _LAST_COMPLETE)
        self.assertEqual(max(prepared.dates), _LAST_COMPLETE)
        self.assertNotIn(_BRIDGE, prepared.dates)
        self.assertNotIn(_ORIGIN, prepared.dates)

    def test_neural_fold_windows_before_origin_and_consistent_with_o2(self):
        sales = _synthetic_sales(n_months=48)
        window = make_forecast_window(_ORIGIN)
        prepared = prepare_monthly_series(sales, "P", window)
        fold, _scaler = prepare_neural_fold(
            prepared.values,
            forecast_origin=int(window.forecast_origin),
            seed=42,
            mode=TargetMode.DIRECT_MIMO,
            require_eligible=True,
        )
        self.assertLess(int(fold.metadata.training_end), _ORIGIN)
        self.assertLessEqual(int(fold.metadata.training_end), _LAST_COMPLETE)
        if fold.metadata.validation_end is not None:
            self.assertLess(int(fold.metadata.validation_end), _ORIGIN)
            self.assertLessEqual(int(fold.metadata.validation_end), _LAST_COMPLETE)
        # Supervised MIMO targets never include bridge or origin calendar months
        # as input history indices beyond last_complete.
        hist_dates = list(prepared.dates)
        self.assertTrue(all(d <= _LAST_COMPLETE for d in hist_dates))


if __name__ == "__main__":
    unittest.main()
