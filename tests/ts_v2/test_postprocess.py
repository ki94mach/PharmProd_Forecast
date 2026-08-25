"""V2 centralized post-processing policy tests."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v2.config import TSForecastConfig
from pkg.ts_v2.models import BaseForecastModel, ForecastResult
from pkg.ts_v2.postprocess import (
    V2_FORBIDDEN_POSTPROCESS_NAMES,
    apply_final_constraints,
    apply_nonnegativity,
    assert_v2_postprocess_allowed,
    export_quantities,
)
from pkg.ts_v2.types import ForecastResult as ForecastResultType


class NegativeForecastModel(BaseForecastModel):
    name = "negative_model"

    def fit(self, train_series):
        return self

    def predict(self, horizon: int, target_dates: Sequence[int]) -> ForecastResult:
        dates = tuple(int(d) for d in target_dates)
        return ForecastResult(
            model_name=self.name,
            predictions=tuple(-float(h) for h in range(1, horizon + 1)),
            target_dates=dates,
            horizons=tuple(range(1, horizon + 1)),
        )


class TestNonnegativityConstraint(unittest.TestCase):
    def test_apply_nonnegativity_without_rounding(self):
        raw = (-3.7, 0.0, 2.5)
        out = apply_nonnegativity(raw)
        self.assertEqual(out, (0.0, 0.0, 2.5))
        self.assertIsInstance(out[0], float)

    def test_apply_final_constraints_keeps_raw_and_constrained(self):
        raw = ForecastResultType(
            model_name="m",
            predictions=(-1.5, 10.2),
            target_dates=(140501, 140502),
            horizons=(1, 2),
        )
        out = apply_final_constraints(raw, config=TSForecastConfig(nonnegative_forecasts=True))
        self.assertEqual(out.raw_predictions, (-1.5, 10.2))
        self.assertEqual(out.constrained_predictions, (0.0, 10.2))
        self.assertEqual(out.metadata["nonneg_adjustment"]["n_adjusted"], 1)

    def test_disabled_nonnegativity_passes_through(self):
        raw = ForecastResultType(
            model_name="m",
            predictions=(-4.0,),
            target_dates=(140501,),
            horizons=(1,),
        )
        out = apply_final_constraints(raw, config=TSForecastConfig(nonnegative_forecasts=False))
        self.assertEqual(out.raw_predictions, (-4.0,))
        self.assertEqual(out.constrained_predictions, (-4.0,))

    def test_export_rounds_only_at_legacy_boundary(self):
        raw = ForecastResultType(
            model_name="m",
            predictions=(10.6, -2.2),
            target_dates=(140501, 140502),
            horizons=(1, 2),
        )
        processed = apply_final_constraints(raw)
        unrounded = export_quantities(processed, round_for_legacy=False)
        rounded = export_quantities(processed, round_for_legacy=True)
        self.assertEqual(unrounded, (10.6, 0.0))
        self.assertEqual(rounded, (11.0, 0.0))


class TestV1SmoothingForbidden(unittest.TestCase):
    def test_forbidden_names_raise(self):
        for name in V2_FORBIDDEN_POSTPROCESS_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    assert_v2_postprocess_allowed(name)

    def test_ts_v2_modules_do_not_reference_v1_smoothing(self):
        root = Path(__file__).resolve().parents[2] / "src" / "pkg" / "ts_v2"
        forbidden_tokens = (
            "redistribute_smoothing",
            "replace_negative_sales",
            "from pkg.forecast import",
            "import pkg.forecast",
            "from pkg.sales_forecasting import",
            "import pkg.sales_forecasting",
        )
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if path.name == "postprocess.py":
                # This module documents forbidden names intentionally.
                continue
            for token in forbidden_tokens:
                if token in text:
                    offenders.append(f"{path.relative_to(root)}: {token}")
        self.assertEqual(offenders, [])

    def test_ts_v2_ast_does_not_import_v1_forecast_modules(self):
        root = Path(__file__).resolve().parents[2] / "src" / "pkg" / "ts_v2"
        banned_modules = {
            "pkg.forecast",
            "pkg.sales_forecasting",
        }
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in banned_modules or alias.name.startswith(
                            "pkg.forecast."
                        ):
                            offenders.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in banned_modules:
                        offenders.append(f"{path.name}: from {node.module}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
