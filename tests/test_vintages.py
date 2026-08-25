"""Tests for canonical TS backfill vintage manifest."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.calendar import (
    iter_shamsi_quarters,
    origin_from_quarter,
    quarter_from_origin,
)
from pkg.benchmark.vintages import (
    EXPECTED_N_QUARTERS,
    VintageManifestError,
    assert_training_cutoff_exclusive,
    build_vintage_specs,
    classify_vintage_status,
    default_manifest_csv_path,
    default_manifest_meta_path,
    load_vintage_manifest,
    validate_vintage_manifest,
    write_vintage_manifest,
)


class TestQuarterOriginMapping(unittest.TestCase):
    def test_round_trip(self):
        for qrt in ("1401Q1", "1401Q2", "1404Q3", "1405Q2"):
            origin = origin_from_quarter(qrt)
            self.assertEqual(quarter_from_origin(origin), qrt)

    def test_first_month_of_quarter(self):
        self.assertEqual(origin_from_quarter("1405Q1"), 140501)
        self.assertEqual(origin_from_quarter("1404Q2"), 140404)
        self.assertEqual(origin_from_quarter("1402Q4"), 140210)
        self.assertEqual(origin_from_quarter("1401Q2"), 140104)

    def test_contiguous_18_quarters(self):
        qrts = iter_shamsi_quarters("1401Q1", "1405Q2")
        self.assertEqual(len(qrts), 18)
        self.assertEqual(qrts[0], "1401Q1")
        self.assertEqual(qrts[-1], "1405Q2")
        self.assertIn("1401Q2", qrts)


class TestStatusClassification(unittest.TestCase):
    def test_fully_partial_forecastable_future(self):
        as_of = 140504
        self.assertEqual(
            classify_vintage_status(140101, as_of_complete_month=as_of),
            "fully_matured",
        )
        self.assertEqual(
            classify_vintage_status(140404, as_of_complete_month=as_of),
            "partially_matured",
        )
        self.assertEqual(
            classify_vintage_status(140505, as_of_complete_month=as_of),
            "forecastable",
        )
        self.assertEqual(
            classify_vintage_status(140506, as_of_complete_month=as_of),
            "future_origin",
        )


class TestBuildAndValidate(unittest.TestCase):
    def test_build_has_18_unique_increasing_origins(self):
        specs = build_vintage_specs(as_of_complete_month=140504)
        self.assertEqual(len(specs), EXPECTED_N_QUARTERS)
        origins = [s.forecast_origin for s in specs]
        self.assertEqual(len(set(origins)), len(origins))
        self.assertEqual(origins, sorted(origins))
        for i in range(1, len(origins)):
            self.assertGreater(origins[i], origins[i - 1])
        for spec in specs:
            self.assertEqual(spec.horizon, 15)
            self.assertEqual(spec.training_cutoff_exclusive, spec.forecast_origin)
            self.assertEqual(spec.forecast_origin, origin_from_quarter(spec.quarter))
        assert_training_cutoff_exclusive(specs)
        result = validate_vintage_manifest(specs, require_meta=False)
        self.assertTrue(result.ok, msg="; ".join(result.errors))

    def test_gaps_detected(self):
        specs = build_vintage_specs(as_of_complete_month=140504)
        broken = list(specs)
        broken.pop(3)  # remove 1401Q4
        result = validate_vintage_manifest(broken, require_meta=False)
        self.assertFalse(result.ok)
        self.assertTrue(any("gap" in e or "unexpected" in e for e in result.errors))

    def test_write_and_reload(self):
        specs = build_vintage_specs(as_of_complete_month=140504)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, meta_path = write_vintage_manifest(
                specs, as_of_complete_month=140504, out_dir=tmp, force=True
            )
            loaded = load_vintage_manifest(csv_path, validate=True)
            self.assertEqual(len(loaded), 18)
            self.assertEqual(
                [(s.quarter, s.forecast_origin) for s in loaded],
                [(s.quarter, s.forecast_origin) for s in specs],
            )
            result = validate_vintage_manifest(
                csv_path=csv_path, meta_path=meta_path, require_meta=True
            )
            self.assertTrue(result.ok, msg="; ".join(result.errors))

    def test_rejects_hardcoded_origin_drift(self):
        specs = build_vintage_specs(as_of_complete_month=140504)
        bad = list(specs)
        # Simulate independently hard-coded wrong origin for 1403Q1
        from pkg.benchmark.vintages import VintageSpec

        s0 = bad[8]  # 1403Q1
        self.assertEqual(s0.quarter, "1403Q1")
        bad[8] = VintageSpec(
            quarter=s0.quarter,
            forecast_origin=140304,  # historical modal, not canonical
            horizon=15,
            status=s0.status,
            notes=s0.notes,
        )
        result = validate_vintage_manifest(bad, require_meta=False)
        self.assertFalse(result.ok)
        self.assertTrue(any("origin_from_quarter" in e for e in result.errors))


@unittest.skipUnless(
    default_manifest_csv_path().exists() and default_manifest_meta_path().exists(),
    "tracked vintage manifest not built yet",
)
class TestTrackedVintageManifest(unittest.TestCase):
    def test_tracked_valid(self):
        specs = load_vintage_manifest(validate=True)
        self.assertEqual(len(specs), 18)
        result = validate_vintage_manifest(require_meta=True)
        self.assertTrue(result.ok, msg="; ".join(result.errors))
        for qrt, origin in result.mapping:
            self.assertEqual(origin, origin_from_quarter(qrt))


if __name__ == "__main__":
    unittest.main()
