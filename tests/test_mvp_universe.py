"""Tests for immutable MVP product universe manifest."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.benchmark.universes import (
    MvpUniverseError,
    assert_mvp_universe_immutable,
    derive_mvp_frame,
    load_mvp_product_names,
    load_mvp_universe,
    mvp_products_csv_path,
    mvp_products_meta_path,
    products_list_sha256,
    validate_mvp_universe,
    write_mvp_universe,
)


class TestDeriveMvpFrame(unittest.TestCase):
    def test_derives_unique_sorted_products_without_inventing_ids(self):
        matched = pd.DataFrame(
            [
                {"product": "Beta", "generic": "G1", "Field": "F", "ProductForm": "P", "Provider": "Pr"},
                {"product": "Alpha", "generic": "G2", "Field": "F", "ProductForm": "P", "Provider": "Pr"},
                {"product": "Beta", "generic": "G1", "Field": "F", "ProductForm": "P", "Provider": "Pr"},
            ]
        )
        frame = derive_mvp_frame(matched)
        self.assertEqual(list(frame["product"]), ["Alpha", "Beta"])
        self.assertTrue(frame["product_id"].isna().all())
        self.assertEqual(list(frame["product_title"]), ["Alpha", "Beta"])

    def test_empty_matched_raises(self):
        with self.assertRaises(MvpUniverseError):
            derive_mvp_frame(pd.DataFrame(columns=["product"]))


class TestValidateMvpUniverse(unittest.TestCase):
    def _write_tmp(self, products: list[str]) -> tuple[Path, Path]:
        tmp = Path(tempfile.mkdtemp())
        rows = []
        for p in products:
            rows.append(
                {
                    "product": p,
                    "product_title": p,
                    "product_id": "",
                    "generic": "G",
                    "field": "F",
                    "product_form": "Form",
                    "provider": "P",
                }
            )
        frame = pd.DataFrame(rows)
        meta = {
            "universe_name": "mvp_products",
            "universe_version": "1",
            "n_products": len(frame),
            "products_sha256": products_list_sha256(products),
            "source": {"panel": "matched_universe.parquet"},
            "description": "test",
        }
        csv_path, meta_path = write_mvp_universe(frame, meta, out_dir=tmp, force=True)
        return csv_path, meta_path

    def test_rejects_duplicates(self):
        csv_path, meta_path = self._write_tmp(["A", "B"])
        # Corrupt CSV with duplicate
        text = csv_path.read_text(encoding="utf-8")
        csv_path.write_text(text + text.splitlines()[-1] + "\n", encoding="utf-8")
        result = validate_mvp_universe(csv_path=csv_path, meta_path=meta_path)
        self.assertFalse(result.ok)
        self.assertTrue(any("duplicate" in e for e in result.errors))

    def test_rejects_empty(self):
        tmp = Path(tempfile.mkdtemp())
        frame = pd.DataFrame(
            columns=[
                "product",
                "product_title",
                "product_id",
                "generic",
                "field",
                "product_form",
                "provider",
            ]
        )
        meta = {
            "n_products": 0,
            "products_sha256": products_list_sha256([]),
            "source": {"panel": "matched_universe.parquet"},
        }
        csv_path, meta_path = write_mvp_universe(frame, meta, out_dir=tmp, force=True)
        result = validate_mvp_universe(csv_path=csv_path, meta_path=meta_path)
        self.assertFalse(result.ok)
        self.assertTrue(any("empty" in e for e in result.errors))

    def test_hash_mismatch_when_csv_tampered(self):
        csv_path, meta_path = self._write_tmp(["Alpha", "Beta"])
        # Tamper product list without updating meta
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        frame.loc[0, "product"] = "Changed"
        frame.loc[0, "product_title"] = "Changed"
        csv_path.write_text(frame.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
        result = validate_mvp_universe(csv_path=csv_path, meta_path=meta_path)
        self.assertFalse(result.ok)
        self.assertTrue(any("sha256" in e for e in result.errors))

    def test_refuses_silent_dim_product_rebuild_via_wrong_source_panel(self):
        csv_path, meta_path = self._write_tmp(["Alpha"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["source"]["panel"] = "live_dim_product_basket"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        result = validate_mvp_universe(csv_path=csv_path, meta_path=meta_path)
        self.assertFalse(result.ok)
        self.assertTrue(any("matched_universe.parquet" in e for e in result.errors))


@unittest.skipUnless(
    mvp_products_csv_path().exists() and mvp_products_meta_path().exists(),
    "tracked MVP universe not built yet",
)
class TestTrackedMvpUniverse(unittest.TestCase):
    def test_tracked_manifest_valid(self):
        result = assert_mvp_universe_immutable()
        self.assertTrue(result.ok)
        self.assertEqual(result.n_products, 55)

    def test_loader_returns_55_sorted_products(self):
        names = load_mvp_product_names()
        self.assertEqual(len(names), 55)
        self.assertEqual(names, sorted(names, key=str.casefold))
        self.assertEqual(len(set(names)), 55)

    def test_no_invented_product_ids(self):
        frame = load_mvp_universe()
        ids = frame["product_id"].astype(str).str.strip()
        self.assertTrue((ids == "").all())

    def test_matches_matched_universe_when_freeze_present(self):
        from pkg.benchmark.config import default_benchmark_root

        root = default_benchmark_root()
        matched = root / "matched_universe.parquet"
        if not matched.exists():
            self.skipTest("freeze not present")
        result = validate_mvp_universe(require_meta=True, check_freeze_checksum=True)
        self.assertTrue(result.ok, msg="; ".join(result.errors))


if __name__ == "__main__":
    unittest.main()
