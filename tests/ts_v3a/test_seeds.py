"""Deterministic seed initialization tests."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pkg.ts_v3a.seeds import set_global_seeds


class TestSeeds(unittest.TestCase):
    def test_python_and_numpy_reproducible(self):
        status = set_global_seeds(42)
        self.assertTrue(status["python"])
        self.assertTrue(status["numpy"])
        a = [random.random() for _ in range(5)]
        import numpy as np

        b = np.random.rand(5).tolist()

        set_global_seeds(42)
        a2 = [random.random() for _ in range(5)]
        b2 = np.random.rand(5).tolist()
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)

    def test_tensorflow_seeded_when_available(self):
        status = set_global_seeds(41, set_tensorflow=True)
        try:
            import tensorflow as tf  # noqa: F401

            self.assertTrue(status["tensorflow"])
            set_global_seeds(41)
            x1 = tf.random.uniform((3,), seed=41).numpy()
            set_global_seeds(41)
            x2 = tf.random.uniform((3,), seed=41).numpy()
            self.assertTrue((x1 == x2).all())
        except ImportError:
            self.assertFalse(status["tensorflow"])


if __name__ == "__main__":
    unittest.main()
