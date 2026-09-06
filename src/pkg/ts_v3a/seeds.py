"""Deterministic seed helpers for V3A neural experiments.

Sets Python ``random``, NumPy, and TensorFlow/Keras seeds where practical.
GPU nondeterminism may still remain depending on TF/CUDA settings.
"""
from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seeds(seed: int, *, set_tensorflow: bool = True) -> dict:
    """Seed Python, NumPy, and optionally TensorFlow/Keras.

    Returns a dict describing which backends were seeded.
    """
    seed = int(seed)
    status: dict = {"seed": seed, "python": True, "numpy": False, "tensorflow": False}

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        pass

    if set_tensorflow:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
            # Best-effort Keras path (TF 2.x).
            try:
                from tensorflow import keras

                if hasattr(keras.utils, "set_random_seed"):
                    keras.utils.set_random_seed(seed)
            except Exception:
                pass
            status["tensorflow"] = True
        except ImportError:
            status["tensorflow"] = False

    return status


def draw_seed_sequence(seeds: tuple[int, ...], index: int) -> int:
    """Pick the ``index``-th seed from an experiment seed tuple (mod length)."""
    if not seeds:
        raise ValueError("seeds tuple is empty")
    return int(seeds[int(index) % len(seeds)])
