"""Limit nested numerical threading for stable parallel backfills.

Job-level parallelism (``--workers``) must not be multiplied by OpenMP / BLAS
threads inside each forecasting library call.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

# Keep these at 1 when running multiple SKU-vintage workers. Do not auto-detect
# core count — callers pass an explicit ``--workers`` value.
DEFAULT_INNER_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def apply_inner_thread_limits(
    *,
    workers: int,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Set process env vars that cap nested library threads.

    When ``workers > 1``, force the default inner-thread caps (unless already
    set by the operator). When ``workers == 1``, leave existing env alone but
    still report the effective configuration.
    """
    applied: dict[str, str] = {}
    mapping = dict(DEFAULT_INNER_THREAD_ENV)
    if overrides:
        mapping.update({str(k): str(v) for k, v in overrides.items()})

    for key, value in mapping.items():
        if workers > 1:
            # Only set if unset, so operators can raise intentionally.
            if key not in os.environ or not str(os.environ.get(key, "")).strip():
                os.environ[key] = value
        applied[key] = str(os.environ.get(key, value))

    # Best-effort torch cap only. Prefer env vars over threadpoolctl — dual
    # OpenMP runtimes (Intel + LLVM) can warn/crash when threadpoolctl loads.
    if workers > 1:
        try:
            import torch  # type: ignore

            torch.set_num_threads(1)
            applied["torch_num_threads"] = "1"
        except Exception:
            applied["torch_num_threads"] = "unavailable"

    applied["requested_workers"] = str(int(workers))
    return applied
