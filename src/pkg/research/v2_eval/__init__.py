"""TS V2 historical backfill evaluation against the legacy TS pipeline.

Read-only analysis of an already-completed backfill experiment. Nothing in this
package refits models, launches a backfill, or writes into the frozen benchmark
or an existing research family directory.

Entry point::

    python -m pkg.research.evaluate_v2_backfill
"""
from __future__ import annotations

from pkg.research.v2_eval.config import V2EvalConfig, default_config

__all__ = ["V2EvalConfig", "default_config"]
