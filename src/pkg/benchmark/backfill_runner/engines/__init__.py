"""Forecast engine registry for historical backfill."""
from __future__ import annotations

from typing import Callable, Dict

from pkg.benchmark.backfill_runner.types import ForecastEngine

_REGISTRY: Dict[str, Callable[[], ForecastEngine]] = {}


def register_engine(name: str, factory: Callable[[], ForecastEngine]) -> None:
    key = str(name).strip().lower()
    _REGISTRY[key] = factory


def get_engine(name: str) -> ForecastEngine:
    key = str(name).strip().lower()
    if key == "v3":
        raise NotImplementedError("V3 forecasting engine is not implemented yet")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown engine {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]()


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def _register_builtins() -> None:
    from pkg.benchmark.backfill_runner.engines.dummy import DummyForecastEngine
    from pkg.benchmark.backfill_runner.engines.v1_adapter import V1ForecastEngine
    from pkg.benchmark.backfill_runner.engines.v2_engine import V2ForecastEngine

    register_engine("dummy", DummyForecastEngine)
    register_engine("v1", V1ForecastEngine)
    register_engine("v2", V2ForecastEngine)


_register_builtins()
