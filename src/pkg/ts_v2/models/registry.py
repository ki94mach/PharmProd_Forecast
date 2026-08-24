"""Central registry of V2 forecasting model factories."""
from __future__ import annotations

from typing import Callable

from pkg.ts_v2.models.base import ForecastModel

ModelFactory = Callable[[], ForecastModel]


class ModelRegistry:
    """Name → factory. Production candidates are configured via ``candidate_models``."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory, *, replace: bool = False) -> None:
        key = str(name)
        if not replace and key in self._factories:
            raise ValueError(f"model {key!r} is already registered")
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(str(name), None)

    def create(self, name: str) -> ForecastModel:
        key = str(name)
        if key not in self._factories:
            raise KeyError(
                f"unknown model {key!r}; registered={self.names()!r}"
            )
        model = self._factories[key]()
        return model

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories.keys())

    def create_all(self, names: tuple[str, ...]) -> list[ForecastModel]:
        return [self.create(n) for n in names]


REGISTRY = ModelRegistry()


def register_model(name: str, factory: ModelFactory, *, replace: bool = False) -> None:
    REGISTRY.register(name, factory, replace=replace)


def available_models() -> tuple[str, ...]:
    """Registered production/test model names (empty until models are added)."""
    return REGISTRY.names()


def get_model(name: str) -> ForecastModel:
    return REGISTRY.create(name)


def models_from_config(candidate_names: tuple[str, ...]) -> list[ForecastModel]:
    return REGISTRY.create_all(candidate_names)
