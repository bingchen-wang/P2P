"""Central registries for swappable P2P components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar


FactoryT = TypeVar("FactoryT", bound=Callable[..., Any])


class Registry(Generic[FactoryT]):
    """Named collection of factories with duplicate protection."""

    def __init__(self, registry_name: str) -> None:
        if not isinstance(registry_name, str) or not registry_name.strip():
            raise ValueError("registry_name must be a nonempty string")

        self.registry_name = registry_name
        self._entries: dict[str, FactoryT] = {}

    def register(self, name: str, factory: FactoryT) -> None:
        """Register one factory under an exact, nonempty name."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("registry entry name must be a nonempty string")
        if not callable(factory):
            raise TypeError("registry factory must be callable")
        if name in self._entries:
            raise ValueError(
                f"{name!r} is already registered in {self.registry_name}"
            )

        self._entries[name] = factory

    def get(self, name: str) -> FactoryT:
        """Return a registered factory."""

        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"unknown {self.registry_name} entry {name!r}; "
                f"available: {available}"
            ) from None

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered factory using keyword arguments."""

        return self.get(name)(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(sorted(self._entries))


BACKENDS: Registry[Callable[..., Any]] = Registry("backends")
MODELS: Registry[Callable[..., Any]] = Registry("models")
STAGES: Registry[Callable[..., Any]] = Registry("stages")
COMMANDS: Registry[Callable[..., Any]] = Registry("commands")
METRICS: Registry[Callable[..., Any]] = Registry("metrics")


__all__ = [
    "BACKENDS",
    "COMMANDS",
    "METRICS",
    "MODELS",
    "STAGES",
    "Registry",
]