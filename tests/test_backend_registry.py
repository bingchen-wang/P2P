"""Tests for the central backend registry."""

from __future__ import annotations

import pytest

from p2p.registry import BACKENDS, Registry


class ExampleBackend:
    def __init__(self, *, value: int) -> None:
        self.value = value


def test_registry_registers_resolves_and_creates() -> None:
    registry = Registry("test backends")
    registry.register("example", ExampleBackend)

    assert registry.get("example") is ExampleBackend
    assert registry.names() == ("example",)

    instance = registry.create("example", value=7)
    assert isinstance(instance, ExampleBackend)
    assert instance.value == 7


def test_registry_rejects_duplicate_names() -> None:
    registry = Registry("test backends")
    registry.register("example", ExampleBackend)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", ExampleBackend)


def test_registry_reports_unknown_names() -> None:
    registry = Registry("test backends")
    registry.register("example", ExampleBackend)

    with pytest.raises(KeyError, match="unknown"):
        registry.get("unknown")


def test_backends_is_the_central_registry() -> None:
    assert isinstance(BACKENDS, Registry)
    assert BACKENDS.registry_name == "backends"