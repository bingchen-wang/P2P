"""Deterministic offline backend for tests and local smoke runs."""

from __future__ import annotations

from typing import Any

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.registry import BACKENDS


class DummyBackend:
    """Return a configured response without making an external call."""

    def __init__(
        self,
        *,
        model_name: str = "dummy",
        answer_text: str = "DummyBackend called.",
        usage: TokenUsage | None = None,
        **_: Any,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")

        self.model_name = model_name
        self.answer_text = answer_text
        self.usage = usage if usage is not None else TokenUsage()

    def chat(self, *, system: str, user: str) -> BackendResponse:
        """Return the configured normalized response."""

        return BackendResponse(
            answer_text=self.answer_text,
            usage=self.usage,
        )


BACKENDS.register("dummy", DummyBackend)


__all__ = ["DummyBackend"]