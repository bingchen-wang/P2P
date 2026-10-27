"""Provider-neutral backend contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token counts returned by every backend."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")

        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                "cached_input_tokens cannot exceed input_tokens"
            )
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                "reasoning_tokens cannot exceed output_tokens"
            )


@dataclass(frozen=True, slots=True)
class BackendResponse:
    """Normalized response returned by every backend adapter."""

    answer_text: str
    usage: TokenUsage = field(default_factory=TokenUsage)

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str):
            raise TypeError("answer_text must be a string")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")


class NoAnswerTextError(RuntimeError):
    """The provider returned no answer text: a refusal, block, or filter.

    The message names the provider's stated reason so that it survives into
    the recorded ``[ERROR]`` answer when every retry fails.
    """


@runtime_checkable
class Backend(Protocol):
    """Minimal interface implemented by backend adapters."""

    model_name: str

    def chat(self, *, system: str, user: str) -> BackendResponse:
        """Return normalized text and token usage."""
        ...


__all__ = [
    "Backend",
    "BackendResponse",
    "NoAnswerTextError",
    "TokenUsage",
]
