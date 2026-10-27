"""Configuration for backend-neutral survey agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Settings used by an injected agent/conductor factory."""

    agent_type: str
    model_name: str
    formality: str = ""
    agent_kwargs: Mapping[str, Any] = field(default_factory=dict)
    delay: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("agent_type", "model_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a nonempty string"
                )

        if not isinstance(self.formality, str):
            raise TypeError("formality must be a string")
        if not isinstance(self.agent_kwargs, Mapping):
            raise TypeError("agent_kwargs must be a mapping")
        if (
            isinstance(self.delay, bool)
            or not isinstance(self.delay, (int, float))
        ):
            raise TypeError("delay must be a number")
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError("delay must be finite and nonnegative")

        object.__setattr__(
            self,
            "agent_kwargs",
            MappingProxyType(dict(self.agent_kwargs)),
        )


__all__ = ["AgentConfig"]