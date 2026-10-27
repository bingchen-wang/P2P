"""Factories connecting agents to the single backend registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import p2p.backends  # Registers built-in backend adapters.
from p2p.agents.base import Agent
from p2p.agents.config import AgentConfig
from p2p.registry import BACKENDS


def create_backend(
    *,
    backend_type: str,
    model_name: str,
    **backend_kwargs: Any,
):
    """Create a backend through ``p2p.registry.BACKENDS``."""

    if "model_name" in backend_kwargs:
        raise ValueError(
            "model_name must be supplied through its explicit argument"
        )

    return BACKENDS.create(
        backend_type,
        model_name=model_name,
        **backend_kwargs,
    )


def create_agent(
    *,
    config: AgentConfig,
    eid: str,
    endowment: str = "",
    backend=None,
    backend_factory: Callable[..., Any] | None = None,
) -> Agent:
    """Create one generic agent using an injected or registered backend."""

    if backend is not None and backend_factory is not None:
        raise ValueError(
            "provide either backend or backend_factory, not both"
        )

    if backend is None:
        resolver = backend_factory or create_backend
        backend = resolver(
            backend_type=config.agent_type,
            model_name=config.model_name,
            **dict(config.agent_kwargs),
        )

    return Agent(
        eid=eid,
        endowment=endowment,
        formality=config.formality,
        backend=backend,
    )


__all__ = ["create_agent", "create_backend"]