"""Backend-neutral survey agent."""

from __future__ import annotations

from dataclasses import dataclass

from p2p.backends.base import BackendResponse, TokenUsage


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Normalized response returned to a survey conductor."""

    eid: str
    answer_text: str
    usage: TokenUsage
    model_name: str


class Agent:
    """Combine persona instructions with an injected backend."""

    def __init__(
        self,
        *,
        eid: str,
        backend,
        endowment: str = "",
        formality: str = "",
    ) -> None:
        for field_name, value in (
            ("eid", eid),
            ("endowment", endowment),
            ("formality", formality),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")

        if not eid.strip():
            raise ValueError("eid must be a nonempty string")
        if not hasattr(backend, "model_name"):
            raise TypeError("backend must expose model_name")
        if not callable(getattr(backend, "chat", None)):
            raise TypeError("backend must expose chat(...)")

        self.eid = eid
        self.backend = backend
        self.endowment = endowment
        self.formality = formality

    def get_system_prompt(self) -> str:
        """Combine nonempty persona and formatting instructions."""

        return "\n\n".join(
            part.strip()
            for part in (self.endowment, self.formality)
            if part.strip()
        )

    def answer(self, question: str) -> AgentResponse:
        """Return one normalized response without accounting for it."""

        if not isinstance(question, str):
            raise TypeError("question must be a string")

        response = self.backend.chat(
            system=self.get_system_prompt(),
            user=question,
        )
        if not isinstance(response, BackendResponse):
            raise TypeError(
                "backend.chat(...) must return BackendResponse"
            )

        return AgentResponse(
            eid=self.eid,
            answer_text=response.answer_text,
            usage=response.usage,
            model_name=self.backend.model_name,
        )


__all__ = ["Agent", "AgentResponse"]