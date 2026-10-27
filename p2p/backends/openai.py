"""OpenAI backend adapter: Chat Completions or Responses, chosen explicitly."""

from __future__ import annotations

from typing import Any

from p2p.backends.base import (
    BackendResponse,
    NoAnswerTextError,
    TokenUsage,
)
from p2p.registry import BACKENDS


_TRANSPORTS = frozenset({"chat_completions", "responses"})


def _count(value: Any, name: str) -> int:
    return getattr(value, name, 0) or 0


class OpenAIBackend:
    """Normalize OpenAI text responses.

    ``transport`` names the API: ``chat_completions`` (the default) or
    ``responses``.  It is never switched silently.  P2P's provider-neutral
    ``max_tokens`` is sent as ``max_completion_tokens`` (Chat Completions) or
    ``max_output_tokens`` (Responses); both bound visible and reasoning
    tokens.  ``store`` is sent on every request because both APIs may store
    responses by default; P2P never retrieves them, so it defaults to false.
    """

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float | None = 0.7,
        max_tokens: int | None = 1024,
        transport: str = "chat_completions",
        store: bool = False,
        client=None,
        api_key: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a nonempty string"
            )
        if transport not in _TRANSPORTS:
            raise ValueError(
                "transport must be 'chat_completions' or 'responses'"
            )
        if not isinstance(store, bool):
            raise ValueError("store must be true or false")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(
                "max_tokens must be a positive integer or null"
            )

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "OpenAI support requires the 'openai' extra"
                ) from error

            client_kwargs: dict[str, Any] = {}
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            client = OpenAI(**client_kwargs)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport
        self.store = store
        self.client = client

    def chat(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        """Return normalized OpenAI text and usage."""

        if self.transport == "responses":
            return self._responses(system=system, user=user)
        return self._chat_completions(system=system, user=user)

    def _chat_completions(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            "store": self.store,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_completion_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(
            **request
        )

        choice = response.choices[0]
        message = choice.message
        answer_text = message.content
        if not answer_text:
            refusal = getattr(message, "refusal", None)
            finish_reason = getattr(choice, "finish_reason", None)
            if refusal:
                raise NoAnswerTextError(
                    f"OpenAI refused: {refusal}"
                )
            if answer_text is None or finish_reason not in (None, "stop"):
                raise NoAnswerTextError(
                    "OpenAI returned no text "
                    f"(finish_reason={finish_reason!r})"
                )
        if not isinstance(answer_text, str):
            raise TypeError(
                "OpenAI response content must be a string"
            )

        usage = getattr(response, "usage", None)
        return BackendResponse(
            answer_text=answer_text.strip(),
            usage=TokenUsage(
                input_tokens=_count(usage, "prompt_tokens"),
                output_tokens=_count(usage, "completion_tokens"),
                cached_input_tokens=_count(
                    getattr(usage, "prompt_tokens_details", None),
                    "cached_tokens",
                ),
                reasoning_tokens=_count(
                    getattr(usage, "completion_tokens_details", None),
                    "reasoning_tokens",
                ),
            ),
        )

    def _responses(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        request: dict[str, Any] = {
            "model": self.model_name,
            "input": user,
            "store": self.store,
        }
        if system:
            request["instructions"] = system
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_output_tokens"] = self.max_tokens

        response = self.client.responses.create(**request)

        answer_text = getattr(response, "output_text", None)
        if not answer_text:
            refusals = [
                content.refusal
                for item in getattr(response, "output", None) or []
                if getattr(item, "type", None) == "message"
                for content in getattr(item, "content", None) or []
                if getattr(content, "type", None) == "refusal"
            ]
            status = getattr(response, "status", None)
            if refusals:
                raise NoAnswerTextError(
                    "OpenAI refused: " + " ".join(refusals)
                )
            if status == "failed":
                error = getattr(response, "error", None)
                raise NoAnswerTextError(
                    "OpenAI response failed: "
                    f"{getattr(error, 'message', error)}"
                )
            if answer_text is None or status not in (None, "completed"):
                details = getattr(response, "incomplete_details", None)
                raise NoAnswerTextError(
                    f"OpenAI returned no text (status={status!r}, "
                    f"reason={getattr(details, 'reason', None)!r})"
                )
        if not isinstance(answer_text, str):
            raise TypeError(
                "OpenAI response output_text must be a string"
            )

        usage = getattr(response, "usage", None)
        return BackendResponse(
            answer_text=answer_text.strip(),
            usage=TokenUsage(
                input_tokens=_count(usage, "input_tokens"),
                output_tokens=_count(usage, "output_tokens"),
                cached_input_tokens=_count(
                    getattr(usage, "input_tokens_details", None),
                    "cached_tokens",
                ),
                reasoning_tokens=_count(
                    getattr(usage, "output_tokens_details", None),
                    "reasoning_tokens",
                ),
            ),
        )


BACKENDS.register("openai", OpenAIBackend)


__all__ = ["OpenAIBackend"]
