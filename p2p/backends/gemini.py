"""Google Gemini backend adapter."""

from __future__ import annotations

import os
from typing import Any, Callable

from p2p.backends.base import (
    BackendResponse,
    NoAnswerTextError,
    TokenUsage,
)
from p2p.registry import BACKENDS


def _reason(value: Any) -> Any:
    """Return an SDK enum's plain value, such as ``'SAFETY'``."""

    return getattr(value, "value", value)


class GeminiBackend:
    """Normalize Gemini GenerateContent responses.

    The model ID has no default: Gemini models are retired on a published
    schedule, so every run names the model it uses.
    """

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float | None = 0.7,
        max_tokens: int | None = 1024,
        system_instruction: str | None = None,
        client=None,
        config_factory: Callable[..., Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a nonempty string"
            )

        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise ImportError(
                    "Gemini support requires the 'gemini' extra"
                ) from error

            resolved_api_key = (
                api_key or os.getenv("GEMINI_API_KEY")
            )
            if not resolved_api_key:
                raise ValueError(
                    "Gemini requires api_key or GEMINI_API_KEY"
                )

            client = genai.Client(
                api_key=resolved_api_key
            )
            if config_factory is None:
                config_factory = (
                    types.GenerateContentConfig
                )

        if config_factory is None:
            config_factory = lambda **kwargs: kwargs

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_instruction = system_instruction
        self.client = client
        self.config_factory = config_factory

    def chat(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        """Return normalized Gemini text and usage."""

        config_values: dict[str, Any] = {}
        if self.temperature is not None:
            config_values["temperature"] = (
                self.temperature
            )
        if self.max_tokens is not None:
            config_values["max_output_tokens"] = (
                self.max_tokens
            )

        instruction = (
            self.system_instruction
            if self.system_instruction is not None
            else system
        )
        if instruction:
            config_values["system_instruction"] = (
                instruction
            )

        request: dict[str, Any] = {
            "model": self.model_name,
            "contents": user,
        }
        if config_values:
            request["config"] = self.config_factory(
                **config_values
            )

        response = self.client.models.generate_content(
            **request
        )

        answer_text = getattr(response, "text", None)
        if not answer_text:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason is not None:
                message = getattr(
                    feedback,
                    "block_reason_message",
                    None,
                )
                raise NoAnswerTextError(
                    "Gemini blocked the prompt "
                    f"(block_reason={_reason(block_reason)!r})"
                    + (f": {message}" if message else "")
                )
            candidates = getattr(response, "candidates", None) or []
            finish_reason = _reason(
                getattr(candidates[0], "finish_reason", None)
                if candidates
                else None
            )
            if answer_text is None or finish_reason not in (None, "STOP"):
                raise NoAnswerTextError(
                    "Gemini returned no text "
                    f"(finish_reason={finish_reason!r})"
                )
        if not isinstance(answer_text, str):
            raise TypeError(
                "Gemini response text must be a string"
            )

        usage_metadata = getattr(
            response,
            "usage_metadata",
            None,
        )
        input_tokens = (
            getattr(
                usage_metadata,
                "prompt_token_count",
                0,
            )
            or 0
        )
        candidate_tokens = (
            getattr(
                usage_metadata,
                "candidates_token_count",
                0,
            )
            or 0
        )
        reasoning_tokens = (
            getattr(
                usage_metadata,
                "thoughts_token_count",
                0,
            )
            or 0
        )
        output_tokens = candidate_tokens + reasoning_tokens
        cached_input_tokens = (
            getattr(
                usage_metadata,
                "cached_content_token_count",
                0,
            )
            or 0
        )

        return BackendResponse(
            answer_text=answer_text.strip(),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )


BACKENDS.register("gemini", GeminiBackend)


__all__ = ["GeminiBackend"]
