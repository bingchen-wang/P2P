"""DeepSeek V4 Chat Completions backend adapter."""

from __future__ import annotations

import os
from typing import Any

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.registry import BACKENDS


_DEFAULT_BASE_URL = "https://api.deepseek.com"
_THINKING_MODES = frozenset({"enabled", "disabled"})
_REASONING_EFFORTS = frozenset({"high", "max"})


def _usage_count(value: Any, name: str) -> int:
    """Return one nonnegative integer from SDK usage metadata."""

    result = getattr(value, name, 0) or 0
    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError(f"DeepSeek usage.{name} must be an integer")
    if result < 0:
        raise ValueError(
            f"DeepSeek usage.{name} must be nonnegative"
        )
    return result


class DeepSeekBackend:
    """Normalize DeepSeek's OpenAI-compatible chat API.

    Thinking is disabled by default because P2P's survey and persona
    workflows generally require short, directly parseable responses. Set
    ``thinking="enabled"`` explicitly for reasoning workloads.
    """

    def __init__(
        self,
        *,
        model_name: str = "deepseek-v4-flash",
        temperature: float | None = 0.7,
        max_tokens: int | None = 1024,
        thinking: str = "disabled",
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        client=None,
        api_key: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a nonempty string"
            )
        if thinking not in _THINKING_MODES:
            raise ValueError(
                "thinking must be 'enabled' or 'disabled'"
            )
        if (
            reasoning_effort is not None
            and reasoning_effort not in _REASONING_EFFORTS
        ):
            raise ValueError(
                "reasoning_effort must be 'high', 'max', or null"
            )
        if thinking == "disabled" and reasoning_effort is not None:
            raise ValueError(
                "reasoning_effort requires thinking='enabled'"
            )
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(
                "max_tokens must be a positive integer or null"
            )

        resolved_base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or _DEFAULT_BASE_URL
        )
        if (
            not isinstance(resolved_base_url, str)
            or not resolved_base_url.strip()
        ):
            raise ValueError("base_url must be a nonempty string")

        if client is None:
            resolved_api_key = (
                api_key or os.getenv("DEEPSEEK_API_KEY")
            )
            if not resolved_api_key:
                raise ValueError(
                    "DeepSeek requires api_key or DEEPSEEK_API_KEY"
                )
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "DeepSeek support requires the 'deepseek' "
                    "or 'openai' extra"
                ) from error

            client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url.rstrip("/"),
            )

        self.model_name = model_name.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.base_url = resolved_base_url.rstrip("/")
        self.client = client

    def chat(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        """Return normalized DeepSeek text and billable token usage."""

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
            "extra_body": {
                "thinking": {
                    "type": self.thinking,
                }
            },
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if self.thinking == "enabled":
            if self.reasoning_effort is not None:
                request["reasoning_effort"] = self.reasoning_effort
        elif self.temperature is not None:
            request["temperature"] = self.temperature

        response = self.client.chat.completions.create(
            **request
        )
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError(
                "DeepSeek response contained no completion choices"
            )

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        answer_text = getattr(message, "content", None)
        if not isinstance(answer_text, str) or not answer_text.strip():
            if finish_reason == "length":
                raise RuntimeError(
                    "DeepSeek exhausted max_tokens before returning "
                    "a final answer"
                )
            raise TypeError(
                "DeepSeek response content must be a nonempty string"
            )

        usage = getattr(response, "usage", None)
        input_tokens = _usage_count(usage, "prompt_tokens")
        output_tokens = _usage_count(usage, "completion_tokens")
        cached_input_tokens = _usage_count(
            usage,
            "prompt_cache_hit_tokens",
        )
        completion_details = getattr(
            usage,
            "completion_tokens_details",
            None,
        )
        reasoning_tokens = _usage_count(
            completion_details,
            "reasoning_tokens",
        )

        if cached_input_tokens > input_tokens:
            raise ValueError(
                "DeepSeek cache-hit tokens exceed prompt tokens"
            )
        if reasoning_tokens > output_tokens:
            raise ValueError(
                "DeepSeek reasoning tokens exceed completion tokens"
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


BACKENDS.register("deepseek", DeepSeekBackend)


__all__ = ["DeepSeekBackend"]
