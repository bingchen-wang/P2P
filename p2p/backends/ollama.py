"""Native local Ollama backend adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any
from urllib.request import Request, urlopen

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.registry import BACKENDS


Transport = Callable[..., Mapping[str, Any]]


class OllamaBackend:
    """Call Ollama's native non-streaming chat endpoint."""

    def __init__(
        self,
        *,
        model_name: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float | None = 0.0,
        max_tokens: int | None = 128,
        timeout: float = 60.0,
        think: bool | str | None = False,
        keep_alive: str | int | None = None,
        options: Mapping[str, Any] | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a nonempty string"
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "base_url must be a nonempty string"
            )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError(
                "timeout must be a positive number"
            )

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = float(timeout)
        self.think = think
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self.transport = (
            transport or self._post_json
        )

    @staticmethod
    def _post_json(
        *,
        url: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Mapping[str, Any]:
        encoded_payload = json.dumps(payload).encode(
            "utf-8"
        )
        request = Request(
            url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            decoded = json.loads(
                response.read().decode("utf-8")
            )

        if not isinstance(decoded, Mapping):
            raise TypeError(
                "Ollama response must be a JSON object"
            )

        return decoded

    def chat(
        self,
        *,
        system: str,
        user: str,
    ) -> BackendResponse:
        """Return normalized Ollama text and usage."""

        messages = []
        if system:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": user,
            }
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        runtime_options = dict(self.options)
        if self.temperature is not None:
            runtime_options["temperature"] = (
                self.temperature
            )
        if self.max_tokens is not None:
            runtime_options["num_predict"] = (
                self.max_tokens
            )
        if runtime_options:
            payload["options"] = runtime_options

        response = self.transport(
            url=f"{self.base_url}/api/chat",
            payload=payload,
            timeout=self.timeout,
        )
        if not isinstance(response, Mapping):
            raise TypeError(
                "Ollama transport must return a mapping"
            )

        message = response.get("message")
        if not isinstance(message, Mapping):
            raise TypeError(
                "Ollama response must contain message"
            )

        answer_text = message.get("content")
        if not isinstance(answer_text, str):
            raise TypeError(
                "Ollama message content must be a string"
            )

        input_tokens = (
            response.get("prompt_eval_count", 0)
            or 0
        )
        output_tokens = (
            response.get("eval_count", 0)
            or 0
        )

        return BackendResponse(
            answer_text=answer_text.strip(),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
            ),
        )


BACKENDS.register("ollama", OllamaBackend)


__all__ = ["OllamaBackend"]