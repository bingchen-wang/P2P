"""Contract tests shared by backend adapters."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

from p2p.backends import (
    DeepSeekBackend,
    DummyBackend,
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
    SimulatedBackend,
)
from p2p.backends.base import (
    Backend,
    BackendResponse,
    TokenUsage,
)
from p2p.registry import BACKENDS


def test_dummy_backend_returns_normalized_response() -> None:
    usage = TokenUsage(
        input_tokens=4,
        output_tokens=2,
        cached_input_tokens=1,
    )
    backend = DummyBackend(
        model_name="dummy-test",
        answer_text="dummy answer",
        usage=usage,
        temperature=0.0,
        max_tokens=16,
    )

    response = backend.chat(
        system="system",
        user="question",
    )

    assert isinstance(backend, Backend)
    assert response == BackendResponse(
        answer_text="dummy answer",
        usage=usage,
    )


def test_all_backends_use_the_central_registry() -> None:
    assert BACKENDS.names() == (
        "deepseek",
        "dummy",
        "gemini",
        "ollama",
        "openai",
        "simulated",
    )
    assert BACKENDS.get("simulated") is SimulatedBackend
    assert BACKENDS.get("deepseek") is DeepSeekBackend
    assert BACKENDS.get("dummy") is DummyBackend
    assert BACKENDS.get("gemini") is GeminiBackend
    assert BACKENDS.get("ollama") is OllamaBackend
    assert BACKENDS.get("openai") is OpenAIBackend


def test_adapters_do_not_accept_trackers_or_return_usage_switches() -> None:
    for backend_class in (
        DeepSeekBackend,
        DummyBackend,
        GeminiBackend,
        OllamaBackend,
        OpenAIBackend,
        SimulatedBackend,
    ):
        constructor_parameters = inspect.signature(
            backend_class
        ).parameters
        chat_parameters = inspect.signature(
            backend_class.chat
        ).parameters

        assert "token_tracker" not in constructor_parameters
        assert "return_usage" not in chat_parameters


def test_openai_backend_normalizes_sdk_response() -> None:
    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        def create(self, **kwargs: Any):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="  OpenAI answer  "
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=4,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=3
                    ),
                    completion_tokens_details=SimpleNamespace(
                        reasoning_tokens=2
                    ),
                ),
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions()
        )
    )
    backend = OpenAIBackend(
        model_name="gpt-test",
        temperature=0.2,
        max_tokens=25,
        client=client,
    )

    response = backend.chat(
        system="system",
        user="question",
    )

    assert response == BackendResponse(
        answer_text="OpenAI answer",
        usage=TokenUsage(
            input_tokens=11,
            output_tokens=4,
            cached_input_tokens=3,
            reasoning_tokens=2,
        ),
    )
    assert calls == [
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "system",
                    "content": "system",
                },
                {
                    "role": "user",
                    "content": "question",
                },
            ],
            "store": False,
            "temperature": 0.2,
            "max_completion_tokens": 25,
        }
    ]


def test_gemini_backend_normalizes_sdk_response() -> None:
    calls: list[dict[str, Any]] = []

    class FakeModels:
        def generate_content(self, **kwargs: Any):
            calls.append(kwargs)
            return SimpleNamespace(
                text="  Gemini answer  ",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=13,
                    candidates_token_count=5,
                    cached_content_token_count=2,
                    thoughts_token_count=3,
                ),
            )

    client = SimpleNamespace(models=FakeModels())
    backend = GeminiBackend(
        model_name="gemini-test",
        temperature=0.3,
        max_tokens=32,
        client=client,
        config_factory=lambda **kwargs: kwargs,
    )

    response = backend.chat(
        system="system",
        user="question",
    )

    assert response == BackendResponse(
        answer_text="Gemini answer",
        usage=TokenUsage(
            input_tokens=13,
            output_tokens=8,
            cached_input_tokens=2,
            reasoning_tokens=3,
        ),
    )
    assert calls == [
        {
            "model": "gemini-test",
            "contents": "question",
            "config": {
                "temperature": 0.3,
                "max_output_tokens": 32,
                "system_instruction": "system",
            },
        }
    ]


def test_ollama_backend_normalizes_native_response() -> None:
    calls: list[dict[str, Any]] = []

    def transport(
        *,
        url: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return {
            "message": {
                "role": "assistant",
                "content": "  Ollama answer  ",
            },
            "prompt_eval_count": 9,
            "eval_count": 3,
        }

    backend = OllamaBackend(
        model_name="gemma3:4b",
        base_url="http://localhost:11434/",
        temperature=0.0,
        max_tokens=24,
        timeout=5,
        think=False,
        keep_alive=0,
        transport=transport,
    )

    response = backend.chat(
        system="system",
        user="question",
    )

    assert response == BackendResponse(
        answer_text="Ollama answer",
        usage=TokenUsage(
            input_tokens=9,
            output_tokens=3,
            cached_input_tokens=0,
        ),
    )
    assert calls == [
        {
            "url": "http://localhost:11434/api/chat",
            "payload": {
                "model": "gemma3:4b",
                "messages": [
                    {
                        "role": "system",
                        "content": "system",
                    },
                    {
                        "role": "user",
                        "content": "question",
                    },
                ],
                "stream": False,
                "think": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 24,
                },
            },
            "timeout": 5.0,
        }
    ]
