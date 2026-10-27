"""Provider-specific contracts for the DeepSeek backend."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from p2p.backends.base import BackendResponse, TokenUsage
from p2p.backends.deepseek import DeepSeekBackend
from p2p.agents.factory import create_backend
from p2p.config import Config
from p2p.stages._backend import load_backend_config


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return self.response


def _client(response: Any) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def _response(
    *,
    content: str | None = "  selected option  ",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 24,
    cache_hit_tokens: int = 40,
    reasoning_tokens: int = 18,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cache_hit_tokens=cache_hit_tokens,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
    )


def test_non_thinking_request_and_usage_normalization() -> None:
    client, completions = _client(
        _response(reasoning_tokens=0)
    )
    backend = DeepSeekBackend(
        model_name="deepseek-v4-flash",
        thinking="disabled",
        temperature=0.2,
        max_tokens=128,
        client=client,
    )

    result = backend.chat(system="system", user="question")

    assert result == BackendResponse(
        answer_text="selected option",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=24,
            cached_input_tokens=40,
            reasoning_tokens=0,
        ),
    )
    assert completions.calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "extra_body": {
                "thinking": {"type": "disabled"}
            },
            "max_tokens": 128,
            "temperature": 0.2,
        }
    ]


def test_thinking_request_omits_sampling_and_tracks_reasoning() -> None:
    client, completions = _client(_response())
    backend = DeepSeekBackend(
        model_name="deepseek-v4-pro",
        thinking="enabled",
        reasoning_effort="max",
        temperature=0.9,
        max_tokens=512,
        client=client,
    )

    result = backend.chat(system="", user="question")

    assert result.usage == TokenUsage(
        input_tokens=100,
        output_tokens=24,
        cached_input_tokens=40,
        reasoning_tokens=18,
    )
    assert completions.calls[0] == {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "question"},
        ],
        "extra_body": {
            "thinking": {"type": "enabled"}
        },
        "max_tokens": 512,
        "reasoning_effort": "max",
    }
    assert "temperature" not in completions.calls[0]


def test_constructor_uses_deepseek_credentials_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[dict[str, Any]] = []
    fake_client, _ = _client(_response())

    def openai_factory(**kwargs: Any):
        constructor_calls.append(kwargs)
        return fake_client

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=openai_factory),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-key")
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL",
        "https://example.deepseek.invalid/",
    )

    backend = DeepSeekBackend()

    assert constructor_calls == [
        {
            "api_key": "secret-test-key",
            "base_url": "https://example.deepseek.invalid",
        }
    ]
    assert backend.base_url == "https://example.deepseek.invalid"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"thinking": "automatic"}, "thinking"),
        (
            {
                "thinking": "disabled",
                "reasoning_effort": "high",
            },
            "requires thinking",
        ),
        (
            {
                "thinking": "enabled",
                "reasoning_effort": "low",
            },
            "reasoning_effort",
        ),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_constructor_rejects_ambiguous_settings(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DeepSeekBackend(
            client=SimpleNamespace(),
            **kwargs,
        )


def test_constructor_requires_provider_key_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekBackend()


def test_length_exhaustion_has_actionable_error() -> None:
    client, _ = _client(
        _response(content="", finish_reason="length")
    )
    backend = DeepSeekBackend(client=client)

    with pytest.raises(RuntimeError, match="exhausted max_tokens"):
        backend.chat(system="system", user="question")


def test_malformed_usage_is_rejected() -> None:
    client, _ = _client(
        _response(
            prompt_tokens=10,
            cache_hit_tokens=11,
            completion_tokens=5,
            reasoning_tokens=0,
        )
    )
    backend = DeepSeekBackend(client=client)

    with pytest.raises(ValueError, match="cache-hit"):
        backend.chat(system="system", user="question")


def test_shipped_profile_is_explicitly_non_thinking() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = root / "config" / "backends" / "deepseek.yaml"
    config = Config(
        source_path=root / "synthetic.yaml",
        data={"backend": str(profile)},
    )

    backend = load_backend_config(config)

    assert set(backend) == {
        "response",
        "endowment",
        "attribute_learner",
    }
    for component in backend.values():
        assert component["type"] == "deepseek"
        assert component["model_name"] == "deepseek-v4-flash"
        assert component["kwargs"]["thinking"] == "disabled"


def test_central_factory_constructs_deepseek_backend() -> None:
    client, _ = _client(_response(reasoning_tokens=0))

    backend = create_backend(
        backend_type="deepseek",
        model_name="deepseek-v4-flash",
        thinking="disabled",
        client=client,
    )

    assert isinstance(backend, DeepSeekBackend)
    assert backend.model_name == "deepseek-v4-flash"
