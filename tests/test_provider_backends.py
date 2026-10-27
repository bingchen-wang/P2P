"""OpenAI and Gemini adapters against the installed SDKs' own types.

No test calls a provider.  Requests go to fake clients; their keys are
checked against the SDK method signatures, and responses are real SDK
objects built offline, so a renamed field fails here rather than in a run.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from p2p.agents.factory import create_backend
from p2p.backends import GeminiBackend, NoAnswerTextError, OpenAIBackend
from p2p.backends.base import BackendResponse, TokenUsage


openai_types = pytest.importorskip("openai.types")
genai_types = pytest.importorskip("google.genai.types")

from openai.resources.chat.completions import Completions  # noqa: E402
from openai.resources.responses import Responses  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402
from openai.types.responses import Response  # noqa: E402


class _Recorder:
    """Fake SDK endpoint: records each request and returns a fixed reply."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.reply

    generate_content = create


def _openai(reply: Any, **kwargs: Any) -> tuple[OpenAIBackend, _Recorder]:
    endpoint = _Recorder(reply)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=endpoint),
        responses=endpoint,
    )
    backend = OpenAIBackend(model_name="gpt-test", client=client, **kwargs)
    return backend, endpoint


def _chat_completion(
    *,
    content: str | None = " Option A ",
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test-2026-01-01",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": refusal,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 6,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
    )


def _response(
    *,
    content: list[dict[str, Any]] | None = None,
    status: str = "completed",
    **extra: Any,
) -> Response:
    output = []
    if content is not None:
        output.append(
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }
        )
    return Response.model_validate(
        {
            "id": "resp-1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-test-2026-01-01",
            "status": status,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": output,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 6,
                "total_tokens": 17,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
            **extra,
        }
    )


def _text(value: str) -> dict[str, Any]:
    return {"type": "output_text", "text": value, "annotations": []}


USAGE = TokenUsage(
    input_tokens=11,
    output_tokens=6,
    cached_input_tokens=3,
    reasoning_tokens=2,
)


# OpenAI: options -------------------------------------------------------


def test_openai_defaults_to_chat_completions_without_storage() -> None:
    backend, _ = _openai(None)

    assert backend.transport == "chat_completions"
    assert backend.store is False


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("transport", "assistants", "transport must be"),
        ("transport", "Responses", "transport must be"),
        ("store", "false", "store must be"),
        ("store", None, "store must be"),
        ("max_tokens", 0, "max_tokens must be"),
        ("max_tokens", True, "max_tokens must be"),
        ("max_tokens", 12.0, "max_tokens must be"),
    ],
)
def test_openai_rejects_invalid_options(option, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAIBackend(
            model_name="gpt-test",
            client=object(),
            **{option: value},
        )


def test_openai_options_reach_the_adapter_from_config_kwargs() -> None:
    backend = create_backend(
        backend_type="openai",
        model_name="gpt-test",
        transport="responses",
        store=True,
        max_tokens=64,
        client=object(),
    )

    assert (backend.transport, backend.store, backend.max_tokens) == (
        "responses",
        True,
        64,
    )


# OpenAI: Chat Completions ---------------------------------------------


def test_chat_completions_request_uses_current_parameter_names() -> None:
    backend, endpoint = _openai(
        _chat_completion(),
        temperature=0.0,
        max_tokens=32,
    )

    response = backend.chat(system="Answer briefly.", user="Question?")

    assert response == BackendResponse(answer_text="Option A", usage=USAGE)
    assert endpoint.calls == [
        {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Question?"},
            ],
            "store": False,
            "temperature": 0.0,
            "max_completion_tokens": 32,
        }
    ]
    accepted = inspect.signature(Completions.create).parameters
    assert set(endpoint.calls[0]) <= set(accepted)


def test_chat_completions_omits_unset_sampling_options() -> None:
    backend, endpoint = _openai(
        _chat_completion(),
        temperature=None,
        max_tokens=None,
        store=True,
    )

    backend.chat(system="s", user="u")

    assert set(endpoint.calls[0]) == {"model", "messages", "store"}
    assert endpoint.calls[0]["store"] is True


def test_chat_completions_refusal_names_the_refusal() -> None:
    backend, _ = _openai(
        _chat_completion(content=None, refusal="I can't help with that.")
    )

    with pytest.raises(NoAnswerTextError, match="refused: I can't help"):
        backend.chat(system="s", user="u")


def test_chat_completions_filtered_reply_names_the_finish_reason() -> None:
    backend, _ = _openai(
        _chat_completion(content=None, finish_reason="content_filter")
    )

    with pytest.raises(NoAnswerTextError, match="'content_filter'"):
        backend.chat(system="s", user="u")


def test_chat_completions_empty_truncated_reply_is_an_error() -> None:
    backend, _ = _openai(_chat_completion(content="", finish_reason="length"))

    with pytest.raises(NoAnswerTextError, match="'length'"):
        backend.chat(system="s", user="u")


def test_chat_completions_keeps_partial_text_and_completed_empty_text() -> None:
    truncated, _ = _openai(
        _chat_completion(content="Option", finish_reason="length")
    )
    empty, _ = _openai(_chat_completion(content="", finish_reason="stop"))

    assert truncated.chat(system="s", user="u").answer_text == "Option"
    assert empty.chat(system="s", user="u").answer_text == ""


# OpenAI: Responses ----------------------------------------------------


def test_responses_request_maps_system_to_instructions() -> None:
    backend, endpoint = _openai(
        _response(content=[_text(" Option "), _text("A ")]),
        transport="responses",
        temperature=0.0,
        max_tokens=32,
    )

    response = backend.chat(system="Answer briefly.", user="Question?")

    assert response == BackendResponse(answer_text="Option A", usage=USAGE)
    assert endpoint.calls == [
        {
            "model": "gpt-test",
            "instructions": "Answer briefly.",
            "input": "Question?",
            "store": False,
            "temperature": 0.0,
            "max_output_tokens": 32,
        }
    ]
    accepted = inspect.signature(Responses.create).parameters
    assert set(endpoint.calls[0]) <= set(accepted)


def test_responses_omits_empty_instructions_and_unset_options() -> None:
    backend, endpoint = _openai(
        _response(content=[_text("A")]),
        transport="responses",
        temperature=None,
        max_tokens=None,
    )

    backend.chat(system="", user="u")

    assert endpoint.calls[0] == {"model": "gpt-test", "input": "u", "store": False}


def test_responses_refusal_names_the_refusal() -> None:
    backend, _ = _openai(
        _response(content=[{"type": "refusal", "refusal": "I can't help."}]),
        transport="responses",
    )

    with pytest.raises(NoAnswerTextError, match="refused: I can't help"):
        backend.chat(system="s", user="u")


def test_responses_incomplete_without_text_names_the_reason() -> None:
    backend, _ = _openai(
        _response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ),
        transport="responses",
    )

    with pytest.raises(NoAnswerTextError, match="'max_output_tokens'"):
        backend.chat(system="s", user="u")


def test_responses_failure_names_the_error() -> None:
    backend, _ = _openai(
        _response(
            status="failed",
            error={"code": "server_error", "message": "upstream timeout"},
        ),
        transport="responses",
    )

    with pytest.raises(NoAnswerTextError, match="upstream timeout"):
        backend.chat(system="s", user="u")


def test_responses_keeps_partial_text_like_chat_completions() -> None:
    backend, _ = _openai(
        _response(
            content=[_text("Option")],
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ),
        transport="responses",
    )

    assert backend.chat(system="s", user="u").answer_text == "Option"


# Gemini ---------------------------------------------------------------


def _gemini(reply: Any, **kwargs: Any) -> tuple[GeminiBackend, _Recorder]:
    endpoint = _Recorder(reply)
    backend = GeminiBackend(
        model_name="gemini-test",
        client=SimpleNamespace(models=endpoint),
        config_factory=genai_types.GenerateContentConfig,
        **kwargs,
    )
    return backend, endpoint


def _gemini_response(
    *,
    text: str | None = " Option A ",
    finish_reason: str = "STOP",
    block_reason: str | None = None,
) -> Any:
    candidates = []
    if block_reason is None:
        parts = [] if text is None else [{"text": text}]
        candidates.append(
            {
                "content": {"role": "model", "parts": parts},
                "finish_reason": finish_reason,
            }
        )
    feedback = (
        None
        if block_reason is None
        else {
            "block_reason": block_reason,
            "block_reason_message": "Prompt was blocked.",
        }
    )
    return genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": candidates,
            "prompt_feedback": feedback,
            "model_version": "gemini-test-001",
            "usage_metadata": {
                "prompt_token_count": 11,
                "candidates_token_count": 4,
                "thoughts_token_count": 2,
                "cached_content_token_count": 3,
            },
        }
    )


def test_gemini_requires_an_explicit_model_id() -> None:
    parameter = inspect.signature(GeminiBackend).parameters["model_name"]

    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="model_name"):
        GeminiBackend(client=object())


def test_gemini_request_builds_a_real_sdk_config() -> None:
    backend, endpoint = _gemini(
        _gemini_response(),
        temperature=0.0,
        max_tokens=32,
    )

    response = backend.chat(system="Answer briefly.", user="Question?")

    assert response == BackendResponse(answer_text="Option A", usage=USAGE)
    (call,) = endpoint.calls
    assert call["model"] == "gemini-test"
    assert call["contents"] == "Question?"
    assert isinstance(call["config"], genai_types.GenerateContentConfig)
    assert call["config"].model_dump(exclude_none=True) == {
        "system_instruction": "Answer briefly.",
        "temperature": 0.0,
        "max_output_tokens": 32,
    }


def test_gemini_blocked_prompt_names_the_block_reason() -> None:
    backend, _ = _gemini(_gemini_response(block_reason="SAFETY"))

    with pytest.raises(
        NoAnswerTextError,
        match=r"block_reason='SAFETY'\): Prompt was blocked",
    ):
        backend.chat(system="s", user="u")


@pytest.mark.parametrize("finish_reason", ["SAFETY", "MAX_TOKENS", "RECITATION"])
def test_gemini_reply_without_text_names_the_finish_reason(finish_reason) -> None:
    backend, _ = _gemini(
        _gemini_response(text=None, finish_reason=finish_reason)
    )

    with pytest.raises(NoAnswerTextError, match=f"'{finish_reason}'"):
        backend.chat(system="s", user="u")


def test_gemini_keeps_partial_text() -> None:
    backend, _ = _gemini(
        _gemini_response(text="Option", finish_reason="MAX_TOKENS")
    )

    assert backend.chat(system="s", user="u").answer_text == "Option"
