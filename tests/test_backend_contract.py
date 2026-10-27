"""Normalized backend response contract."""

from __future__ import annotations

from dataclasses import fields

import pytest

from p2p.backends.base import BackendResponse, TokenUsage


def test_backend_response_has_only_normalized_text_and_usage() -> None:
    usage = TokenUsage(
        input_tokens=12,
        output_tokens=5,
        cached_input_tokens=3,
    )
    response = BackendResponse(answer_text="normalized answer", usage=usage)

    assert tuple(field.name for field in fields(TokenUsage)) == (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    )
    assert tuple(field.name for field in fields(BackendResponse)) == (
        "answer_text",
        "usage",
    )
    assert response.answer_text == "normalized answer"
    assert response.usage == usage


def test_usage_defaults_to_zero() -> None:
    response = BackendResponse(answer_text="answer")

    assert response.usage == TokenUsage()


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1},
        {"output_tokens": -1},
        {"cached_input_tokens": -1},
        {"reasoning_tokens": -1},
        {"input_tokens": 1, "cached_input_tokens": 2},
        {"output_tokens": 1, "reasoning_tokens": 2},
    ],
)
def test_usage_rejects_invalid_counts(usage: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TokenUsage(**usage)
