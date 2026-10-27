"""Opt-in paid smoke for the live DeepSeek API."""

from __future__ import annotations

import os

import pytest

from p2p.backends.deepseek import DeepSeekBackend


pytestmark = [
    pytest.mark.paid,
    pytest.mark.skipif(
        os.environ.get("P2P_RUN_DEEPSEEK") != "1",
        reason="set P2P_RUN_DEEPSEEK=1 to enable the paid test",
    ),
]


def test_deepseek_v4_flash_non_thinking_smoke() -> None:
    backend = DeepSeekBackend(
        model_name=os.environ.get(
            "P2P_DEEPSEEK_MODEL",
            "deepseek-v4-flash",
        ),
        thinking="disabled",
        temperature=0,
        max_tokens=8,
    )

    response = backend.chat(
        system="Return only the requested word.",
        user="Return exactly: OK",
    )

    assert response.answer_text
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.reasoning_tokens == 0
