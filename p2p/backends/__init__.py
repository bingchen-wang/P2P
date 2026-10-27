"""Backend adapters and normalized response contracts."""

from p2p.backends.base import (
    Backend,
    BackendResponse,
    NoAnswerTextError,
    TokenUsage,
)
from p2p.backends.deepseek import DeepSeekBackend
from p2p.backends.dummy import DummyBackend
from p2p.backends.gemini import GeminiBackend
from p2p.backends.ollama import OllamaBackend
from p2p.backends.openai import OpenAIBackend
from p2p.backends.simulated import SimulatedBackend


__all__ = [
    "Backend",
    "BackendResponse",
    "DeepSeekBackend",
    "DummyBackend",
    "GeminiBackend",
    "NoAnswerTextError",
    "OllamaBackend",
    "OpenAIBackend",
    "SimulatedBackend",
    "TokenUsage",
]
