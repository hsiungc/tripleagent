# src/tripleagent/models/factory.py

from __future__ import annotations

from .config import ModelConfig
from .backends.openai import OpenAIBackend
from .backends.huggingface import HuggingFaceBackend


def create_backend(config: ModelConfig):
    provider = (config.provider or "").lower().strip()

    if provider == "openai":
        return OpenAIBackend(config=config)

    if provider in ("hf", "huggingface"):
        return HuggingFaceBackend(config=config)

    # if provider == "anthropic":
    #     return AnthropicBackend(config=config)

    # if provider == "gemini":
    #     return GeminiBackend(config=config)

    raise ValueError(
        f"Unsupported provider '{config.provider}' in ModelConfig. "
    )