from ..config import ModelConfig
from .backends.huggingface import HuggingFaceBackend
from .backends.openai import OpenAIBackend
from .base import ChatBackend


def create_backend(config: ModelConfig) -> ChatBackend:
    if config.provider == "openai":
        return OpenAIBackend(config)
    elif config.provider in ("hf_local", "hf_inference"):
        return HuggingFaceBackend(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
