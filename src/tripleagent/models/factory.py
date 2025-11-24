from ..config import ModelConfig
from .base import ChatBackend

from .backends.openai import OpenAIBackend
from .backends.huggingface import HuggingFaceBackend

def create_backend(config: ModelConfig) -> ChatBackend:
    if config.provider == "openai":
        return OpenAIBackend(config)
    elif config.provider in ("hf_local", "hf_inference"):
        return HuggingFaceBackend(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")