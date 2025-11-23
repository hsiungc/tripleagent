from ..config import ModelConfig
from .base import ChatBackend

from .backends.openai import OpenAIBackend

def create_backend(config: ModelConfig) -> ChatBackend:
    if config.provider == "openai":
        return OpenAIBackend(config)
    # elif config.provider == "azure":
    #     from .backends.azure_backend import AzureBackend
    #     return AzureBackend(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")