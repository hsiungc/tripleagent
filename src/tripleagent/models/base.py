from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ModelConfig

Message = Dict[str, Any]
ToolSpec = Dict[str, Any]


class ChatBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

@dataclass
class Model:
    name: str
    backend: ChatBackend
    config: ModelConfig

    @classmethod
    def from_yaml(cls, yaml_path: str, section: str = "model") -> "Model":
        from .factory import create_backend
        
        config = ModelConfig.from_yaml(yaml_path, section=section)
        backend = create_backend(config)
        return cls(name=config.name, backend=backend, config=config)

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.backend.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            model=model or self.name,
            **kwargs,
        )