from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import ModelConfig

Message = Dict[str, Any]
ToolSpec = Dict[str, Any]


class ChatBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        #     prompt = self._messages_to_prompt(messages)
        #     response = self.generate(prompt)

        #     normalized_response: Message = {
        #         "role": "assistant",
        #         "content": response
        #     }

        #     return {
        #         "response": normalized_response,
        #         "usage": None
        #     }

        # def _messages_to_prompt(self, messages: list[Message]) -> str:
        #     lines: List[str] = []
        #     for m in messages:
        #         role = m.get("role", "user")
        #         content = m.get("content", "")
        #         if isinstance(content, list):
        #             parts = []
        #             for chunk in content:
        #                 if isinstance(chunk, dict) and chunk.get("type") == "text":
        #                     parts.append(chunk.get("text", ""))
        #             content = "".join(parts)
        #         lines.append(f"{role}: {content}")
        #     lines.append("assistant: ")
        #     return "\n".join(lines)
        raise NotImplementedError


@dataclass
class Model:
    name: str
    backend: ChatBackend
    config: ModelConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Model":
        config = ModelConfig.from_yaml(yaml_path)

        from .factory import create_backend

        backend = create_backend(config)
        return cls(name=config.name, backend=backend, config=config)

    # Conversational chat interface
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self.backend.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            model=model,
            **kwargs,
        )
