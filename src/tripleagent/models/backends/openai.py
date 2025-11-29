import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..config import ModelConfig
from ..base import ChatBackend, Message, ToolSpec


@dataclass
class OpenAIBackend(ChatBackend):
    config: ModelConfig

    def __post_init__(self) -> None:
        api_key = os.getenv(self.config.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {self.config.api_key_env}"
            )

        client_params: Dict[str, Any] = {"api_key": api_key}
        # if self.config.api_base:
        #     client_params["api_base"] = self.config.api_base
        self.client = AsyncOpenAI(**client_params)

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model or self.config.name,
            "messages": messages,
        }

        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        params.setdefault("max_tokens", self.config.max_new_tokens)
        params.setdefault("temperature", self.config.temperature)

        params.update(kwargs)

        response = await self.client.chat.completions.create(**params)

        choice = response.choices[0]
        message = choice.message

        normalized_response: Message = {
            "role": message.role,
            "content": message.content,
        }

        if message.tool_calls:
            normalized_response["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "response": normalized_response,
            "usage": usage,
        }
