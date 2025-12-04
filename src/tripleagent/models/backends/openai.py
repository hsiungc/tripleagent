from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from ..config import ModelConfig
from ..base import ChatBackend, Message, ToolSpec


class OpenAIBackend(ChatBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

        api_key = os.getenv(config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found in environment variable {config.api_key_env or 'OPENAI_API_KEY'}."
            )
        self._api_key = api_key

        base = (config.api_base or "https://api.openai.com").rstrip("/")
        self._endpoint = f"{base}"

        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.config.name,
            "messages": messages,
        }

        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_new_tokens"]

        if tools:
            payload["tools"] = tools
            if tool_choice in ("auto", "none"):
                payload["tool_choice"] = tool_choice
            elif tool_choice is not None:
                payload["tool_choice"] = tool_choice

        print(f"[OpenAIBackend] Sending tools={bool(tools)} tool_choice={tool_choice}")

        resp = await self._client.post(self._endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from OpenAI: {data}")

        message = choices[0].get("message", {})
        usage = data.get("usage", {})

        return {
            "response": message,
            "usage": usage,
        }
