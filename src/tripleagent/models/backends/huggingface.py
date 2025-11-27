import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ...config import ModelConfig
from ..base import ChatBackend, Message, ToolSpec


@dataclass
class HuggingFaceBackend(ChatBackend):
    config: ModelConfig

    def __post_init__(self) -> None:
        self._use_inference = bool(self.config.api_base)

        # HF Inference API setup
        if self._use_inference:
            if not self.config.api_key_env:
                raise ValueError(
                    "HuggingFace Inference API key environment variable not set."
                )

            api_key = os.getenv(self.config.api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"HuggingFace Inference API key not found in environment variable {self.config.api_key_env}."
                )

            base = self.config.api_base.rstrip("/")
            model_name = self.config.name
            self._endpoint = f"{base}/{model_name}"

            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={"Authorization": f"Bearer {api_key}"},
            )

        else:
            # Local model setup
            model_name = self.config.name
            extra = Dict[str, Any] = self.config.extra or {}

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self._pipeline = pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer, **extra
            )

    @staticmethod
    def _messages_to_prompt(messages: List[Message]) -> str:
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                content = "\n".join(parts)

            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = self._messages_to_prompt(messages)

        if self._use_inference:
            # HR Inference API call
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature,
                },
            }
            response = await self._client.post(self._endpoint, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                text = response.text[:500]
                raise RuntimeError(
                    f"HuggingFace Inference API error: {e.response.status_code} - {text}"
                ) from e

            data = response.json()

            if isinstance(data, list) and data and isinstance(data[0], dict):
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            else:
                text = str(data)

        else:
            # Local model inference
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": False,
            }

            pad_token_id: Optional[int] = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                gen_kwargs["pad_token_id"] = pad_token_id

            outputs = self._pipeline(prompt, **gen_kwargs)
            text = outputs[0].get("generated_text", "") if outputs else ""

            normalized_message: Message = {
                "role": "assistant",
                "content": text,
            }

            return {
                "message": normalized_message,
                "usage": None,
            }
