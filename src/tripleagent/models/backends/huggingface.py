from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ...config import ModelConfig
from ..base import ChatBackend


@dataclass
class HuggingFaceBackend(ChatBackend):
    config: ModelConfig

    def __post_init__(self) -> None:
        self._use_inference = bool(self.config.api_base)
        
        # HF Inference API setup
        if self._use_inference:
            if not self.config.api_key_env:
                raise ValueError("HuggingFace Inference API key environment variable not set.")
            
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"HuggingFace Inference API key not found in environment variable {self.config.api_key_env}.")
            base = self.config.api_base.rstrip('/')
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
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **extra
            )

    async def chat(
        self,
        prompt: str
    ) -> str:
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
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list and data and isinstance(data[0], dict):
                if "generated_text" in data[0]:
                    return data[0]['generated_text']

            if isinstance(data, dict) and "error" in data:
                return data["generated_text"]
            
            return str(data)
            
        else:
            # Local model inference
            kwargs: Dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": False,
            }
            
            pad_token_id: Optional[int] = getattr(self.tokenizer, "eos_token_id", None)
            if pad_id is not None:
                kwargs["pad_token_id"] = pad_token_id
            
            outputs = self._pipeline(prompt, **kwargs)
            return outputs[0]['generated_text']