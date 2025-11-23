from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    provider: str
    name: str = field(default="gpt2")
    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.0)
    api_base = Optional[str]
    api_key_env: str = field(default="OPENAI_API_KEY")
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(**config_dict)
    
    @staticmethod
    def from_yaml(yaml_path: str) -> "ModelConfig":
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return ModelConfig.from_dict(config_dict["model"])