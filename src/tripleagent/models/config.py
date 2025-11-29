from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    provider: str
    name: str = "gpt2"
    max_new_tokens: int = 512
    temperature: float = 0.0
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        return cls(**config_dict)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        section: str = "model",
    ) -> "ModelConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if section not in config_dict:
            raise KeyError(
                f"Section '{section}' not found in YAML file {yaml_path}."
            )

        return cls.from_dict(config_dict[section])
