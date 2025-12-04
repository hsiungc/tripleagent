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

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(**config_dict)

    @staticmethod
    def from_yaml(
        yaml_path: str,
        section: str = "model",
    ) -> "ModelConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)

        cfg_dict = full_cfg.get(section)
        if cfg_dict is None:
            raise KeyError(f"Section '{section}' not found in {yaml_path}")

        return ModelConfig.from_dict(cfg_dict)
