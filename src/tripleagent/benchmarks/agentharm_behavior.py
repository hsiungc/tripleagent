from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import json
import shutil

from huggingface_hub import snapshot_download

Example = Dict[str, Any]


def load_agentharm_behaviors(
    name: Literal["harmful", "benign", "chat"] = "harmful",
    split: Literal["val", "test_public", "test_private"] = "test_public",
    cache_dir: Optional[Path] = None,
) -> List[Example]:
    if cache_dir is None:
        cache_dir = Path(".cache") / "agentharm_dataset" / "AgentHarm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id="ai-safety-institute/AgentHarm",
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns="**/*.json",
            resume_download=True,
        )
    except Exception as ex:
        print(f"[AgentHarm] Error downloading dataset: {ex}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise

    split_dict = {
        "harmful": {
            "val": "benchmark/harmful_behaviors_validation.json",
            "test_public": "benchmark/harmful_behaviors_test_public.json",
            "test_private": "benchmark/harmful_behaviors_test_private.json",
        },
        "benign": {
            "val": "benchmark/benign_behaviors_validation.json",
            "test_public": "benchmark/benign_behaviors_test_public.json",
            "test_private": "benchmark/benign_behaviors_test_private.json",
        },
        "chat": {
            "val": "benchmark/chat_validation.json",
            "test_public": "benchmark/chat_public_test.json",
            "test_private": "benchmark/chat_private_test.json",
        },
    }

    if name not in split_dict:
        raise ValueError(f"Dataset {name} is not available.")
    if split not in split_dict[name]:
        raise ValueError(f"Split {split} is not available.")

    input_json_file = cache_dir / split_dict[name][split]
    if not input_json_file.exists():
        raise FileNotFoundError(f"[AgentHarm] File not found: {input_json_file}")

    with input_json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    behaviors = data.get("behaviors", [])
    if not isinstance(behaviors, list):
        raise ValueError("[AgentHarm] Expected 'behaviors' to be a list in the JSON.")

    return behaviors


def filter_behaviors(
    behaviors: List[Example],
    behavior_ids: Optional[List[str]] = None,
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
) -> List[Example]:
    filtered = behaviors

    if behavior_ids:
        id_set = set(behavior_ids)
        filtered = [b for b in filtered if str(b.get("id")) in id_set]

    if detailed_behaviors is not None:
        filtered = [
            b for b in filtered
            if b.get("detailed_prompt") == detailed_behaviors
        ]

    if hint_included is not None:
        filtered = [
            b for b in filtered
            if b.get("hint_included") == hint_included
        ]

    return filtered
