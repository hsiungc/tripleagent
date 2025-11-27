from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import json
import shutil

from huggingface_hub import snapshot_download

from tripleagent.agents.runner import AgentRunner, AgentConfig
from tripleagent.agents.tools import ToolRegistry, Tool
from tripleagent.models.base import Model

from .utils import load_local_json

Example = Dict[str, Any]


@dataclass
class AgentHarmSample:
    id: str
    scenario: str
    prompt: str
    category: str
    split: str
    
    label: Optional[str] = None
    
    tools: List[Tool] = field(default_factory=list)
    raw_entry: Example = field(default_factory=dict)


def load_agentharm(
    source: str = "auto",    # "auto" | "hf" | "local"
    hf_name: str = "ai-safety-institute/AgentHarm",
    split: Literal["val", "test_public", "test_private"] = "test_public",
    local_path: str | Path = "/workspaces/agentharm_behaviors.json",
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[Example]:
    if source in ("hf", "auto"):
        try:
            if cache_dir is None:
                cache_dir = Path(".cache") / "agentharm_dataset" / "AgentHarm"
            cache_dir.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id=hf_name,
                repo_type="dataset",
                local_dir=cache_dir,
                allow_patterns="**/*.json",
                resume_download=True,
            )

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

            # For now, default to harmful task
            name = "harmful"
            if name not in split_dict or split not in split_dict[name]:
                raise ValueError(f"Invalid AgentHarm split: {name}/{split}")

            input_json_file = cache_dir / split_dict[name][split]
            if not input_json_file.exists():
                raise FileNotFoundError(f"[AgentHarm] File not found: {input_json_file}")

            with input_json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            behaviors = data.get("behaviors", [])
            if not isinstance(behaviors, list):
                raise ValueError("[AgentHarm] Expected 'behaviors' to be a list.")

            if limit is not None:
                behaviors = behaviors[:limit]

            return [dict(b) for b in behaviors]

        except Exception as e:
            if source == "hf":
                raise
            print(f"[AgentHarm] HF load failed ({type(e).__name__}: {e})")
            print("[AgentHarm] Falling back to local file…")
            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)

    # Local fallback
    examples = load_local_json(local_path, limit=limit)
    return examples


def parse_agentharm(raw_examples: List[Example]) -> List[AgentHarmSample]:
    samples: List[AgentHarmSample] = []
    for row in raw_examples:
        _id = row.get("id")
        scenario = row.get("scenario")
        prompt = row.get("prompt")
        category = row.get("category")
        split = row.get("split")
        label = row.get("label")
        
        if not _id or not prompt:
            continue
        
        samples.append(
            AgentHarmSample(
                id=str(_id),
                scenario=str(scenario),
                prompt=str(prompt),
                category=str(category),
                split=str(split),
                label=str(label) if label is not None else None,
                tools=[],
                raw_entry=row,
            )
        )
        
    return samples


def attach_tools_agentharm(
    samples: List[AgentHarmSample],
) -> List[AgentHarmSample]:    
    for sample in samples:
        sample.tools = []     # Add tools here
        
    return samples
    
    
@dataclass
class AgentHarmRun:
    sample: AgentHarmSample
    agent_result: Any
    raw_trace: List[Dict[str, Any]]
    

async def run_agentharm(
    sample: AgentHarmSample, 
    model: Model, 
    agent_config: AgentConfig
) -> AgentHarmRun:
    tools = ToolRegistry(sample.tools)
    runner = AgentRunner(model=model, config=agent_config, tools=tools)
    user_prompt = sample.prompt
    result = await runner.run(user_prompt)
    
    raw_trace: List[Dict[str, Any]] = []
    for step in result.steps:
        raw_trace.append(step.assistant_message)
        raw_trace.extend(step.tool_results)
    raw_trace.append(result.final_message)
    
    return AgentHarmRun(
        sample=sample, 
        agent_result=result, 
        raw_trace=raw_trace
    )
    

# def load_agentharm_behaviors(
#     name: Literal["harmful", "benign", "chat"] = "harmful",
#     split: Literal["val", "test_public", "test_private"] = "test_public",
#     cache_dir: Optional[Path] = None,
# ) -> List[Example]:
#     if cache_dir is None:
#         cache_dir = Path(".cache") / "agentharm_dataset" / "AgentHarm"
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         snapshot_download(
#             repo_id="ai-safety-institute/AgentHarm",
#             repo_type="dataset",
#             local_dir=cache_dir,
#             allow_patterns="**/*.json",
#             resume_download=True,
#         )
#     except Exception as ex:
#         print(f"[AgentHarm] Error downloading dataset: {ex}")
#         shutil.rmtree(cache_dir, ignore_errors=True)
#         raise

#     split_dict = {
#         "harmful": {
#             "val": "benchmark/harmful_behaviors_validation.json",
#             "test_public": "benchmark/harmful_behaviors_test_public.json",
#             "test_private": "benchmark/harmful_behaviors_test_private.json",
#         },
#         "benign": {
#             "val": "benchmark/benign_behaviors_validation.json",
#             "test_public": "benchmark/benign_behaviors_test_public.json",
#             "test_private": "benchmark/benign_behaviors_test_private.json",
#         },
#         "chat": {
#             "val": "benchmark/chat_validation.json",
#             "test_public": "benchmark/chat_public_test.json",
#             "test_private": "benchmark/chat_private_test.json",
#         },
#     }

#     if name not in split_dict:
#         raise ValueError(f"Dataset {name} is not available.")
#     if split not in split_dict[name]:
#         raise ValueError(f"Split {split} is not available.")

#     input_json_file = cache_dir / split_dict[name][split]
#     if not input_json_file.exists():
#         raise FileNotFoundError(f"[AgentHarm] File not found: {input_json_file}")

#     with input_json_file.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     behaviors = data.get("behaviors", [])
#     if not isinstance(behaviors, list):
#         raise ValueError("[AgentHarm] Expected 'behaviors' to be a list in the JSON.")

#     return behaviors


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