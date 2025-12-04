from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from huggingface_hub import snapshot_download

from tripleagent.agents.runner import AgentConfig, AgentRunner
from tripleagent.models.base import Model
from tripleagent.benchmarks.helpers.agentharm_tools import build_tools_for_sample
from tripleagent.benchmarks.helpers.agentharm_prompts import get_system_prompt
from tripleagent.scoring.agentharm import judge_agentharm
from tripleagent.reporting.agentharm import summarize_agentharm

from .utils import load_local_json

Example = Dict[str, Any]
Message = Dict[str, Any]


@dataclass
class AgentHarmSample:
    id: str
    scenario: str
    prompt: str
    category: str
    split: str
    is_harmful: bool          # True for harmful set, False for benign
    raw_entry: Example


@dataclass
class AgentHarmRun:
    sample: AgentHarmSample
    agent_result: Any         # AgentRunResult-like object from AgentRunner
    raw_trace: List[Message]


# -------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------

def load_agentharm(
    source: str = "auto",  # "auto" | "hf" | "local"
    hf_name: str = "ai-safety-institute/AgentHarm",
    task_name: Literal["harmful", "benign", "chat"] = "harmful",
    split: Literal["val", "test_public", "test_private"] = "test_public",
    local_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[Example]:
    """
    Load raw AgentHarm behaviors (JSON dicts), either from HF or a local file.
    """
    # HF path (default)
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

            if task_name not in split_dict or split not in split_dict[task_name]:
                raise ValueError(f"[AgentHarm] Invalid split: {task_name}/{split}")

            input_json_file = cache_dir / split_dict[task_name][split]
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
    if local_path is None:
        raise ValueError(
            "[AgentHarm] local_path must be provided when 'source' is 'local' "
            "or HF download failed."
        )

    examples = load_local_json(Path(local_path), limit=limit)
    return examples


def parse_agentharm(
    raw_examples: List[Example],
    task_name: Literal["harmful", "benign", "chat"] = "harmful",
) -> List[AgentHarmSample]:
    samples: List[AgentHarmSample] = []
    is_harmful = task_name == "harmful"

    for row in raw_examples:
        _id = row.get("id")
        scenario = row.get("scenario", "")
        prompt = row.get("prompt", "")
        category = row.get("category", "unknown")
        split = row.get("split", "unknown")

        if not _id or not prompt:
            continue

        samples.append(
            AgentHarmSample(
                id=str(_id),
                scenario=str(scenario),
                prompt=str(prompt),
                category=str(category),
                split=str(split),
                is_harmful=is_harmful,
                raw_entry=row,
            )
        )

    return samples


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
            b for b in filtered if b.get("detailed_prompt") == detailed_behaviors
        ]

    if hint_included is not None:
        filtered = [b for b in filtered if b.get("hint_included") == hint_included]

    return filtered


async def run_agentharm(
    sample: AgentHarmSample,
    model: Model,
    agent_config: AgentConfig,
) -> AgentHarmRun:
    tool_registry = build_tools_for_sample(
        raw_example=sample.raw_entry,
        task_name="harmful" if sample.is_harmful else "benign",
    )

    runner = AgentRunner(
        model=model,
        tools=tool_registry,
        config=agent_config,
    )

    result = await runner.run(sample.prompt)

    trace: List[Message] = []
    for step in result.steps:
        trace.append(step.assistant_message)
        trace.extend(step.tool_results)
    trace.append(result.final_message)

    return AgentHarmRun(
        sample=sample,
        agent_result=result,
        raw_trace=trace,
    )


async def experiment_agentharm(
    model: Model,
    judge_model: Model,
    cfg: dict,
    run_dir: Path,
) -> dict:
    task_name: Literal["harmful", "benign", "chat"] = cfg.get("task_name", "harmful")

    raw = load_agentharm(
        source=cfg.get("source", "hf"),
        hf_name=cfg.get("hf_name", "ai-safety-institute/AgentHarm"),
        task_name=task_name,
        split=cfg.get("split", "test_public"),
        local_path=cfg.get("local_path"),
        limit=cfg.get("limit"),
    )

    raw = filter_behaviors(
        raw,
        behavior_ids=cfg.get("behavior_ids"),
        detailed_behaviors=cfg.get("detailed_behaviors"),
        hint_included=cfg.get("hint_included"),
    )

    samples = parse_agentharm(raw, task_name=task_name)

    # Official AgentHarm prompts from the paper (via helper)
    system_prompt_template = cfg.get("system_prompt_template", "default")
    prompt_technique = cfg.get("prompt_technique", "standard")

    system_prompt = get_system_prompt(
        system_prompt_template=system_prompt_template,
        prompt_technique=prompt_technique,
    )

    agent_cfg = AgentConfig(
        system_prompt=system_prompt,
        max_iterations=cfg.get("max_iterations", 3),
        tool_choice=cfg.get("tool_choice", "auto"),
        temperature=0.0,
        max_new_tokens=cfg.get("max_new_tokens", 512),
    )

    refusal_judge_model = judge_model
    semantic_judge_model = judge_model

    scores = []
    for sample in samples:
        run_result = await run_agentharm(
            sample=sample,
            model=model,
            agent_config=agent_cfg,
        )
        score = await judge_agentharm(
            sample=sample,
            run_result=run_result,
            refusal_judge_model=refusal_judge_model,
            semantic_judge_model=semantic_judge_model,
        )
        scores.append(score)

    summary = summarize_agentharm(scores)

    (run_dir / "agentharm_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "agentharm_scores.json").write_text(
        json.dumps([s.__dict__ for s in scores], indent=2),
        encoding="utf-8",
    )

    return summary