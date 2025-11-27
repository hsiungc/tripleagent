from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tripleagent.agents.runner import AgentConfig, AgentRunner
from tripleagent.agents.tools import Tool, ToolRegistry
from tripleagent.models.base import Model

from .utils import load_local_json

Example = Dict[str, Any]


@dataclass
class AgentSafetyBenchSample:
    id: str
    risk_category: List[str]
    instruction: str
    environments: List[Dict[str, Any]]
    failure_modes: List[str]
    fulfillable: bool

    # Fill in later
    tools: List[Tool] = field(default_factory=list)
    raw_entry: Example = field(default_factory=dict)


# Load raw examples
def load_agentsafetybench(
    source: str = "auto",  # "auto" | "hf" | "local"
    hf_name: str = "thu-coai/Agent-SafetyBench",
    split: str = "train",
    local_path: str | Path = "/workspaces/agentsafety_data.json",
    limit: Optional[int] = None,
) -> List[Example]:
    if source == "hf":
        raise RuntimeError(
            "HuggingFace datasets loading for AgentSafetyBench is disabled in this project "
            "because the HF dataset currently raises DatasetGenerationError. "
            "Use source='local' and a local JSON file instead."
        )

    if source not in ("local", "auto"):
        raise ValueError(f"Unknown source '{source}'. Expected 'local' or 'auto'.")

    path = Path(local_path)

    # Local file
    examples = load_local_json(path, limit=limit)
    return examples


def parse_agentsafetybench(raw_examples: List[Example]) -> List[AgentSafetyBenchSample]:
    samples: List[AgentSafetyBenchSample] = []

    for row in raw_examples:
        _id = row.get("id")
        risks = row.get("risks", "unknown")
        instruction = row.get("instruction") or ""

        environments = row.get("environments") or []
        if not isinstance(environments, list):
            raise ValueError(
                f"Expected 'environments' to be a list, got {type(environments)}."
            )

        environments: List[Dict[str, Any]] = [dict(e) for e in environments]

        fm_raw = row.get("failure_modes", [])
        if isinstance(fm_raw, str):
            failure_modes = [fm.strip() for fm in fm_raw.split(",") if fm.strip()]
        else:
            failure_modes = [str(fm) for fm in fm_raw]

        fulfillable = bool(row.get("fulfillable", True))

        if not _id or not instruction:
            continue  # skip for now

        if isinstance(risks, list):
            risk_category: List[str] = [str(r) for r in risks]
        else:
            risk_category = [str(risks)]

        samples.append(
            AgentSafetyBenchSample(
                id=str(_id),
                risk_category=risk_category,
                instruction=str(instruction),
                environments=environments,
                failure_modes=failure_modes,
                fulfillable=fulfillable,
                tools=[],  # for later
                raw_entry=row,
            )
        )

    return samples


def attach_tools_agentsafetybench(
    samples: List[AgentSafetyBenchSample],
) -> List[AgentSafetyBenchSample]:
    for sample in samples:
        sample.tools = []  # Add tools here

    return samples


@dataclass
class AgentSafetyBenchRun:
    sample: AgentSafetyBenchSample
    agent_result: Any
    raw_trace: List[Dict[str, Any]]


async def run_agentsafetybench(
    model: Model, sample: AgentSafetyBenchSample, agent_config: AgentConfig
) -> AgentSafetyBenchRun:
    tools = ToolRegistry(sample.tools)
    runner = AgentRunner(model=model, config=agent_config, tools=tools)
    user_prompt = sample.instruction
    result = await runner.run(user_prompt)

    raw_trace: List[Dict[str, Any]] = []
    for step in result.steps:
        raw_trace.append(step.assistant_message)
        raw_trace.extend(step.tool_results)
    raw_trace.append(result.final_message)

    return AgentSafetyBenchRun(
        sample=sample,
        agent_result=result,
        raw_trace=raw_trace,
    )
