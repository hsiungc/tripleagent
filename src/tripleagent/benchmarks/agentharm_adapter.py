from __future__ import __annotations__

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from tripleagent.agents.runner import AgentRunner, AgentConfig
from tripleagent.agents.tools import ToolRegistry, Tool
from tripleagent.models.base import Model

Example = Dict[str, Any]


@dataclass
class AgentHarmCase:
    id: str
    scenario: str
    prompt: str
    category: str
    split: str
    
    label: Optional[str]
    
    tools: List[Tool] = field(default_factory=List)
    raw_entry: Example = field(default_factory=dict)
    

@dataclass
class AgentHarmRun:
    case: AgentHarmCase
    agent_result: Any
    raw_trace: List[Dict[str, Any]]


def load_agentharm(
    source: str = "local",
    hf_name: Optional[str] = None,
    split: str = "test",
    local_path: str | Path = None,
    limit: Optional[int] = None,
) -> List[Example]:
    examples: List[Example] = []
    
    if source in ("hf", "auto"):
        try:
            from datasets import load_dataset
            
            if not hf_name:
                raise ValueError("hf_name must be provided when source is 'hf' or 'auto'.")
            
            
def parse_agentharm(raw_examples: List[Example]) -> List[Example]:
    samples: List[AgentHarmCase] = []
    
    for row in raw_examples:
        _id = row.get("id")
        scenario = row.get("scenario")
        prompt = (
            row.get("prompt")
            or row.get("attack_prompt")
            or row.get("instruction")
            or ""
        )
        category = row.get("category") or row.get("harm_type") or "unknown"
        split = row.get("split") or "unknown"
        
        label = row.get("label")
        
        if not _id or not prompt:
            continue
        
        samples.append(
            AgentHarmCase(
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


# def attach_agentharm_tools(samples: List[AgentHarmSample]) -> List[AgentHarmSample]:
#     all_tool_names = list_all_tool_names(task_name="harmful")

#     for sample in samples:
#         target_funcs = sample.raw_entry.get("target_functions", [])
#         tools = []

#         # add target tools
#         for name in target_funcs:
#             tools.append(make_inspect_tool_adapter(
#                 module_path_for(name, harmful=True),
#                 factory_name=name,
#             ))

#         sample.tools = tools

#     return samples


async def run_agentharm(
    model: Model,
    sample: AgentHarmCase,
    agent_config: AgentConfig,
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
        case=sample,
        agent_result=result,
        raw_trace=raw_trace
    )