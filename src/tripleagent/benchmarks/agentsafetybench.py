from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from tripleagent.agents.tools import ToolRegistry, Tool
from tripleagent.agents.runner import AgentRunner, AgentConfig
from tripleagent.models.base import Model

Example = Dict[str, Any]


@dataclass
class AgentSafetyBenchCase:
    id: str
    risk_category: str
    instruction: str
    environments: List[Dict[str, Any]]
    failure_modes: List[str]
    fulfillable: bool

    # Fill in later
    tools: List[Tool] = field(default_factory=list)
    raw_entry: Example = field(default_factory=dict)

# Load raw examples
def load_agentsafetybench(
    source: str = "auto",   # auto, hf, local
    hf_name: str = "thu-coai/Agent-SafetyBench",
    split: str = "train",
    local_path: str | Path = "/workspaces/agentsafety_data.json",    # Update to the correct location
    limit: Optional[int] = None,
) -> List[Example]:
    # Include here a HF dataset loading function if the data is available on Hugging Face, otherwise load from local file
    
    examples: List[Example] = []
    
    if source in ("hf", "auto"):
        try:
            from datasets import load_dataset
            
            if hf_name is None:
                raise ValueError("hf_name is None; expected a dataset name string.")
            
            data = load_dataset(hf_name, split=split)
            if limit is not None:
                n = min(limit, len(data))
                data = data.select(range(n))
            examples = [dict(row) for row in data]
            return examples
        except Exception as e:
            if source == "hf":
                raise
            print(f"[AgentSafetyBench] HF load failed ({type(e).__name__}: {e})")
            print("[AgentSafetyBench] Falling back to local file…")
            
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Download the data and place it in the correct location.")
    
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    
    stripped_text = text.lstrip()
    if not stripped_text:
        raise ValueError(f"{path} is empty.")
    
    try:
        data = json.loads(stripped_text)
        if isinstance(data, dict):
            examples = [data]
        elif isinstance(data, list):
            examples = data
        else:
            raise ValueError("Unexpected JSON root type.")
    except json.JSONDecodeError:
        lines = [
            line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
        examples = [json.loads(line) for line in lines]
    
    if limit is not None:
        examples = examples[:limit]
        
    return examples

def parse_agentsafetybench(raw_examples: List[Example]) -> List[AgentSafetyBenchCase]:
    samples: List[AgentSafetyBenchCase] = []
    
    for row in raw_examples:
        _id = row.get("id")
        risk_category = row.get("risks", "unknown")
        instruction = row.get("instruction") or ""
        environments = row.get("environments", []),
        
        fm_raw = row.get("failure_modes", [])
        if isinstance(fm_raw, str):
            failure_modes = [fm.strip() for fm in fm_raw.split(",") if fm.strip()]
        else:
            failure_modes = [str(fm) for fm in fm_raw]

        fulfillable = bool(row.get("fulfillable", True))
        
        if not _id or not instruction:
            continue # skip for now

        samples.append(
            AgentSafetyBenchCase(
                id=str(_id),
                risk_category=str(risk_category),
                instruction=str(instruction),
                environments=environments,
                failure_modes=failure_modes,
                fulfillable=fulfillable,
                tools=[],       # for later
                raw_entry=row,
            )
        )
        
    return samples

def attach_tools(
    samples: List[AgentSafetyBenchCase],
) -> List[AgentSafetyBenchCase]:
    for sample in samples:
        # sample.tools = [EXAMPLE TOOL(sample)]     # Add tools here
    return samples


@dataclass
class AgentSafetyBenchRun:
    case: AgentSafetyBenchCase
    agent_result: Any       # Agent result from agent runner
    raw_trace: List[Dict[str, Any]]
    
async def run_agentsafetybench(
    model: Model,
    case: AgentSafetyBenchCase,
    agent_config: AgentConfig
) -> AgentSafetyBenchRun:
    tools = ToolRegistry(getattr(case, "tools", []))
    runner = AgentRunner(model=model, config=agent_config, tools=tools)
    
    user_prompt = case.instruction
    
    result = await runner.run(user_prompt)
    
    raw_trace: List[Dict[str, Any]] = []
    for step in result.steps:
        raw_trace.append(step.assistant_message)
        raw_trace.extend(step.tool_results)
    raw_trace.append(result.final_message)
    
    return AgentSafetyBenchRun(
        case=case,
        agent_result=result,
        raw_trace=raw_trace,
    )
    