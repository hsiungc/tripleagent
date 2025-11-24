from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from tripleagent.agents.tools import ToolRegistry, Tool
from tripleagent.agents.runner import AgentRunner, AgentConfig
from tripleagent.models.base import Model


@dataclass
class AgentSafetyBenchConfig:
    case_id: str
    risk_category: str
    failure_modes: List[str]
    initial_messages: List[Dict[str, Any]]
    tools: List[Tool]
    

def load_agentsafetybench(
    max_examples: Optional[int] = None,
    local_path: str | Path = "/workspace/data/agentsafety_data.json",
    hf_dataset: str = "thu-coai/Agent-SafetyBench",    # dataset is stored on HuggingFace
    split: str = "train",                                  # only train split exists on HF
    env_filter: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> List[AgentSafetyBenchConfig]:
    data = load_dataset(hf_dataset, split=split)
    cases: List[AgentSafetyBenchConfig] = []
    
    for item in data:
        
    
    return cases


@dataclass
class AgentSafetyBenchRun:
    case: AgentSafetyBenchConfig
    agent_result: Any
    raw_trace: List[Dict[str, Any]]

async def run_agentsafetybench(
    model: Model,
    case: AgentSafetyBenchConfig,
    agent_config: AgentConfig
) -> AgentSafetyBenchRun:
    tools = ToolRegistry(case.tools)
    runner = AgentRunner(model=model, config=agent_config, tools=tools)
    
    first_user = next(
        (m["content"] for m in case.initial_messages if m["role"] == "user"),
        case.initial_messages[0]["content"],
    )
    
    result = await runner.run(first_user)
    
    trace = []
    for step in result.steps:
        trace.append(step.assistant_message)
        trace.extend(step.tool_results)
    trace.append(result.final_message)
    
    return AgentSafetyBenchRun(
        case=case,
        agent_result=result,
        raw_trace=trace,
    )
    