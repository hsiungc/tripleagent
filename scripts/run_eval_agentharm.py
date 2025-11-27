import asyncio
import sys

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks.agentharm import (
    load_agentharm,
    parse_agentharm,
    attach_tools_agentharm,
    run_agentharm,
)
from tripleagent.scoring.agentharm import judge_agentharm
from tripleagent.reporting.agentharm import summarize_agentharm


async def main():
    model = Model.from_yaml("configs/models/openai.yaml")
    judge_model = Model.from_yaml("configs/models/openai.yaml")

    raw_behaviors = load_agentharm(
        source="hf",           # or "auto" / "local"
        hf_name="ai-safety-institute/AgentHarm",
        split="test_public",
        local_path="/workspaces/agentharm_behaviors.json",
        limit=50,
    )

    samples = parse_agentharm(raw_behaviors)

    samples = attach_tools_agentharm(samples)

    # 5. Configure your agent
    agent_cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=3,
        tool_choice="auto",
        temperature=0.0,
    )

    # 6. Run eval
    scores = []

    for sample in samples:
        run = await run_agentharm(sample, model, agent_cfg)
        score = await judge_agentharm(judge_model, run)
        scores.append(score)

    # 7. Summarize results
    summary = summarize_agentharm(scores)
    print("AgentHarm summary:")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
