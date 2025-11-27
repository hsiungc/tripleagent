import asyncio

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

    raw = load_agentharm(
        source="hf",
        hf_name="ai-safety-institute/AgentHarm",
        split="test_public",
        limit=50,
    )
    samples = parse_agentharm(raw)
    samples = attach_tools_agentharm(samples)

    agent_cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=3,
        tool_choice="auto",
        temperature=0.0,
    )

    scores = []
    for sample in samples:
        run = await run_agentharm(sample, model, agent_cfg)
        score = await judge_agentharm(judge_model, run)
        scores.append(score)

    summary = summarize_agentharm(scores)
    print("AgentHarm summary:")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())