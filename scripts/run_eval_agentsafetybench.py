import asyncio

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig

from tripleagent.benchmarks.agentsafetybench import (
    load_agentsafetybench,
    parse_agentsafetybench,
    attach_tools_agentsafetybench,
    run_agentsafetybench,
)
from tripleagent.scoring.agentsafetybench import judge_agentsafetybench
from tripleagent.reporting.agentsafetybench import summarize_agentsafetybench


async def main():
    model = Model.from_yaml("configs/models/openai.yaml")
    judge_model = Model.from_yaml("configs/models/openai.yaml")

    # 2. Load raw examples (HF or local)
    raw = load_agentsafetybench(
        source="local",
        hf_name="thu-coai/Agent-SafetyBench",
        split="train",
        limit=50,
    )

    samples = parse_agentsafetybench(raw)
    samples = attach_tools_agentsafetybench(samples)

    agent_cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=1,
        tool_choice="none",
        temperature=0.0,
    )

    # 5. Run benchmark + judge each run
    scores = []

    for sample in samples:
        run = await run_agentsafetybench(model, sample, agent_cfg)
        score = await judge_agentsafetybench(judge_model, run)
        scores.append(score)

    # 6. Summarize results
    summary = summarize_agentsafetybench(scores)
    print("AgentSafetyBench summary:")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
