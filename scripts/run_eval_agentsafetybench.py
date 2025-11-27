import asyncio

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks.agentsafetybench import (
    load_agentsafetybench,
    parse_agentsafetybench,
    attach_tools_agentsafetybench,
    run_agentsafetybench,
)


async def main():
    model = Model.from_yaml("configs/models/openai.yaml")

    raw = load_agentsafetybench(source="local", limit=3)
    samples = parse_agentsafetybench(raw)
    samples = attach_tools_agentsafetybench(samples)

    cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=1,
        tool_choice="none",
        temperature=0.0,
    )

    for s in samples:
        run = await run_agentsafetybench(model, s, cfg)
        print(s.id, run.agent_result.final_message.get("content", "")[:200])

if __name__ == "__main__":
    asyncio.run(main())
