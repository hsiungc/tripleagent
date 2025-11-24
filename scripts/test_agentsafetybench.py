import asyncio

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks.agentsafetybench import (
    load_agentsafetybench,
    parse_agentsafetybench,
    run_agentsafetybench,
)


async def main():
    model = Model.from_yaml("configs/models/openai.yaml")

    raw_examples = load_agentsafetybench(
        source="local",                     # "hf", "local", or "auto"
        local_path="/workspaces/agentsafety_data.json",
        limit=3,
    )

    samples = parse_agentsafetybench(raw_examples)

    agent_config = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=1,
        tool_choice="none",
        temperature=0.0,
    )

    for case in samples:
        print(f"\n=== Case {case.id} | risk={case.risk_category} | fulfillable={case.fulfillable}")
        print("Prompt:", case.instruction[:120], "...")
        run_result = await run_agentsafetybench(model, case, agent_config)
        final = run_result.agent_result.final_message.get("content", "")
        print("Model answer:", final[:200], "...")


if __name__ == "__main__":
    asyncio.run(main())