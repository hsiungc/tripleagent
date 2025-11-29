import asyncio
import json

from tripleagent.benchmarks.agentsafetybench import (
    load_agentsafetybench,
    parse_agentsafetybench,
    run_agentsafetybench,
)
from tripleagent.agents.runner import AgentConfig
from tripleagent.models.base import Model
from tripleagent.models.factory import create_backend


async def main() -> None:
    # 1) Load just 1 example
    raw_examples = load_agentsafetybench(
        source="local",
        local_path="/workspaces/agentsafety_data.json",  # adjust
        limit=1,
    )
    samples = parse_agentsafetybench(raw_examples)

    sample = samples[0]
    print(f"Testing AgentSafetyBench sample id={sample.id}")

    # 2) Instantiate your model (replace this with your actual constructor)
    # For example, if you have a config-based factory:
    #
    #   model = load_model("configs/models/openai_gpt4o.yaml")
    #
    # or directly:
    #
    #   from tripleagent.models.openai_backend import OpenAIModel
    #   model = OpenAIModel(name="gpt-4o", ...)
    #
    # For now I'll leave this as a placeholder:
    model = ...  # TODO: construct your Model here

    agent_cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_iterations=3,
        tool_choice="auto",
        temperature=0.0,
        max_new_tokens=256,
    )

    # 3) Run a single case
    run = await run_agentsafetybench(
        model=model,
        sample=sample,
        agent_config=agent_cfg,
    )

    print("\n=== Agent final message ===")
    print(run.agent_result.final_message)

    print("\n=== Trace (assistant + tools) ===")
    for step in run.agent_result.steps:
        print(f"\n--- Step {step.step_number} ---")
        print("Assistant:", step.assistant_message)
        print("Tool calls:", json.dumps(step.tool_calls, indent=2))
        print("Tool results:")
        for tr in step.tool_results:
            print(" ", tr)


if __name__ == "__main__":
    asyncio.run(main())
