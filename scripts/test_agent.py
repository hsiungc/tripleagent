import asyncio
from pathlib import Path

import yaml

from tripleagent.agents.example_tools import EchoTool
from tripleagent.agents.runner import AgentConfig, AgentRunner
from tripleagent.agents.tools import ToolRegistry
from tripleagent.models.base import Model

ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "configs" / "models" / "openai.yaml"

with CONFIG_PATH.open("r") as f:
    config_data = yaml.safe_load(f)
    print("Loaded model config")
    print("Exists?     =", CONFIG_PATH.exists())


async def main():
    model = Model.from_yaml(CONFIG_PATH)

    tools = ToolRegistry([EchoTool()])

    config = AgentConfig(
        system_prompt="You are a helpful assistant that may call tools.",
        max_iterations=4,
        tool_choice="auto",
        temperature=0.0,
    )

    runner = AgentRunner(
        model=model,
        tools=tools,
        config=config,
    )

    user_prompt = (
        "Call the echo tool with the text 'hello from tool land', "
        "then explain what the tool returned."
    )

    result = await runner.run(user_prompt)

    print("Final assistant message:\n", result.final_message)
    print("\nSteps:")
    for step in result.steps:
        print(f"- Step {step.step_number}")
        print("  assistant:", step.assistant_message)
        print("  tool_calls:", step.tool_calls)
        print("  tool_results:", step.tool_results)
    print("\nUsage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())
