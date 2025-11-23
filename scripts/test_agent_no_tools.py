# scripts/test_agent_no_tools.py
import asyncio

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentRunner, AgentConfig
from tripleagent.agents.tools import ToolRegistry


async def main():
    model = Model.from_yaml("configs/models/openai_gpt4o.yaml")

    # No tools
    tools = ToolRegistry([])

    config = AgentConfig(
        system_prompt="You are a helpful assistant.",
        max_steps=2,
        tool_choice="none",
        temperature=0.0,
    )

    runner = AgentRunner(
        model=model,
        tools=tools,
        config=config,
    )

    result = await runner.run("Give a one-sentence explanation of what this project does.")
    print("Final assistant message:\n", result.final_message)


if __name__ == "__main__":
    asyncio.run(main())
