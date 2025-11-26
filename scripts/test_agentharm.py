import asyncio
from pathlib import Path

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks.agentharm_adapter import (
    load_agentharm,
    parse_agentharm,
    run_agentharm,
)


async def main():
    model = Model.from_yaml("configs/models/openai.yaml")  # or HF

    raw = load_agentharm(
        source="local",
        local_path=Path("/workspaces/agentharm_data.jsonl"),
        limit=3,
    )
    samples = parse_agentharm(raw)

    cfg = AgentConfig(
        system_prompt="You are a safety-conscious assistant.",
        max_steps=1,
        tool_choice="none",
        temperature=0.0,
    )

    for s in samples:
        print(f"\n=== AgentHarm case {s.id} | category={s.category} | split={s.split}")
        print("Prompt:", s.prompt[:160], "...")
        run = await run_agentharm(model, s, cfg)
        final = run.agent_result.final_message.get("content", "")
        print("Model answer:", final[:200], "...")


if __name__ == "__main__":
    asyncio.run(main())
