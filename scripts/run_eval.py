# scripts/run_eval.py
import asyncio
from tripleagent.config import ModelConfig
from tripleagent.models.base import Model
from tripleagent.data.agentsafetybench import load_agentsafetybench
from tripleagent.runner import AgentRunner  # your loop class

async def main():
    model_cfg = ModelConfig.from_yaml("configs/models/openai.yaml")
    model = Model.from_yaml("configs/models/openai.yaml")

    samples = load_agentsafetybench("workspaces/agentsafety_data.jsonl", limit=1)
    runner = AgentRunner(model=model, config=...)  # your agent loop config

    await runner.run_samples(samples, output_path="runs/gpt4o_agentsafetybench_50.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
