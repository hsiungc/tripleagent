import asyncio
import sys

import yaml

from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks import agentharm, agentsafetybench
from tripleagent.models.base import Model

BENCHMARKS = {
    "agentsafety": agentsafetybench,
    "agentharm": agentharm,
}


async def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]

    bench_name = exp_cfg["benchmark"]["name"]
    bench_module = BENCHMARKS[bench_name]

    model = Model.from_yaml(exp_cfg["model_under_test"]["config_path"])
    judge_model = Model.from_yaml(exp_cfg["judge_model"]["config_path"])
    agent_config = AgentConfig(**exp_cfg["agent"])

    summary = await bench_module.run_experiment(
        model=model,
        judge_model=judge_model,
        agent_config=agent_config,
        bench_cfg=exp_cfg["benchmark"],
    )

    print(summary)


if __name__ == "__main__":
    config = sys.argv[1]
    asyncio.run(main(config))
