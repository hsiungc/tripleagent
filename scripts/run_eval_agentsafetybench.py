# scripts/run_experiment.py
import asyncio
import yaml

from tripleagent.models.base import Model
from tripleagent.agents.runner import AgentConfig
from tripleagent.benchmarks.agentsafetybench_adapter import (
    load_agentsafetybench,
    parse_agentsafetybench,
    attach_tools,
    run_agentsafetybench,
)
from tripleagent.scoring.agentsafetybench import judge_agentsafetybench
from tripleagent.reporting.agentsafetybench import summarize_agentsafetybench


async def agentsafetybench_exp(cfg: dict) -> None:
    exp = cfg["experiment"]

    model = Model.from_yaml(exp["model_under_test"]["config_path"])
    judge_model = Model.from_yaml(exp["judge_model"]["config_path"])

    raw = load_agentsafetybench(
        source=exp["benchmark"]["source"],
        local_path=exp["benchmark"]["local_path"],
        limit=exp["benchmark"]["limit"],
    )
    samples = parse_agentsafetybench(raw)
    # samples = attach_tools(samples)

    a_cfg = AgentConfig(**exp["agent"])

    scores = []
    for case in samples:
        run = await run_agentsafetybench(model, case, a_cfg)
        score = await judge_agentsafetybench(judge_model, run)
        scores.append(score)

    summary = summarize_agentsafetybench(scores)
    print(summary)


async def main(path: str) -> None:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    bench_name = cfg["experiment"]["benchmark"]["name"]
    if bench_name == "agentsafety":
        await agentsafetybench_exp(cfg)
    else:
        raise NotImplementedError(f"Unknown benchmark {bench_name}")


if __name__ == "__main__":
    import sys
    asyncio.run(main(sys.argv[1]))