from tripleagent.benchmarks.agentharm_adapter import (
    load_agentharm,
    parse_agentharm,
    run_agentharm,
)
from tripleagent.scoring.agentharm import judge_agentharm_run
from tripleagent.reporting.agentharm import summarize_agentharm


async def agentharm_exp(cfg: dict) -> None:
    exp = cfg["experiment"]

    model = Model.from_yaml(exp["model_under_test"]["config_path"])
    judge_model = Model.from_yaml(exp["judge_model"]["config_path"])

    raw = load_agentharm(
        source=exp["benchmark"]["source"],
        local_path=exp["benchmark"]["local_path"],
        limit=exp["benchmark"]["limit"],
    )
    samples = parse_agentharm(raw)

    a_cfg = AgentConfig(**exp["agent"])

    scores = []
    for sample in samples:
        run = await run_agentharm(model, sample, a_cfg)
        score = await judge_agentharm_run(judge_model, run)
        scores.append(score)

    summary = summarize_agentharm(scores)
    print(summary)


# if __name__ == "__main__":
#     bench_name = cfg["experiment"]["benchmark"]["name"]
# if bench_name == "agentsafety":
#     await agentsafety_exp(cfg)
# elif bench_name == "agentharm":
#     await agentharm_exp(cfg)
# else:
#     raise NotImplementedError(f"Unknown benchmark {bench_name}")
