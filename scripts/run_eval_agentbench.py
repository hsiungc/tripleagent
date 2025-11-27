# scripts/run_eval_agentbench.py

import argparse
import pathlib
import sys
from typing import Any, Dict

import yaml

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tripleagent.benchmarks.agentbench import AgentBenchConfig, run_agentbench 


def load_agentbench_config(path: str | pathlib.Path) -> AgentBenchConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    bench = cfg.get("benchmark", {})
    return AgentBenchConfig(
        task_config=bench["task_config"],
        agent_config=bench["agent_config"],
        output_dir=bench.get("output_dir", "outputs/agentbench"),
        workers=bench.get("workers", 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentBench evaluation.")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/experiments/agentbench.yaml",
        help="Path to experiment config YAML.",
    )
    args = parser.parse_args()

    cfg = load_agentbench_config(args.config)
    run_agentbench(cfg)


if __name__ == "__main__":
    main()
