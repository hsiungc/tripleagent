import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml

from tripleagent.models.base import Model
from tripleagent.benchmarks.agentsafetybench import experiment_agentsafetybench
from tripleagent.benchmarks.agentharm import experiment_agentharm
from tripleagent.benchmarks.agentbench import experiment_agentbench


async def main(exp_path: str) -> None:
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    model = Model.from_yaml(exp_path, section="model")
    judge_model = Model.from_yaml(exp_path, section="judge_model")

    benchmarks_cfg = exp_cfg["benchmarks"]

    tripleagent_root = Path(__file__).resolve().parent
    runs_root = tripleagent_root / "outputs" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if "output_dir" in exp_cfg:
        custom = Path(exp_cfg["output_dir"])
        if not custom.is_absolute():
            run_dir = runs_root / custom
        else:
            run_dir = custom
    else:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_dir = runs_root / ts

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_experiment] Writing all benchmark outputs to: {run_dir}")

    summaries = {}

    if benchmarks_cfg.get("agentsafetybench", {}).get("enabled", False):
        summaries["agentsafetybench"] = await experiment_agentsafetybench(
            model=model,
            judge_model=judge_model,
            cfg=benchmarks_cfg["agentsafetybench"],
            run_dir=run_dir,
        )

    if benchmarks_cfg.get("agentharm", {}).get("enabled", False):
        summaries["agentharm"] = await experiment_agentharm(
            model=model,
            judge_model=judge_model,
            cfg=benchmarks_cfg["agentharm"],
            run_dir=run_dir,
        )

    if benchmarks_cfg.get("agentbench", {}).get("enabled", False):
        summaries["agentbench"] = await experiment_agentbench(
            cfg=benchmarks_cfg["agentbench"],
            run_dir=run_dir,
            model=model,
            exp_path=exp_path,
        )

    (run_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print("Combined summary written to", run_dir / "combined_summary.json")


if __name__ == "__main__":
    exp_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/default.yaml"
    asyncio.run(main(exp_path))