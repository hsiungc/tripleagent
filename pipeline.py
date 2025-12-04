import asyncio
from datetime import datetime
import json
import sys
from pathlib import Path

import yaml

from tripleagent.models.base import Model
from tripleagent.benchmarks.agentsafetybench import experiment_agentsafetybench
from tripleagent.benchmarks.agentharm import experiment_agentharm
from tripleagent.benchmarks.agentbench import experiment_agentbench


async def main(exp_path: str) -> None:
    print(f"[run_experiment] Loading experiment config from {exp_path}", flush=True)
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    outputs_root = Path(exp_cfg.get("output_root", "outputs"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = outputs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_experiment] Run directory: {run_dir}", flush=True)

    (run_dir / "experiment_config.json").write_text(
        json.dumps(exp_cfg, indent=2),
        encoding="utf-8",
    )

    print("[run_experiment] Loading models…", flush=True)
    model = Model.from_yaml(exp_path, section="model")
    judge_model = Model.from_yaml(exp_path, section="judge_model")
    print(
        f"[run_experiment] main model={model.name}, judge model={judge_model.name}",
        flush=True,
    )

    benchmarks_cfg = exp_cfg.get("benchmarks", {})
    summaries: dict[str, dict] = {}

    # ---- AgentBench ----
    if benchmarks_cfg.get("agentbench", {}).get("enabled", False):
        print("[run_experiment] Running AgentBench…", flush=True)
        summaries["agentbench"] = await experiment_agentbench(
            cfg=benchmarks_cfg["agentbench"],
            run_dir=run_dir,
            # model=model,
        )
        print("[run_experiment] AgentBench done.", flush=True)
        
    # ---- Agent-SafetyBench ----
    if benchmarks_cfg.get("agentsafetybench", {}).get("enabled", False):
        print("[run_experiment] Running Agent-SafetyBench…", flush=True)
        summaries["agentsafetybench"] = await experiment_agentsafetybench(
            model=model,
            judge_model=judge_model,
            cfg=benchmarks_cfg["agentsafetybench"],
            run_dir=run_dir,
        )
        print("[run_experiment] Agent-SafetyBench done.", flush=True)

    # ---- AgentHarm ----
    if benchmarks_cfg.get("agentharm", {}).get("enabled", False):
        print("[run_experiment] Running AgentHarm…", flush=True)
        summaries["agentharm"] = await experiment_agentharm(
            model=model,
            judge_model=judge_model,
            cfg=benchmarks_cfg["agentharm"],
            run_dir=run_dir,
        )
        print("[run_experiment] AgentHarm done.", flush=True)

    combined_path = run_dir / "combined_summary.json"
    combined_path.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print("[run_experiment] Combined summary written to", combined_path, flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(
                "Usage: python scripts/run_experiment.py configs/experiments/<file>.yaml"
            )
            sys.exit(1)

        asyncio.run(main(sys.argv[1]))
    except Exception as e:
        import traceback

        print("[run_experiment] ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
