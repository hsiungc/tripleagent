# src/tripleagent/benchmarks/agentbench.py

import asyncio
import os
import sys
from pathlib import Path
import json

AGENTBENCH_ROOT = Path(__file__).resolve().parents[2] / "tripleagent" / "thirdparty" / "agentbench"

async def experiment_agentbench(cfg: dict, run_dir: Path) -> dict:
    task_cfg = AGENTBENCH_ROOT / "configs" / "tasks" / "knowledgegraph" / "dev_small.yaml"
    agent_cfg = AGENTBENCH_ROOT / "configs" / "agents" / "api_agents" / "gpt-4.yaml"

    output_dir = run_dir / "agentbench_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",  # unbuffered
        "-m",
        "tripleagent.thirdparty.agentbench.eval",
        "--task",
        str(task_cfg),
        "--agent",
        str(agent_cfg),
        "--output",
        str(output_dir),
    ]

    print("[AgentBench] Running:", " ".join(cmd))
    print("[AgentBench] AgentBench root:", AGENTBENCH_ROOT)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(AGENTBENCH_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        # hard cap runtime so it can't hang forever
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        print("[AgentBench] Timed out, killed subprocess.")
        print("--- AgentBench stdout (partial) ---")
        print(stdout.decode("utf-8", errors="ignore"))
        print("--- AgentBench stderr (partial) ---")
        print(stderr.decode("utf-8", errors="ignore"))
        # record a graceful failure in summary and bail
        summary = {
            "main": None,
            "F1": None,
            "EM": None,
            "executability": None,
            "error": "AgentBench timed out",
        }
        (run_dir / "agentbench_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    if stdout:
        print("--- AgentBench stdout ---")
        print(stdout.decode("utf-8", errors="ignore"))
    if stderr:
        print("--- AgentBench stderr ---")
        print(stderr.decode("utf-8", errors="ignore"))

    if proc.returncode != 0:
        print(f"[AgentBench] eval failed with code {proc.returncode}")
        summary = {
            "main": None,
            "F1": None,
            "EM": None,
            "executability": None,
            "error": f"AgentBench exited with code {proc.returncode}",
        }
        (run_dir / "agentbench_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    # if AgentBench writes a metrics JSON, load it; otherwise, just stub something
    metrics_path = output_dir / "knowledgegraph_metrics.json"
    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "main": 0.0,
            "F1": 0.0,
            "EM": 0.0,
            "executability": 0.0,
        }

    (run_dir / "agentbench_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


# import asyncio
# import json
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict

# from tripleagent.models.base import Model

# Example = Dict[str, Any]


# @dataclass
# class AgentBenchConfig:
#     task_config: str
#     agent_config: str
#     output_dir: str
#     main: float | None
#     F1: float | None
#     EM: float | None
#     executability: float | None


# def _find_agentbench_root() -> Path:
#     return Path(__file__).resolve().parents[1] / "thirdparty" / "agentbench"


# @dataclass
# class AgentBenchResult:
#     task_name: str
#     metrics: Dict[str, float]
#     raw_metrics_path: Path


# async def experiment_agentbench(
#     cfg: dict,
#     run_dir: Path,
#     model: Model,  # currently unused: AgentBench still uses its own agent YAML
# ) -> dict:
#     agentbench_root = _find_agentbench_root()

#     task_cfg_rel = cfg.get("task_config")
#     agent_cfg_rel = cfg.get("agent_config")

#     if not task_cfg_rel or not agent_cfg_rel:
#         raise ValueError(
#             "[AgentBench] 'task_config' and 'agent_config' must be set in "
#             "benchmarks.agentbench in your experiment YAML."
#         )

#     task_cfg = agentbench_root / task_cfg_rel
#     agent_cfg = agentbench_root / agent_cfg_rel

#     if not task_cfg.exists():
#         raise FileNotFoundError(f"[AgentBench] Task config not found: {task_cfg}")
#     if not agent_cfg.exists():
#         raise FileNotFoundError(f"[AgentBench] Agent config not found: {agent_cfg}")

#     ab_output_dir = run_dir / "agentbench_outputs"
#     ab_output_dir.mkdir(parents=True, exist_ok=True)

#     cmd = [
#         sys.executable,
#         "-m",
#         "tripleagent.thirdparty.agentbench.eval",
#         "--task",
#         str(task_cfg),
#         "--agent",
#         str(agent_cfg),
#         "--output",
#         str(ab_output_dir),
#     ]

#     print("[AgentBench] Running:", " ".join(cmd))
#     print("[AgentBench] AgentBench root:", agentbench_root)

#     proc = await asyncio.create_subprocess_exec(
#         *cmd,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#     )
#     stdout, stderr = await proc.communicate()
#     stdout_txt = stdout.decode("utf-8", errors="ignore")
#     stderr_txt = stderr.decode("utf-8", errors="ignore")

#     if stdout_txt:
#         print("[AgentBench stdout]")
#         print(stdout_txt)
#     if stderr_txt:
#         print("[AgentBench stderr]")
#         print(stderr_txt)

#     if proc.returncode != 0:
#         raise RuntimeError(
#             f"AgentBench eval failed with code {proc.returncode}\n"
#             f"--- AgentBench stdout ---\n{stdout_txt}\n"
#             f"--- AgentBench stderr ---\n{stderr_txt}"
#         )

#     metrics_path = ab_output_dir / "results.json"
#     if not metrics_path.exists():
#         raise FileNotFoundError(
#             f"[AgentBench] results.json not found in {ab_output_dir}. "
#             "Check AgentBench's output structure."
#         )

#     metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

#     summary_path = run_dir / "agentbench_summary.json"
#     summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
#     print("[AgentBench] Metrics written to", summary_path)

#     return metrics