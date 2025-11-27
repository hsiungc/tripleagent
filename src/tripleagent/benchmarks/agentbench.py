from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TRIPLEAGENT_ROOT = Path(__file__).resolve().parents[1]
AGENTBENCH_ROOT = TRIPLEAGENT_ROOT / "thirdparty" / "agentbench"


@dataclass
class AgentBenchConfig:
    task_config: str
    agent_config: str
    output_dir: str = "outputs/agentbench"
    workers: int = 4


def run_agentbench(cfg: AgentBenchConfig) -> None:
    eval_py = AGENTBENCH_ROOT / "eval.py"

    cmd = [
        sys.executable,
        str(eval_py),
        "--task",
        str(AGENTBENCH_ROOT / cfg.task_config),
        "--agent",
        str(AGENTBENCH_ROOT / cfg.agent_config),
        "--output",
        str(AGENTBENCH_ROOT / cfg.output_dir),
        "--workers",
        str(cfg.workers),
    ]

    print("[AgentBench] Running:", " ".join(cmd))

    env = os.environ.copy()
    agentbench_pythonpath = str(AGENTBENCH_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = agentbench_pythonpath + os.pathsep + existing
    else:
        env["PYTHONPATH"] = agentbench_pythonpath

    subprocess.run(
        cmd,
        check=True,
        cwd=str(AGENTBENCH_ROOT),
        env=env,
    )