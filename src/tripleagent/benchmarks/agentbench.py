from __future__ import annotations

import json
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from tripleagent.reporting.agentbench import summarize_agentbench
from tripleagent.models.base import Model

TRIPLEAGENT_ROOT = Path(__file__).resolve().parents[1]
AGENTBENCH_ROOT = TRIPLEAGENT_ROOT / "thirdparty" / "agentbench"


@dataclass
class AgentBenchConfig:
    task_config: str
    agent_config: str
    output_dir: str = "outputs/agentbench"
    workers: int = 4


def run_agentbench(
    cfg: AgentBenchConfig,
    output_dir: Path,
) -> Path:
    eval_py = AGENTBENCH_ROOT / "eval.py"

    output_dir = output_dir.resolve()

    cmd = [
        sys.executable,
        str(eval_py),
        "--task",
        cfg.task_config,
        "--agent",
        cfg.agent_config,
        "--output",
        str(output_dir),
        "--workers",
        str(cfg.workers),
    ]

    print("[AgentBench] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=AGENTBENCH_ROOT, check=True)

    return output_dir    

    

async def experiment_agentbench(cfg: dict, run_dir: Path, model: Model) -> dict:
    task_out_dir = run_dir / "agentbench_knowledgegraph"
    metrics_path = task_out_dir / "metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}

    summary = {
        "task": cfg.get("task_name", "knowledgegraph"),
        "metrics": metrics,
    }

    (run_dir / "agentbench_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary