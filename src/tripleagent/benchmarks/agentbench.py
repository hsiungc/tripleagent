from __future__ import annotations

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

    

async def experiment_agentbench(
    cfg: Dict[str, Any],
    run_dir: Path,
    model: Model,
    exp_path: str,
) -> Dict[str, Any]:
    agentbench_root = Path(__file__).resolve().parents[2] / "thirdparty" / "agentbench"
    eval_py = agentbench_root / "eval.py"

    task_config_rel = cfg.get("task_config", "configs/tasks/knowledgegraph/dev.yaml")
    task_cfg_path = agentbench_root / task_config_rel

    # Use the shared run_dir directly
    out_dir = run_dir

    workers = str(cfg.get("workers", 4))
    model_section = cfg.get("model_section", "model")

    cmd = [
        sys.executable,
        str(eval_py),
        "--task",
        str(task_cfg_path),
        "--exp-config",
        str(exp_path),
        "--model-section",
        model_section,
        "--output",
        str(out_dir),
        "--workers",
        workers,
    ]

    print("[AgentBench] Running:", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"AgentBench eval.py exited with code {proc.returncode}.\n"
            f"stdout: {stdout.decode() if stdout else ''}\n"
            f"stderr: {stderr.decode() if stderr else ''}"
        )

    return {
        "output_dir": str(out_dir),
        "task_config": str(task_cfg_path),
        "exp_config": str(exp_path),
        "model_section": model_section,
    }