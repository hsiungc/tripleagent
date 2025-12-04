from __future__ import annotations

# import json
# from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentBenchScore:
    task_name: str
    main: float | None
    F1: float | None
    EM: float | None
    executability: float | None
    raw_metrics: Dict[str, Any]


# def summarize_agentbench(output_dir: str | Path) -> Dict[str, Any]:
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         raise FileNotFoundError(output_dir)

#     metrics: Dict[str, Any] = {}
#     for path in output_dir.rglob("*.json"):
#         if path.name == "config.json":
#             continue
#         try:
#             data = json.loads(path.read_text(encoding="utf-8"))
#         except Exception:
#             continue

#         if isinstance(data, dict) and "metrics" in data:
#             metrics[path.stem] = data["metrics"]

#     return {
#         "raw_output_dir": str(output_dir),
#         "tasks": list(metrics.keys()),
#         "metrics": metrics,
#     }