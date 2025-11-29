from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def make_run_dir(run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = run_name or "run"
    run_dir = RUNS_ROOT / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
