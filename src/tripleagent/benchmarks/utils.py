import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

Example = Dict[str, Any]


def load_local_json(
    path: str | Path,
    limit: Optional[int] = None,
) -> List[Example]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"{path} is empty.")

    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {path}: {e}") from e

    if isinstance(raw, dict):
        examples: List[Example] = [cast(Example, raw)]
    elif isinstance(raw, list):
        examples = [cast(Example, dict(obj)) for obj in raw]
    else:
        raise ValueError("Unexpected JSON root type. Expected dict or list.")

    if limit is not None:
        examples = examples[:limit]

    return examples