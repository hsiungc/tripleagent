from dataclasses import asdict, is_dataclass
from typing import Any, List


def scores_to_dicts(scores: List[Any]) -> List[dict]:
    out = []
    for s in scores:
        if is_dataclass(s) and not isinstance(s, type):
            out.append(asdict(s))
        elif isinstance(s, dict):
            out.append(s)
        else:
            out.append({"value": str(s)})
    return out
