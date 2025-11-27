from collections import Counter
from typing import Any, Dict, List

from tripleagent.benchmarks.agentharm import AgentHarmRun
from tripleagent.scoring.agentharm import AgentHarmScore


def summarize_agentharm(scores: List[AgentHarmScore]) -> Dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {}

    harmful_count = sum(1 for s in scores if s.harmful)
    refused_count = sum(1 for s in scores if s.refused)

    harmful_by_cat = Counter()
    for s in scores:
        if s.harmful:
            harmful_by_cat[s.category] += 1

    return {
        "total_cases": total,
        "harmful_rate": harmful_count / total,
        "refusal_rate": refused_count / total,
        "harmful_by_category": dict(harmful_by_cat),
    }
