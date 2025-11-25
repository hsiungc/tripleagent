from collections import Counter
from typing import List, Dict, Any

from tripleagent.scoring.agentsafetybench import AgentSafetyBenchScore


def summarize_agentsafetybench(scores: List[AgentSafetyBenchScore]) -> Dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {}
    
    safe_count = sum(1 for s in scores if s.safe)
    helpful_count = sum(1 for s in scores if s.helpful)
    
    safe_by_risk = Counter()
    for s in scores:
        if s.safe:
            safe_by_risk[s.risk_category] += 1
    
    return {
        "total_cases": total,
        "safe_rate": safe_count / total,
        "helpful_rate": helpful_count / total,
        "safe_by_risk_category": dict(safe_by_risk),
    }