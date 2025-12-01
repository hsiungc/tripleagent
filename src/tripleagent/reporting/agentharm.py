from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from tripleagent.scoring.agentharm import AgentHarmScore


def summarize_agentharm(scores: List[AgentHarmScore]) -> Dict[str, Any]:
    n = len(scores)

    valid_scores = [s.score for s in scores if s.score is not None]
    avg_score = float(mean(valid_scores)) if valid_scores else 0.0

    full_scores = [
        1.0 if (s.score is not None and s.score == 1.0) else 0.0
        for s in scores
        if s.score is not None
    ]
    avg_full_score = float(mean(full_scores)) if full_scores else 0.0

    refusals = [1.0 if s.refusal else 0.0 for s in scores]
    avg_refusals = float(mean(refusals)) if refusals else 0.0

    non_ref_scores = [
        s.score for s in scores
        if s.score is not None and not s.refusal
    ]
    avg_score_non_refusals = float(mean(non_ref_scores)) if non_ref_scores else 0.0

    cat_scores: Dict[str, List[float]] = {}
    cat_refusals: Dict[str, List[float]] = {}

    for s in scores:
        cat = s.category or "unknown"

        if s.score is not None:
            cat_scores.setdefault(cat, []).append(s.score)

        cat_refusals.setdefault(cat, []).append(1.0 if s.refusal else 0.0)

    category_avg_score: Dict[str, float] = {}
    for cat, vals in cat_scores.items():
        category_avg_score[f"{cat}_avg_scores"] = float(mean(vals)) if vals else 0.0

    category_avg_refusals: Dict[str, float] = {}
    for cat, vals in cat_refusals.items():
        category_avg_refusals[f"{cat}_avg_refusals"] = float(mean(vals)) if vals else 0.0

    return {
        "n_samples": n,
        "avg_score": avg_score,
        "avg_full_score": avg_full_score,
        "avg_refusals": avg_refusals,
        "avg_score_non_refusals": avg_score_non_refusals,
        "category_avg_score": category_avg_score,
        "category_avg_refusals": category_avg_refusals,
    }
