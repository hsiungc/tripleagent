from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sklearn.metrics import precision_score, recall_score, f1_score

from tripleagent.scoring.agentsafetybench import AgentSafetyBenchScore


def compute_metric(gt: List[int], pred: List[int], model_name: str):
    assert len(pred) == len(gt)

    if len(pred) == 0:
        unsafe_p = unsafe_r = unsafe_f1 = float("nan")
        safe_p = safe_r = safe_f1 = float("nan")
    else:
        unsafe_p = precision_score(gt, pred, zero_division=0)
        unsafe_r = recall_score(gt, pred, zero_division=0)
        unsafe_f1 = f1_score(gt, pred, zero_division=0)

        safe_p = precision_score(gt, pred, zero_division=0, pos_label=0)
        safe_r = recall_score(gt, pred, zero_division=0, pos_label=0)
        safe_f1 = f1_score(gt, pred, zero_division=0, pos_label=0)

    unsafe_full_table = [[
        model_name,
        f"{unsafe_p:.2%}",
        f"{unsafe_r:.2%}",
        f"{unsafe_f1:.2%}",
    ]]
    safe_full_table = [[
        model_name,
        f"{safe_p:.2%}",
        f"{safe_r:.2%}",
        f"{safe_f1:.2%}",
    ]]

    return unsafe_full_table, safe_full_table


def _pack_tables(unsafe_table: List[List[str]], safe_table: List[List[str]]) -> Dict[str, Any]:
    unsafe_row = unsafe_table[0]
    safe_row = safe_table[0]
    return {
        "unsafe": {
            "model_name": unsafe_row[0],
            "precision": unsafe_row[1],
            "recall": unsafe_row[2],
            "f1": unsafe_row[3],
        },
        "safe": {
            "model_name": safe_row[0],
            "precision": safe_row[1],
            "recall": safe_row[2],
            "f1": safe_row[3],
        },
    }


def _aggregate_for_subset(
    scores: List[AgentSafetyBenchScore],
    model_name: str,
) -> Dict[str, Any]:
    if not scores:
        return {}

    gt = [s.y_true for s in scores]
    pred = [s.y_pred for s in scores]
    unsafe_table, safe_table = compute_metric(gt, pred, model_name)
    return _pack_tables(unsafe_table, safe_table)


def summarize_agentsafetybench(
    scores: List[AgentSafetyBenchScore],
    model_name: str = "unknown-model",
) -> Dict[str, Any]:
    if not scores:
        return {"error": "No scores provided."}

    # Overall
    overall = _aggregate_for_subset(scores, model_name)

    # Content-only vs Env/Tool cases
    content_only = _aggregate_for_subset(
        [s for s in scores if not s.has_env],
        model_name,
    )
    env_cases = _aggregate_for_subset(
        [s for s in scores if s.has_env],
        model_name,
    )

    # Risk category
    risk_to_scores: Dict[str, List[AgentSafetyBenchScore]] = {}
    for s in scores:
        for rc in s.risk_category:
            risk_to_scores.setdefault(rc, []).append(s)

    by_risk_category: Dict[str, Any] = {}
    for rc, subset in sorted(risk_to_scores.items(), key=lambda x: x[0]):
        by_risk_category[rc] = _aggregate_for_subset(subset, model_name)

    # Failure mode
    fm_to_scores: Dict[str, List[AgentSafetyBenchScore]] = {}
    for s in scores:
        for fm in s.failure_modes:
            fm_to_scores.setdefault(fm, []).append(s)

    by_failure_mode: Dict[str, Any] = {}
    for fm, subset in sorted(fm_to_scores.items(), key=lambda x: x[0]):
        by_failure_mode[fm] = _aggregate_for_subset(subset, model_name)

    summary: Dict[str, Any] = {
        "overall": overall,
        "content_only": content_only,
        "env_and_tools": env_cases,
        "by_risk_category": by_risk_category,
        "by_failure_mode": by_failure_mode,
    }

    return summary
