import json
from typing import Any, Dict, List

from tripleagent.models.base import Model


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(chunk.get("text", ""))
        return "\n".join(parts)

    return str(content)


def trace_to_text(raw_trace: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in raw_trace:
        role = msg.get("role", "assistant")
        content = extract_text_from_content(msg.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def call_json_judge(
    judge_model: Model,
    prompt: str,
) -> Dict[str, Any]:
    resp = await judge_model.backend.chat(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
        tool_choice=None,
    )
    msg = resp.get("message", {})
    content = extract_text_from_content(msg.get("content", ""))

    try:
        data: Dict[str, Any] = json.loads(content)
    except Exception:
        data = {"parse_error": True, "raw": content}

    return data
