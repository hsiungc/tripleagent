from __future__ import annotations

import inspect as py_inspect

from typing import Any, Dict, List, Tuple, Literal

from tripleagent.agents.tools import ToolRegistry
from tripleagent.thirdparty.agentharm import harmful_tools, benign_tools


class AgentHarmTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        fn: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn
    
    async def __call__(self, args: Dict[str, Any]) -> Any:
        if py_inspect.iscoroutinefunction(self._fn):
            return await self._fn(**args)
        else:
            return self._fn(**args)
        

def _build_parameters_from_signature(fn: Any) -> Dict[str, Any]:
    sig = py_inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        param_type = "string"
        if param.annotation in (int, "int"):
            param_type = "integer"
        elif param.annotation in (float, "float"):
            param_type = "number"
        elif param.annotation in (bool, "bool"):
            param_type = "boolean"

        props[name] = {
            "type": param_type,
        }
        if param.default is param.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": props,
        "required": required,
    }
    
    
def build_tools_for_sample(
    raw_example: Dict[str, Any],
    task_name: Literal["harmful", "benign", "chat"],
    n_irrelevant_tools: int = 0,
) -> ToolRegistry:
    registry = ToolRegistry()

    module = harmful_tools if task_name == "harmful" else benign_tools

    target_fns = raw_example.get("target_functions") or []
    if not isinstance(target_fns, list):
        target_fns = [target_fns]

    for tool_name in target_fns:
        if not hasattr(module, tool_name):
            print(f"[AgentHarm] Warning: tool '{tool_name}' not found in {task_name}_tools.")
            continue

        tool_factory = getattr(module, tool_name)
        inner_tool = tool_factory()

        desc = (inner_tool.__doc__ or "").strip()
        params = _build_parameters_from_signature(inner_tool)

        wrapped = AgentHarmTool(
            name=tool_name,
            description=desc,
            parameters=params,
            fn=inner_tool,
        )

        registry._tools[tool_name] = wrapped

    print(
        f"[AgentHarm] Built {len(registry.list_tools())} tools for sample id={raw_example.get('id')} ({task_name})."
    )
    return registry