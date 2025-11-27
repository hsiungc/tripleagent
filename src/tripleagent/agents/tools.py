from typing import Any, Dict, List, Protocol, runtime_checkable

Message = Dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: Dict[str, Any]

    async def __call__(self, args: Dict[str, Any]) -> Any: ...


class ToolRegistry:
    def __init__(self, tools: List[Tool] | None = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            specs.append(spec)
        return specs

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())
