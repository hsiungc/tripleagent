from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tripleagent.agents.tools import Tool, ToolRegistry
from tripleagent.thirdparty.agentsafetybench.environments import EnvManager

_env_manager = EnvManager()


class EnvTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        env: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._env = env

    async def __call__(self, args: Dict[str, Any]) -> Any:
        print(f"[EnvTool] {self.name} called with args={args}")
        return self._env.call_tool(self.name, args)


def build_envs_and_tools(raw_example: Dict[str, Any]) -> Tuple[List[Any], ToolRegistry]:
    envs: List[Any] = []
    registry = ToolRegistry()

    envs_info = raw_example.get("environments") or []
    for env_info in envs_info:
        name = env_info.get("name") or ""
        if not name:
            continue

        params = env_info.get("parameters") or None
        env = _env_manager.init_env(name, params)
        if env is None:
            print(
                f"[AgentSafetyBench] Warning: environment '{name}' "
                f"not registered; treating sample id={raw_example.get('id')} as partially content-only."
            )
            continue

        envs.append(env)

        tool_names = env_info.get("tools", [])
        tool_descs = env.get_tool_descs(tool_names)

        for td in tool_descs:
            tool_name = td["name"]
            description = td.get("description", "")
            parameters = td.get("parameters", {})

            tool_obj = EnvTool(
                name=tool_name,
                description=description,
                parameters=parameters,
                env=env,
            )

            registry.register(tool_obj)

    print(
        f"[AgentSafetyBench] Built {len(envs)} envs and "
        f"{len(registry.list_tools())} tools for sample id={raw_example.get('id')}"
    )
    return envs, registry