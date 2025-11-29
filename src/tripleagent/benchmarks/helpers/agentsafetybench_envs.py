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


def build_envs_and_tools(
    raw_example: Dict[str, Any]
) -> Tuple[List[Any], ToolRegistry]:
    envs: List[Any] = []
    tools: List[Tool] = []

    envs_info = raw_example.get("environments") or []
    for env_info in envs_info:
        env_name = env_info.get("name") or ""
        if not env_name:
            continue

        params = env_info.get("parameters") or None
        env = _env_manager.init_env(env_name, params)
        if env is None:
            raise ValueError(
                f"Environment {env_name} not found for example id={raw_example.get('id')}"
            )

        envs.append(env)

        tool_names = env_info.get("tools", [])
        tool_descs = env.get_tool_descs(tool_names)

        for td in tool_descs:
            t_name = td["name"]
            t_desc = td.get("description", "")
            t_params = td.get("parameters", {})

            tools.append(
                EnvTool(
                    name=t_name,
                    description=t_desc,
                    parameters=t_params,
                    env=env,
                )
            )

    registry = ToolRegistry(tools)
    return envs, registry