from dataclasses import dataclass
from typing import Any, Dict

from tripleagent.agents.tools import Tool
from .client import MCPClient


@dataclass
class MCPTool(Tool):
    name: str
    description: str
    parameters: Dict[str, Any]
    server_id: str
    client: MCPClient
    
    async def __call__(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self.client.call_tool(
            server_id=self.server_id,
            tool_name=self.name,
            arguments=arguments,
        )
    