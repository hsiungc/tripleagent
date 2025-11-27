from typing import Any, Dict


class MCPClient:
    def __init__(self, connection_info: Dict[str, Any]) -> None:
        self.connection_info = connection_info
        # TODO: initialize MCP connection here

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:

        # TODO: implement MCP RPC call
        raise NotImplementedError
