# tripleagent/agent/example_tools.py
from typing import Any, Dict


class EchoTool:
    """
    A trivial tool that just echoes back the input text.
    Great for validating the tool-calling loop.
    """

    name = "echo"
    description = "Echoes back the given text."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo back.",
            }
        },
        "required": ["text"],
    }

    async def __call__(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        text = arguments.get("text", "")
        return {
            "echo": text,
        }
