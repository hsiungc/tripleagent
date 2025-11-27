import itertools
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models.base import Model
from .tools import ToolRegistry

Message = Dict[str, Any]


@dataclass
class AgentConfig:
    system_prompt: str
    max_iterations: int = 0
    tool_choice: Optional[str] = "auto"
    model_name_override: Optional[str] = None
    temperature: float = 0.0
    max_new_tokens: int = 512


@dataclass
class AgentStep:
    step_number: int
    assistant_message: Message
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Message]


@dataclass
class AgentRunResult:
    final_message: Message
    steps: List[AgentStep]
    usage: List[Dict[str, Any]]


class AgentRunner:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        config: AgentConfig,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config

    async def run(self, user_input: str) -> AgentRunResult:
        messages: List[Message] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_input},
        ]

        steps: List[AgentStep] = []
        usage_stats: List[Dict[str, Any]] = []
        tools_schema = self.tools.get_tool_specs()

        if self.config.max_iterations > 0:
            iterator = range(1, self.config.max_iterations + 1)
        else:
            iterator = itertools.count(1)

        for step_num in iterator:
            response = await self.model.backend.chat(
                messages,
                tools=tools_schema if tools_schema else None,
                tool_choice=self.config.tool_choice if tools_schema else None,
                model=self.config.model_name_override or self.model.name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
            )

            assistant_message = response["response"]
            usage = response.get("usage", {})
            if usage:
                usage_stats.append(usage)

            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                messages.append(assistant_message)
                return AgentRunResult(
                    final_message=assistant_message,
                    steps=steps,
                    usage=usage_stats,
                )

            tool_results: List[Message] = []
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                raw_args = fn.get("arguments", {})

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}

                tool = self.tools.get(name)
                result = await tool(args)

                tool_message: Message = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result),
                }
                tool_results.append(tool_message)

            step = AgentStep(
                step_number=step_num,
                assistant_message=assistant_message,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            steps.append(step)

            messages.append(assistant_message)
            messages.extend(tool_results)

            if not tool_calls:
                break

        final_message = (
            steps[-1].assistant_message
            if steps
            else {
                "role": "assistant",
                "content": "No response generated.",
            }
        )

        return AgentRunResult(
            final_message=final_message,
            steps=steps,
            usage=usage_stats,
        )

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Message:
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        result_content = f"Executed {tool_name} with arguments {json.dumps(arguments)}"

        return {
            "role": "tool",
            "content": result_content,
        }
