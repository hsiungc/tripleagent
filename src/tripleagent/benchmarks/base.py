from typing import Any, Dict, Protocol

from tripleagent.agents.runner import AgentConfig
from tripleagent.models.base import Model


class BenchmarkAdapter(Protocol):
    name: str

    async def run_experiment(
        self,
        model: Model,
        judge_model: Model,
        agent_config: AgentConfig,
        bench_cfg: Dict[str, Any],
    ) -> Dict[str, Any]: ...
