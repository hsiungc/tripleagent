# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List

# import json

# @dataclass
# class AgentSafetySample:
#     id: int
#     risks: List[str]
#     instruction: str
#     environments: List[Dict[str, Any]]
#     failure_modes: List[int]
#     fulfillable: bool

# def load_agentsafetybench(path: str, limit: int | None = None) -> List[AgentSafetySample]:
#     data = json.loads(Path(path).read_text())
#     if limit is not None:
#         data = data[:limit]
#     samples = []
#     for row in data:
#         samples.append(
#             AgentSafetySample(
#                 id=row["id"],
#                 risks=row["risks"],
#                 instruction=row["instruction"],
#                 environments=row["environments"],
#                 failure_modes=[int(x) for x in row["failure_modes"].split(",")],
#                 fulfillable=bool(row["fulfillable"]),
#             )
#         )
#     return samples
