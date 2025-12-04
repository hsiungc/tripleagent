# import importlib
# import sys
# import traceback
# from copy import deepcopy

# class EnvManager:
#     def __init__(self):
#         pass
    
#     def init_env(self, env_name, env_params):
#         # print(sys.path)
#         try:
#             env_module = importlib.import_module(env_name)
#         except Exception as e:
#             print(e)
#             traceback.print_exc()
#             return None
#         # print(env_module)
#         env = getattr(env_module, env_name)
        
#         return env(parameters=deepcopy(env_params))


from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional


class EnvManager:
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def init_env(self, name: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if name in self._cache:
            EnvClass = self._cache[name]
        else:
            try:
                module = import_module(
                    f"tripleagent.thirdparty.agentsafetybench.environments.{name}"
                )
            except ModuleNotFoundError:
                print(
                    f"[EnvManager] Module for environment '{name}' not found "
                    f"in tripleagent.thirdparty.agentsafetybench.environments."
                )
                return None

            EnvClass = getattr(module, name, None)
            if EnvClass is None:
                print(
                    f"[EnvManager] Class '{name}' not found in module "
                    f"tripleagent.thirdparty.agentsafetybench.environments.{name}"
                )
                return None

            self._cache[name] = EnvClass

        return EnvClass(parameters=parameters or {})
