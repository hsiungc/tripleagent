import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar

import time
import importlib
import argparse

from os.path import join, isdir, isfile, relpath
from glob import glob

from .src import YAMLConfig, print_rank_0, serialize
from .src.task import Task
from .src.agent import Agent
from .create_assignment import InstanceFactory, Assigment, deep_merge
from .src.utils import ColorMessage

# TripleAgent
from tripleagent.models.config import ModelConfig

def parse_args_to_assignment() -> Assigment:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("evaluation", "Evaluation configurations")

    group.add_argument("--task", type=str, required=True, help="Task config to load")

    # Original AgentBench mode: explicit agent config (YAML or JSON)
    group.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent config to load (YAML path or JSON string)",
    )

    # New: tripleagent experiment YAML
    group.add_argument(
        "--exp-config",
        type=str,
        default=None,
        help="Path to tripleagent experiment YAML (with a 'model' section)",
    )
    group.add_argument(
        "--model-section",
        type=str,
        default="model",
        help="Section name in exp-config to use for model (default: 'model')",
    )

    group.add_argument("--output", type=str, default=None, help="Output root directory")
    group.add_argument("--workers", type=int, default=None, help="Number of workers for evaluation")

    args = parser.parse_args()
    
    if not args.output:
        args.output = "outputs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
        # --- Load TASK as before (YAML path or JSON string) ---
    try:
        task = json.loads(args.task)
        if isinstance(task, str):
            raise Exception()
    except Exception:
        with open(args.task, "r", encoding="utf-8") as f:
            task = yaml.safe_load(f)

    if args.exp_config:
        model_cfg = ModelConfig.from_yaml(
            yaml_path=args.exp_config,
            section=args.model_section,
        )

        provider = (model_cfg.provider or "").lower()
        if provider != "openai":
            raise ValueError(
                "AgentBench HTTPAgent integration currently assumes provider='openai'. "
                f"Got provider={model_cfg.provider!r}."
            )

        api_base = model_cfg.api_base or "https://api.openai.com/v1"
        api_url = api_base.rstrip("/") + "/chat/completions"
        api_key_env = model_cfg.api_key_env or "OPENAI_API_KEY"

        agent = {
            "module": "src.agents.HTTPAgent",
            "parameters": {
                "name": f"tripleagent_{model_cfg.name}",
                "url": api_url,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer ${api_key_env}",
                },
                "body": {
                    "model": model_cfg.name,
                },
                "prompter": {
                    "name": "role_content_dict",
                    "args": {
                        "message_key": "messages",
                        "role_key": "role",
                        "content_key": "content",
                        "user_role": "user",
                        "agent_role": "assistant",
                    },
                },
            },
        }
    else:
        # Original behavior: use --agent (YAML path or JSON)
        if not args.agent:
            raise ValueError(
                "Either --agent OR --exp-config must be provided. You passed neither."
            )

        try:
            agent = json.loads(args.agent)
            if isinstance(agent, str):
                raise Exception()
        except Exception:
            with open(args.agent, "r", encoding="utf-8") as f:
                agent = yaml.safe_load(f)

    print(task)
    print(agent)

    if "workers" not in task["parameters"]:
        task["parameters"]["workers"] = 1
    if args.workers:
        task["parameters"]["workers"] = args.workers
    
    return Assigment(agent=InstanceFactory(**agent), task=InstanceFactory(**task), output=args.output)

# register a signal handler to release task
def get_single_handler(task):
    def signal_handler(sig, frame):
        print(ColorMessage.red(f"Received signal {sig}, exiting ..."))
        if isinstance(task, Task):
            task.release()
        sys.exit(0)
    return signal_handler

def main():
    assignment = parse_args_to_assignment()
    create_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not assignment.output:
        assignment.output = "outputs" + "/" + create_time
        
    os.makedirs(assignment.output, exist_ok=True)
    
    print(ColorMessage.cyan("[Evaluation] Loading Agent ..."))
    agent = assignment.agent.create()
    print(ColorMessage.cyan("[Evaluation] Successfully loaded Agent."))
    print(ColorMessage.cyan("[Evaluation] Loading Task ..."))
    task = assignment.task.create()
    task.output_root_dir = assignment.output
    print(ColorMessage.cyan("[Evaluation] Successfully loaded Task."))
    config_path = os.path.join(assignment.output, "config.json")
    with open(config_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(assignment.to_json(), indent=4, ensure_ascii=False))

    start = time.time()
    # register a signal handler to release task
    import signal
    signal.signal(signal.SIGTERM, get_single_handler(task))
    task.evaluate(agent)
    task.release()
    del task
    print(ColorMessage.cyan(f"[Evaluation] Finish in {time.time() - start:.1f}s"))


if __name__ == "__main__":
    main()
