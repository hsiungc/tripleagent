from openai import OpenAI
from ...agent import Agent
import os
import json
from copy import deepcopy
from typing import List
import httpx


class OpenAIChatCompletion(Agent):
    def __init__(self, api_args=None, **config):
        if not api_args:
            api_args = {}
        print("[OpenAIChatCompletion] api_args =", api_args)
        print("[OpenAIChatCompletion] config   =", config)

        api_args = deepcopy(api_args)

        # --- API key ---
        api_key = api_args.pop("key", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set api_args.key or OPENAI_API_KEY."
            )

        # --- Base URL (optional) ---
        api_base = api_args.pop("base", None) or os.getenv("OPENAI_API_BASE") or None
        if api_base and not api_base.startswith("http"):
            print(
                f"[OpenAIChatCompletion] Ignoring invalid OPENAI_API_BASE: {api_base}"
            )
            api_base = None

        # --- Model name ---
        model_name = api_args.pop("model", None)
        if not model_name:
            raise ValueError("OpenAI model is required (api_args.model).")

        self.model_name = model_name

        # Optional: allow api_args to override max_tokens/temperature, but keep them small
        self.api_args = {
            "max_tokens": api_args.pop("max_tokens", 256),
            "temperature": api_args.pop("temperature", 0.0),
            **api_args,
        }

        print("[OpenAIChatCompletion] Using model:", self.model_name)
        print("[OpenAIChatCompletion] Base URL   :", api_base or "<default>")

        # Short-ish timeout so it can't hang forever
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=30.0,  # seconds
        )

        super().__init__(**config)

    def inference(self, history: List[dict]) -> str:
        return "TEMP_DUMMY_ANSWER"
        # AgentBench uses role="agent"; map to "assistant"
        # messages = json.loads(json.dumps(history))
        # for m in messages:
        #     if m.get("role") == "agent":
        #         m["role"] = "assistant"

        # print(
        #     f"[OpenAIChatCompletion] Calling OpenAI with "
        #     f"{len(messages)} messages, api_args={self.api_args}"
        # )

        # try:
        #     resp = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=messages,
        #         **self.api_args,
        #     )
        # except httpx.TimeoutException:
        #     print("[OpenAIChatCompletion] Request timed out.")
        #     return "I'm sorry, but I could not complete this request due to a timeout."
        # except Exception as e:
        #     print("[OpenAIChatCompletion] Error while calling OpenAI:", repr(e))
        #     return f"ERROR: {e}"

        # if not resp.choices:
        #     print("[OpenAIChatCompletion] No choices returned.")
        #     return ""

        # content = resp.choices[0].message.content or ""
        # print(
        #     "[OpenAIChatCompletion] Response length:",
        #     len(content),
        # )
        # return content


class OpenAICompletion(Agent):
    def __init__(self, api_args=None, **config):
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)

        api_key = api_args.pop("key", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required; set api_args.key or OPENAI_API_KEY."
            )

        model_name = api_args.pop("model", None)
        if not model_name:
            raise ValueError(
                "OpenAI model is required, please assign api_args.model."
            )

        if "base" in api_args:
            api_args.pop("base")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.api_args = api_args

        super().__init__(**config)

    def inference(self, history: List[dict]) -> str:
        prompt = ""
        for h in history:
            role = "Assistant" if h["role"] == "agent" else h["role"].title()
            content = h["content"]
            prompt += f"{role}: {content}\n\n"
        prompt += "Assistant: "

        resp = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self.api_args,
        )

        return resp.choices[0].text or ""
    
    
# import openai
# from ...agent import Agent
# import os
# import json
# import sys
# import time
# import re
# import math
# import random
# import datetime
# import argparse
# import requests
# from typing import List, Callable
# import dataclasses
# from copy import deepcopy


# class OpenAIChatCompletion(Agent):
#     def __init__(self, api_args=None, **config):
#         if not api_args:
#             api_args = {}
#         print("api_args={}".format(api_args))
#         print("config={}".format(config))
        
#         api_args = deepcopy(api_args)
#         api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
#         if not api_key:
#             raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
#         os.environ['OPENAI_API_KEY'] = api_key
#         openai.api_key = api_key
#         print("OpenAI API key={}".format(openai.api_key))
#         api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
#         if api_base:
#             os.environ['OPENAI_API_BASE'] = api_base
#             openai.api_base = api_base
#         print("openai.api_base={}".format(openai.api_base))
#         api_args["model"] = api_args.pop("model", None)
#         if not api_args["model"]:
#             raise ValueError("OpenAI model is required, please assign api_args.model.")
#         self.api_args = api_args
#         super().__init__(**config)

#     def inference(self, history: List[dict]) -> str:
#         history = json.loads(json.dumps(history))
#         for h in history:
#             if h['role'] == 'agent':
#                 h['role'] = 'assistant'

#         resp = openai.ChatCompletion.create(
#             messages=history,
#             **self.api_args
#         )

#         return resp["choices"][0]["message"]["content"]


# class OpenAICompletion(Agent):
#     def __init__(self, api_args=None, **config):
#         if not api_args:
#             api_args = {}
#         api_args = deepcopy(api_args)
#         api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
#         if not api_key:
#             raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
#         os.environ['OPENAI_API_KEY'] = api_key
#         openai.api_key = api_key
#         print("OpenAI API key={}".format(openai.api_key))
#         api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
#         if api_base:
#             os.environ['OPENAI_API_BASE'] = api_base
#             openai.api_base = api_base
#         print("openai.api_base={}".format(openai.api_base))
#         api_args["model"] = api_args.pop("model", None)
#         if not api_args["model"]:
#             raise ValueError("OpenAI model is required, please assign api_args.model.")
#         self.api_args = api_args
#         super().__init__(**config)

#     def inference(self, history: List[dict]) -> str:
#         prompt = ""
#         for h in history:
#             role = 'Assistant' if h['role'] == 'agent' else h['role']
#             content = h['content']
#             prompt += f"{role}: {content}\n\n"
#         prompt += 'Assistant: '

#         resp = openai.Completion.create(
#             prompt=prompt,
#             **self.api_args
#         )

#         return resp["choices"][0]["text"]


