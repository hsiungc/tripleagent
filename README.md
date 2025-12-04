# TripleAgent

A Unified Capability, Behavioral Safety, and Harmful-Compliance Evaluation Pipeline for LLM Agents.

TripleAgent is a unified, YAML-driven evaluation pipeline designed to test LLM agents across capability, behavioral safety, and harmful-compliance dimensions. The framework integrates multiple external benchmarks under a single interface while sharing:

* Model backends

* Tooling and agent-runner infrastructure

* Scoring and reporting layers

* A consistent experiment configuration flow

The objective is to make it easy to compare model performance across radically different types of agent evaluations.


## Benchmarks

The following benchmarks are used in TripleAgent:

#### **AgentBench (Capabilities)**

* Evaluates agent reasoning, planning, and tool-based action execution across diverse environments.

* TripleAgent wraps and launches the benchmark through a unified config (KnowledgeGraph demo included).

* Full fidelity requires external services (SPARQL/Freebase, simulators), so current integration focuses on pipeline wiring + metrics parsing.


#### **AgentSafetyBench (Behavioral Safety)**

* Tests whether agents behave safely in scenario-driven, tool-enabled environments.

* Each scenario tests behavioral safety risks (e.g., data leakage, harmful commands, unauthorized actions) by observing what the agent actually does, not just what it says.

* Supports environment + tool wiring and per-risk-category safety metrics.


#### **AgentHarm**

* Challenges models with adversarial prompts designed to induce unsafe or harmful actions.

* Uses a refusal judge (implemented) and semantic judge (partially integrated) per the paper.

* Reports compliance/refusal metrics including avg_score, avg_refusals, and category-level safety.


## Current Repository Snapshot (Dec 2025)

TripleAgent integrates three major safety/capability benchmarks with partial fidelity:

* AgentBench: scaffolding + launcher + output parsing for selected environments (e.g., KnowledgeGraph).

* AgentSafetyBench: end-to-end pipeline for scenario-based safety classification; environment and tool integration included.

* AgentHarm: end-to-end harmful-compliance evaluation including refusal scoring and category metrics (semantic scoring partially implemented).



## Repository Structure

```
tripleagent/
  configs/
    experiments/
      agentharm.yaml
      agentsafetybench.yaml
      agentbench.yaml
  src/
    agents/
    tripleagent/
      benchmarks/
        agentharm.py
        agentsafetybench.py
        agentbench.py
      models/
      reporting/
      scoring/
      thirdparty/
        agentbench/
        agentharm/
        agentsafetybench/
  pipeline.py
  pyproject.toml
```

## Instructions

**1. Prerequisites**

* Python ≥ 3.10

* ```pip``` and ```venv``` (or your preferred env manager)

* API keys for any model backends you plan to use, e.g.: OPENAI_API_KEY
    * (Optional) HF token, etc.

**2. Clone & create a virtual environment**
```
git clone <your-repo-url> tripleagent
cd tripleagent

python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

To install using the pyproject.toml file:
```
pip install -e .
```

**4. Configure API keys**

For example, with OpenAI:
```
export OPENAI_API_KEY="sk-..."
```

Set any other provider keys as required by your model configs.


## Running TripleAgent
TripleAgent is a pick-and-choose pipeline. Select a benchmark in the experiment YAML and run the corresponding script.

Each experiment config lives under ```configs/```.


**1. AgentBench**

Run a scaffolded AgentBench v0.1 experiment (KnowledgeGraph):
```
python -m pipeline configs/agentbench.yaml
```

**2. AgentSafetyBench**

Run an AgentSafetyBench experiment:
```
python -m pipeline configs/agentsafetybench.yaml
```

**3. AgentHarm**

Run an AgentHarm experiment:
```
python -m pipeline configs/agentharm.yaml
```

## Citations

**AgentBench**
```
@article{liu2023agentbench,
  title   = {AgentBench: Evaluating LLMs as Agents},
  author  = {Xiao Liu and Hao Yu and Hanchen Zhang and Yifan Xu and Xuanyu Lei and Hanyu Lai and Yu Gu and Hangliang Ding and Kaiwen Men and Kejuan Yang and Shudan Zhang and Xiang Deng and Aohan Zeng and Zhengxiao Du and Chenhui Zhang and Sheng Shen and Tianjun Zhang and Yu Su and Huan Sun and Minlie Huang and Yuxiao Dong and Jie Tang},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2308.03688}
}
```

**Agent-SafetyBench**
```
@article{zhang2024agent,
  title={Agent-SafetyBench: Evaluating the Safety of LLM Agents},
  author={Zhang, Zhexin and Cui, Shiyao and Lu, Yida and Zhou, Jingzhuo and Yang, Junxiao and Wang, Hongning and Huang, Minlie},
  journal={arXiv preprint arXiv:2412.14470},
  year={2024}
}
```

**AgentHarm**
```
@article{andriushchenko2024agentharm,
  title={AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents},
  author={Andriushchenko, Maksym and Souly, Alexandra and Dziemian, Mateusz and Duenas, Derek and Lin, Maxwell and Wang, Justin and Hendrycks, Dan and Zou, Andy and Kolter, Zico and Fredrikson, Matt and Winsor, Eric and Wynne, Jerome and Gal, Yarin and Davies, Xander},
  journal={arXiv preprint arXiv:2410.09024},
  year={2024}
}
```