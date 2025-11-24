from datasets import load_dataset, logging

logging.set_verbosity_debug()

data = load_dataset("thu-coai/Agent-SafetyBench")

rows = data[:5]
print(rows)