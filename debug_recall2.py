import json
from pathlib import Path
from scripts.graph.graphrag_retriever_v2 import parse_references

with open("data/sft/validation.jsonl") as f:
    for line in f:
        row = json.loads(line)
        gt = row.get("output", "")
        expected = parse_references(gt)
        if expected:
            print(f"Found expected: {expected}")
            break
