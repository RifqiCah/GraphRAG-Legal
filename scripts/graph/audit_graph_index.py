import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.graphrag_retriever import (
    load_schema,
    infer_node_label,
    node_schema_label,
    is_edge_schema_valid,
)


def load_index(path="indexes/graph/graph_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    index = load_index()
    graph = index["graph"]
    schema_info = load_schema("ontology/schema.json")

    valid_relations = schema_info["valid_relations"]
    missing_nodes = index.get("missing_nodes", [])

    node_type_counts = Counter()
    nodes_with_text_by_type = Counter()
    nodes_without_text_by_type = Counter()

    for node_id, attrs in graph.nodes(data=True):
        label = node_schema_label(graph, node_id)
        node_type_counts[label] += 1
        if attrs.get("text"):
            nodes_with_text_by_type[label] += 1
        else:
            nodes_without_text_by_type[label] += 1

    edge_label_counts = Counter()
    invalid_relation_label_counts = Counter()
    schema_valid_edges = 0
    schema_invalid_edges = 0
    unknown_relation_edges = 0

    invalid_examples = []
    unknown_relation_examples = []

    for source, target, attrs in graph.edges(data=True):
        predicate = attrs.get("predicate", "")
        edge_label_counts[predicate] += 1

        if predicate not in valid_relations:
            unknown_relation_edges += 1
            invalid_relation_label_counts[predicate] += 1
            if len(unknown_relation_examples) < 20:
                unknown_relation_examples.append({
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "source_type": node_schema_label(graph, source),
                    "target_type": node_schema_label(graph, target),
                })
            continue

        if is_edge_schema_valid(graph, source, target, predicate, schema_info):
            schema_valid_edges += 1
        else:
            schema_invalid_edges += 1
            invalid_relation_label_counts[predicate] += 1
            if len(invalid_examples) < 20:
                invalid_examples.append({
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "source_type": node_schema_label(graph, source),
                    "target_type": node_schema_label(graph, target),
                })

    missing_by_inferred_type = Counter(infer_node_label(n) for n in missing_nodes)

    missing_samples_by_type = defaultdict(list)
    for n in missing_nodes:
        t = infer_node_label(n)
        if len(missing_samples_by_type[t]) < 20:
            missing_samples_by_type[t].append(n)

    summary = {
        "nodes": {
            "total": graph.number_of_nodes(),
            "by_type": dict(node_type_counts),
            "with_text_by_type": dict(nodes_with_text_by_type),
            "without_text_by_type": dict(nodes_without_text_by_type),
        },
        "edges": {
            "total": graph.number_of_edges(),
            "schema_valid_edges": schema_valid_edges,
            "schema_invalid_edges": schema_invalid_edges,
            "unknown_relation_edges": unknown_relation_edges,
            "by_label": dict(edge_label_counts),
            "invalid_or_unknown_by_label": dict(invalid_relation_label_counts),
        },
        "missing_nodes": {
            "total": len(missing_nodes),
            "by_inferred_type": dict(missing_by_inferred_type),
            "samples_by_type": dict(missing_samples_by_type),
        },
        "examples": {
            "schema_invalid_edges": invalid_examples,
            "unknown_relation_edges": unknown_relation_examples,
        },
    }

    write_json("indexes/graph/audit_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
