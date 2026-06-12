#!/usr/bin/env python3
"""
Search vector_index_v2.

Works with the dependency-free hashing TF-IDF index built by
scripts/graph/build_vector_index_v2.py.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.build_vector_index_v2 import vectorize


def sparse_dot(query_vector: Dict[int, float], row: Dict[str, Any]) -> float:
    indices = row["vector"]["indices"]
    values = row["vector"]["values"]
    score = 0.0
    for index, value in zip(indices, values):
        score += query_vector.get(index, 0.0) * float(value)
    return score


def load_index(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def search(index: Dict[str, Any], query: str, top_k: int, regulasi_id: str = "", section_type: str = "") -> List[Dict[str, Any]]:
    config = index["config"]
    query_vector = vectorize(
        query,
        idf=index["idf"],
        dimensions=config["dimensions"],
        use_bigrams=config["use_bigrams"],
    )

    if not query_vector:
        return []

    metadata_by_chunk = {row["chunk_id"]: row for row in index["metadata"]}
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for vector_row in index["vectors"]:
        chunk_id = vector_row["chunk_id"]
        metadata = metadata_by_chunk.get(chunk_id, {})
        if regulasi_id and metadata.get("regulasi_id") != regulasi_id:
            continue
        if section_type and metadata.get("section_type") != section_type:
            continue
        score = sparse_dot(query_vector, vector_row)
        if score > 0:
            scored.append((score, metadata))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, metadata in scored[:top_k]:
        results.append(
            {
                "score": round(float(score), 6),
                "chunk_id": metadata.get("chunk_id", ""),
                "regulasi_id": metadata.get("regulasi_id", ""),
                "source_node_id": metadata.get("source_node_id", ""),
                "section_type": metadata.get("section_type", ""),
                "reference_label": metadata.get("reference_label", ""),
                "text": metadata.get("text", ""),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search vector_index_v2.")
    parser.add_argument("query", nargs="*", help="Query text.")
    parser.add_argument("--index", type=Path, default=Path("indexes/vector_v2/vector_index.pkl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--regulasi-id", default="")
    parser.add_argument("--section-type", default="")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Query is required.")

    index = load_index(args.index)
    results = search(index, query, args.top_k, args.regulasi_id, args.section_type)

    if args.json:
        print(json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2))
        return

    print(f"query: {query}")
    print(f"top_k: {args.top_k}")
    for i, row in enumerate(results, start=1):
        text = " ".join(row["text"].split())
        if len(text) > 260:
            text = text[:257] + "..."
        print()
        print(f"{i}. score={row['score']} {row['reference_label']}")
        print(f"   chunk_id={row['chunk_id']} source_node_id={row['source_node_id']}")
        print(f"   {text}")


if __name__ == "__main__":
    main()
