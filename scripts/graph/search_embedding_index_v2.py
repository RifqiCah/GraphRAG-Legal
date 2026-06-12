#!/usr/bin/env python3
"""
Search model-based embedding index v2.

This embeds the query with the same embedding backend used by
build_embedding_index_v2.py, then runs brute-force cosine/dot search over the
local normalized vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.build_embedding_index_v2 import DEFAULT_MODEL, apply_prompt_prefix, l2_normalize


def load_index(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def make_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def dot(a: List[float], b: List[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def embed_query(client: OpenAI, model: str, query: str, normalize: bool) -> List[float]:
    response = client.embeddings.create(model=model, input=[query])
    vector = list(response.data[0].embedding)
    return l2_normalize(vector) if normalize else vector


def embed_query_local(model_name: str, device: str, query: str, normalize: bool) -> List[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    vector = model.encode(
        [query],
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vector.astype(float).tolist()


def search(
    index: Dict[str, Any],
    query_vector: List[float],
    top_k: int,
    regulasi_id: str = "",
    section_type: str = "",
) -> List[Dict[str, Any]]:
    metadata_by_chunk = {row["chunk_id"]: row for row in index["metadata"]}
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for row in index["embeddings"]:
        metadata = metadata_by_chunk.get(row["chunk_id"], {})
        if regulasi_id and metadata.get("regulasi_id") != regulasi_id:
            continue
        if section_type and metadata.get("section_type") != section_type:
            continue
        score = dot(query_vector, row["embedding"])
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
    parser = argparse.ArgumentParser(description="Search model-based embedding index v2.")
    parser.add_argument("query", nargs="*")
    parser.add_argument("--index", type=Path, default=Path("indexes/vector_v2_model/vector_index.pkl"))
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--regulasi-id", default="")
    parser.add_argument("--section-type", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Query is required.")

    index = load_index(args.index)
    backend = index.get("backend", "openai_compatible")
    model = args.model or os.getenv("EMBEDDING_MODEL") or index.get("embedding_model") or DEFAULT_MODEL
    api_key = args.api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("EMBEDDING_BASE_URL") or index.get("base_url") or None
    normalize = bool(index.get("normalize", True))
    device = args.device or index.get("device") or "cuda"

    if backend == "sentence_transformers":
        query = apply_prompt_prefix(query, index.get("query_prefix", "query"))
        query_vector = embed_query_local(model, device, query, normalize=normalize)
    elif backend == "openai_compatible":
        if not api_key:
            raise RuntimeError("Missing embedding API key. Set EMBEDDING_API_KEY or OPENAI_API_KEY.")
        client = make_client(api_key=api_key, base_url=base_url)
        query_vector = embed_query(client, model, query, normalize=normalize)
    else:
        raise RuntimeError(f"Unknown embedding backend: {backend}")
    results = search(index, query_vector, args.top_k, args.regulasi_id, args.section_type)

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
