#!/usr/bin/env python3
"""
Build vector index v2 from chunks_v2.

Default backend is dependency-free hashing TF-IDF. It is a lightweight vector
baseline for retrieval experiments when neural embedding packages are not
installed yet. The output contract is intentionally backend-agnostic so it can
be swapped later with sentence-transformers, OpenAI embeddings, Chroma, FAISS,
or Qdrant without changing chunks_v2.

Outputs:
- indexes/vector_v2/vector_index.pkl
- indexes/vector_v2/vectors.jsonl
- indexes/vector_v2/metadata.jsonl
- indexes/vector_v2/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_EXCLUDED_REGULASI_IDS = {"REG-UUD-1945"}
DEFAULT_DIMENSIONS = 262144
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "b",
    "c",
    "d",
    "dan",
    "atau",
    "yang",
    "di",
    "ke",
    "dari",
    "dalam",
    "dengan",
    "untuk",
    "pada",
    "oleh",
    "sebagai",
    "adalah",
    "ini",
    "itu",
    "serta",
    "dapat",
    "harus",
    "ayat",
    "pasal",
    "nomor",
    "tahun",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def tokenize(text: str, use_bigrams: bool = True) -> List[str]:
    words = [
        token.lower()
        for token in TOKEN_PATTERN.findall(normalize_text(text))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
    if not use_bigrams:
        return words
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def load_chunks(chunks_dir: Path, excluded: set) -> List[Dict[str, Any]]:
    chunks = []
    for path in sorted(chunks_dir.glob("chunks_REG-*.jsonl")):
        regulasi_id = path.stem.replace("chunks_", "")
        if regulasi_id in excluded:
            continue
        for row in read_jsonl(path):
            text = row.get("text", "")
            if not str(text).strip():
                continue
            metadata = row.get("metadata", {}) or {}
            chunks.append(
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "regulasi_id": row.get("regulasi_id", regulasi_id),
                    "source_node_id": row.get("source_node_id", ""),
                    "section_type": row.get("section_type", ""),
                    "reference_label": row.get("reference_label", ""),
                    "text": text,
                    "metadata": metadata,
                }
            )
    return chunks


def build_idf(chunks: List[Dict[str, Any]], use_bigrams: bool, min_df: int, max_df_ratio: float) -> Dict[str, float]:
    document_frequency: Counter[str] = Counter()
    total_docs = len(chunks)

    for chunk in chunks:
        document_frequency.update(set(tokenize(chunk["text"], use_bigrams=use_bigrams)))

    max_df = max(1, int(total_docs * max_df_ratio))
    idf = {}
    for token, df in document_frequency.items():
        if df < min_df or df > max_df:
            continue
        idf[token] = math.log((1 + total_docs) / (1 + df)) + 1.0
    return idf


def vectorize(text: str, idf: Dict[str, float], dimensions: int, use_bigrams: bool) -> Dict[int, float]:
    counts = Counter(token for token in tokenize(text, use_bigrams=use_bigrams) if token in idf)
    if not counts:
        return {}

    values: Dict[int, float] = {}
    for token, count in counts.items():
        hashed = stable_hash(token)
        index = hashed % dimensions
        sign = -1.0 if ((hashed >> 63) & 1) else 1.0
        weight = sign * (1.0 + math.log(count)) * idf[token]
        values[index] = values.get(index, 0.0) + weight

    norm = math.sqrt(sum(value * value for value in values.values()))
    if not norm:
        return {}
    return {index: value / norm for index, value in values.items() if value}


def sparse_to_row(vector: Dict[int, float]) -> Dict[str, List[Any]]:
    ordered = sorted(vector.items())
    return {
        "indices": [index for index, _ in ordered],
        "values": [round(value, 8) for _, value in ordered],
    }


def build_vectors(
    chunks: List[Dict[str, Any]],
    idf: Dict[str, float],
    dimensions: int,
    use_bigrams: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    vector_rows = []
    metadata_rows = []

    for chunk in chunks:
        vector = vectorize(chunk["text"], idf=idf, dimensions=dimensions, use_bigrams=use_bigrams)
        vector_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "vector": sparse_to_row(vector),
            }
        )
        metadata_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "regulasi_id": chunk["regulasi_id"],
                "source_node_id": chunk["source_node_id"],
                "section_type": chunk["section_type"],
                "reference_label": chunk["reference_label"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "nnz": len(vector),
            }
        )

    return vector_rows, metadata_rows


def make_summary(
    chunks: List[Dict[str, Any]],
    vector_rows: List[Dict[str, Any]],
    idf: Dict[str, float],
    args: argparse.Namespace,
    excluded: set,
) -> Dict[str, Any]:
    nnz_values = [len(row["vector"]["indices"]) for row in vector_rows]
    section_counts = Counter(chunk.get("section_type", "") for chunk in chunks)
    regulasi_counts = Counter(chunk.get("regulasi_id", "") for chunk in chunks)

    return {
        "schema_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "hashing_tfidf_v1",
        "embedding_model": "local_hashing_tfidf",
        "dimensions": args.dimensions,
        "use_bigrams": args.use_bigrams,
        "min_df": args.min_df,
        "max_df_ratio": args.max_df_ratio,
        "excluded_regulasi_ids": sorted(excluded),
        "total_chunks": len(chunks),
        "total_vectors": len(vector_rows),
        "vocabulary_terms": len(idf),
        "avg_nonzero_dimensions": round(sum(nnz_values) / max(1, len(nnz_values)), 2),
        "max_nonzero_dimensions": max(nnz_values) if nnz_values else 0,
        "chunks_by_section_type": dict(section_counts),
        "chunks_by_regulasi_id": dict(regulasi_counts),
        "outputs": {
            "vector_index": str(args.out_dir / "vector_index.pkl"),
            "vectors_jsonl": str(args.out_dir / "vectors.jsonl"),
            "metadata_jsonl": str(args.out_dir / "metadata.jsonl"),
            "summary_json": str(args.out_dir / "summary.json"),
        },
        "notes": [
            "This is a dependency-free vector baseline, not a neural semantic embedding model.",
            "The output can be rebuilt later with sentence-transformers/OpenAI/Chroma/FAISS without changing chunks_v2.",
        ],
    }


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    excluded = set(args.exclude or [])
    if args.include_uud:
        excluded.discard("REG-UUD-1945")

    chunks = load_chunks(args.chunks_dir, excluded)
    idf = build_idf(
        chunks,
        use_bigrams=args.use_bigrams,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio,
    )
    vector_rows, metadata_rows = build_vectors(
        chunks,
        idf=idf,
        dimensions=args.dimensions,
        use_bigrams=args.use_bigrams,
    )
    summary = make_summary(chunks, vector_rows, idf, args, excluded)

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "vector_index.pkl").open("wb") as f:
            pickle.dump(
                {
                    "schema_version": "2.0.0",
                    "backend": "hashing_tfidf_v1",
                    "config": {
                        "dimensions": args.dimensions,
                        "use_bigrams": args.use_bigrams,
                        "min_df": args.min_df,
                        "max_df_ratio": args.max_df_ratio,
                    },
                    "idf": idf,
                    "vectors": vector_rows,
                    "metadata": metadata_rows,
                    "summary": summary,
                },
                f,
            )
        write_jsonl(args.out_dir / "vectors.jsonl", vector_rows)
        write_jsonl(args.out_dir / "metadata.jsonl", metadata_rows)
        write_json(args.out_dir / "summary.json", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index v2 from chunks_v2.")
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("indexes/vector_v2"))
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df-ratio", type=float, default=0.92)
    parser.add_argument("--use-bigrams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude", action="append", default=sorted(DEFAULT_EXCLUDED_REGULASI_IDS))
    parser.add_argument("--include-uud", action="store_true", help="Include REG-UUD-1945 even though it is excluded by default.")
    parser.add_argument("--dry-run", action="store_true", help="Build in memory and print summary without writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_index(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
