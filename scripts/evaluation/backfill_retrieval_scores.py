#!/usr/bin/env python3
"""Backfill mode-independent raw dense retrieval scores into prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def apply_prefix(text: str, prefix: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix or text.lower().startswith(prefix.lower() + ":"):
        return text
    return f"{prefix}: {text}"


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: header CSV tidak ditemukan")
        rows = list(reader)
    if "vector_query" not in reader.fieldnames:
        raise ValueError(f"{path}: kolom vector_query tidak ditemukan")
    return list(reader.fieldnames), rows


def score_queries(
    vector_index: Dict,
    queries: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Dict[str, Tuple[float, float, List[str]]]:
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    normalize = bool(vector_index.get("normalize", True))
    query_prefix = vector_index.get("query_prefix", "query")
    prefixed = [apply_prefix(query, query_prefix) for query in queries]
    query_matrix = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    embedding_rows = vector_index["embeddings"]
    chunk_ids = [row["chunk_id"] for row in embedding_rows]
    passage_matrix = np.asarray(
        [row["embedding"] for row in embedding_rows],
        dtype="float32",
    )

    scored: Dict[str, Tuple[float, float, List[str]]] = {}
    score_batch_size = max(1, min(batch_size, 128))
    for start in range(0, len(queries), score_batch_size):
        end = min(len(queries), start + score_batch_size)
        scores = query_matrix[start:end] @ passage_matrix.T
        for offset, query in enumerate(queries[start:end]):
            order = np.argsort(-scores[offset], kind="stable")[:5]
            top_scores = [round(float(scores[offset, idx]), 6) for idx in order]
            if len(top_scores) < 5:
                raise ValueError("Indeks menghasilkan kurang dari lima kandidat")
            scored[query] = (
                top_scores[0],
                round(top_scores[0] - top_scores[4], 6),
                [chunk_ids[idx] for idx in order],
            )
    return scored


def with_score_columns(fieldnames: Sequence[str]) -> List[str]:
    score_names = {"retrieval_top1_score", "retrieval_score_margin"}
    base = [name for name in fieldnames if name not in score_names]
    insert_at = base.index("vector_query") + 1
    return [
        *base[:insert_at],
        "retrieval_top1_score",
        "retrieval_score_margin",
        *base[insert_at:],
    ]


def backfill_file(
    path: Path,
    fieldnames: Sequence[str],
    rows: List[Dict[str, str]],
    scores: Dict[str, Tuple[float, float, List[str]]],
    backup: bool,
) -> None:
    for row in rows:
        top1, margin, top_ids = scores[row["vector_query"]]
        if row.get("retrieved_chunk_ids"):
            import json

            stored_ids = json.loads(row["retrieved_chunk_ids"])
            if stored_ids[:5] != top_ids:
                raise ValueError(
                    f"{path}: top-5 hasil hitung ulang berbeda pada label "
                    f"{row.get('label', '')}: {top_ids} != {stored_ids[:5]}"
                )
        row["retrieval_top1_score"] = f"{top1:.6f}"
        row["retrieval_score_margin"] = f"{margin:.6f}"

    if backup:
        backup_path = path.with_suffix(path.suffix + ".pre_retrieval_scores.bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    output_fields = with_score_columns(fieldnames)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Prediction CSV files to update in place.",
    )
    parser.add_argument(
        "--vector-index",
        type=Path,
        default=PROJECT_ROOT / "indexes/vector_v2_model/vector_index.pkl",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.vector_index.open("rb") as handle:
        vector_index = pickle.load(handle)

    loaded = {}
    queries = []
    seen = set()
    for path in args.files:
        fieldnames, rows = read_csv(path)
        loaded[path] = (fieldnames, rows)
        for row in rows:
            query = row["vector_query"]
            if query not in seen:
                seen.add(query)
                queries.append(query)

    model_name = (
        args.model
        or vector_index.get("embedding_model")
        or "intfloat/multilingual-e5-base"
    )
    scores = score_queries(
        vector_index=vector_index,
        queries=queries,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    for path, (fieldnames, rows) in loaded.items():
        backfill_file(
            path=path,
            fieldnames=fieldnames,
            rows=rows,
            scores=scores,
            backup=not args.no_backup,
        )
        print(f"updated {path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
