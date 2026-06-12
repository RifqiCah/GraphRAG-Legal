#!/usr/bin/env python3
"""
Build model-based embedding index v2 from chunks_v2.

This script supports local sentence-transformers and OpenAI-compatible
embeddings endpoints by code. It keeps chunks_v2 as the citation/source-of-truth
layer, then embeds a contextualized retrieval text for each chunk.

Recommended env:
- HF_HOME             recommended for local Hugging Face cache
- EMBEDDING_API_KEY
- EMBEDDING_BASE_URL   optional; leave empty for official OpenAI
- EMBEDDING_MODEL      example: intfloat/multilingual-e5-base

Outputs:
- indexes/vector_v2_model/vector_index.pkl
- indexes/vector_v2_model/embeddings.jsonl
- indexes/vector_v2_model/metadata.jsonl
- indexes/vector_v2_model/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_EXCLUDED_REGULASI_IDS = {"REG-UUD-1945"}
DEFAULT_MODEL = "intfloat/multilingual-e5-base"


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


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def l2_normalize(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in values))
    if not norm:
        return values
    return [float(x) / norm for x in values]


def load_chunks(chunks_dir: Path, excluded: set, limit: int = 0) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("chunks_REG-*.jsonl")):
        regulasi_id = path.stem.replace("chunks_", "")
        if regulasi_id in excluded:
            continue
        for row in read_jsonl(path):
            text = normalize_whitespace(row.get("text", ""))
            if not text:
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
            if limit and len(chunks) >= limit:
                return chunks
    return chunks


def build_embedding_text(chunk: Dict[str, Any], max_chars: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    title = metadata.get("judul", "")
    jenis = metadata.get("jenis", "")
    nomor = metadata.get("nomor", "")
    tahun = metadata.get("tahun", "")

    parts = [
        f"Regulasi: {jenis} Nomor {nomor} Tahun {tahun}".strip(),
        f"Judul: {title}".strip(),
        f"Referensi: {chunk.get('reference_label', '')}".strip(),
        f"Bagian: {chunk.get('section_type', '')}".strip(),
        f"Teks: {chunk.get('text', '')}".strip(),
    ]
    text = "\n".join(part for part in parts if part and not part.endswith(":"))
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def apply_prompt_prefix(text: str, prefix: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix:
        return text
    if text.lower().startswith(prefix.lower() + ":"):
        return text
    return f"{prefix}: {text}"


def existing_chunk_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        chunk_id = row.get("chunk_id")
        if chunk_id:
            ids.add(chunk_id)
    return ids


def make_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def embed_batch(
    client: OpenAI,
    model: str,
    texts: List[str],
    max_retries: int,
    retry_sleep: float,
) -> List[List[float]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [list(item.embedding) for item in response.data]
        except Exception as exc:  # noqa: BLE001 - log provider error transparently.
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_for = retry_sleep * attempt
            print(f"Embedding request failed attempt {attempt}/{max_retries}: {exc}. Retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"Embedding request failed after {max_retries} attempts: {last_error}")


def load_sentence_transformer(model: str, device: str, cache_folder: str = "", local_files_only: bool = False):
    from sentence_transformers import SentenceTransformer

    kwargs = {
        "device": device,
        "local_files_only": local_files_only,
    }
    # Prefer the standard Hugging Face cache discovery via HF_HOME.
    # Passing HF_HOME itself as cache_folder makes sentence-transformers look in
    # a different layout than HF_HOME/hub, so only pass this when explicitly set.
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    return SentenceTransformer(model, **kwargs)


def embed_batch_local(
    model_obj,
    texts: List[str],
    batch_size: int,
    normalize: bool,
) -> List[List[float]]:
    embeddings = model_obj.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [row.astype(float).tolist() for row in embeddings]


def load_embeddings(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    load_dotenv()

    excluded = set(args.exclude or [])
    if args.include_uud:
        excluded.discard("REG-UUD-1945")

    backend = args.backend
    model = args.model or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    api_key = args.api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("EMBEDDING_BASE_URL") or None

    chunks = load_chunks(args.chunks_dir, excluded, limit=args.limit)
    embedding_rows_path = args.out_dir / "embeddings.jsonl"
    metadata_rows_path = args.out_dir / "metadata.jsonl"

    existing_ids = existing_chunk_ids(embedding_rows_path) if args.resume and not args.force else set()
    pending_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]

    summary_base = {
        "schema_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "embedding_model": model,
        "base_url": base_url or "official_openai_default",
        "device": args.device if backend == "sentence_transformers" else "",
        "passage_prefix": args.passage_prefix if backend == "sentence_transformers" else "",
        "local_files_only": args.local_files_only if backend == "sentence_transformers" else False,
        "excluded_regulasi_ids": sorted(excluded),
        "total_chunks_selected": len(chunks),
        "already_embedded": len(existing_ids),
        "pending_chunks": len(pending_chunks),
        "batch_size": args.batch_size,
        "max_input_chars": args.max_input_chars,
        "normalize": args.normalize,
    }

    if args.dry_run:
        sample = []
        for chunk in chunks[: min(3, len(chunks))]:
            embedding_text = build_embedding_text(chunk, args.max_input_chars)
            if backend == "sentence_transformers":
                embedding_text = apply_prompt_prefix(embedding_text, args.passage_prefix)
            sample.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "reference_label": chunk["reference_label"],
                    "embedding_text": embedding_text,
                }
            )
        return {**summary_base, "dry_run": True, "sample_embedding_inputs": sample}

    if backend == "openai_compatible" and not api_key:
        raise RuntimeError(
            "Missing embedding API key. Set EMBEDDING_API_KEY or OPENAI_API_KEY, "
            "or pass --api-key."
        )

    if args.force:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for path in [embedding_rows_path, metadata_rows_path]:
            if path.exists():
                path.unlink()

    client = None
    local_model = None
    if backend == "openai_compatible":
        client = make_client(api_key=api_key, base_url=base_url)
    elif backend == "sentence_transformers":
        local_model = load_sentence_transformer(
            model,
            args.device,
            cache_folder=args.cache_folder,
            local_files_only=args.local_files_only,
        )
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    written = 0
    for start in range(0, len(pending_chunks), args.batch_size):
        batch = pending_chunks[start : start + args.batch_size]
        texts = [build_embedding_text(chunk, args.max_input_chars) for chunk in batch]
        if backend == "sentence_transformers":
            texts = [apply_prompt_prefix(text, args.passage_prefix) for text in texts]
            vectors = embed_batch_local(
                model_obj=local_model,
                texts=texts,
                batch_size=args.batch_size,
                normalize=args.normalize,
            )
        else:
            vectors = embed_batch(
                client=client,
                model=model,
                texts=texts,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )

        embedding_rows = []
        metadata_rows = []
        for chunk, text, vector in zip(batch, texts, vectors):
            if args.normalize and backend != "sentence_transformers":
                vector = l2_normalize(vector)
            embedding_rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "embedding": vector,
                    "dimension": len(vector),
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
                    "embedding_text": text,
                    "metadata": chunk["metadata"],
                }
            )

        append_jsonl(embedding_rows_path, embedding_rows)
        append_jsonl(metadata_rows_path, metadata_rows)
        written += len(batch)
        print(f"embedded {written}/{len(pending_chunks)} pending chunks")

    embeddings = load_embeddings(embedding_rows_path)
    metadata = list(read_jsonl(metadata_rows_path)) if metadata_rows_path.exists() else []
    dimensions = Counter(row.get("dimension", 0) for row in embeddings)
    section_counts = Counter(row.get("section_type", "") for row in metadata)
    regulasi_counts = Counter(row.get("regulasi_id", "") for row in metadata)

    summary = {
        **summary_base,
        "dry_run": False,
        "total_embeddings": len(embeddings),
        "dimensions": dict(dimensions),
        "chunks_by_section_type": dict(section_counts),
        "chunks_by_regulasi_id": dict(regulasi_counts),
        "outputs": {
            "vector_index": str(args.out_dir / "vector_index.pkl"),
            "embeddings_jsonl": str(embedding_rows_path),
            "metadata_jsonl": str(metadata_rows_path),
            "summary_json": str(args.out_dir / "summary.json"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "vector_index.pkl").open("wb") as f:
        pickle.dump(
                {
                    "schema_version": "2.0.0",
                    "backend": backend,
                    "embedding_model": model,
                    "base_url": base_url,
                    "device": args.device if backend == "sentence_transformers" else "",
                    "passage_prefix": args.passage_prefix if backend == "sentence_transformers" else "",
                    "query_prefix": args.query_prefix if backend == "sentence_transformers" else "",
                    "local_files_only": args.local_files_only if backend == "sentence_transformers" else False,
                    "normalize": args.normalize,
                    "embeddings": embeddings,
                "metadata": metadata,
                "summary": summary,
            },
            f,
        )
    write_json(args.out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-based embedding index v2.")
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("indexes/vector_v2_model"))
    parser.add_argument("--backend", choices=["sentence_transformers", "openai_compatible"], default="sentence_transformers")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-folder", default="", help="Optional explicit sentence-transformers cache folder. Usually leave empty and use HF_HOME.")
    parser.add_argument("--passage-prefix", default="passage")
    parser.add_argument("--query-prefix", default="query")
    parser.add_argument("--local-files-only", action="store_true", help="Load local Hugging Face cache only; do not contact huggingface.co.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Embed only first N chunks for pilot/testing.")
    parser.add_argument("--max-input-chars", type=int, default=8000)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--exclude", action="append", default=sorted(DEFAULT_EXCLUDED_REGULASI_IDS))
    parser.add_argument("--include-uud", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_index(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
