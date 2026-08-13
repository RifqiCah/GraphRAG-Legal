#!/usr/bin/env python3
"""
GraphRAG retriever v2.

Retrieval flow:
1. Resolve exact legal references from the query.
2. Run semantic vector search over chunk embeddings.
3. Expand matched nodes through the local knowledge graph.
4. Pack traceable context blocks for the generation layer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_GRAPH_INDEX = Path("indexes/graph_v2/graph_index.pkl")
DEFAULT_VECTOR_INDEX = Path("indexes/vector_v2_model/vector_index.pkl")

STRUCTURAL_PREDICATES = {
    "MEMILIKI_BAB",
    "MEMILIKI_BAGIAN",
    "MEMILIKI_PARAGRAF",
    "MEMILIKI_PASAL",
    "MEMILIKI_AYAT",
    "MEMILIKI_SUBITEM",
    "MEMILIKI_KONSIDERANS",
    "MEMILIKI_DASAR_HUKUM",
    "MEMILIKI_DIKTUM",
    "MEMILIKI_LAMPIRAN",
    "MEMILIKI_PENJELASAN",
    "MEMILIKI_PENJELASAN_UMUM",
    "MEMILIKI_PENJELASAN_PASAL",
    "MEMILIKI_PENUTUP",
    "MEMILIKI_CHUNK",
    "MENJELASKAN",
}

LEGAL_PREDICATES = {
    "MENGACU_PADA",
    "BERDASARKAN",
    "MENGAMANDEMEN",
    "MENCABUT",
    "MENCABUT_PASAL",
    "MULAI_BERLAKU",
    "DIUNDANGKAN_PADA",
    "BERTENTANGAN_DENGAN",
}

SEMANTIC_PREDICATES = {
    "MENGATUR_TOPIK",
    "MENGATUR_SUBJEK",
    "MEMBERIKAN_HAK",
    "MEMBEBANKAN_KEWAJIBAN",
    "MELARANG",
    "MENDEFINISIKAN",
}

EXPANSION_PREDICATES = STRUCTURAL_PREDICATES | LEGAL_PREDICATES | SEMANTIC_PREDICATES
LEAF_TYPES = {"Chunk", "Topik", "Subjek_Hukum", "Peristiwa_Hukum"}

SECTION_PRIORITY = {
    "ayat": 0.06,
    "pasal": 0.05,
    "subitem": 0.05,
    "penjelasan_pasal": 0.03,
    "penjelasan_umum": 0.01,
    "penutup": -0.02,
    "menimbang": -0.04,
    "mengingat": -0.08,
}

REFERENCE_STOPWORDS = {
    "apa",
    "bagaimana",
    "jelaskan",
    "menurut",
    "berdasarkan",
    "dalam",
    "pada",
    "terkait",
    "mengenai",
    "tentang",
    "sesuai",
    "dengan",
    "diatur",
    "aturan",
    "ketentuan",
}


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_spaces(text: Any) -> str:
    return " ".join(str(text or "").split())


def truncate(text: Any, max_chars: int = 360) -> str:
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def slug_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip().upper()).strip("-")


def normalize_pasal(value: Any) -> str:
    return slug_part(value)


def normalize_ayat(value: Any) -> str:
    return slug_part(value)


def build_regulasi_id(kind: str, number: str, year: str, regime: str = "") -> str:
    kind = slug_part(kind)
    number = slug_part(number)
    year = slug_part(year)
    regime = slug_part(regime)
    if kind == "UU" and regime:
        return f"REG-UU-{number}-{regime}-{year}"
    return f"REG-{kind}-{number}-{year}"


def parse_pasal_ayat(text: str, start: int = 0, end: Optional[int] = None) -> Tuple[str, str]:
    window = text[start:end] if end is not None else text[start:]
    pasal_match = re.search(
        r"\bPasal\s+([0-9]+[A-Za-z]?|[IVXLCDM]+)\b",
        window,
        flags=re.IGNORECASE,
    )
    ayat_match = re.search(
        r"\bayat\s*\(?\s*([0-9]+[A-Za-z]?|[IVXLCDM]+)\s*\)?",
        window,
        flags=re.IGNORECASE,
    )
    pasal = normalize_pasal(pasal_match.group(1)) if pasal_match else ""
    ayat = normalize_ayat(ayat_match.group(1)) if ayat_match else ""
    return pasal, ayat


def parse_references(query: str, fallback_regulasi_id: str = "") -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    text = str(query or "")

    def add_ref(raw: str, regulasi_id: str, pasal: str = "", ayat: str = "", node_id: str = "") -> None:
        key = (regulasi_id, pasal, ayat, node_id)
        if key in seen:
            return
        seen.add(key)
        refs.append(
            {
                "raw": normalize_spaces(raw),
                "regulasi_id": regulasi_id,
                "pasal": pasal,
                "ayat": ayat,
                "node_id": node_id,
            }
        )

    direct_node_pattern = re.compile(
        r"\b(?:REG|PSL|AYT|CHK|SUB|KONS|DASAR|DIKTUM|BAB|BAGIAN|PAR|PJL)-[A-Za-z0-9-]+\b"
    )
    for match in direct_node_pattern.finditer(text):
        raw = match.group(0)
        regulasi_match = re.search(r"REG-(?:PP|UU|UUD)-[A-Za-z0-9-]+", raw)
        pasal, ayat = parse_pasal_ayat(text, match.end(), min(len(text), match.end() + 120))
        add_ref(raw, regulasi_match.group(0).upper() if regulasi_match else "", pasal, ayat, raw.upper())

    if re.search(r"\bUUD\b\s*(?:Tahun\s*)?1945\b", text, flags=re.IGNORECASE):
        pasal, ayat = parse_pasal_ayat(text)
        add_ref("UUD 1945", "REG-UUD-1945", pasal, ayat)

    reg_pattern = re.compile(
        r"\b(PP|UU)\s*(?:No\.?|Nomor)?\s*"
        r"([0-9]+)\s*"
        r"(?:[/\-\s]*(PNPS|PENPRES|PRP))?\s*"
        r"(?:/|\s+Tahun\s+|\s+)"
        r"(\d{4})\b",
        flags=re.IGNORECASE,
    )
    for match in reg_pattern.finditer(text):
        kind, number, regime, year = match.groups()
        regulasi_id = build_regulasi_id(kind, number, year, regime or "")
        pasal, ayat = parse_pasal_ayat(text, match.end(), min(len(text), match.end() + 160))
        if not pasal:
            pasal, ayat = parse_pasal_ayat(text, max(0, match.start() - 80), match.start())
        add_ref(match.group(0), regulasi_id, pasal, ayat)

    if fallback_regulasi_id and not refs:
        pasal, ayat = parse_pasal_ayat(text)
        if pasal:
            add_ref(fallback_regulasi_id, fallback_regulasi_id, pasal, ayat)

    return refs


def clean_vector_query(query: str) -> str:
    """Remove formal legal citations before embedding semantic intent."""
    cleaned = str(query or "")
    cleaned = re.sub(
        r"\b(?:REG|PSL|AYT|CHK|SUB|KONS|DASAR|DIKTUM|BAB|BAGIAN|PAR|PJL)-[A-Za-z0-9-]+\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:UUD)\s*(?:Tahun\s*)?1945\b"
        r"(?:\s+Pasal\s+(?:[0-9]+[A-Za-z]?|[IVXLCDM]+))?"
        r"(?:\s+ayat\s*\(?\s*(?:[0-9]+[A-Za-z]?|[IVXLCDM]+)\s*\)?)?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:PP|UU)\s*(?:No\.?|Nomor)?\s*"
        r"[0-9]+\s*(?:[/\-\s]*(?:PNPS|PENPRES|PRP))?\s*"
        r"(?:/|\s+Tahun\s+|\s+)\d{4}\b"
        r"(?:\s+Pasal\s+(?:[0-9]+[A-Za-z]?|[IVXLCDM]+))?"
        r"(?:\s+ayat\s*\(?\s*(?:[0-9]+[A-Za-z]?|[IVXLCDM]+)\s*\)?)?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bPasal\s+(?:[0-9]+[A-Za-z]?|[IVXLCDM]+)\b"
        r"(?:\s+ayat\s*\(?\s*(?:[0-9]+[A-Za-z]?|[IVXLCDM]+)\s*\)?)?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:menurut|berdasarkan|sesuai\s+dengan|diatur\s+dalam)\b\s*$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,:;()\[\]{}]+", " ", cleaned)
    return normalize_spaces(cleaned)


def meaningful_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", str(text or "").lower())
    return [token for token in tokens if token not in REFERENCE_STOPWORDS and len(token) > 1]


def build_query_analysis(
    query: str,
    refs: Sequence[Dict[str, str]],
    query_cleaning: bool,
    auto_skip_vector_exact_only: bool,
) -> Dict[str, Any]:
    vector_query = clean_vector_query(query) if query_cleaning else normalize_spaces(query)
    tokens = meaningful_tokens(vector_query)
    exact_only = bool(refs) and len(tokens) == 0
    vector_skipped_reason = ""
    if auto_skip_vector_exact_only and exact_only:
        vector_skipped_reason = "exact_reference_only"
    if not vector_query and not vector_skipped_reason:
        vector_query = normalize_spaces(query)
    return {
        "original_query": normalize_spaces(query),
        "vector_query": vector_query,
        "query_cleaning_enabled": query_cleaning,
        "auto_skip_vector_exact_only": auto_skip_vector_exact_only,
        "exact_reference_only": exact_only,
        "meaningful_tokens": tokens,
        "vector_skipped_reason": vector_skipped_reason,
    }


def node_brief(node: Dict[str, Any], max_text_chars: int = 420) -> Dict[str, Any]:
    properties = node.get("properties") or {}
    keep_props = {}
    for key in (
        "jenis",
        "nomor",
        "tahun",
        "judul",
        "nomor_pasal",
        "nomor_ayat",
        "nomor",
        "reference_label",
        "section_type",
        "source_node_id",
    ):
        if key in properties:
            keep_props[key] = properties[key]
    return {
        "id": node.get("id", ""),
        "type": node.get("type", ""),
        "regulasi_id": node.get("regulasi_id", ""),
        "label": node.get("label", ""),
        "text": truncate(node.get("text", ""), max_text_chars),
        "is_stub": bool(node.get("is_stub", False)),
        "properties": keep_props,
    }


def edge_brief(edge: Dict[str, Any], max_evidence_chars: int = 260) -> Dict[str, Any]:
    evidence = edge.get("evidence") or {}
    return {
        "id": edge.get("id") or edge.get("edge_id", ""),
        "predicate": edge.get("predicate", ""),
        "source_id": edge.get("source_id", ""),
        "target_id": edge.get("target_id", ""),
        "source_type": edge.get("source_type", ""),
        "target_type": edge.get("target_type", ""),
        "confidence": edge.get("confidence", None),
        "extraction_method": edge.get("extraction_method", ""),
        "evidence_text": truncate(evidence.get("text", ""), max_evidence_chars),
    }


def resolve_exact_matches(graph: Dict[str, Any], refs: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    nodes = graph["nodes"]
    matches: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()

    for ref in refs:
        candidate_tiers: List[List[str]] = []
        node_id = ref.get("node_id") or ""
        regulasi_id = ref.get("regulasi_id") or ""
        pasal = ref.get("pasal") or ""
        ayat = ref.get("ayat") or ""

        if regulasi_id and pasal and ayat:
            candidate_tiers.append([f"AYT-{regulasi_id}-{pasal}-{ayat}"])
        if regulasi_id and pasal:
            candidate_tiers.append([f"PSL-{regulasi_id}-{pasal}"])
        if node_id and node_id in nodes:
            candidate_tiers.append([node_id])
        if regulasi_id:
            candidate_tiers.append([regulasi_id])

        found: List[str] = []
        for candidates in candidate_tiers:
            found = [candidate for candidate in candidates if candidate in nodes]
            if found:
                break

        if not found and regulasi_id and pasal:
            for current_id, node in nodes.items():
                if node.get("regulasi_id") != regulasi_id:
                    continue
                props = node.get("properties") or {}
                if node.get("type") == "Pasal" and slug_part(props.get("nomor_pasal")) == pasal:
                    found.append(current_id)
                if node.get("type") == "Ayat" and ayat and current_id.startswith(f"AYT-{regulasi_id}-{pasal}-"):
                    if slug_part(props.get("nomor_ayat")) == ayat:
                        found.append(current_id)

        found = list(dict.fromkeys(found))
        for found_id in found:
            if found_id in seen_nodes:
                continue
            seen_nodes.add(found_id)
            matches.append(
                {
                    "reference": ref,
                    "node_id": found_id,
                    "node": node_brief(nodes[found_id]),
                    "score": 1.0,
                }
            )

    return matches


def apply_prefix(text: str, prefix: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix:
        return text
    if text.lower().startswith(prefix.lower() + ":"):
        return text
    return f"{prefix}: {text}"


def resolve_device(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_query_local(
    model_name: str,
    device: str,
    query: str,
    normalize: bool,
    local_files_only: bool,
) -> List[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    vector = model.encode(
        [query],
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vector.astype(float).tolist()


def load_sentence_transformer(model_name: str, device: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device, local_files_only=local_files_only)


def encode_query_with_model(model, query: str, normalize: bool) -> List[float]:
    vector = model.encode(
        [query],
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vector.astype(float).tolist()


def score_vector_index(
    vector_index: Dict[str, Any],
    query_vector: List[float],
    top_k: int,
    regulasi_id: str = "",
    section_type: str = "",
) -> List[Dict[str, Any]]:
    metadata_by_chunk = {row["chunk_id"]: row for row in vector_index["metadata"]}
    scored: List[Tuple[float, Dict[str, Any]]] = []

    try:
        import numpy as np

        query_np = np.asarray(query_vector, dtype="float32")
        rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        vectors = []
        for embedding_row in vector_index["embeddings"]:
            metadata = metadata_by_chunk.get(embedding_row["chunk_id"], {})
            if regulasi_id and metadata.get("regulasi_id") != regulasi_id:
                continue
            if section_type and metadata.get("section_type") != section_type:
                continue
            rows.append((embedding_row, metadata))
            vectors.append(embedding_row["embedding"])
        if vectors:
            matrix = np.asarray(vectors, dtype="float32")
            scores = matrix @ query_np
            scored = [(float(score), metadata) for score, (_, metadata) in zip(scores, rows)]
    except Exception:
        for embedding_row in vector_index["embeddings"]:
            metadata = metadata_by_chunk.get(embedding_row["chunk_id"], {})
            if regulasi_id and metadata.get("regulasi_id") != regulasi_id:
                continue
            if section_type and metadata.get("section_type") != section_type:
                continue
            score = sum(float(a) * float(b) for a, b in zip(query_vector, embedding_row["embedding"]))
            scored.append((score, metadata))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: List[Dict[str, Any]] = []
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
                "metadata": metadata.get("metadata", {}),
            }
        )
    return results


def vector_search(
    vector_index: Dict[str, Any],
    query: str,
    top_k: int,
    regulasi_id: str = "",
    section_type: str = "",
    model_name: str = "",
    device: str = "auto",
    local_files_only: Optional[bool] = None,
    encoder_model: Any = None,
) -> List[Dict[str, Any]]:
    backend = vector_index.get("backend", "sentence_transformers")
    if backend != "sentence_transformers":
        raise RuntimeError(f"GraphRAG v2 currently expects sentence_transformers index, got: {backend}")

    model_name = model_name or vector_index.get("embedding_model") or "intfloat/multilingual-e5-base"
    normalize = bool(vector_index.get("normalize", True))
    local_files_only = bool(vector_index.get("local_files_only", True)) if local_files_only is None else local_files_only
    query_prefix = vector_index.get("query_prefix", "query")
    prefixed_query = apply_prefix(query, query_prefix)
    device = resolve_device(device)
    if encoder_model is not None:
        query_vector = encode_query_with_model(encoder_model, prefixed_query, normalize)
    else:
        query_vector = embed_query_local(model_name, device, prefixed_query, normalize, local_files_only)
    return score_vector_index(vector_index, query_vector, top_k, regulasi_id, section_type)


def graph_distance(graph: Dict[str, Any], start_id: str, target_ids: Set[str], max_hops: int = 2) -> Optional[int]:
    if not start_id or not target_ids:
        return None
    if start_id in target_ids:
        return 0

    adjacency = graph["adjacency"]
    visited = {start_id}
    queue: List[Tuple[str, int]] = [(start_id, 0)]
    while queue:
        node_id, distance = queue.pop(0)
        if distance >= max_hops:
            continue
        neighbors = []
        for edge_ref in adjacency.get("out", {}).get(node_id, []):
            neighbors.append(edge_ref.get("target_id", ""))
        for edge_ref in adjacency.get("in", {}).get(node_id, []):
            neighbors.append(edge_ref.get("source_id", ""))
        for neighbor_id in neighbors:
            if not neighbor_id or neighbor_id in visited:
                continue
            if neighbor_id in target_ids:
                return distance + 1
            visited.add(neighbor_id)
            queue.append((neighbor_id, distance + 1))
    return None


def rerank_vector_results(
    graph: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    exact_matches: Sequence[Dict[str, Any]],
    top_k: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not results:
        return []

    exact_node_ids = {match["node_id"] for match in exact_matches}
    exact_regulasi_ids = {
        match.get("node", {}).get("regulasi_id", "")
        for match in exact_matches
        if match.get("node", {}).get("regulasi_id")
    }
    exact_has_article_level = any(
        (match.get("node", {}).get("type") in {"Pasal", "Ayat", "SubItem"})
        for match in exact_matches
    )

    w_vector = weights.get("w_vector", 1.0) if weights else 1.0
    w_reg = weights.get("w_reg", 1.0) if weights else 1.0
    w_exact = weights.get("w_exact", 1.0) if weights else 1.0
    w_graph = weights.get("w_graph", 1.0) if weights else 1.0
    w_section = weights.get("w_section", 1.0) if weights else 1.0

    reranked = []
    for rank, result in enumerate(results, start=1):
        source_node_id = result.get("source_node_id", "")
        section_type = str(result.get("section_type", "")).lower()
        vector_score = float(result.get("score", 0.0))

        same_regulation_bonus = 0.0
        if exact_regulasi_ids and result.get("regulasi_id") in exact_regulasi_ids:
            same_regulation_bonus = 0.08

        exact_node_bonus = 0.0
        if source_node_id in exact_node_ids or result.get("chunk_id") in exact_node_ids:
            exact_node_bonus = 0.25

        distance = graph_distance(graph, source_node_id, exact_node_ids, max_hops=2)
        if distance == 0:
            graph_proximity_score = 0.18
        elif distance == 1:
            graph_proximity_score = 0.12
        elif distance == 2:
            graph_proximity_score = 0.06
        else:
            graph_proximity_score = 0.0

        section_priority_score = SECTION_PRIORITY.get(section_type, 0.0)
        if exact_has_article_level and section_type in {"mengingat", "menimbang", "penutup"}:
            section_priority_score -= 0.04

        # Keep the original vector rank as a tiny stable prior for ties.
        rank_prior = max(0.0, 0.01 - (rank * 0.0005))
        final_score = (
            (w_vector * vector_score)
            + (w_reg * same_regulation_bonus)
            + (w_exact * exact_node_bonus)
            + (w_graph * graph_proximity_score)
            + (w_section * section_priority_score)
            + rank_prior
        )

        scored = dict(result)
        scored["vector_score"] = round(vector_score, 6)
        scored["score"] = round(final_score, 6)
        scored["rerank_features"] = {
            "same_regulation_bonus": round(same_regulation_bonus, 6),
            "exact_node_bonus": round(exact_node_bonus, 6),
            "graph_distance": distance,
            "graph_proximity_score": round(graph_proximity_score, 6),
            "section_priority_score": round(section_priority_score, 6),
            "rank_prior": round(rank_prior, 6),
        }
        reranked.append(scored)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def edge_priority(edge: Dict[str, Any], seed_type: str = "") -> Tuple[int, str]:
    predicate = edge.get("predicate", "")
    if predicate in {"MEMILIKI_CHUNK", "MEMILIKI_AYAT", "MEMILIKI_SUBITEM"}:
        priority = 0
    elif predicate in LEGAL_PREDICATES:
        priority = 1
    elif predicate in SEMANTIC_PREDICATES:
        priority = 2
    elif predicate == "MENJELASKAN":
        priority = 3
    elif predicate in STRUCTURAL_PREDICATES:
        priority = 4
    else:
        priority = 9
    if seed_type == "Regulasi" and predicate.startswith("MEMILIKI_"):
        priority += 2
    return priority, predicate


def full_edge(edge_ref: Dict[str, Any], edge_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    edge_id = edge_ref.get("edge_id") or edge_ref.get("id")
    return edge_by_id.get(edge_id, edge_ref)


def expand_graph(
    graph: Dict[str, Any],
    seed_node_ids: Iterable[str],
    max_hops: int = 1,
    max_edges_per_node: int = 24,
) -> Dict[str, Any]:
    nodes = graph["nodes"]
    adjacency = graph["adjacency"]
    edge_by_id = {edge.get("id"): edge for edge in graph["edges"]}

    selected_nodes: Dict[str, Dict[str, Any]] = {}
    selected_edges: Dict[str, Dict[str, Any]] = {}
    queue: List[Tuple[str, int]] = []
    visited: Set[Tuple[str, int]] = set()

    for node_id in seed_node_ids:
        if node_id in nodes:
            queue.append((node_id, 0))
            selected_nodes[node_id] = node_brief(nodes[node_id])

    while queue:
        node_id, hop = queue.pop(0)
        if (node_id, hop) in visited:
            continue
        visited.add((node_id, hop))
        node = nodes.get(node_id, {})
        seed_type = node.get("type", "")

        incident: List[Dict[str, Any]] = []
        for edge_ref in adjacency.get("out", {}).get(node_id, []):
            edge = full_edge(edge_ref, edge_by_id)
            if edge.get("predicate") in EXPANSION_PREDICATES:
                incident.append(edge)
        for edge_ref in adjacency.get("in", {}).get(node_id, []):
            edge = full_edge(edge_ref, edge_by_id)
            if edge.get("predicate") in EXPANSION_PREDICATES:
                incident.append(edge)

        incident.sort(key=lambda item: edge_priority(item, seed_type))

        for edge in incident[:max_edges_per_node]:
            edge_id = edge.get("id") or f"{edge.get('source_id')}|{edge.get('predicate')}|{edge.get('target_id')}"
            selected_edges[edge_id] = edge_brief(edge)
            for adjacent_id in (edge.get("source_id"), edge.get("target_id")):
                if adjacent_id and adjacent_id in nodes:
                    selected_nodes[adjacent_id] = node_brief(nodes[adjacent_id])
                    adjacent_type = nodes[adjacent_id].get("type", "")
                    if hop + 1 < max_hops and adjacent_type not in LEAF_TYPES:
                        queue.append((adjacent_id, hop + 1))

    return {
        "seed_node_ids": [node_id for node_id in seed_node_ids if node_id in nodes],
        "nodes": list(selected_nodes.values()),
        "edges": list(selected_edges.values()),
        "max_hops": max_hops,
    }


def child_texts(graph: Dict[str, Any], node_id: str, limit: int = 8) -> List[Dict[str, str]]:
    nodes = graph["nodes"]
    edge_by_id = {edge.get("id"): edge for edge in graph["edges"]}
    rows: List[Dict[str, str]] = []
    for edge_ref in graph["adjacency"].get("out", {}).get(node_id, []):
        edge = full_edge(edge_ref, edge_by_id)
        if edge.get("predicate") not in {"MEMILIKI_AYAT", "MEMILIKI_SUBITEM", "MEMILIKI_CHUNK"}:
            continue
        target = nodes.get(edge.get("target_id"), {})
        text = normalize_spaces(target.get("text", ""))
        if not text:
            continue
        rows.append(
            {
                "node_id": target.get("id", ""),
                "reference_label": target.get("label", ""),
                "text": text,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def edge_sentence(edge: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> str:
    source = node_by_id.get(edge.get("source_id"), {})
    target = node_by_id.get(edge.get("target_id"), {})
    source_label = source.get("label") or edge.get("source_id")
    target_label = target.get("label") or edge.get("target_id")
    return f"{source_label} --{edge.get('predicate')}--> {target_label}"


def pack_context(
    graph: Dict[str, Any],
    exact_matches: Sequence[Dict[str, Any]],
    vector_results: Sequence[Dict[str, Any]],
    graph_context: Dict[str, Any],
    max_blocks: int = 8,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for match in exact_matches:
        node_id = match["node_id"]
        node = graph["nodes"].get(node_id, {})
        text = normalize_spaces(node.get("text", ""))
        if not text:
            children = child_texts(graph, node_id, limit=6)
            text = " ".join(row["text"] for row in children)
        key = f"exact:{node_id}"
        if text and key not in seen_keys:
            seen_keys.add(key)
            blocks.append(
                {
                    "source": "exact_reference",
                    "score": match.get("score", 1.0),
                    "node_id": node_id,
                    "chunk_id": "",
                    "regulasi_id": node.get("regulasi_id", ""),
                    "reference_label": node.get("label", node_id),
                    "text": truncate(text, 1200),
                }
            )

    exact_node_ids = {match["node_id"] for match in exact_matches}
    for result in vector_results:
        if result.get("source_node_id") in exact_node_ids or result.get("chunk_id") in exact_node_ids:
            continue
        key = f"vector:{result.get('chunk_id')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        blocks.append(
            {
                "source": "vector_search",
                "score": result.get("score", 0.0),
                "node_id": result.get("source_node_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "regulasi_id": result.get("regulasi_id", ""),
                "reference_label": result.get("reference_label", ""),
                "section_type": result.get("section_type", ""),
                "text": truncate(result.get("text", ""), 1200),
            }
        )
        if len(blocks) >= max_blocks:
            break

    node_by_id = {node["id"]: node for node in graph_context.get("nodes", [])}
    graph_facts = []
    for edge in graph_context.get("edges", []):
        if edge.get("predicate") in {"MEMILIKI_CHUNK"}:
            continue
        graph_facts.append(edge_sentence(edge, node_by_id))
        if len(graph_facts) >= 24:
            break

    if graph_facts:
        blocks.append(
            {
                "source": "graph_expansion",
                "score": None,
                "node_id": "",
                "chunk_id": "",
                "regulasi_id": "",
                "reference_label": "Graph expansion facts",
                "text": "\n".join(f"- {fact}" for fact in graph_facts),
            }
        )

    return blocks[: max_blocks + 1]


def retrieve_from_loaded_resources(
    graph: Dict[str, Any],
    original_query: str,
    top_k: int,
    max_hops: int,
    max_edges_per_node: int,
    max_context_blocks: int,
    regulasi_id: str = "",
    section_type: str = "",
    no_vector: bool = False,
    query_cleaning: bool = True,
    auto_skip_vector_exact_only: bool = True,
    rerank: bool = True,
    vector_candidate_multiplier: int = 4,
    weights: Optional[Dict[str, float]] = None,
    vector_index: Optional[Dict[str, Any]] = None,
    model_name: str = "",
    device: str = "auto",
    allow_download: bool = False,
    encoder_model: Any = None,
) -> Dict[str, Any]:
    refs = parse_references(original_query, regulasi_id)
    exact_matches = resolve_exact_matches(graph, refs)
    query_analysis = build_query_analysis(
        original_query,
        refs,
        query_cleaning=query_cleaning,
        auto_skip_vector_exact_only=auto_skip_vector_exact_only,
    )

    vector_results: List[Dict[str, Any]] = []
    dense_vector_results: List[Dict[str, Any]] = []
    vector_enabled = not no_vector and not query_analysis["vector_skipped_reason"]
    if vector_enabled:
        if vector_index is None:
            raise RuntimeError("vector_index is required when vector retrieval is enabled.")
        candidate_top_k = top_k
        if rerank:
            candidate_top_k = max(top_k, top_k * vector_candidate_multiplier)
        vector_results = vector_search(
            vector_index=vector_index,
            query=query_analysis["vector_query"] or original_query,
            top_k=candidate_top_k,
            regulasi_id=regulasi_id,
            section_type=section_type,
            model_name=model_name,
            device=device,
            local_files_only=not allow_download,
            encoder_model=encoder_model,
        )
        # Keep the raw dense top-k before GraphRAG reranking. This is the
        # mode-independent confidence signal available before routing.
        dense_vector_results = [dict(row) for row in vector_results[:top_k]]
        if rerank:
            vector_results = rerank_vector_results(graph, vector_results, exact_matches, top_k, weights)

    seed_node_ids: List[str] = []
    seed_node_ids.extend(match["node_id"] for match in exact_matches)
    seed_node_ids.extend(result["source_node_id"] for result in vector_results if result.get("source_node_id"))
    seed_node_ids.extend(result["chunk_id"] for result in vector_results if result.get("chunk_id"))
    seed_node_ids = list(dict.fromkeys(seed_node_ids))

    graph_context = expand_graph(
        graph,
        seed_node_ids=seed_node_ids,
        max_hops=max_hops,
        max_edges_per_node=max_edges_per_node,
    )
    context_blocks = pack_context(graph, exact_matches, vector_results, graph_context, max_context_blocks)

    return {
        "schema_version": "2.0.0",
        "retriever": "graphrag_retriever_v2",
        "query": original_query,
        "inputs": {
            "graph_index": "",
            "vector_index": "",
            "top_k": top_k,
            "max_hops": max_hops,
            "regulasi_id": regulasi_id,
            "section_type": section_type,
            "vector_enabled": vector_enabled,
            "rerank_enabled": rerank,
        },
        "query_analysis": query_analysis,
        "parsed_references": refs,
        "exact_matches": exact_matches,
        "dense_vector_results": dense_vector_results,
        "vector_results": vector_results,
        "graph_context": graph_context,
        "context_blocks": context_blocks,
    }


def retrieve(args: argparse.Namespace) -> Dict[str, Any]:
    graph = load_pickle(args.graph_index)
    vector_index = None
    if not args.no_vector:
        vector_index = load_pickle(args.vector_index)

    result = retrieve_from_loaded_resources(
        graph=graph,
        original_query=" ".join(args.query),
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_edges_per_node=args.max_edges_per_node,
        max_context_blocks=args.max_context_blocks,
        regulasi_id=args.regulasi_id,
        section_type=args.section_type,
        no_vector=args.no_vector,
        query_cleaning=args.query_cleaning,
        auto_skip_vector_exact_only=args.auto_skip_vector_exact_only,
        rerank=args.rerank,
        vector_candidate_multiplier=args.vector_candidate_multiplier,
        vector_index=vector_index,
        model_name=args.model,
        device=args.device,
        allow_download=args.allow_download,
    )
    result["inputs"]["graph_index"] = str(args.graph_index)
    result["inputs"]["vector_index"] = str(args.vector_index)
    return result


def print_human(result: Dict[str, Any]) -> None:
    print("GraphRAG Retrieval V2")
    print(f"query: {result['query']}")
    query_analysis = result.get("query_analysis", {})
    if query_analysis:
        print(f"vector_query: {query_analysis.get('vector_query', '') or '-'}")
        if query_analysis.get("vector_skipped_reason"):
            print(f"vector_skipped_reason: {query_analysis['vector_skipped_reason']}")
    print(
        "summary: "
        f"exact={len(result['exact_matches'])}, "
        f"vector={len(result['vector_results'])}, "
        f"graph_nodes={len(result['graph_context']['nodes'])}, "
        f"graph_edges={len(result['graph_context']['edges'])}"
    )

    if result["exact_matches"]:
        print("\nExact matches:")
        for i, match in enumerate(result["exact_matches"], start=1):
            node = match["node"]
            print(f"{i}. {node['id']} [{node['type']}] {node.get('label', '')}")
            if node.get("text"):
                print(f"   {truncate(node['text'], 240)}")

    if result["vector_results"]:
        print("\nVector results:")
        for i, row in enumerate(result["vector_results"], start=1):
            vector_part = ""
            if "vector_score" in row:
                vector_part = f" vector={row['vector_score']}"
            print(f"{i}. score={row['score']}{vector_part} {row['reference_label']}")
            print(f"   chunk_id={row['chunk_id']} source_node_id={row['source_node_id']}")
            if row.get("rerank_features"):
                features = row["rerank_features"]
                print(
                    "   rerank="
                    f"same_reg={features.get('same_regulation_bonus')}, "
                    f"exact={features.get('exact_node_bonus')}, "
                    f"dist={features.get('graph_distance')}, "
                    f"graph={features.get('graph_proximity_score')}, "
                    f"section={features.get('section_priority_score')}"
                )
            print(f"   {truncate(row['text'], 260)}")

    if result["context_blocks"]:
        print("\nPacked context:")
        for i, block in enumerate(result["context_blocks"], start=1):
            score = "" if block.get("score") is None else f" score={block.get('score')}"
            print(f"{i}. [{block['source']}]{score} {block.get('reference_label', '')}")
            ids = []
            if block.get("node_id"):
                ids.append(f"node={block['node_id']}")
            if block.get("chunk_id"):
                ids.append(f"chunk={block['chunk_id']}")
            if ids:
                print(f"   {' '.join(ids)}")
            print(f"   {truncate(block.get('text', ''), 520)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphRAG v2 retrieval: exact reference + vector search + graph expansion."
    )
    parser.add_argument("query", nargs="*", help="Natural language query or exact legal reference.")
    parser.add_argument("--graph-index", type=Path, default=DEFAULT_GRAPH_INDEX)
    parser.add_argument("--vector-index", type=Path, default=DEFAULT_VECTOR_INDEX)
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=os.getenv("GRAPHRAG_DEVICE", "auto"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=1)
    parser.add_argument("--max-edges-per-node", type=int, default=24)
    parser.add_argument("--max-context-blocks", type=int, default=8)
    parser.add_argument("--regulasi-id", default="")
    parser.add_argument("--section-type", default="")
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--query-cleaning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-skip-vector-exact-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vector-candidate-multiplier", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.query:
        raise SystemExit("Query is required.")

    result = retrieve(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
