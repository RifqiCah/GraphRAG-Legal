import argparse
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_EXCLUDED_REGULASI_IDS = {"REG-UUD-1945"}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def compact_properties(data: Dict[str, Any], skip_keys: Optional[set] = None) -> Dict[str, Any]:
    skip_keys = skip_keys or set()
    result = {}
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, list):
            result[key] = value
        elif isinstance(value, dict):
            result[key] = value
    return result


def merge_node(
    nodes: Dict[str, Dict[str, Any]],
    node_id: str,
    node_type: str,
    regulasi_id: str = "",
    text: str = "",
    label: str = "",
    properties: Optional[Dict[str, Any]] = None,
    is_stub: bool = False,
) -> None:
    if not node_id:
        return

    properties = properties or {}
    existing = nodes.get(node_id)

    if existing:
        if text and not existing.get("text"):
            existing["text"] = normalize_text(text)
        if label and not existing.get("label"):
            existing["label"] = label
        if regulasi_id and not existing.get("regulasi_id"):
            existing["regulasi_id"] = regulasi_id
        existing["properties"].update({k: v for k, v in properties.items() if v not in ("", None, [], {})})
        existing["is_stub"] = existing.get("is_stub", False) and is_stub
        return

    nodes[node_id] = {
        "id": node_id,
        "type": node_type,
        "regulasi_id": regulasi_id,
        "label": label,
        "text": normalize_text(text),
        "properties": properties,
        "is_stub": is_stub,
    }


def add_subitem_nodes(
    nodes: Dict[str, Dict[str, Any]],
    sub_items: List[Dict[str, Any]],
    regulasi_id: str,
    parent_id: str,
) -> None:
    for item in sub_items or []:
        item_id = item.get("id")
        merge_node(
            nodes,
            item_id,
            "SubItem",
            regulasi_id=regulasi_id,
            text=item.get("teks_isi", ""),
            label=f"{item.get('jenis', '')} {item.get('nomor', '')}".strip(),
            properties=compact_properties(item, {"children", "teks_isi"}),
        )
        for child in item.get("children", []) or []:
            add_subitem_nodes(nodes, [child], regulasi_id, item_id or parent_id)


def add_structure_nodes(nodes: Dict[str, Dict[str, Any]], data: Dict[str, Any]) -> None:
    regulasi_id = data["regulasi_id"]
    metadata = data.get("metadata", {})

    merge_node(
        nodes,
        regulasi_id,
        "Regulasi",
        regulasi_id=regulasi_id,
        text=metadata.get("judul", ""),
        label=metadata.get("judul", regulasi_id),
        properties=metadata,
    )

    preamble = data.get("preamble", {})
    for item in preamble.get("menimbang", []) or []:
        merge_node(
            nodes,
            item.get("id"),
            "Konsiderans",
            regulasi_id=regulasi_id,
            text=item.get("teks_isi", ""),
            label=f"Menimbang {item.get('nomor', '')}".strip(),
            properties=compact_properties(item, {"teks_isi"}),
        )

    for item in preamble.get("mengingat", []) or []:
        merge_node(
            nodes,
            item.get("id"),
            "Dasar_Hukum",
            regulasi_id=regulasi_id,
            text=item.get("teks_isi", ""),
            label=f"Mengingat {item.get('nomor', '')}".strip(),
            properties=compact_properties(item, {"teks_isi"}),
        )

    if preamble.get("memutuskan"):
        merge_node(
            nodes,
            f"DIKTUM-{regulasi_id}-MEMUTUSKAN",
            "Diktum",
            regulasi_id=regulasi_id,
            text=preamble.get("memutuskan", ""),
            label="Memutuskan",
            properties={"jenis": "memutuskan"},
        )
    if preamble.get("menetapkan"):
        merge_node(
            nodes,
            f"DIKTUM-{regulasi_id}-MENETAPKAN",
            "Diktum",
            regulasi_id=regulasi_id,
            text=preamble.get("menetapkan", ""),
            label="Menetapkan",
            properties={"jenis": "menetapkan"},
        )

    body = data.get("body", {})
    for bab in body.get("bab", []) or []:
        merge_node(
            nodes,
            bab.get("id"),
            "Bab",
            regulasi_id=regulasi_id,
            text=bab.get("judul", ""),
            label=f"Bab {bab.get('nomor_bab', '')}".strip(),
            properties=compact_properties(bab, {"bagian", "paragraf", "pasal_ids"}),
        )
        for bagian in bab.get("bagian", []) or []:
            merge_node(
                nodes,
                bagian.get("id"),
                "Bagian",
                regulasi_id=regulasi_id,
                text=bagian.get("judul", ""),
                label=f"Bagian {bagian.get('nomor_bagian', '')}".strip(),
                properties=compact_properties(bagian, {"paragraf", "pasal_ids"}),
            )
            for paragraf in bagian.get("paragraf", []) or []:
                merge_node(
                    nodes,
                    paragraf.get("id"),
                    "Paragraf",
                    regulasi_id=regulasi_id,
                    text=paragraf.get("judul", ""),
                    label=f"Paragraf {paragraf.get('nomor_paragraf', '')}".strip(),
                    properties=compact_properties(paragraf, {"pasal_ids"}),
                )

    for pasal in body.get("pasal", []) or []:
        pasal_id = pasal.get("id")
        merge_node(
            nodes,
            pasal_id,
            "Pasal",
            regulasi_id=regulasi_id,
            text=pasal.get("teks_isi", ""),
            label=f"Pasal {pasal.get('nomor_pasal', '')}".strip(),
            properties=compact_properties(pasal, {"ayat", "sub_items", "teks_isi"}),
        )
        add_subitem_nodes(nodes, pasal.get("sub_items", []), regulasi_id, pasal_id)

        for ayat in pasal.get("ayat", []) or []:
            ayat_id = ayat.get("id")
            merge_node(
                nodes,
                ayat_id,
                "Ayat",
                regulasi_id=regulasi_id,
                text=ayat.get("teks_isi", ""),
                label=f"Pasal {pasal.get('nomor_pasal', '')} ayat ({ayat.get('nomor_ayat', '')})",
                properties=compact_properties(ayat, {"sub_items", "teks_isi"}),
            )
            add_subitem_nodes(nodes, ayat.get("sub_items", []), regulasi_id, ayat_id)

    explanation = data.get("explanation", {})
    if explanation.get("available"):
        explanation_id = f"PJL-{regulasi_id}"
        merge_node(
            nodes,
            explanation_id,
            "Penjelasan",
            regulasi_id=regulasi_id,
            text="",
            label=f"Penjelasan {regulasi_id}",
            properties={"source_file": explanation.get("source_file", "")},
        )
        if explanation.get("umum"):
            merge_node(
                nodes,
                f"PJL-{regulasi_id}-UMUM",
                "Penjelasan_Umum",
                regulasi_id=regulasi_id,
                text=explanation.get("umum", ""),
                label="Penjelasan Umum",
                properties={"source_file": explanation.get("source_file", "")},
            )
        for item in explanation.get("pasal_demi_pasal", []) or []:
            merge_node(
                nodes,
                item.get("id"),
                "Penjelasan_Pasal",
                regulasi_id=regulasi_id,
                text=item.get("teks_isi", ""),
                label=f"Penjelasan Pasal {item.get('nomor_pasal', '')}".strip(),
                properties=compact_properties(item, {"teks_isi"}),
            )

    for item in data.get("attachments", []) or []:
        merge_node(
            nodes,
            item.get("id"),
            "Lampiran",
            regulasi_id=regulasi_id,
            text=item.get("teks_isi", ""),
            label=item.get("judul", "Lampiran"),
            properties=compact_properties(item, {"teks_isi"}),
        )

    closing = data.get("closing", {})
    if closing.get("teks_isi"):
        merge_node(
            nodes,
            f"PENUTUP-{regulasi_id}",
            "Penutup",
            regulasi_id=regulasi_id,
            text=closing.get("teks_isi", ""),
            label=f"Penutup {regulasi_id}",
            properties=compact_properties(closing, {"teks_isi"}),
        )


def add_chunk_nodes(nodes: Dict[str, Dict[str, Any]], chunks_dir: Path, excluded: set) -> None:
    for path in sorted(chunks_dir.glob("chunks_*.jsonl")):
        regulasi_id = path.stem.replace("chunks_", "")
        if regulasi_id in excluded:
            continue
        for chunk in iter_jsonl(path):
            merge_node(
                nodes,
                chunk.get("chunk_id"),
                "Chunk",
                regulasi_id=chunk.get("regulasi_id", regulasi_id),
                text=chunk.get("text", ""),
                label=chunk.get("reference_label", chunk.get("chunk_id", "")),
                properties=compact_properties(chunk, {"text"}),
            )


def load_triplets(triplets_dir: Path, excluded: set) -> List[Dict[str, Any]]:
    triplets = []
    for path in sorted(triplets_dir.glob("triplets_REG-*.json")):
        regulasi_id = path.stem.replace("triplets_", "")
        if regulasi_id in excluded:
            continue
        data = read_json(path)
        for idx, triplet in enumerate(data.get("triplets", []), start=1):
            row = dict(triplet)
            row["_triplet_file"] = str(path)
            row["_triplet_index"] = idx
            triplets.append(row)
    return triplets


def ensure_triplet_nodes(nodes: Dict[str, Dict[str, Any]], triplet: Dict[str, Any]) -> Tuple[bool, bool]:
    created_source = False
    created_target = False
    for side in ("source", "target"):
        node_id = triplet.get(f"{side}_id", "")
        node_type = triplet.get(f"{side}_type", "Unknown")
        if not node_id:
            continue
        if node_id not in nodes:
            evidence = triplet.get("evidence", {}) or {}
            properties = dict(triplet.get("properties", {}) or {})
            text = properties.get(f"{side}_name") or properties.get("target_name") or evidence.get("text", "")
            merge_node(
                nodes,
                node_id,
                node_type,
                regulasi_id=triplet.get("properties", {}).get("regulasi_id", ""),
                text=text,
                label=properties.get(f"{side}_name", properties.get("target_name", node_id)),
                properties=properties,
                is_stub=True,
            )
            if side == "source":
                created_source = True
            else:
                created_target = True
    return created_source, created_target


def build_graph(nodes: Dict[str, Dict[str, Any]], triplets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    edges = []
    stub_nodes_created = []

    for triplet in triplets:
        source_id = triplet.get("source_id", "")
        target_id = triplet.get("target_id", "")
        source_known = source_id in nodes
        target_known = target_id in nodes
        ensure_triplet_nodes(nodes, triplet)
        if source_id and not source_known and source_id in nodes:
            stub_nodes_created.append(source_id)
        if target_id and not target_known and target_id in nodes:
            stub_nodes_created.append(target_id)

    for edge_idx, triplet in enumerate(triplets, start=1):
        source_id = triplet.get("source_id")
        target_id = triplet.get("target_id")
        predicate = triplet.get("predicate")
        if not source_id or not target_id or not predicate:
            continue

        edge = {
            "id": f"EDGE-{edge_idx:08d}",
            "key": f"{predicate}#{edge_idx}",
            "predicate": predicate,
            "source_id": source_id,
            "target_id": target_id,
            "source_type": triplet.get("source_type", ""),
            "target_type": triplet.get("target_type", ""),
            "confidence": triplet.get("confidence"),
            "extraction_method": triplet.get("extraction_method", ""),
            "evidence": triplet.get("evidence", {}),
            "properties": triplet.get("properties", {}),
            "validation": triplet.get("validation", {}),
            "triplet_file": triplet.get("_triplet_file", ""),
            "triplet_index": triplet.get("_triplet_index", 0),
        }
        edges.append(edge)

    return edges, stub_nodes_created


def make_adjacency(edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    adjacency = {"out": {}, "in": {}}
    for edge in edges:
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        item = {
            "edge_id": edge["id"],
            "predicate": edge["predicate"],
            "source_id": source_id,
            "target_id": target_id,
        }
        adjacency["out"].setdefault(source_id, []).append(item)
        adjacency["in"].setdefault(target_id, []).append(item)
    return adjacency


def make_summary(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    excluded: set,
    stub_nodes_created: List[str],
) -> Dict[str, Any]:
    node_types = Counter(attrs.get("type", "Unknown") for attrs in nodes.values())
    edge_predicates = Counter(attrs.get("predicate", "UNKNOWN") for attrs in edges)
    edge_methods = Counter(attrs.get("extraction_method", "unknown") for attrs in edges)
    stub_types = Counter(
        nodes[node_id].get("type", "Unknown")
        for node_id in stub_nodes_created
        if node_id in nodes
    )

    chunk_ids = {node_id for node_id, attrs in nodes.items() if attrs.get("type") == "Chunk"}
    chunks_with_incoming = {
        attrs["target_id"]
        for attrs in edges
        if attrs.get("predicate") == "MEMILIKI_CHUNK"
    }
    orphan_chunks = sorted(chunk_ids - chunks_with_incoming)

    return {
        "schema_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "excluded_regulasi_ids": sorted(excluded),
        "nodes": {
            "total": len(nodes),
            "by_type": dict(node_types),
            "stub_total": len(set(stub_nodes_created)),
            "stub_by_type": dict(stub_types),
        },
        "edges": {
            "total": len(edges),
            "by_predicate": dict(edge_predicates),
            "by_extraction_method": dict(edge_methods),
        },
        "qc": {
            "orphan_chunks": len(orphan_chunks),
            "sample_orphan_chunks": orphan_chunks[:20],
        },
    }


def serializable_nodes(nodes: Dict[str, Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for node_id, attrs in sorted(nodes.items()):
        yield {"id": node_id, **attrs}


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    excluded = set(args.exclude or [])
    if args.include_uud:
        excluded.discard("REG-UUD-1945")

    nodes: Dict[str, Dict[str, Any]] = {}
    for path in sorted(args.processed_dir.glob("REG-*.json")):
        regulasi_id = path.stem
        if regulasi_id in excluded:
            continue
        add_structure_nodes(nodes, read_json(path))

    add_chunk_nodes(nodes, args.chunks_dir, excluded)
    triplets = load_triplets(args.triplets_dir, excluded)
    edges, stub_nodes_created = build_graph(nodes, triplets)
    adjacency = make_adjacency(edges)
    summary = make_summary(nodes, edges, excluded, stub_nodes_created)

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "graph_index.pkl").open("wb") as f:
            pickle.dump(
                {
                    "schema_version": "2.0.0",
                    "nodes": nodes,
                    "edges": edges,
                    "adjacency": adjacency,
                    "triplets": triplets,
                    "summary": summary,
                },
                f,
            )
        write_json(args.out_dir / "summary.json", summary)
        write_jsonl(args.out_dir / "nodes.jsonl", serializable_nodes(nodes))
        write_jsonl(args.out_dir / "edges.jsonl", edges)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local Knowledge Graph index from v2 outputs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v2"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks_v2"))
    parser.add_argument("--triplets-dir", type=Path, default=Path("data/triplets_v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("indexes/graph_v2"))
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
