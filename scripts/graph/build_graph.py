import argparse
import json
import pickle
from pathlib import Path

import networkx as nx


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def build_processed_nodes(processed_dir: Path):
    nodes = {}

    for path in sorted(processed_dir.glob("*.json")):
        data = load_json(path)
        regulasi_id = data.get("regulasi_id")
        metadata = data.get("metadata", {})

        if not regulasi_id:
            continue

        nodes[regulasi_id] = {
            "id": regulasi_id,
            "type": "REGULASI",
            "text": metadata.get("judul", ""),
            "title": metadata.get("judul", ""),
            "metadata": metadata,
        }

        for pasal in data.get("pasal", []):
            nomor_pasal = str(pasal.get("nomor_pasal", "")).strip()

            for ayat in pasal.get("ayat", []):
                nomor_ayat = str(ayat.get("nomor_ayat", "0")).strip()
                node_id = f"PSL-{regulasi_id}-{nomor_pasal}-{nomor_ayat}"

                teks_isi = normalize_text(ayat.get("teks_isi", ""))
                sub_items = ayat.get("sub_items", [])

                nodes[node_id] = {
                    "id": node_id,
                    "type": "PASAL",
                    "regulasi_id": regulasi_id,
                    "nomor_pasal": nomor_pasal,
                    "nomor_ayat": nomor_ayat,
                    "text": teks_isi,
                    "sub_items": sub_items,
                    "metadata": metadata,
                }

    return nodes


def build_graph(nodes, triplets_dir: Path):
    graph = nx.MultiDiGraph()

    for node_id, attrs in nodes.items():
        graph.add_node(node_id, **attrs)

    all_triplets = []
    missing_nodes = set()

    for path in sorted(triplets_dir.glob("triplets_*.json")):
        data = load_json(path)

        for triplet in data.get("triplets", []):
            source_id = triplet.get("source_id")
            target_id = triplet.get("target_id")
            predicate = triplet.get("predicate")

            if not source_id or not target_id or not predicate:
                continue

            if source_id not in graph:
                graph.add_node(
                    source_id,
                    id=source_id,
                    type=infer_node_type(source_id),
                    text="",
                    metadata={},
                )
                missing_nodes.add(source_id)

            if target_id not in graph:
                graph.add_node(
                    target_id,
                    id=target_id,
                    type=infer_node_type(target_id),
                    text="",
                    metadata={},
                )
                missing_nodes.add(target_id)

            edge_attrs = {
                "predicate": predicate,
                "confidence": triplet.get("confidence"),
                "extraction_method": triplet.get("extraction_method"),
                "raw_match": triplet.get("raw_match", ""),
                "properties": triplet.get("properties", {}),
            }

            graph.add_edge(source_id, target_id, key=predicate, **edge_attrs)
            all_triplets.append(triplet)

    return graph, all_triplets, missing_nodes


def infer_node_type(node_id: str) -> str:
    if node_id.startswith("REG-"):
        return "REGULASI"
    if node_id.startswith("PSL-"):
        return "PASAL"
    if node_id.startswith("SBJ-"):
        return "SUBJECT"
    return "UNKNOWN"


def save_index(graph, nodes, triplets, missing_nodes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "graph": graph,
        "nodes": nodes,
        "triplets": triplets,
        "missing_nodes": sorted(missing_nodes),
    }

    with open(out_dir / "graph_index.pkl", "wb") as f:
        pickle.dump(index, f)

    summary = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_nodes_with_text": sum(
            1 for _, attrs in graph.nodes(data=True) if attrs.get("text")
        ),
        "num_triplets": len(triplets),
        "num_missing_nodes": len(missing_nodes),
        "sample_missing_nodes": sorted(missing_nodes)[:20],
        "sample_nodes": list(nodes.keys())[:10],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--triplets_dir", default="data/triplets")
    parser.add_argument("--out_dir", default="indexes/graph")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    triplets_dir = Path(args.triplets_dir)
    out_dir = Path(args.out_dir)

    nodes = build_processed_nodes(processed_dir)
    graph, triplets, missing_nodes = build_graph(nodes, triplets_dir)
    summary = save_index(graph, nodes, triplets, missing_nodes, out_dir)

    print("GraphRAG index berhasil dibuat.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
