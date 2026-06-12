import argparse
import pickle
from pathlib import Path


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def score_node(query, node_id, attrs):
    query = query.lower()
    text = str(attrs.get("text", "")).lower()
    title = str(attrs.get("title", "")).lower()
    node_id_l = node_id.lower()

    score = 0
    for token in query.split():
        if token in text:
            score += 2
        if token in title:
            score += 2
        if token in node_id_l:
            score += 1
    return score


def retrieve_seed_nodes(graph, query, top_k=3):
    candidates = []

    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("text"):
            continue

        score = score_node(query, node_id, attrs)
        if score > 0:
            candidates.append((score, node_id, attrs))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:top_k]


def expand_1hop(graph, seed_nodes, max_context=5):
    selected = []
    seen = set()

    def add_node(node_id, relation="SEED"):
        if node_id in seen:
            return
        attrs = graph.nodes[node_id]
        text = attrs.get("text", "")
        if not text:
            return
        seen.add(node_id)
        selected.append({
            "node_id": node_id,
            "relation": relation,
            "text": text,
            "type": attrs.get("type"),
            "regulasi_id": attrs.get("regulasi_id"),
            "nomor_pasal": attrs.get("nomor_pasal"),
            "nomor_ayat": attrs.get("nomor_ayat"),
        })

    for _, node_id, _ in seed_nodes:
        add_node(node_id, "SEED")

        for _, target, edge_data in graph.out_edges(node_id, data=True):
            if len(selected) >= max_context:
                break
            add_node(target, edge_data.get("predicate", "OUT"))

        for source, _, edge_data in graph.in_edges(node_id, data=True):
            if len(selected) >= max_context:
                break
            add_node(source, edge_data.get("predicate", "IN"))

        if len(selected) >= max_context:
            break

    return selected[:max_context]


def build_context(context_nodes):
    parts = []
    for i, item in enumerate(context_nodes, start=1):
        ref = item["node_id"]
        if item.get("nomor_pasal"):
            ref = f"{item.get('regulasi_id')} Pasal {item.get('nomor_pasal')} ayat {item.get('nomor_ayat')}"
        parts.append(
            f"[Konteks {i}]\n"
            f"Relasi: {item['relation']}\n"
            f"Referensi: {ref}\n"
            f"Teks: {item['text']}"
        )
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="indexes/graph/graph_index.pkl")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_context", type=int, default=5)
    args = parser.parse_args()

    index = load_index(Path(args.index))
    graph = index["graph"]

    seed_nodes = retrieve_seed_nodes(graph, args.query, args.top_k)
    context_nodes = expand_1hop(graph, seed_nodes, args.max_context)
    context = build_context(context_nodes)

    print("=== SEED NODES ===")
    for score, node_id, attrs in seed_nodes:
        print(f"score={score} | {node_id} | type={attrs.get('type')}")

    print("\n=== GRAPH CONTEXT ===")
    print(context)


if __name__ == "__main__":
    main()
