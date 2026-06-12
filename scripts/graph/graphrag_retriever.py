import argparse
import json
import pickle
import re
from pathlib import Path


DEFAULT_SCHEMA_PATH = "ontology/schema.json"
DEFAULT_INDEX_PATH = "indexes/graph/graph_index.pkl"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph_index(index_path=DEFAULT_INDEX_PATH):
    with open(index_path, "rb") as f:
        return pickle.load(f)


def normalize_text(text):
    return " ".join(str(text or "").split())


def load_schema(schema_path=DEFAULT_SCHEMA_PATH):
    schema = load_json(Path(schema_path))
    edge_defs = schema.get("edge_definitions", {})

    explicit = edge_defs.get("explicit_relations", [])
    semantic = edge_defs.get("semantic_relations", [])

    relation_specs = {}
    explicit_labels = []
    semantic_labels = []

    for rel in explicit:
        label = rel.get("label")
        if label:
            relation_specs[label] = {
                **rel,
                "category": "explicit",
            }
            explicit_labels.append(label)

    for rel in semantic:
        label = rel.get("label")
        if label:
            relation_specs[label] = {
                **rel,
                "category": "semantic",
            }
            semantic_labels.append(label)

    relation_priority = {
        "MENGACU_PADA": 0,
        "MENGAMANDEMEN": 1,
        "MENCABUT_PASAL": 1,
        "MENCABUT": 1,
        "MEMILIKI_PASAL": 2,
        "MENGATUR_TOPIK": 3,
        "MENDAHULUI": 4,
        "BERTENTANGAN_DENGAN": 8,
    }

    valid_relations = set(relation_specs.keys())

    return {
        "raw": schema,
        "relation_specs": relation_specs,
        "explicit_labels": explicit_labels,
        "semantic_labels": semantic_labels,
        "valid_relations": valid_relations,
        "relation_priority": relation_priority,
    }


def infer_node_label(node_id):
    if node_id.startswith("REG-"):
        return "Regulasi"
    if node_id.startswith("PSL-"):
        return "Pasal"
    if node_id.startswith("SBJ-"):
        return "Subjek_Hukum"
    return "Unknown"


def graph_type_to_schema_label(graph_type):
    mapping = {
        "REGULASI": "Regulasi",
        "PASAL": "Pasal",
        "SUBJECT": "Subjek_Hukum",
        "SUBJEK_HUKUM": "Subjek_Hukum",
    }
    return mapping.get(str(graph_type or "").upper(), None)


def node_schema_label(graph, node_id):
    if node_id not in graph:
        return infer_node_label(node_id)

    attrs = graph.nodes[node_id]
    label = graph_type_to_schema_label(attrs.get("type"))
    return label or infer_node_label(node_id)


def allowed_target_types(spec):
    target_type = spec.get("target_type")
    if isinstance(target_type, list):
        return set(target_type)
    if isinstance(target_type, str):
        return {target_type}
    return set()


def is_edge_schema_valid(graph, source_id, target_id, predicate, schema_info):
    relation_specs = schema_info["relation_specs"]

    if predicate not in relation_specs:
        return False

    spec = relation_specs[predicate]
    source_label = node_schema_label(graph, source_id)
    target_label = node_schema_label(graph, target_id)

    expected_source = spec.get("source_type")
    expected_targets = allowed_target_types(spec)

    if expected_source and source_label != expected_source:
        return False

    if expected_targets and target_label not in expected_targets:
        return False

    return True


def normalize_regulation_kind(kind):
    kind = str(kind or "").strip()

    mapping = {
        "PP": "PP",
        "UU": "UU",
        "UUD": "UUD",
        "PERPRES": "Perpres",
        "PERMENDIKBUD": "Permendikbud",
    }

    return mapping.get(kind.upper(), kind)


def normalize_regulation_number(number):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(number or "").strip().upper()).strip("-")


def build_regulasi_id(kind, number, year=None):
    kind = normalize_regulation_kind(kind)
    number_raw = str(number or "").strip()
    year_raw = str(year or "").strip()

    if not kind or not number_raw:
        return None

    # Special handling for tokens like "1/PNPS/1965".
    pnps_match = re.match(
        r"(?i)^([A-Za-z0-9]+)[/\-](PNPS|PENPRES|PRP)[/\-](\d{4})$",
        number_raw,
    )
    if kind == "UU" and pnps_match:
        nomor = normalize_regulation_number(pnps_match.group(1))
        regime = normalize_regulation_number(pnps_match.group(2))
        tahun = normalize_regulation_number(pnps_match.group(3))
        return f"REG-UU-{nomor}-{regime}-{tahun}"

    nomor = normalize_regulation_number(number_raw)
    tahun = normalize_regulation_number(year_raw)

    if tahun:
        return f"REG-{kind}-{nomor}-{tahun}"
    return f"REG-{kind}-{nomor}"


def parse_reference(reference):
    """
    Contoh yang didukung:
    - Pasal 94 ayat (5)
    - PP 48 Tahun 2008, ... Pasal 32
    - PP 4 Tahun 2022 ... Pasal 5 ayat (2)
    - PP No 37 2009 Dosen per Pasal 6
    - PP Nomor 74 Tahun 2008 Pasal 6
    - UU 14 Tahun 2005 Pasal 1
    - UUD 1945
    """
    ref = str(reference or "")

    regulasi_id = None

    if re.search(r"\bUUD\b.*1945", ref, flags=re.IGNORECASE):
        regulasi_id = "REG-UUD-1945"
    else:
        reg_match = re.search(
            r"\b(PP|UU|Perpres|Permendikbud)"
            r"\s*(?:No\.?|Nomor)?\s*"
            r"([A-Za-z0-9\/\.-]+)"
            r"\s*(?:Tahun\s*(\d{4}))?",
            ref,
            flags=re.IGNORECASE,
        )

        if reg_match:
            regulasi_id = build_regulasi_id(
                reg_match.group(1),
                reg_match.group(2),
                reg_match.group(3),
            )

    pasal_match = re.search(
        r"\bPasal\s+([0-9A-Za-z]+)",
        ref,
        flags=re.IGNORECASE,
    )

    ayat_match = re.search(
        r"\bayat\s*\(?([0-9A-Za-z]+)\)?",
        ref,
        flags=re.IGNORECASE,
    )

    return {
        "regulasi_id": regulasi_id,
        "pasal": pasal_match.group(1) if pasal_match else None,
        "ayat": ayat_match.group(1) if ayat_match else None,
    }


def find_nodes_by_reference(graph, reference):
    parsed = parse_reference(reference)
    regulasi_id = parsed["regulasi_id"]
    pasal = parsed["pasal"]
    ayat = parsed["ayat"]

    matches = []

    if regulasi_id and pasal and ayat:
        exact_id = f"PSL-{regulasi_id}-{pasal}-{ayat}"
        if exact_id in graph:
            return [exact_id]

    if regulasi_id and pasal:
        prefix = f"PSL-{regulasi_id}-{pasal}-"
        for node_id in graph.nodes:
            if node_id.startswith(prefix):
                matches.append(node_id)

    if matches:
        return sorted(
            matches,
            key=lambda x: (
                0 if x.endswith("-0") else 1,
                x,
            ),
        )

    if regulasi_id and regulasi_id in graph:
        return [regulasi_id]

    return []


def node_has_text(graph, node_id):
    if node_id not in graph:
        return False
    return bool(normalize_text(graph.nodes[node_id].get("text", "")))


def is_pasal_node(node_id):
    return str(node_id).startswith("PSL-")


def is_regulasi_node(node_id):
    return str(node_id).startswith("REG-")


def keyword_score(query, node_id, attrs):
    query = str(query or "").lower()
    text = str(attrs.get("text", "")).lower()
    title = str(attrs.get("title", "")).lower()
    node_id_l = node_id.lower()

    stopwords = {
        "apa", "yang", "harus", "dalam", "dan", "atau", "di", "ke", "dari",
        "untuk", "pada", "dengan", "sebagai", "terkait", "bagaimana",
        "siapa", "kapan", "dimana", "mengapa", "adalah", "itu", "oleh",
        "agar", "dapat", "sebutkan", "jelaskan", "berikan", "mengenai",
        "isi", "pasal", "ayat", "tahun", "nomor"
    }

    tokens = [
        t for t in re.findall(r"[a-zA-Z0-9]+", query)
        if len(t) > 2 and t not in stopwords
    ]

    score = 0.0
    for token in tokens:
        if token in text:
            score += 2.0
        if token in title:
            score += 3.0
        if token in node_id_l:
            score += 1.0

    text_len = len(text)
    if text_len > 2500:
        score -= 2.0
    elif text_len > 1500:
        score -= 1.0

    if attrs.get("type") == "PASAL":
        score += 0.5

    return score


def fallback_keyword_search(graph, query, top_k=3):
    candidates = []

    for node_id, attrs in graph.nodes(data=True):
        if not attrs.get("text"):
            continue

        score = keyword_score(query, node_id, attrs)
        if score > 0:
            candidates.append((score, node_id))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return [node_id for _, node_id in candidates[:top_k]]


def add_context_node(graph, node_id, selected, seen, relation, direction):
    if node_id in seen or node_id not in graph:
        return

    attrs = graph.nodes[node_id]
    text = normalize_text(attrs.get("text", ""))

    if not text:
        return

    seen.add(node_id)
    selected.append({
        "node_id": node_id,
        "relation": relation,
        "direction": direction,
        "schema_label": node_schema_label(graph, node_id),
        "type": attrs.get("type"),
        "regulasi_id": attrs.get("regulasi_id"),
        "nomor_pasal": attrs.get("nomor_pasal"),
        "nomor_ayat": attrs.get("nomor_ayat"),
        "text": text,
    })


def collect_schema_valid_neighbors(graph, seed, schema_info):
    relation_priority = schema_info["relation_priority"]
    valid_relations = schema_info["valid_relations"]

    neighbors = []

    for _, target, edge_data in graph.out_edges(seed, data=True):
        predicate = edge_data.get("predicate", "")

        if predicate not in valid_relations:
            continue

        if not is_edge_schema_valid(graph, seed, target, predicate, schema_info):
            continue

        priority = relation_priority.get(predicate, 9)
        confidence = edge_data.get("confidence") or 0
        neighbors.append((priority, -confidence, predicate, target, "OUT"))

    for source, _, edge_data in graph.in_edges(seed, data=True):
        predicate = edge_data.get("predicate", "")

        if predicate not in valid_relations:
            continue

        if not is_edge_schema_valid(graph, source, seed, predicate, schema_info):
            continue

        priority = relation_priority.get(predicate, 9)
        confidence = edge_data.get("confidence") or 0
        neighbors.append((priority, -confidence, predicate, source, "IN"))

    neighbors.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return neighbors


def expand_context(graph, seed_nodes, schema_info, max_context=5):
    selected = []
    seen = set()

    for seed in seed_nodes:
        add_context_node(graph, seed, selected, seen, "SEED", "SELF")

        neighbors = collect_schema_valid_neighbors(graph, seed, schema_info)

        for _, _, predicate, neighbor, direction in neighbors:
            if len(selected) >= max_context:
                break
            add_context_node(graph, neighbor, selected, seen, predicate, direction)

        if len(selected) >= max_context:
            break

    return selected[:max_context]


def format_reference(item):
    if item.get("schema_label") == "Pasal" and item.get("regulasi_id"):
        return (
            f"{item.get('regulasi_id')} "
            f"Pasal {item.get('nomor_pasal')} "
            f"ayat {item.get('nomor_ayat')}"
        )
    return item["node_id"]


def build_context_text(context_nodes, max_chars_per_node=1200):
    parts = []

    for idx, item in enumerate(context_nodes, start=1):
        text = item["text"]
        if len(text) > max_chars_per_node:
            text = text[:max_chars_per_node].rstrip() + "..."

        parts.append(
            f"[Konteks {idx}]\n"
            f"Tipe Node: {item.get('schema_label')}\n"
            f"Relasi: {item['relation']} ({item['direction']})\n"
            f"Referensi: {format_reference(item)}\n"
            f"Teks: {text}"
        )

    return "\n\n".join(parts)


def build_no_context_message(reason):
    if reason == "reference_node_without_text_or_pasal":
        return (
            "Konteks hukum spesifik untuk referensi ini tidak tersedia di knowledge graph. "
            "Jawab berdasarkan pertanyaan dan pengetahuan hasil fine-tuning, tanpa mengarang detail baru."
        )

    if reason == "reference_not_found":
        return (
            "Referensi tidak ditemukan secara langsung di knowledge graph. "
            "Gunakan konteks hasil pencarian kata kunci jika tersedia, dan jangan mengarang detail baru."
        )

    if reason == "empty_context_after_expansion":
        return (
            "Konteks hukum yang memiliki teks tidak ditemukan setelah ekspansi graph. "
            "Jawab berdasarkan pertanyaan dan pengetahuan hasil fine-tuning, tanpa mengarang detail baru."
        )

    return (
        "Konteks hukum tidak tersedia. "
        "Jawab berdasarkan pertanyaan dan pengetahuan hasil fine-tuning, tanpa mengarang detail baru."
    )


def retrieve_graphrag_context(
    graph,
    schema_info,
    query,
    reference="",
    top_k=3,
    max_context=5,
    max_chars_per_node=1200,
):
    seed_nodes = find_nodes_by_reference(graph, reference)

    retrieval_mode = "reference"
    coverage_warning = ""

    if seed_nodes:
        has_text_seed = any(node_has_text(graph, node_id) for node_id in seed_nodes)
        has_pasal_seed = any(is_pasal_node(node_id) for node_id in seed_nodes)

        if not has_text_seed and not has_pasal_seed:
            coverage_warning = "reference_node_without_text_or_pasal"
            return {
                "retrieval_mode": retrieval_mode,
                "coverage_warning": coverage_warning,
                "seed_nodes": seed_nodes[:top_k],
                "context_nodes": [],
                "context_text": build_no_context_message(coverage_warning),
            }
    else:
        seed_nodes = fallback_keyword_search(graph, query, top_k=top_k)
        retrieval_mode = "keyword_fallback"
        coverage_warning = "reference_not_found"

    seed_nodes = seed_nodes[:top_k]
    context_nodes = expand_context(
        graph=graph,
        seed_nodes=seed_nodes,
        schema_info=schema_info,
        max_context=max_context,
    )

    if not context_nodes:
        if not coverage_warning:
            coverage_warning = "empty_context_after_expansion"

        return {
            "retrieval_mode": retrieval_mode,
            "coverage_warning": coverage_warning,
            "seed_nodes": seed_nodes,
            "context_nodes": [],
            "context_text": build_no_context_message(coverage_warning),
        }

    context_text = build_context_text(
        context_nodes,
        max_chars_per_node=max_chars_per_node,
    )

    return {
        "retrieval_mode": retrieval_mode,
        "coverage_warning": coverage_warning,
        "seed_nodes": seed_nodes,
        "context_nodes": context_nodes,
        "context_text": context_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_context", type=int, default=5)
    parser.add_argument("--max_chars_per_node", type=int, default=1200)
    args = parser.parse_args()

    index = load_graph_index(args.index)
    graph = index["graph"]
    schema_info = load_schema(args.schema)

    result = retrieve_graphrag_context(
        graph=graph,
        schema_info=schema_info,
        query=args.query,
        reference=args.reference,
        top_k=args.top_k,
        max_context=args.max_context,
        max_chars_per_node=args.max_chars_per_node,
    )

    print("=== RETRIEVAL MODE ===")
    print(result["retrieval_mode"])

    print("\n=== COVERAGE WARNING ===")
    print(result.get("coverage_warning", ""))

    print("\n=== SEED NODES ===")
    for node_id in result["seed_nodes"]:
        print(node_id)

    print("\n=== GRAPH CONTEXT ===")
    print(result["context_text"])


if __name__ == "__main__":
    main()
