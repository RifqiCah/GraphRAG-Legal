import json
from pathlib import Path
from scripts.graph.graphrag_retriever_v2 import parse_references, retrieve_from_loaded_resources, load_pickle, load_sentence_transformer
from scripts.evaluation.tune_retriever import evaluate_offline_recall

graph = load_pickle(Path("indexes/graph_v2/graph_index.pkl"))
vector_index = load_pickle(Path("indexes/vector_v2_model/vector_index.pkl"))
encoder_model = load_sentence_transformer("intfloat/multilingual-e5-base", device="cuda", local_files_only=True)

with open("data/sft/validation.jsonl") as f:
    for line in f:
        row = json.loads(line)
        gt = row.get("output", "")
        expected = parse_references(gt)
        if expected:
            query = row.get("instruction", "")
            res = retrieve_from_loaded_resources(
                graph=graph, original_query=query, top_k=8, max_hops=0,
                max_edges_per_node=24, max_context_blocks=8, weights={},
                vector_index=vector_index, encoder_model=encoder_model, rerank=False
            )
            score = evaluate_offline_recall(gt, res["context_blocks"])
            if score == 0.0:
                print(f"Failed Query: {query}")
                print(f"Expected: {expected}")
                print("Context retrieved:")
                for b in res["context_blocks"]:
                    print(f"  Got: reg={b.get('regulasi_id')} node={b.get('node_id')} ref={b.get('reference_label')}")
