#!/usr/bin/env python3
"""
Ablation Study for LegalRef-GraphRAG
Tests incremental features:
dense RAG -> + legal-unit chunking -> + graph expansion -> + B_reg -> + B_exact -> + B_section -> full LegalRef-GraphRAG
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import pandas as pd

from scripts.graph.graphrag_retriever_v2 import (
    load_pickle,
    retrieve_from_loaded_resources,
    parse_references,
    load_sentence_transformer,
)
from scripts.evaluation.tune_retriever import evaluate_offline_recall

def run_ablation(
    val_data: List[Dict],
    graph: Dict,
    vector_index: Dict,
    encoder_model: Any,
    config_name: str,
    rerank: bool,
    max_hops: int,
    weights: Dict[str, float]
) -> float:
    total_recall = 0.0
    for row in tqdm(val_data, desc=config_name):
        query = row.get("input", "")
        ground_truth = row.get("output", "")
        
        res = retrieve_from_loaded_resources(
            graph=graph,
            original_query=query,
            top_k=8,
            max_hops=max_hops,
            max_edges_per_node=24,
            max_context_blocks=8,
            weights=weights,
            vector_index=vector_index,
            encoder_model=encoder_model,
            rerank=rerank,
        )
        
        # If graph expansion is disabled, remove graph_expansion blocks manually 
        # (since retrieve_from_loaded_resources might still append exact match graph facts)
        blocks = res["context_blocks"]
        if max_hops == 0:
            blocks = [b for b in blocks if b.get("source") != "graph_expansion"]
            
        recall = evaluate_offline_recall(ground_truth, blocks)
        total_recall += recall
        
    return total_recall / len(val_data)

def main():
    graph_index_path = Path("indexes/graph_v2/graph_index.pkl")
    vector_index_path = Path("indexes/vector_v2_model/vector_index.pkl")
    validation_path = Path("data/sft/validation.jsonl")
    
    if not graph_index_path.exists() or not vector_index_path.exists() or not validation_path.exists():
        print("Required files not found. Ensure indexes and validation set exist.")
        sys.exit(1)

    print("Loading indexes...")
    graph = load_pickle(graph_index_path)
    vector_index = load_pickle(vector_index_path)
    
    model_name = vector_index.get("embedding_model", "intfloat/multilingual-e5-base")
    print(f"Loading embedding model {model_name}...")
    encoder_model = load_sentence_transformer(model_name, device="cuda", local_files_only=True)
    
    print("Loading validation set...")
    val_data = []
    with open(validation_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100: # Limit to 100 for speed
                break
            val_data.append(json.loads(line))
            
    print(f"Loaded {len(val_data)} validation samples.")
    
    # 1. Dense RAG + Legal Unit Chunking (Our baseline vector index is already legal-unit chunked)
    score_base = run_ablation(
        val_data, graph, vector_index, encoder_model, 
        config_name="Dense RAG + Legal Unit Chunking",
        rerank=False, max_hops=0, weights={}
    )
    
    # 2. + Graph Expansion
    score_graph = run_ablation(
        val_data, graph, vector_index, encoder_model, 
        config_name="+ Graph Expansion",
        rerank=False, max_hops=1, weights={}
    )
    
    # 3. + B_reg
    score_breg = run_ablation(
        val_data, graph, vector_index, encoder_model, 
        config_name="+ B_reg",
        rerank=True, max_hops=1, weights={"w_vector": 1.0, "w_reg": 1.0, "w_exact": 0.0, "w_graph": 0.0, "w_section": 0.0}
    )
    
    # 4. + B_exact
    score_bexact = run_ablation(
        val_data, graph, vector_index, encoder_model, 
        config_name="+ B_exact",
        rerank=True, max_hops=1, weights={"w_vector": 1.0, "w_reg": 1.0, "w_exact": 1.0, "w_graph": 0.0, "w_section": 0.0}
    )
    
    # 5. Full LegalRef-GraphRAG (All bonuses including B_section and B_graph)
    # Note: Use the best weights from tune_retriever.py here ideally. We use 1.0 as fallback.
    score_full = run_ablation(
        val_data, graph, vector_index, encoder_model, 
        config_name="Full LegalRef-GraphRAG (+ B_section)",
        rerank=True, max_hops=1, weights={"w_vector": 1.0, "w_reg": 1.0, "w_exact": 1.0, "w_graph": 1.0, "w_section": 1.0}
    )
    
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS (Offline Recall on Validation Set)")
    print("="*50)
    
    results = [
        {"Stage": "1. Dense RAG + Legal Unit Chunking", "Recall": score_base},
        {"Stage": "2. + Graph Expansion", "Recall": score_graph},
        {"Stage": "3. + B_reg", "Recall": score_breg},
        {"Stage": "4. + B_exact", "Recall": score_bexact},
        {"Stage": "5. Full LegalRef-GraphRAG (+ B_section)", "Recall": score_full},
    ]
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    df.to_csv("results/ablation_study.csv", index=False)
    print("\nSaved to results/ablation_study.csv")

if __name__ == "__main__":
    main()
