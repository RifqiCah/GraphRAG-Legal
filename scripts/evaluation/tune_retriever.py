#!/usr/bin/env python3
"""
Tuning Reranking Weights for Ablation Study.
Optimizes w_reg, w_exact, w_graph, w_section offline using validation set.
"""

import json
import itertools
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

from scripts.graph.graphrag_retriever_v2 import (
    load_pickle,
    retrieve_from_loaded_resources,
    parse_references,
    load_sentence_transformer,
)

def evaluate_offline_recall(ground_truth: str, context_blocks: List[Dict[str, Any]]) -> float:
    expected_refs = parse_references(ground_truth)
    if not expected_refs:
        return 1.0 # Nothing to recall
    
    # Combine all context text
    full_context = "\n".join([b.get("text", "") for b in context_blocks])
    context_refs = parse_references(full_context)
    
    found = 0
    for e_ref in expected_refs:
        e_reg = e_ref.get("regulasi_id", "")
        e_pasal = e_ref.get("pasal", "")
        e_ayat = e_ref.get("ayat", "")
        
        is_found = False
        for b in context_blocks:
            c_reg = b.get("regulasi_id", "")
            c_node = b.get("node_id", "")
            
            # Match by regulasi_id
            if e_reg and c_reg == e_reg:
                if not e_pasal:
                    is_found = True
                    break
                # Match pasal in node_id, eg: PSL-REG-UU-1-2023-10
                if e_pasal and f"-{e_pasal}" in c_node:
                    # if expected ayat, check if it's in node_id
                    if e_ayat:
                        if f"-{e_pasal}-{e_ayat}" in c_node:
                            is_found = True
                            break
                    else:
                        is_found = True
                        break
        
        if is_found:
            found += 1
            
    return found / len(expected_refs)

def evaluate_offline_mrr(ground_truth: str, context_blocks: List[Dict[str, Any]]) -> float:
    expected_refs = parse_references(ground_truth)
    if not expected_refs:
        return 1.0
    
    mrr_sum = 0.0
    for e_ref in expected_refs:
        e_reg = e_ref.get("regulasi_id", "")
        e_pasal = e_ref.get("pasal", "")
        e_ayat = e_ref.get("ayat", "")
        
        rank = 0
        for i, b in enumerate(context_blocks):
            c_reg = b.get("regulasi_id", "")
            c_node = b.get("node_id", "")
            
            if e_reg and c_reg == e_reg:
                if not e_pasal:
                    rank = i + 1
                    break
                if e_pasal and f"-{e_pasal}" in c_node:
                    if e_ayat:
                        if f"-{e_pasal}-{e_ayat}" in c_node:
                            rank = i + 1
                            break
                    else:
                        rank = i + 1
                        break
        
        if rank > 0:
            mrr_sum += 1.0 / rank
            
    return mrr_sum / len(expected_refs)

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
    
    # Preload encoder to speed up loop
    model_name = vector_index.get("embedding_model", "intfloat/multilingual-e5-base")
    print(f"Loading embedding model {model_name}...")
    encoder_model = load_sentence_transformer(model_name, device="cuda", local_files_only=True)
    
    print("Loading validation set...")
    val_data = []
    with open(validation_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100: # Limit to 100 samples for speed during grid search
                break
            val_data.append(json.loads(line))
            
    print(f"Loaded {len(val_data)} validation samples.")
    
    # Grid search space
    w_vectors = [1.0]
    w_regs = [0.0, 0.5, 1.0]
    w_exacts = [0.0, 0.5, 1.0]
    w_graphs = [0.0, 0.5, 1.0]
    w_sections = [0.0, 0.5, 1.0]
    
    combinations = list(itertools.product(w_vectors, w_regs, w_exacts, w_graphs, w_sections))
    print(f"Total combinations to test: {len(combinations)}")
    
    best_recall = -1.0
    best_weights = {}
    
    for w_vec, w_r, w_e, w_g, w_s in tqdm(combinations, desc="Grid Search"):
        weights = {
            "w_vector": w_vec,
            "w_reg": w_r,
            "w_exact": w_e,
            "w_graph": w_g,
            "w_section": w_s,
        }
        
        total_recall = 0.0
        for row in val_data:
            query = row.get("input", "")
            ground_truth = row.get("output", "")
            
            res = retrieve_from_loaded_resources(
                graph=graph,
                original_query=query,
                top_k=8,
                max_hops=1,
                max_edges_per_node=24,
                max_context_blocks=8,
                weights=weights,
                vector_index=vector_index,
                encoder_model=encoder_model,
                rerank=True,
            )
            
            recall = evaluate_offline_recall(ground_truth, res["context_blocks"])
            total_recall += recall
            
        mean_recall = total_recall / len(val_data)
        if mean_recall > best_recall:
            best_recall = mean_recall
            best_weights = weights
            
    print("\n=== TUNING RESULTS ===")
    print(f"Best Mean Recall: {best_recall:.4f}")
    print("Best Weights:")
    for k, v in best_weights.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
