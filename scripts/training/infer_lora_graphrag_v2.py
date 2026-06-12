import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.graphrag_retriever_v2 import (
    load_pickle,
    load_sentence_transformer,
    retrieve_from_loaded_resources,
    truncate,
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(input_text, legal_context):
    return (
        "### Instruksi:\n"
        "Gunakan konteks hukum berikut untuk menjawab pertanyaan. "
        "Jika konteks memuat rujukan antar pasal atau perubahan regulasi, gunakan sebagai dasar jawaban. "
        "Jika konteks tidak cukup, jawab seperlunya tanpa mengarang detail baru.\n\n"
        "Konteks Hukum:\n"
        f"{legal_context}\n\n"
        "Pertanyaan:\n"
        f"{input_text}\n"
        "### Jawaban:\n"
    )


def count_tokens(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def fit_prompt_to_budget(tokenizer, input_text, retrieval_result, max_chars_per_block, max_input_tokens):
    original_blocks = list(retrieval_result.get("context_blocks", []))
    if not original_blocks:
        graph_context = "Konteks hukum yang relevan tidak ditemukan."
        prompt = build_prompt(input_text, graph_context)
        return graph_context, prompt, count_tokens(tokenizer, prompt)

    min_chars_per_block = 160
    char_budget = max_chars_per_block
    active_blocks = list(original_blocks)

    while active_blocks:
        trial_result = dict(retrieval_result)
        trial_result["context_blocks"] = active_blocks
        graph_context = format_graphrag_context(trial_result, char_budget)
        prompt = build_prompt(input_text, graph_context)
        prompt_token_count = count_tokens(tokenizer, prompt)
        if prompt_token_count <= max_input_tokens:
            return graph_context, prompt, prompt_token_count

        if char_budget > min_chars_per_block:
            char_budget = max(min_chars_per_block, char_budget - 100)
            continue

        if len(active_blocks) > 1:
            active_blocks = active_blocks[:-1]
            char_budget = max_chars_per_block
            continue

        break

    trial_result = dict(retrieval_result)
    trial_result["context_blocks"] = active_blocks[:1]
    graph_context = format_graphrag_context(trial_result, min_chars_per_block)
    prompt = build_prompt(input_text, graph_context)
    prompt_token_count = count_tokens(tokenizer, prompt)
    return graph_context, prompt, prompt_token_count


def generate_answer(model, tokenizer, prompt, max_input_tokens=2048, max_new_tokens=1024):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def format_graphrag_context(result, max_chars_per_block):
    blocks = result.get("context_blocks", [])
    if not blocks:
        return "Konteks hukum yang relevan tidak ditemukan."

    lines = []
    for idx, block in enumerate(blocks, start=1):
        lines.append(f"[Konteks {idx}] {block.get('reference_label', '')}".strip())
        if block.get("regulasi_id"):
            lines.append(f"Regulasi: {block.get('regulasi_id', '')}")
        if block.get("node_id"):
            lines.append(f"Node: {block.get('node_id', '')}")
        if block.get("chunk_id"):
            lines.append(f"Chunk: {block.get('chunk_id', '')}")
        if block.get("score") is not None:
            lines.append(f"Skor: {block.get('score')}")
        lines.append(f"Teks: {truncate(block.get('text', ''), max_chars_per_block)}")
        lines.append("")
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_path", default="outputs/lora/llama2-7b-legal")
    parser.add_argument("--test_file", default="data/sft/test.jsonl")
    parser.add_argument("--output_file", default="outputs/predictions/lora_graphrag_v2_predictions.csv")
    parser.add_argument("--graph_index", default="indexes/graph_v2/graph_index.pkl")
    parser.add_argument("--vector_index", default="indexes/vector_v2_model/vector_index.pkl")
    parser.add_argument("--model", default="")
    parser.add_argument("--retrieval_device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_hops", type=int, default=1)
    parser.add_argument("--max_edges_per_node", type=int, default=24)
    parser.add_argument("--max_context_blocks", type=int, default=8)
    parser.add_argument("--max_chars_per_block", type=int, default=700)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-cleaning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-skip-vector-exact-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vector-candidate-multiplier", type=int, default=4)
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Inference Fine-tune + GraphRAG v2 ===", flush=True)
    print(f"Base model            : {args.base_model}", flush=True)
    print(f"Adapter path          : {args.adapter_path}", flush=True)
    print(f"Test file             : {args.test_file}", flush=True)
    print(f"Graph index           : {args.graph_index}", flush=True)
    print(f"Vector index          : {args.vector_index}", flush=True)
    print(f"Retrieval device      : {args.retrieval_device}", flush=True)
    print(f"top_k                 : {args.top_k}", flush=True)
    print(f"max_hops              : {args.max_hops}", flush=True)
    print(f"max_context_blocks    : {args.max_context_blocks}", flush=True)
    print(f"max_chars_per_block   : {args.max_chars_per_block}", flush=True)
    print(f"max_input_tokens      : {args.max_input_tokens}", flush=True)
    print(f"max_new_tokens        : {args.max_new_tokens}", flush=True)
    print(f"Output file           : {args.output_file}", flush=True)

    print("Loading graph and vector indexes...", flush=True)
    graph_index = load_pickle(Path(args.graph_index))
    graph = graph_index
    vector_index = None
    encoder_model = None
    encoder_model_name = args.model
    if not args.no_vector:
        vector_index = load_pickle(Path(args.vector_index))
        encoder_model_name = args.model or vector_index.get("embedding_model") or "intfloat/multilingual-e5-base"
        print(f"Encoder model         : {encoder_model_name}", flush=True)
        print("Loading vector encoder...", flush=True)
        encoder_model = load_sentence_transformer(
            model_name=encoder_model_name,
            device=args.retrieval_device,
            local_files_only=not args.allow_download,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    test_data = load_jsonl(args.test_file)
    if args.limit:
        test_data = test_data[:args.limit]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "input",
                "reference",
                "label",
                "ground_truth",
                "vector_query",
                "vector_skipped_reason",
                "graph_seed_nodes",
                "graph_context",
                "prompt_token_count",
                "prediction",
            ],
        )
        writer.writeheader()

        for idx, item in enumerate(test_data):
            input_text = item["input"]
            ground_truth = item["output"]
            metadata = item.get("metadata", {})
            reference = metadata.get("reference", "")
            label = metadata.get("label", "")

            retrieval_result = retrieve_from_loaded_resources(
                graph=graph,
                original_query=input_text,
                top_k=args.top_k,
                max_hops=args.max_hops,
                max_edges_per_node=args.max_edges_per_node,
                max_context_blocks=args.max_context_blocks,
                regulasi_id="",
                section_type="",
                no_vector=args.no_vector,
                query_cleaning=args.query_cleaning,
                auto_skip_vector_exact_only=args.auto_skip_vector_exact_only,
                rerank=args.rerank,
                vector_candidate_multiplier=args.vector_candidate_multiplier,
                vector_index=vector_index,
                model_name=encoder_model_name,
                device=args.retrieval_device,
                allow_download=args.allow_download,
                encoder_model=encoder_model,
            )
            graph_context, prompt, prompt_token_count = fit_prompt_to_budget(
                tokenizer=tokenizer,
                input_text=input_text,
                retrieval_result=retrieval_result,
                max_chars_per_block=args.max_chars_per_block,
                max_input_tokens=args.max_input_tokens,
            )

            prediction = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )

            writer.writerow(
                {
                    "index": idx,
                    "input": input_text,
                    "reference": reference,
                    "label": label,
                    "ground_truth": ground_truth,
                    "vector_query": retrieval_result.get("query_analysis", {}).get("vector_query", ""),
                    "vector_skipped_reason": retrieval_result.get("query_analysis", {}).get("vector_skipped_reason", ""),
                    "graph_seed_nodes": json.dumps(retrieval_result.get("graph_context", {}).get("seed_node_ids", []), ensure_ascii=False),
                    "graph_context": graph_context,
                    "prompt_token_count": prompt_token_count,
                    "prediction": prediction,
                }
            )
            f.flush()

            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{len(test_data)}", flush=True)

    print("Inference Fine-tune + GraphRAG v2 selesai.", flush=True)
    print(f"Hasil disimpan di: {output_path}", flush=True)


if __name__ == "__main__":
    main()
