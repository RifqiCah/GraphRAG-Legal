import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.graphrag_retriever_v2 import (
    clean_vector_query,
    load_pickle,
    load_sentence_transformer,
    truncate,
    vector_search,
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def configure_reproducibility(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_score_features(results):
    if not results:
        return None, None
    top1 = float(results[0]["score"])
    if len(results) < 5:
        return top1, None
    return top1, round(top1 - float(results[4]["score"]), 6)


def build_prompt(input_text, legal_context):
    return (
        "### Instruksi:\n"
        "Gunakan konteks hukum berikut untuk menjawab pertanyaan. "
        "Jika konteks tidak cukup, jawab seperlunya tanpa mengarang detail baru.\n\n"
        "Konteks Hukum:\n"
        f"{legal_context}\n\n"
        "Pertanyaan:\n"
        f"{input_text}\n"
        "### Jawaban:\n"
    )


def count_tokens(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def fit_prompt_to_budget(tokenizer, input_text, rag_results, max_chars_per_chunk, max_input_tokens):
    if not rag_results:
        legal_context = "Konteks hukum yang relevan tidak ditemukan."
        prompt = build_prompt(input_text, legal_context)
        return legal_context, prompt, count_tokens(tokenizer, prompt)

    min_chars_per_chunk = 160
    char_budget = max_chars_per_chunk
    active_results = list(rag_results)

    while active_results:
        legal_context = format_rag_context(active_results, char_budget)
        prompt = build_prompt(input_text, legal_context)
        prompt_token_count = count_tokens(tokenizer, prompt)
        if prompt_token_count <= max_input_tokens:
            return legal_context, prompt, prompt_token_count

        if char_budget > min_chars_per_chunk:
            char_budget = max(min_chars_per_chunk, char_budget - 100)
            continue

        if len(active_results) > 1:
            active_results = active_results[:-1]
            char_budget = max_chars_per_chunk
            continue

        break

    legal_context = format_rag_context(active_results[:1], min_chars_per_chunk)
    prompt = build_prompt(input_text, legal_context)
    prompt_token_count = count_tokens(tokenizer, prompt)
    return legal_context, prompt, prompt_token_count


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


def format_rag_context(results, max_chars_per_chunk):
    if not results:
        return "Konteks hukum yang relevan tidak ditemukan."

    lines = []
    for idx, row in enumerate(results, start=1):
        lines.append(f"[Dokumen {idx}] {row.get('reference_label', '')}".strip())
        lines.append(f"Regulasi: {row.get('regulasi_id', '')}")
        lines.append(f"Chunk: {row.get('chunk_id', '')}")
        lines.append(f"Skor: {row.get('score', '')}")
        lines.append(f"Teks: {truncate(row.get('text', ''), max_chars_per_chunk)}")
        lines.append("")
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_path", default="outputs/lora/llama2-7b-legal")
    parser.add_argument("--test_file", default="data/sft/test.jsonl")
    parser.add_argument("--output_file", default="outputs/predictions/lora_rag_v2_predictions.csv")
    parser.add_argument("--vector_index", default="indexes/vector_v2_model/vector_index.pkl")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_chars_per_chunk", type=int, default=700)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-cleaning", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Inference Fine-tune + RAG v2 ===", flush=True)
    print(f"Base model          : {args.base_model}", flush=True)
    print(f"Adapter path        : {args.adapter_path}", flush=True)
    print(f"Test file           : {args.test_file}", flush=True)
    print(f"Vector index        : {args.vector_index}", flush=True)
    print(f"Device              : {args.device}", flush=True)
    print(f"top_k               : {args.top_k}", flush=True)
    print(f"max_chars_per_chunk : {args.max_chars_per_chunk}", flush=True)
    print(f"max_input_tokens    : {args.max_input_tokens}", flush=True)
    print(f"max_new_tokens      : {args.max_new_tokens}", flush=True)
    print(f"seed                : {args.seed}", flush=True)
    print("decoding            : do_sample=False, temperature=0.0", flush=True)
    print(f"Output file         : {args.output_file}", flush=True)

    configure_reproducibility(args.seed)

    vector_index = load_pickle(Path(args.vector_index))
    encoder_model_name = args.model or vector_index.get("embedding_model") or "intfloat/multilingual-e5-base"
    print(f"Encoder model       : {encoder_model_name}", flush=True)
    print("Loading vector encoder...", flush=True)
    encoder_model = load_sentence_transformer(
        model_name=encoder_model_name,
        device=args.device,
        local_files_only=True,
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
                "retrieval_top1_score",
                "retrieval_score_margin",
                "retrieved_chunk_ids",
                "retrieved_regulasi_ids",
                "retrieved_context",
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

            vector_query = clean_vector_query(input_text) if args.query_cleaning else " ".join(str(input_text).split())
            if not vector_query:
                vector_query = " ".join(str(input_text).split())

            rag_results = vector_search(
                vector_index=vector_index,
                query=vector_query,
                top_k=args.top_k,
                model_name=encoder_model_name,
                device=args.device,
                local_files_only=True,
                encoder_model=encoder_model,
            )
            retrieval_top1_score, retrieval_score_margin = dense_score_features(rag_results)

            legal_context, prompt, prompt_token_count = fit_prompt_to_budget(
                tokenizer=tokenizer,
                input_text=input_text,
                rag_results=rag_results,
                max_chars_per_chunk=args.max_chars_per_chunk,
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
                    "vector_query": vector_query,
                    "retrieval_top1_score": retrieval_top1_score,
                    "retrieval_score_margin": retrieval_score_margin,
                    "retrieved_chunk_ids": json.dumps([row.get("chunk_id", "") for row in rag_results], ensure_ascii=False),
                    "retrieved_regulasi_ids": json.dumps([row.get("regulasi_id", "") for row in rag_results], ensure_ascii=False),
                    "retrieved_context": legal_context,
                    "prompt_token_count": prompt_token_count,
                    "prediction": prediction,
                }
            )
            f.flush()

            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{len(test_data)}", flush=True)

    print("Inference Fine-tune + RAG v2 selesai.", flush=True)
    print(f"Hasil disimpan di: {output_path}", flush=True)


if __name__ == "__main__":
    main()
