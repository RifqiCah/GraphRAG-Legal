import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.graphrag_retriever import (
    load_graph_index,
    load_schema,
    retrieve_graphrag_context,
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(input_text, graph_context):
    system_prompt = (
        "Anda adalah asisten hukum untuk menjawab pertanyaan berdasarkan regulasi Indonesia.\n"
        "Anda akan diberi konteks hasil retrieval dari knowledge graph regulasi.\n\n"
        "Ringkasan struktur knowledge graph:\n"
        "- Tipe node utama: Regulasi, Pasal, dan Subjek_Hukum.\n"
        "- Node Regulasi merepresentasikan dokumen hukum seperti UU, PP, atau UUD.\n"
        "- Node Pasal merepresentasikan pasal atau ayat tertentu dalam regulasi.\n"
        "- Node Subjek_Hukum merepresentasikan entitas/topik hukum yang diatur.\n"
        "- Relasi MEMILIKI_PASAL berarti regulasi memiliki pasal tertentu.\n"
        "- Relasi MENGACU_PADA berarti suatu pasal/regulasi merujuk ke pasal atau regulasi lain.\n"
        "- Relasi MENCABUT, MENCABUT_PASAL, dan MENGAMANDEMEN menunjukkan perubahan status atau isi hukum.\n"
        "- Relasi MENDAHULUI menunjukkan hubungan temporal antar regulasi.\n"
        "- Relasi MENGATUR_TOPIK menunjukkan pasal yang mengatur subjek/topik hukum tertentu.\n\n"
        "Aturan menjawab:\n"
        "1. Gunakan bagian Konteks Hukum sebagai sumber utama jawaban jika konteks tersedia.\n"
        "2. Utamakan konteks dengan relasi SEED karena paling dekat dengan referensi pertanyaan.\n"
        "3. Jika ada relasi MENGACU_PADA, gunakan sebagai rujukan pendukung.\n"
        "4. Jika ada relasi MENCABUT, MENCABUT_PASAL, atau MENGAMANDEMEN, perhatikan karena dapat memengaruhi keberlakuan isi pasal.\n"
        "5. Gunakan teks pada node Pasal sebagai sumber substansi utama.\n"
        "6. Jika konteks menyatakan bahwa referensi tidak tersedia di knowledge graph, jawab berdasarkan pertanyaan dan hasil fine-tuning tanpa mengarang detail baru.\n"
        "7. Jangan menyebut istilah teknis seperti GraphRAG, node, edge, retrieval, seed node, atau coverage warning dalam jawaban akhir.\n"
        "8. Jangan membuat informasi baru yang tidak didukung oleh konteks atau pertanyaan.\n"
        "9. Jawablah ringkas, langsung, dan sesuai gaya jawaban dataset.\n"
        "10. Berikan jawaban final saja tanpa analisis proses.\n"
    )

    return (
        "### Sistem:\n"
        f"{system_prompt}\n\n"
        "### Konteks Hukum:\n"
        f"{graph_context}\n\n"
        "### Instruksi:\n"
        f"{input_text}\n\n"
        "### Jawaban:\n"
    )


def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--adapter_path", default="outputs/lora/llama2-7b-legal")
    parser.add_argument("--test_file", default="data/sft/test.jsonl")
    parser.add_argument("--output_file", default="outputs/predictions/lora_graphrag_predictions.csv")
    parser.add_argument("--graph_index", default="indexes/graph/graph_index.pkl")
    parser.add_argument("--schema", default="ontology/schema.json")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_context", type=int, default=5)
    parser.add_argument("--max_chars_per_node", type=int, default=700)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Inference Fine-tune + GraphRAG ===", flush=True)
    print(f"Base model          : {args.base_model}", flush=True)
    print(f"Adapter path        : {args.adapter_path}", flush=True)
    print(f"Test file           : {args.test_file}", flush=True)
    print(f"Graph index         : {args.graph_index}", flush=True)
    print(f"Schema              : {args.schema}", flush=True)
    print(f"top_k               : {args.top_k}", flush=True)
    print(f"max_context         : {args.max_context}", flush=True)
    print(f"max_chars_per_node  : {args.max_chars_per_node}", flush=True)
    print(f"max_new_tokens      : {args.max_new_tokens}", flush=True)
    print(f"Output file         : {args.output_file}", flush=True)

    print("Loading GraphRAG index...", flush=True)
    graph_index = load_graph_index(args.graph_index)
    graph = graph_index["graph"]
    schema_info = load_schema(args.schema)

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
                "graph_retrieval_mode",
                "graph_coverage_warning",
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

            graph_result = retrieve_graphrag_context(
                graph=graph,
                schema_info=schema_info,
                query=input_text,
                reference=reference,
                top_k=args.top_k,
                max_context=args.max_context,
                max_chars_per_node=args.max_chars_per_node,
            )

            graph_context = graph_result["context_text"]
            prompt = build_prompt(input_text, graph_context)
            prompt_token_count = len(tokenizer(prompt)["input_ids"])

            prediction = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            writer.writerow({
                "index": idx,
                "input": input_text,
                "reference": reference,
                "label": label,
                "ground_truth": ground_truth,
                "graph_retrieval_mode": graph_result.get("retrieval_mode", ""),
                "graph_coverage_warning": graph_result.get("coverage_warning", ""),
                "graph_seed_nodes": json.dumps(graph_result.get("seed_nodes", []), ensure_ascii=False),
                "graph_context": graph_context,
                "prompt_token_count": prompt_token_count,
                "prediction": prediction,
            })

            f.flush()

            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{len(test_data)}", flush=True)

    print("Inference Fine-tune + GraphRAG selesai.", flush=True)
    print(f"Hasil disimpan di: {output_path}", flush=True)


if __name__ == "__main__":
    main()
