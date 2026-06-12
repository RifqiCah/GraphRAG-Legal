import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(input_text):
    return (
        "### Instruksi:\n"
        f"{input_text}\n\n"
        "### Jawaban:\n"
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--test_file", default="data/sft/test.jsonl")
    parser.add_argument("--output_file", default="outputs/predictions/part1_zeroshot/zeroshot_predictions.csv")
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Inference Zero-shot only ===", flush=True)
    print(f"Base model        : {args.base_model}", flush=True)
    print(f"Test file         : {args.test_file}", flush=True)
    print(f"max_input_tokens  : {args.max_input_tokens}", flush=True)
    print(f"max_new_tokens    : {args.max_new_tokens}", flush=True)
    print(f"Output file       : {args.output_file}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
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

            prompt = build_prompt(input_text)
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
                    "prediction": prediction,
                }
            )
            f.flush()

            if (idx + 1) % 10 == 0:
                print(f"Generated {idx + 1}/{len(test_data)}", flush=True)

    print("Inference Zero-shot selesai.", flush=True)
    print(f"Hasil disimpan di: {output_path}", flush=True)


if __name__ == "__main__":
    main()
