import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_percent = 100 * trainable_params / total_params if total_params else 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percent": trainable_percent,
    }


def print_parameter_report(title, stats):
    print(f"\n=== {title} ===", flush=True)
    print(f"Total params     : {stats['total_params']:,}", flush=True)
    print(f"Trainable params : {stats['trainable_params']:,}", flush=True)
    print(f"Frozen params    : {stats['frozen_params']:,}", flush=True)
    print(f"Trainable %      : {stats['trainable_percent']:.6f}%", flush=True)


def build_prompt(example: Dict[str, Any]) -> str:
    return (
        "### Instruksi:\n"
        f"{example['input']}\n\n"
        "### Jawaban:\n"
    )


def tokenize_example(example, tokenizer, max_source_length: int, max_target_length: int):
    prompt = build_prompt(example)
    answer = str(example["output"]).strip() + tokenizer.eos_token

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_length,
    )

    answer_tokens = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]
    labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

    max_total_length = max_source_length + max_target_length

    input_ids = input_ids[:max_total_length]
    attention_mask = attention_mask[:max_total_length]
    labels = labels[:max_total_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        pad_token_id = self.tokenizer.pad_token_id

        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            labels = f["labels"]

            pad_len = max_len - len(input_ids)

            input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
            attention_mask_batch.append(attention_mask + [0] * pad_len)
            labels_batch.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()

    # Sesuai paper: LLaMA2-7B
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="Base model LLaMA2-7B sesuai paper",
    )

    parser.add_argument("--train_file", default="data/sft/train.jsonl")
    parser.add_argument("--eval_file", default="data/sft/validation.jsonl")
    parser.add_argument("--output_dir", default="outputs/lora/llama2-7b-legal")

    # Konfigurasi utama sesuai paper
    parser.add_argument("--max_source_length", type=int, default=2048)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)

    # LoRA biasa, bukan QLoRA
    # Paper menyebut PEFT LoRA, tetapi tidak merinci r/alpha/dropout.
    # Nilai ini adalah konfigurasi LoRA umum untuk LLaMA-family.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Setting training pendukung
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_total_length = args.max_source_length + args.max_target_length

    print("=== CUDA CHECK ===", flush=True)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)

    if torch.cuda.is_available():
        print(f"current_device: {torch.cuda.current_device()}", flush=True)
        print(f"device_name: {torch.cuda.get_device_name(0)}", flush=True)

    print("\n=== Training Configuration ===", flush=True)
    print(f"Model              : {args.model_name}", flush=True)
    print("Fine-tuning method : PEFT LoRA", flush=True)
    print(f"Epoch              : {args.num_train_epochs}", flush=True)
    print(f"Batch size         : {args.per_device_train_batch_size}", flush=True)
    print(f"Max source length  : {args.max_source_length}", flush=True)
    print(f"Max target length  : {args.max_target_length}", flush=True)
    print(f"Max total length   : {max_total_length}", flush=True)
    print(f"Output dir         : {args.output_dir}", flush=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.eval_file,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    tokenized_dataset = dataset.map(
        lambda example: tokenize_example(
            example,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    # Tidak mengubah metode menjadi QLoRA.
    # Ini hanya menghemat memori aktivasi saat backprop.
    model.gradient_checkpointing_enable()

    before_lora_stats = count_parameters(model)
    print_parameter_report("PARAMETERS BEFORE LORA", before_lora_stats)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    after_lora_stats = count_parameters(model)
    print_parameter_report("PARAMETERS AFTER LORA", after_lora_stats)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        optim="adamw_torch",
    )

    data_collator = CausalLMCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "model_name": args.model_name,
        "fine_tuning_method": "PEFT LoRA",
        "quantization": "none",
        "output_dir": str(output_dir),
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "max_total_length": max_total_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "parameters_before_lora": before_lora_stats,
        "parameters_after_lora": after_lora_stats,
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training selesai.", flush=True)
    print(f"Adapter LoRA tersimpan di: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
