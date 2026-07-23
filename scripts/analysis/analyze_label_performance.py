#!/usr/bin/env python3
"""
Analyze RAG vs GraphRAG performance by heuristic question categories.

Default run:
    .venv/bin/python scripts/analysis/analyze_label_performance.py

Optional:
    .venv/bin/python scripts/analysis/analyze_label_performance.py \
      --test-data data/sft/test.jsonl \
      --reports-dir outputs/reports \
      --predictions-dir outputs/predictions \
      --output-dir reports/label_analysis

The script reads existing per-row evaluation metrics when available. It does
not rerun model inference.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


METRIC_COLUMNS = ["bleu_4", "meteor", "rouge_l", "embedding_cosine"]
GROUP_COLUMNS = ["dataset_task", "question_word", "question_intent", "reference_type", "answer_length_bin"]
FULL_DATASET_TASK_COUNTS = {
    "legal_overlapping_analysis": 1087,
    "element_extraction": 890,
    "legal_qna": 1794,
    "legal_summarization": 1415,
    "drafting_revisions": 1571,
    "drafting_provisions": 1750,
}
TASK_ORDER = list(FULL_DATASET_TASK_COUNTS) + ["unknown_task"]
TASK_DISPLAY_LABELS = {
    "legal_overlapping_analysis": "Overlap analysis",
    "element_extraction": "Element extraction",
    "legal_qna": "Legal Q&A",
    "legal_summarization": "Summarization",
    "drafting_revisions": "Drafting revisions",
    "drafting_provisions": "Drafting provisions",
    "unknown_task": "Unknown task",
}


@dataclass(frozen=True)
class RunConfig:
    model: str
    training_state: str
    method: str
    source_path: Path
    source_kind: str

    @property
    def config_id(self) -> str:
        return f"{self.model}_{self.training_state}_{self.method}".replace(" ", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze label/category performance for RAG vs GraphRAG.")
    parser.add_argument("--test-data", type=Path, default=Path("data/sft/test.jsonl"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/label_analysis"))
    parser.add_argument(
        "--include-only",
        default="",
        help="Comma-separated substrings for config_id or source path filtering.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--recompute-missing-metrics",
        action="store_true",
        help="Recompute metrics from predictions only when per-row metric CSVs are missing.",
    )
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_spaces(text: object) -> str:
    return " ".join(str(text or "").split())


def parse_input_parts(input_text: str) -> Tuple[str, str]:
    text = str(input_text or "")
    reference = ""
    question = ""

    ref_match = re.search(r"Referensi:\s*(.*?)(?:\n\s*Pertanyaan:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if ref_match:
        reference = normalize_spaces(ref_match.group(1))

    question_match = re.search(r"Pertanyaan:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if question_match:
        question = normalize_spaces(question_match.group(1))
    else:
        question = normalize_spaces(text)

    return reference, question


def extract_label_prefix(sample_id: str) -> str:
    label = normalize_spaces(sample_id).lower()
    if not label:
        return ""
    return label.split("-", 1)[0]


def map_prefix_to_dataset_task(label_prefix: str) -> str:
    prefix = normalize_spaces(label_prefix).lower()
    if not prefix:
        return "unknown_task"
    if any(token in prefix for token in ["overlap", "overlapping", "conflict", "similarity"]):
        return "legal_overlapping_analysis"
    if any(token in prefix for token in ["extract", "extraction", "element"]):
        return "element_extraction"
    if prefix in {"qna", "qa"} or any(token in prefix for token in ["qna", "qa"]):
        return "legal_qna"
    if any(token in prefix for token in ["summary", "summarization", "summarize"]):
        return "legal_summarization"
    if any(token in prefix for token in ["revision", "revise"]):
        return "drafting_revisions"
    if any(token in prefix for token in ["draft", "provision"]):
        return "drafting_provisions"
    return "unknown_task"


def infer_question_word(question: str) -> str:
    q = normalize_spaces(question).lower().strip(" ,.;:!?")
    if not q:
        return "lainnya"
    if q.startswith("di mana") or q.startswith("dimana"):
        return "di mana"
    for word in ["apa", "siapa", "kapan", "mengapa", "kenapa", "bagaimana", "berapa", "apakah"]:
        if re.match(rf"^{word}\b", q):
            return "mengapa" if word == "kenapa" else word
    return "lainnya"


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def infer_question_intent(question: str) -> str:
    q = normalize_spaces(question).lower()
    rules = [
        ("sanction", ["sanksi", "hukuman", "denda", "pidana"]),
        ("authority", ["siapa", "berwenang", "kewenangan", "menetapkan", "mengatur"]),
        ("procedure", ["bagaimana", "tata cara", "prosedur", "mekanisme", "langkah"]),
        ("requirement", ["syarat", "persyaratan", "agar", "untuk dapat"]),
        ("obligation", ["harus", "wajib", "kewajiban", "dilakukan"]),
        ("permission", ["boleh", "dapatkah", "diizinkan", "diperbolehkan"]),
        ("definition", ["apa yang dimaksud", "definisi", "pengertian", "disebut"]),
        ("time", ["kapan", "jangka waktu", "paling lama", "batas waktu"]),
        ("amount", ["berapa", "jumlah", "besaran"]),
    ]
    for intent, patterns in rules:
        if contains_any(q, patterns):
            return intent
    return "general_qna"


def infer_reference_type(reference: str) -> str:
    ref = normalize_spaces(reference)
    low = ref.lower()
    if not low:
        return "no_reference"
    pasal_count = len(re.findall(r"\bpasal\b", low))
    if ";" in ref or re.search(r",\s*pasal\b", low) or re.search(r"\bdan\s+pasal\b", low) or pasal_count > 1:
        return "multi_reference"
    has_pasal = bool(re.search(r"\bpasal\s+[0-9A-Za-z]+", low))
    has_ayat = bool(re.search(r"\bayat\s*\(?[0-9A-Za-z]+\)?", low))
    if has_pasal and has_ayat:
        return "pasal_ayat"
    if has_pasal:
        return "pasal_only"
    if has_ayat:
        return "ayat_only"
    return "other_reference"


def extract_article_number(reference: str) -> str:
    match = re.search(r"\bpasal\s+([0-9A-Za-z]+)", str(reference or ""), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def infer_answer_length_bin(answer: str) -> str:
    token_count = len(re.findall(r"\S+", str(answer or "")))
    if token_count <= 20:
        return "short"
    if token_count <= 60:
        return "medium"
    return "long"


def load_test_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test data tidak ditemukan: {path}")

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metadata = obj.get("metadata") or {}
            input_text = str(obj.get("input", ""))
            reference_from_input, question = parse_input_parts(input_text)
            reference_answer = str(obj.get("output", ""))
            reference = str(metadata.get("reference", ""))
            sample_id = str(metadata.get("label", ""))
            label_prefix = extract_label_prefix(sample_id)
            dataset_task = map_prefix_to_dataset_task(label_prefix)
            rows.append(
                {
                    "index": idx,
                    "input": input_text,
                    "sample_id": sample_id,
                    "item_label": sample_id,
                    "label_prefix": label_prefix,
                    "dataset_task": dataset_task,
                    "reference": reference,
                    "reference_from_input": reference_from_input,
                    "question": question,
                    "reference_answer": reference_answer,
                    "question_word": infer_question_word(question),
                    "question_intent": infer_question_intent(question),
                    "reference_type": infer_reference_type(reference),
                    "article_number": extract_article_number(reference),
                    "answer_length_bin": infer_answer_length_bin(reference_answer),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Test data kosong atau tidak bisa dibaca: {path}")
    return df


def parse_config_from_name(path: Path) -> Optional[Tuple[str, str, str]]:
    name = path.name.lower()
    if "newsplit" not in name:
        return None
    if not (name.startswith("part1_llama2_") or name.startswith("part1_wizardlm_") or "llama2_" in name or "wizardlm_" in name):
        return None

    if "wizardlm" in name:
        model = "wizardlm-13b"
    elif "llama2" in name:
        model = "llama2-7b"
    else:
        return None

    if "zeroshot" in name or "zero-shot" in name:
        training_state = "zero-shot"
    elif "finetune" in name or "fine-tuned" in name or "fine_tuned" in name:
        training_state = "fine-tuned"
    else:
        return None

    if "graphrag" in name:
        method = "graphrag"
    elif re.search(r"(^|_)rag(_|\.|-)", name):
        method = "rag"
    else:
        method = "only"

    return model, training_state, method


def include_config(config: RunConfig, include_only: str) -> bool:
    filters = [item.strip().lower() for item in include_only.split(",") if item.strip()]
    if not filters:
        return True
    haystacks = [config.config_id.lower(), str(config.source_path).lower()]
    return any(token in haystack for token in filters for haystack in haystacks)


def discover_eval_configs(reports_dir: Path, include_only: str) -> Dict[Tuple[str, str, str], RunConfig]:
    configs: Dict[Tuple[str, str, str], RunConfig] = {}
    for path in sorted(reports_dir.glob("*_newsplit_eval_rows.csv")):
        parsed = parse_config_from_name(path)
        if not parsed:
            continue
        config = RunConfig(*parsed, source_path=path, source_kind="eval_rows")
        if include_config(config, include_only):
            configs[parsed] = config
    return configs


def discover_prediction_configs(predictions_dir: Path, include_only: str) -> Dict[Tuple[str, str, str], RunConfig]:
    configs: Dict[Tuple[str, str, str], RunConfig] = {}
    for path in sorted(predictions_dir.glob("**/*_newsplit.csv")):
        parsed = parse_config_from_name(path)
        if not parsed:
            continue
        config = RunConfig(*parsed, source_path=path, source_kind="predictions")
        if include_config(config, include_only):
            configs[parsed] = config
    return configs


def require_columns(df: pd.DataFrame, path: Path, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Kolom wajib tidak ditemukan di {path}: {missing}. Kolom tersedia: {available}")


def read_eval_rows(config: RunConfig) -> pd.DataFrame:
    df = pd.read_csv(config.source_path)
    require_columns(
        df,
        config.source_path,
        ["index", "bleu", "meteor", "rouge_l", "embedding_cosine", "prediction", "ground_truth"],
    )
    df = df.copy()
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
    df = df.rename(columns={"bleu": "bleu_4", "ground_truth": "reference_answer_from_eval"})
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def recompute_prediction_rows(config: RunConfig, embedding_model: str, device: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.evaluation.eval_predictions_qa import evaluate_rows, load_rows
    from scripts.graph.graphrag_retriever_v2 import load_sentence_transformer

    rows = load_rows(config.source_path)
    encoder_model = load_sentence_transformer(model_name=embedding_model, device=device, local_files_only=True)
    per_row, _ = evaluate_rows(rows, encoder_model=encoder_model)
    df = pd.DataFrame(per_row)
    df = df.rename(columns={"bleu": "bleu_4", "ground_truth": "reference_answer_from_eval"})
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_all_metrics(
    eval_configs: Dict[Tuple[str, str, str], RunConfig],
    prediction_configs: Dict[Tuple[str, str, str], RunConfig],
    dataset_df: pd.DataFrame,
    recompute_missing: bool,
    embedding_model: str,
    device: str,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []
    all_keys = sorted(set(eval_configs) | set(prediction_configs))

    category_cols = [
        "index",
        "sample_id",
        "item_label",
        "label_prefix",
        "dataset_task",
        "reference",
        "reference_from_input",
        "question",
        "reference_answer",
        "question_word",
        "question_intent",
        "reference_type",
        "article_number",
        "answer_length_bin",
    ]

    for key in all_keys:
        config = eval_configs.get(key)
        if config is not None:
            df = read_eval_rows(config)
        elif recompute_missing and key in prediction_configs:
            config = prediction_configs[key]
            df = recompute_prediction_rows(config, embedding_model=embedding_model, device=device)
        else:
            continue

        if "index" not in df.columns:
            df = df.reset_index().rename(columns={"index": "index"})
        df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")
        merged = df.merge(dataset_df[category_cols], on="index", how="left", suffixes=("", "_dataset"))

        model, training_state, method = key
        merged["model"] = model
        merged["training_state"] = training_state
        merged["method"] = method
        merged["source_file"] = str(config.source_path)
        frames.append(merged)
        diagnostics.append(
            {
                "model": model,
                "training_state": training_state,
                "method": method,
                "rows": int(len(merged)),
                "source_kind": config.source_kind,
                "source_file": str(config.source_path),
            }
        )

    if not frames:
        raise RuntimeError("Tidak ada file evaluasi/prediksi yang bisa dimuat.")

    return pd.concat(frames, ignore_index=True), diagnostics


def metric_col(method: str, metric: str) -> str:
    return f"{method}_{metric}"


def build_pairwise_delta(all_metrics: pd.DataFrame) -> pd.DataFrame:
    pairs: List[pd.DataFrame] = []
    base_cols = [
        "index",
        "sample_id",
        "item_label",
        "label_prefix",
        "dataset_task",
        "reference",
        "reference_from_input",
        "question",
        "reference_answer",
        "question_word",
        "question_intent",
        "reference_type",
        "article_number",
        "answer_length_bin",
    ]

    for (model, training_state), group in all_metrics.groupby(["model", "training_state"], sort=True):
        rag = group[group["method"] == "rag"].copy()
        graph = group[group["method"] == "graphrag"].copy()
        if rag.empty or graph.empty:
            continue

        rag_cols = base_cols + METRIC_COLUMNS + ["prediction", "prompt_token_count"]
        graph_cols = ["index"] + METRIC_COLUMNS + ["prediction", "prompt_token_count"]
        rag_cols = [col for col in rag_cols if col in rag.columns]
        graph_cols = [col for col in graph_cols if col in graph.columns]

        merged = rag[rag_cols].merge(
            graph[graph_cols],
            on="index",
            how="inner",
            suffixes=("_rag", "_graphrag"),
        )
        merged.insert(1, "model", model)
        merged.insert(2, "training_state", training_state)

        for metric in METRIC_COLUMNS:
            merged[f"rag_{metric}"] = pd.to_numeric(merged[f"{metric}_rag"], errors="coerce")
            merged[f"graphrag_{metric}"] = pd.to_numeric(merged[f"{metric}_graphrag"], errors="coerce")
            if metric == "bleu_4":
                delta_name = "delta_bleu"
            elif metric == "embedding_cosine":
                delta_name = "delta_cosine"
            else:
                delta_name = f"delta_{metric}"
            merged[delta_name] = merged[f"graphrag_{metric}"] - merged[f"rag_{metric}"]

        if "prediction_rag" in merged.columns:
            merged = merged.rename(columns={"prediction_rag": "rag_prediction"})
        if "prediction_graphrag" in merged.columns:
            merged = merged.rename(columns={"prediction_graphrag": "graphrag_prediction"})

        merged["delta_score"] = merged[["delta_meteor", "delta_rouge_l", "delta_cosine"]].sum(axis=1, skipna=True)
        merged["reverse_score"] = -merged["delta_score"]
        merged["interpretation"] = merged.apply(interpret_row, axis=1)

        drop_cols = [f"{metric}_rag" for metric in METRIC_COLUMNS] + [f"{metric}_graphrag" for metric in METRIC_COLUMNS]
        merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
        pairs.append(merged)

    if not pairs:
        raise RuntimeError("Tidak ada pasangan RAG dan GraphRAG yang bisa dibandingkan.")
    return pd.concat(pairs, ignore_index=True)


def interpret_deltas(values: Iterable[float]) -> str:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return "Campuran"
    positive = sum(1 for v in vals if v > 0)
    negative = sum(1 for v in vals if v < 0)
    if positive > len(vals) / 2:
        return "GraphRAG unggul"
    if negative > len(vals) / 2:
        return "RAG unggul"
    return "Campuran"


def interpret_row(row: pd.Series) -> str:
    return interpret_deltas([row.get("delta_bleu"), row.get("delta_meteor"), row.get("delta_rouge_l"), row.get("delta_cosine")])


def group_metrics(pairwise: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = pairwise.groupby(["model", "training_state", group_col], dropna=False, sort=True)
    for (model, training_state, group_value), df in grouped:
        record: Dict[str, object] = {
            "model": model,
            "training_state": training_state,
            "group_column": group_col,
            "group_value": group_value,
            "count": int(len(df)),
        }
        for metric in METRIC_COLUMNS:
            rag_name = f"rag_{metric}"
            graph_name = f"graphrag_{metric}"
            if metric == "bleu_4":
                delta_name = "delta_bleu"
            elif metric == "embedding_cosine":
                delta_name = "delta_cosine"
            else:
                delta_name = f"delta_{metric}"
            out_metric = "bleu" if metric == "bleu_4" else metric
            record[f"mean_{out_metric}_rag"] = round(float(df[rag_name].mean()), 6)
            record[f"mean_{out_metric}_graphrag"] = round(float(df[graph_name].mean()), 6)
            record[delta_name] = round(float(df[delta_name].mean()), 6)
        record["delta_score"] = round(
            float(df[["delta_meteor", "delta_rouge_l", "delta_cosine"]].sum(axis=1, skipna=True).mean()),
            6,
        )
        record["interpretation"] = interpret_deltas(
            [record["delta_bleu"], record["delta_meteor"], record["delta_rouge_l"], record["delta_cosine"]]
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["model", "training_state", "group_value"]).reset_index(drop=True)


def group_bleu4_metrics(pairwise: pd.DataFrame, group_col: str, training_state: Optional[str] = None) -> pd.DataFrame:
    df = pairwise.copy()
    if training_state:
        df = df[df["training_state"] == training_state].copy()
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(["model", "training_state", group_col], dropna=False, sort=True)
    for (model, state, group_value), sub in grouped:
        delta = pd.to_numeric(sub["delta_bleu"], errors="coerce")
        record = {
            "model": model,
            "training_state": state,
            "group_column": group_col,
            "group_value": group_value,
            "count": int(len(sub)),
            "mean_bleu4_rag": round(float(pd.to_numeric(sub["rag_bleu_4"], errors="coerce").mean()), 6),
            "mean_bleu4_graphrag": round(float(pd.to_numeric(sub["graphrag_bleu_4"], errors="coerce").mean()), 6),
            "delta_bleu": round(float(delta.mean()), 6),
            "positive_delta_count": int((delta > 0).sum()),
            "negative_delta_count": int((delta < 0).sum()),
            "zero_delta_count": int((delta == 0).sum()),
            "positive_delta_percentage": round(float((delta > 0).mean() * 100), 2),
            "negative_delta_percentage": round(float((delta < 0).mean() * 100), 2),
        }
        if record["delta_bleu"] > 0:
            record["bleu4_winner"] = "GraphRAG"
        elif record["delta_bleu"] < 0:
            record["bleu4_winner"] = "RAG"
        else:
            record["bleu4_winner"] = "Tie"
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["training_state", "model", "group_column", "group_value"]).reset_index(drop=True)


def bleu_tokenize(text: object) -> List[str]:
    return normalize_spaces(text).lower().split()


def compute_corpus_bleu4(references: Sequence[object], predictions: Sequence[object]) -> float:
    refs = [[bleu_tokenize(ref)] for ref in references]
    preds = [bleu_tokenize(pred) for pred in predictions]
    if not refs or not preds:
        return 0.0
    return float(
        corpus_bleu(
            refs,
            preds,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def group_corpus_bleu4_metrics(pairwise: pd.DataFrame, group_col: str, training_state: Optional[str] = None) -> pd.DataFrame:
    df = pairwise.copy()
    if training_state:
        df = df[df["training_state"] == training_state].copy()
    rows: List[Dict[str, object]] = []

    for model_key, model_df in list(df.groupby("model", sort=True)) + [("all", df)]:
        grouped = model_df.groupby(["training_state", group_col], dropna=False, sort=True)
        for (state, group_value), sub in grouped:
            rag_bleu = compute_corpus_bleu4(sub["reference_answer"].tolist(), sub["rag_prediction"].tolist())
            graph_bleu = compute_corpus_bleu4(sub["reference_answer"].tolist(), sub["graphrag_prediction"].tolist())
            delta = graph_bleu - rag_bleu
            rows.append(
                {
                    "model": model_key,
                    "training_state": state,
                    "group_column": group_col,
                    "group_value": group_value,
                    "count": int(len(sub)),
                    "corpus_bleu4_rag": round(rag_bleu, 6),
                    "corpus_bleu4_graphrag": round(graph_bleu, 6),
                    "delta_corpus_bleu4": round(delta, 6),
                    "bleu4_winner": "GraphRAG" if delta > 0 else "RAG" if delta < 0 else "Tie",
                }
            )
    return pd.DataFrame(rows).sort_values(["training_state", "model", "group_column", "group_value"]).reset_index(drop=True)


def aggregate_by_dimension(pairwise: pd.DataFrame, group_col: str, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    keys = extra_cols or []
    grouped = pairwise.groupby(keys + [group_col], dropna=False, sort=True)
    rows: List[Dict[str, object]] = []
    for key, df in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        record: Dict[str, object] = {}
        for col, value in zip(keys + [group_col], key):
            record[col] = value
        record["count"] = int(len(df))
        for delta in ["delta_bleu", "delta_meteor", "delta_rouge_l", "delta_cosine"]:
            record[delta] = round(float(df[delta].mean()), 6)
        record["delta_score"] = round(float(df[["delta_meteor", "delta_rouge_l", "delta_cosine"]].sum(axis=1).mean()), 6)
        record["interpretation"] = interpret_deltas(
            [record["delta_bleu"], record["delta_meteor"], record["delta_rouge_l"], record["delta_cosine"]]
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("delta_score", ascending=False).reset_index(drop=True)


def select_example_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "training_state",
        "index",
        "sample_id",
        "label_prefix",
        "dataset_task",
        "reference",
        "question",
        "question_word",
        "question_intent",
        "reference_type",
        "answer_length_bin",
        "reference_answer",
        "rag_prediction",
        "graphrag_prediction",
        "rag_bleu_4",
        "graphrag_bleu_4",
        "delta_bleu",
        "rag_meteor",
        "graphrag_meteor",
        "delta_meteor",
        "rag_rouge_l",
        "graphrag_rouge_l",
        "delta_rouge_l",
        "rag_embedding_cosine",
        "graphrag_embedding_cosine",
        "delta_cosine",
        "delta_score",
        "interpretation",
    ]
    return df[[col for col in cols if col in df.columns]].copy()


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 10) -> str:
    if df.empty:
        return "_Tidak ada data._"
    view = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype(str).str.replace("\n", " ", regex=False).str.slice(0, 160)

    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def svg_text(text: object) -> str:
    return html.escape(str(text or ""), quote=True)


def task_display_label(task: object) -> str:
    task_str = str(task or "")
    return TASK_DISPLAY_LABELS.get(task_str, task_str.replace("_", " "))


def dataset_task_score_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    df = pairwise.copy()
    df["rag_total_score"] = df["rag_meteor"] + df["rag_rouge_l"] + df["rag_embedding_cosine"]
    df["graphrag_total_score"] = df["graphrag_meteor"] + df["graphrag_rouge_l"] + df["graphrag_embedding_cosine"]
    grouped = (
        df.groupby(["training_state", "dataset_task"], as_index=False)
        .agg(
            count=("index", "count"),
            rag_total_score=("rag_total_score", "mean"),
            graphrag_total_score=("graphrag_total_score", "mean"),
            delta_score=("delta_score", "mean"),
        )
    )
    grouped["dataset_task"] = pd.Categorical(grouped["dataset_task"], categories=TASK_ORDER, ordered=True)
    return grouped.sort_values(["training_state", "dataset_task"]).reset_index(drop=True)


def write_total_score_svg(score_df: pd.DataFrame, training_state: str, path: Path) -> None:
    data = score_df[score_df["training_state"] == training_state].copy()
    if data.empty:
        return

    labels = data["dataset_task"].astype(str).tolist()
    max_score = max(float(data["rag_total_score"].max()), float(data["graphrag_total_score"].max()), 1.0)
    x_max = 2.5 if max_score <= 2.5 else float(np.ceil(max_score * 10) / 10)

    width = 960
    left = 170
    right = 70
    top = 96
    row_h = 62
    bar_h = 18
    gap = 5
    axis_h = len(labels) * row_h
    height = top + axis_h + 92
    chart_w = width - left - right
    zero_x = left

    def x(value: float) -> float:
        return zero_x + (float(value) / x_max) * chart_w

    title_state = "Zero-shot" if training_state == "zero-shot" else "Fine-tuned"
    subtitle = "Total score = mean(METEOR + ROUGE-L + Embedding-Cosine)"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="36" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#222">RAG vs GraphRAG by Task ({svg_text(title_state)})</text>',
        f'<text x="{left}" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">{svg_text(subtitle)}</text>',
        f'<rect x="{left}" y="76" width="14" height="14" fill="#4C78A8"/><text x="{left + 20}" y="88" font-family="Arial, sans-serif" font-size="13" fill="#333">RAG</text>',
        f'<rect x="{left + 82}" y="76" width="14" height="14" fill="#F58518"/><text x="{left + 102}" y="88" font-family="Arial, sans-serif" font-size="13" fill="#333">GraphRAG</text>',
    ]

    tick_count = 5 if x_max <= 2.5 else 6
    for i in range(tick_count + 1):
        value = x_max * i / tick_count
        tick_x = x(value)
        parts.append(f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{top + axis_h}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{top + axis_h + 24}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">{value:.1f}</text>')

    for i, (_, row) in enumerate(data.iterrows()):
        y_mid = top + i * row_h + row_h / 2
        rag = float(row["rag_total_score"])
        graph = float(row["graphrag_total_score"])
        delta = float(row["delta_score"])
        label = task_display_label(row["dataset_task"])

        parts.append(f'<text x="{left - 12}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#222">{svg_text(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{y_mid + row_h / 2 - 6:.1f}" x2="{left + chart_w}" y2="{y_mid + row_h / 2 - 6:.1f}" stroke="#f2f2f2" stroke-width="1"/>')

        rag_w = x(rag) - zero_x
        graph_w = x(graph) - zero_x
        rag_y = y_mid - bar_h - gap / 2
        graph_y = y_mid + gap / 2
        parts.append(f'<rect x="{zero_x}" y="{rag_y:.1f}" width="{rag_w:.1f}" height="{bar_h}" rx="2" fill="#4C78A8"/>')
        parts.append(f'<rect x="{zero_x}" y="{graph_y:.1f}" width="{graph_w:.1f}" height="{bar_h}" rx="2" fill="#F58518"/>')
        parts.append(f'<text x="{x(rag) + 6:.1f}" y="{rag_y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#333">{rag:.3f}</text>')
        parts.append(f'<text x="{x(graph) + 6:.1f}" y="{graph_y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#333">{graph:.3f}</text>')
        delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
        parts.append(f'<text x="{width - 18}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 24}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555">Mean total score</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_score_svg(score_df: pd.DataFrame, path: Path) -> None:
    if score_df.empty:
        return
    pivot_rows = []
    for task in TASK_ORDER:
        sub = score_df[score_df["dataset_task"].astype(str) == task]
        if sub.empty:
            continue
        pivot_rows.append(
            {
                "dataset_task": task,
                "zero-shot": float(sub[sub["training_state"] == "zero-shot"]["delta_score"].mean()) if not sub[sub["training_state"] == "zero-shot"].empty else 0.0,
                "fine-tuned": float(sub[sub["training_state"] == "fine-tuned"]["delta_score"].mean()) if not sub[sub["training_state"] == "fine-tuned"].empty else 0.0,
            }
        )
    data = pd.DataFrame(pivot_rows)
    max_abs = max(float(data[["zero-shot", "fine-tuned"]].abs().max().max()), 0.01)
    x_abs = max(0.10, np.ceil(max_abs * 100) / 100)

    width = 960
    left = 180
    right = 40
    top = 94
    row_h = 58
    bar_h = 17
    gap = 5
    axis_h = len(data) * row_h
    height = top + axis_h + 92
    chart_w = width - left - right
    zero_x = left + chart_w / 2

    def x(value: float) -> float:
        return zero_x + (float(value) / x_abs) * (chart_w / 2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="36" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#222">Delta GraphRAG vs RAG by Task</text>',
        f'<text x="{left}" y="62" font-family="Arial, sans-serif" font-size="13" fill="#555">Delta score = GraphRAG total score - RAG total score. Positive means GraphRAG better.</text>',
        f'<rect x="{left}" y="76" width="14" height="14" fill="#72B7B2"/><text x="{left + 20}" y="88" font-family="Arial, sans-serif" font-size="13" fill="#333">Zero-shot</text>',
        f'<rect x="{left + 112}" y="76" width="14" height="14" fill="#E45756"/><text x="{left + 132}" y="88" font-family="Arial, sans-serif" font-size="13" fill="#333">Fine-tuned</text>',
    ]

    tick_values = np.linspace(-x_abs, x_abs, 9)
    for value in tick_values:
        tick_x = x(float(value))
        stroke = "#333" if abs(value) < 1e-9 else "#e7e7e7"
        parts.append(f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{top + axis_h}" stroke="{stroke}" stroke-width="1"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{top + axis_h + 24}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">{value:+.2f}</text>')

    for i, (_, row) in enumerate(data.iterrows()):
        y_mid = top + i * row_h + row_h / 2
        label = task_display_label(row["dataset_task"])
        parts.append(f'<text x="{left - 12}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#222">{svg_text(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{y_mid + row_h / 2 - 6:.1f}" x2="{left + chart_w}" y2="{y_mid + row_h / 2 - 6:.1f}" stroke="#f2f2f2" stroke-width="1"/>')

        for key, color, y_offset in [("zero-shot", "#72B7B2", -bar_h - gap / 2), ("fine-tuned", "#E45756", gap / 2)]:
            value = float(row[key])
            end_x = x(value)
            rect_x = min(zero_x, end_x)
            rect_w = abs(end_x - zero_x)
            rect_y = y_mid + y_offset
            parts.append(f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" height="{bar_h}" rx="2" fill="{color}"/>')
            text_anchor = "start" if value >= 0 else "end"
            text_x = end_x + 6 if value >= 0 else end_x - 6
            parts.append(f'<text x="{text_x:.1f}" y="{rect_y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="{text_anchor}" fill="#333">{value:+.3f}</text>')

    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 24}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555">Delta total score (GraphRAG - RAG)</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bleu4_delta_svg(bleu_df: pd.DataFrame, path: Path, title: str = "Fine-tuned BLEU-4 Delta by Task") -> None:
    data = bleu_df[
        (bleu_df["training_state"] == "fine-tuned")
        & (bleu_df["group_column"] == "dataset_task")
    ].copy()
    if data.empty:
        return
    overall = (
        data.groupby("group_value", as_index=False)
        .agg(count=("count", "sum"), delta_bleu=("delta_bleu", "mean"))
    )
    overall["group_value"] = pd.Categorical(overall["group_value"], categories=TASK_ORDER, ordered=True)
    overall = overall.sort_values("group_value")

    max_abs = max(float(overall["delta_bleu"].abs().max()), 0.01)
    x_abs = max(0.08, np.ceil(max_abs * 100) / 100)
    width = 920
    left = 180
    right = 46
    top = 86
    row_h = 48
    bar_h = 20
    axis_h = len(overall) * row_h
    height = top + axis_h + 78
    chart_w = width - left - right
    zero_x = left + chart_w / 2

    def x(value: float) -> float:
        return zero_x + (float(value) / x_abs) * (chart_w / 2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#222">{svg_text(title)}</text>',
        f'<text x="{left}" y="58" font-family="Arial, sans-serif" font-size="13" fill="#555">Delta BLEU-4 = GraphRAG BLEU-4 - RAG BLEU-4. Positive means GraphRAG better.</text>',
    ]
    for value in np.linspace(-x_abs, x_abs, 9):
        tick_x = x(float(value))
        stroke = "#333" if abs(value) < 1e-9 else "#e7e7e7"
        parts.append(f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{top + axis_h}" stroke="{stroke}" stroke-width="1"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{top + axis_h + 22}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">{value:+.2f}</text>')

    for i, (_, row) in enumerate(overall.iterrows()):
        y_mid = top + i * row_h + row_h / 2
        value = float(row["delta_bleu"])
        end_x = x(value)
        rect_x = min(zero_x, end_x)
        rect_w = abs(end_x - zero_x)
        color = "#2f7f5f" if value > 0 else "#b84040" if value < 0 else "#777"
        label = task_display_label(row["group_value"])
        parts.append(f'<text x="{left - 12}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#222">{svg_text(label)}</text>')
        parts.append(f'<rect x="{rect_x:.1f}" y="{y_mid - bar_h / 2:.1f}" width="{rect_w:.1f}" height="{bar_h}" rx="2" fill="{color}"/>')
        text_anchor = "start" if value >= 0 else "end"
        text_x = end_x + 6 if value >= 0 else end_x - 6
        parts.append(f'<text x="{text_x:.1f}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="{text_anchor}" fill="#333">{value:+.4f}</text>')

    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555">Mean delta BLEU-4</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bleu4_rag_vs_graphrag_svg(
    bleu_df: pd.DataFrame,
    training_state: str,
    path: Path,
    model: str = "all",
) -> None:
    data = bleu_df[
        (bleu_df["training_state"] == training_state)
        & (bleu_df["group_column"] == "dataset_task")
        & (bleu_df["model"] == model)
    ].copy()
    if data.empty:
        return
    data["group_value"] = pd.Categorical(data["group_value"], categories=TASK_ORDER, ordered=True)
    data = data.sort_values("group_value")

    max_score = max(float(data["corpus_bleu4_rag"].max()), float(data["corpus_bleu4_graphrag"].max()), 0.1)
    x_max = max(0.5, float(np.ceil(max_score * 10) / 10))
    width = 920
    left = 180
    right = 60
    top = 92
    row_h = 58
    bar_h = 17
    gap = 5
    axis_h = len(data) * row_h
    height = top + axis_h + 82
    chart_w = width - left - right

    def x(value: float) -> float:
        return left + (float(value) / x_max) * chart_w

    state_title = "Zero-shot" if training_state == "zero-shot" else "Fine-tuned"
    model_title = {
        "llama2-7b": "LLaMA 2 7B",
        "wizardlm-13b": "WizardLM 13B",
        "all": "All Models",
    }.get(model, model)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#222">RAG vs GraphRAG by Task ({svg_text(model_title)}, {svg_text(state_title)})</text>',
        f'<text x="{left}" y="58" font-family="Arial, sans-serif" font-size="13" fill="#555">BLEU-4 per dataset task. Higher is better.</text>',
        f'<rect x="{left}" y="72" width="14" height="14" fill="#4C78A8"/><text x="{left + 20}" y="84" font-family="Arial, sans-serif" font-size="13" fill="#333">RAG</text>',
        f'<rect x="{left + 82}" y="72" width="14" height="14" fill="#F58518"/><text x="{left + 102}" y="84" font-family="Arial, sans-serif" font-size="13" fill="#333">GraphRAG</text>',
    ]

    tick_count = 5
    for i in range(tick_count + 1):
        value = x_max * i / tick_count
        tick_x = x(value)
        parts.append(f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{top + axis_h}" stroke="#e7e7e7" stroke-width="1"/>')
        parts.append(f'<text x="{tick_x:.1f}" y="{top + axis_h + 22}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">{value:.1f}</text>')

    for i, (_, row) in enumerate(data.iterrows()):
        y_mid = top + i * row_h + row_h / 2
        rag = float(row["corpus_bleu4_rag"])
        graph = float(row["corpus_bleu4_graphrag"])
        delta = float(row["delta_corpus_bleu4"])
        label = task_display_label(row["group_value"])

        rag_y = y_mid - bar_h - gap / 2
        graph_y = y_mid + gap / 2
        parts.append(f'<text x="{left - 12}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="13" text-anchor="end" fill="#222">{svg_text(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{y_mid + row_h / 2 - 6:.1f}" x2="{left + chart_w}" y2="{y_mid + row_h / 2 - 6:.1f}" stroke="#f2f2f2" stroke-width="1"/>')
        parts.append(f'<rect x="{left}" y="{rag_y:.1f}" width="{x(rag) - left:.1f}" height="{bar_h}" rx="2" fill="#4C78A8"/>')
        parts.append(f'<rect x="{left}" y="{graph_y:.1f}" width="{x(graph) - left:.1f}" height="{bar_h}" rx="2" fill="#F58518"/>')
        parts.append(f'<text x="{x(rag) + 6:.1f}" y="{rag_y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#333">{rag:.3f}</text>')
        parts.append(f'<text x="{x(graph) + 6:.1f}" y="{graph_y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#333">{graph:.3f}</text>')
        delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
        parts.append(f'<text x="{width - 14}" y="{y_mid + 5:.1f}" font-family="Arial, sans-serif" font-size="12" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 22}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#555">BLEU-4</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_combined_bleu4_rag_vs_graphrag_svg(bleu_df: pd.DataFrame, path: Path) -> None:
    data = bleu_df[
        (bleu_df["group_column"] == "dataset_task")
        & (bleu_df["model"].isin(["llama2-7b", "wizardlm-13b"]))
    ].copy()
    if data.empty:
        return

    models = [("llama2-7b", "LLaMA 2 7B"), ("wizardlm-13b", "WizardLM 13B")]
    states = [("zero-shot", "Zero-shot"), ("fine-tuned", "Fine-tuned")]
    tasks = [task for task in TASK_ORDER if task != "unknown_task"]

    max_score = max(float(data["corpus_bleu4_rag"].max()), float(data["corpus_bleu4_graphrag"].max()), 0.1)
    x_max = max(0.5, float(np.ceil(max_score * 10) / 10))

    width = 1500
    height = 1030
    outer_left = 58
    outer_top = 128
    panel_w = 680
    panel_h = 374
    col_gap = 38
    row_gap = 86
    label_w = 150
    right_pad = 34
    chart_w = panel_w - label_w - right_pad
    row_h = 42
    bar_h = 12
    gap = 4

    def panel_x(col: int) -> int:
        return outer_left + col * (panel_w + col_gap)

    def panel_y(row: int) -> int:
        return outer_top + row * (panel_h + row_gap)

    def x(value: float, px: float) -> float:
        return px + label_w + (float(value) / x_max) * chart_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="58" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222">RAG vs GraphRAG by Task (BLEU-4)</text>',
        '<text x="58" y="64" font-family="Arial, sans-serif" font-size="13" fill="#555">Separate panels by model and training state. Higher BLEU-4 is better.</text>',
        '<rect x="1120" y="26" width="14" height="14" fill="#4C78A8"/><text x="1140" y="38" font-family="Arial, sans-serif" font-size="13" fill="#333">RAG</text>',
        '<rect x="1190" y="26" width="14" height="14" fill="#F58518"/><text x="1210" y="38" font-family="Arial, sans-serif" font-size="13" fill="#333">GraphRAG</text>',
    ]

    tick_count = 5
    for r, (state, state_label) in enumerate(states):
        for c, (model, model_label) in enumerate(models):
            px = panel_x(c)
            py = panel_y(r)
            sub = data[(data["model"] == model) & (data["training_state"] == state)].copy()
            sub["group_value"] = pd.Categorical(sub["group_value"], categories=tasks, ordered=True)
            sub = sub.sort_values("group_value")

            parts.append(f'<text x="{px}" y="{py - 28}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#222">{svg_text(model_label)} · {svg_text(state_label)}</text>')
            parts.append(f'<rect x="{px}" y="{py - 14}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#dddddd" stroke-width="1"/>')

            chart_left = px + label_w
            chart_top = py + 16
            for i in range(tick_count + 1):
                value = x_max * i / tick_count
                tx = x(value, px)
                parts.append(f'<line x1="{tx:.1f}" y1="{chart_top - 6}" x2="{tx:.1f}" y2="{chart_top + len(tasks) * row_h}" stroke="#ececec" stroke-width="1"/>')
                parts.append(f'<text x="{tx:.1f}" y="{chart_top + len(tasks) * row_h + 24}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#555">{value:.1f}</text>')

            for i, task in enumerate(tasks):
                row = sub[sub["group_value"].astype(str) == task]
                if row.empty:
                    continue
                row = row.iloc[0]
                y_mid = chart_top + i * row_h + row_h / 2
                rag = float(row["corpus_bleu4_rag"])
                graph = float(row["corpus_bleu4_graphrag"])
                delta = float(row["delta_corpus_bleu4"])
                label = task_display_label(task)
                rag_y = y_mid - bar_h - gap / 2
                graph_y = y_mid + gap / 2

                parts.append(f'<text x="{chart_left - 8}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end" fill="#222">{svg_text(label)}</text>')
                parts.append(f'<line x1="{chart_left}" y1="{y_mid + row_h / 2 - 4:.1f}" x2="{chart_left + chart_w}" y2="{y_mid + row_h / 2 - 4:.1f}" stroke="#f5f5f5" stroke-width="1"/>')
                parts.append(f'<rect x="{chart_left}" y="{rag_y:.1f}" width="{x(rag, px) - chart_left:.1f}" height="{bar_h}" rx="2" fill="#4C78A8"/>')
                parts.append(f'<rect x="{chart_left}" y="{graph_y:.1f}" width="{x(graph, px) - chart_left:.1f}" height="{bar_h}" rx="2" fill="#F58518"/>')
                parts.append(f'<text x="{x(rag, px) + 4:.1f}" y="{rag_y + 10:.1f}" font-family="Arial, sans-serif" font-size="9.5" fill="#333">{rag:.3f}</text>')
                parts.append(f'<text x="{x(graph, px) + 4:.1f}" y="{graph_y + 10:.1f}" font-family="Arial, sans-serif" font-size="9.5" fill="#333">{graph:.3f}</text>')
                delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
                parts.append(f'<text x="{px + panel_w - 8}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

            parts.append(f'<text x="{chart_left + chart_w / 2:.1f}" y="{py + panel_h - 18}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">BLEU-4</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_finetuned_bleu4_two_model_svg(bleu_df: pd.DataFrame, path: Path) -> None:
    data = bleu_df[
        (bleu_df["group_column"] == "dataset_task")
        & (bleu_df["training_state"] == "fine-tuned")
        & (bleu_df["model"].isin(["llama2-7b", "wizardlm-13b"]))
    ].copy()
    if data.empty:
        return

    models = [("llama2-7b", "LLaMA 2 7B"), ("wizardlm-13b", "WizardLM 13B")]
    tasks = [task for task in TASK_ORDER if task != "unknown_task"]
    max_score = max(float(data["corpus_bleu4_rag"].max()), float(data["corpus_bleu4_graphrag"].max()), 0.1)
    x_max = max(0.5, float(np.ceil(max_score * 10) / 10))

    width = 1500
    height = 560
    outer_left = 58
    outer_top = 128
    panel_w = 680
    panel_h = 374
    col_gap = 38
    label_w = 150
    right_pad = 34
    chart_w = panel_w - label_w - right_pad
    row_h = 42
    bar_h = 12
    gap = 4

    def panel_x(col: int) -> int:
        return outer_left + col * (panel_w + col_gap)

    def x(value: float, px: float) -> float:
        return px + label_w + (float(value) / x_max) * chart_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="58" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222">Fine-tuned RAG vs GraphRAG by Task (BLEU-4)</text>',
        '<text x="58" y="64" font-family="Arial, sans-serif" font-size="13" fill="#555">Separate panels by model. Higher BLEU-4 is better.</text>',
        '<rect x="1120" y="26" width="14" height="14" fill="#4C78A8"/><text x="1140" y="38" font-family="Arial, sans-serif" font-size="13" fill="#333">RAG</text>',
        '<rect x="1190" y="26" width="14" height="14" fill="#F58518"/><text x="1210" y="38" font-family="Arial, sans-serif" font-size="13" fill="#333">GraphRAG</text>',
    ]

    tick_count = 5
    for c, (model, model_label) in enumerate(models):
        px = panel_x(c)
        py = outer_top
        sub = data[data["model"] == model].copy()
        sub["group_value"] = pd.Categorical(sub["group_value"], categories=tasks, ordered=True)
        sub = sub.sort_values("group_value")

        parts.append(f'<text x="{px}" y="{py - 28}" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#222">{svg_text(model_label)}</text>')
        parts.append(f'<rect x="{px}" y="{py - 14}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#dddddd" stroke-width="1"/>')

        chart_left = px + label_w
        chart_top = py + 16
        for i in range(tick_count + 1):
            value = x_max * i / tick_count
            tx = x(value, px)
            parts.append(f'<line x1="{tx:.1f}" y1="{chart_top - 6}" x2="{tx:.1f}" y2="{chart_top + len(tasks) * row_h}" stroke="#ececec" stroke-width="1"/>')
            parts.append(f'<text x="{tx:.1f}" y="{chart_top + len(tasks) * row_h + 24}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#555">{value:.1f}</text>')

        for i, task in enumerate(tasks):
            row = sub[sub["group_value"].astype(str) == task]
            if row.empty:
                continue
            row = row.iloc[0]
            y_mid = chart_top + i * row_h + row_h / 2
            rag = float(row["corpus_bleu4_rag"])
            graph = float(row["corpus_bleu4_graphrag"])
            delta = float(row["delta_corpus_bleu4"])
            label = task_display_label(task)
            rag_y = y_mid - bar_h - gap / 2
            graph_y = y_mid + gap / 2

            parts.append(f'<text x="{chart_left - 8}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="11" text-anchor="end" fill="#222">{svg_text(label)}</text>')
            parts.append(f'<line x1="{chart_left}" y1="{y_mid + row_h / 2 - 4:.1f}" x2="{chart_left + chart_w}" y2="{y_mid + row_h / 2 - 4:.1f}" stroke="#f5f5f5" stroke-width="1"/>')
            parts.append(f'<rect x="{chart_left}" y="{rag_y:.1f}" width="{x(rag, px) - chart_left:.1f}" height="{bar_h}" rx="2" fill="#4C78A8"/>')
            parts.append(f'<rect x="{chart_left}" y="{graph_y:.1f}" width="{x(graph, px) - chart_left:.1f}" height="{bar_h}" rx="2" fill="#F58518"/>')
            parts.append(f'<text x="{x(rag, px) + 4:.1f}" y="{rag_y + 10:.1f}" font-family="Arial, sans-serif" font-size="9.5" fill="#333">{rag:.3f}</text>')
            parts.append(f'<text x="{x(graph, px) + 4:.1f}" y="{graph_y + 10:.1f}" font-family="Arial, sans-serif" font-size="9.5" fill="#333">{graph:.3f}</text>')
            delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
            parts.append(f'<text x="{px + panel_w - 8}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

        parts.append(f'<text x="{chart_left + chart_w / 2:.1f}" y="{py + panel_h - 18}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#555">BLEU-4</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_single_model_paper_bleu4_svg(bleu_df: pd.DataFrame, model: str, path: Path) -> None:
    data = bleu_df[
        (bleu_df["group_column"] == "dataset_task")
        & (bleu_df["training_state"] == "fine-tuned")
        & (bleu_df["model"] == model)
    ].copy()
    if data.empty:
        return

    tasks = [task for task in TASK_ORDER if task != "unknown_task"]
    data["group_value"] = pd.Categorical(data["group_value"], categories=tasks, ordered=True)
    data = data.sort_values("group_value")

    model_title = {
        "llama2-7b": "LLaMA 2 7B",
        "wizardlm-13b": "WizardLM 13B",
    }.get(model, model)

    width = 920
    height = 520
    box_x = 26
    box_y = 54
    box_w = 858
    box_h = 412
    left = 214
    right = 64
    top = 86
    row_h = 42
    bar_h = 12
    gap = 4
    chart_w = box_x + box_w - right - left
    x_max = 0.5

    def x(value: float) -> float:
        return left + min(float(value), x_max) / x_max * chart_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{box_x}" y="28" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#222">{svg_text(model_title)}</text>',
        f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" fill="none" stroke="#e1e1e1" stroke-width="1"/>',
    ]

    for i in range(6):
        value = i / 10
        tx = x(value)
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 8}" x2="{tx:.1f}" y2="{top + len(tasks) * row_h}" stroke="#e8e8e8" stroke-width="1"/>')
        parts.append(f'<text x="{tx:.1f}" y="{top + len(tasks) * row_h + 22}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#555">{value:.1f}</text>')

    for i, task in enumerate(tasks):
        row = data[data["group_value"].astype(str) == task]
        if row.empty:
            continue
        row = row.iloc[0]
        y_mid = top + i * row_h + row_h / 2
        rag = float(row["corpus_bleu4_rag"])
        graph = float(row["corpus_bleu4_graphrag"])
        delta = float(row["delta_corpus_bleu4"])
        label = task_display_label(task)
        rag_y = y_mid - bar_h - gap / 2
        graph_y = y_mid + gap / 2

        parts.append(f'<text x="{left - 12}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="10" text-anchor="end" fill="#333">{svg_text(label)}</text>')
        parts.append(f'<rect x="{left}" y="{rag_y:.1f}" width="{x(rag) - left:.1f}" height="{bar_h}" fill="#4C78A8"/>')
        parts.append(f'<rect x="{left}" y="{graph_y:.1f}" width="{x(graph) - left:.1f}" height="{bar_h}" fill="#F58518"/>')
        parts.append(f'<text x="{x(rag) + 4:.1f}" y="{rag_y + 10:.1f}" font-family="Arial, sans-serif" font-size="8.5" fill="#333">{rag:.3f}</text>')
        parts.append(f'<text x="{x(graph) + 4:.1f}" y="{graph_y + 10:.1f}" font-family="Arial, sans-serif" font-size="8.5" fill="#333">{graph:.3f}</text>')
        delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
        parts.append(f'<text x="{box_x + box_w - 22}" y="{y_mid + 4:.1f}" font-family="Arial, sans-serif" font-size="9" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{box_y + box_h - 10}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#555">BLEU-4</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_two_model_single_paper_bleu4_svg(bleu_df: pd.DataFrame, path: Path) -> None:
    data = bleu_df[
        (bleu_df["group_column"] == "dataset_task")
        & (bleu_df["training_state"] == "fine-tuned")
        & (bleu_df["model"].isin(["llama2-7b", "wizardlm-13b"]))
    ].copy()
    if data.empty:
        return

    models = [("llama2-7b", "LLaMA 2 7B"), ("wizardlm-13b", "WizardLM 13B")]
    tasks = [task for task in TASK_ORDER if task != "unknown_task"]
    max_score = max(float(data["corpus_bleu4_rag"].max()), float(data["corpus_bleu4_graphrag"].max()), 0.1)
    x_max = max(0.5, float(np.ceil(max_score * 10) / 10))

    width = 920
    height = 900
    box_x = 26
    box_w = 858
    box_h = 354
    first_box_y = 60
    panel_gap = 76
    left = 214
    right = 64
    top_pad = 38
    row_h = 38
    bar_h = 11
    gap = 4
    chart_w = box_x + box_w - right - left

    def x(value: float) -> float:
        return left + min(float(value), x_max) / x_max * chart_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="26" y="28" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#222">Fine-tuned RAG vs GraphRAG by Task (BLEU-4)</text>',
        '<rect x="660" y="18" width="14" height="14" fill="#4C78A8"/><text x="680" y="30" font-family="Arial, sans-serif" font-size="12" fill="#333">RAG</text>',
        '<rect x="732" y="18" width="14" height="14" fill="#F58518"/><text x="752" y="30" font-family="Arial, sans-serif" font-size="12" fill="#333">GraphRAG</text>',
    ]

    for panel_idx, (model, model_title) in enumerate(models):
        box_y = first_box_y + panel_idx * (box_h + panel_gap)
        chart_top = box_y + top_pad
        sub = data[data["model"] == model].copy()
        sub["group_value"] = pd.Categorical(sub["group_value"], categories=tasks, ordered=True)
        sub = sub.sort_values("group_value")

        parts.append(f'<text x="{box_x}" y="{box_y - 10}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#222">{svg_text(model_title)}</text>')
        parts.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" fill="none" stroke="#e1e1e1" stroke-width="1"/>')

        for i in range(6):
            value = i / 10
            tx = x(value)
            parts.append(f'<line x1="{tx:.1f}" y1="{chart_top - 8}" x2="{tx:.1f}" y2="{chart_top + len(tasks) * row_h}" stroke="#e8e8e8" stroke-width="1"/>')
            parts.append(f'<text x="{tx:.1f}" y="{chart_top + len(tasks) * row_h + 20}" font-family="Arial, sans-serif" font-size="9.5" text-anchor="middle" fill="#555">{value:.1f}</text>')

        for i, task in enumerate(tasks):
            row = sub[sub["group_value"].astype(str) == task]
            if row.empty:
                continue
            row = row.iloc[0]
            y_mid = chart_top + i * row_h + row_h / 2
            rag = float(row["corpus_bleu4_rag"])
            graph = float(row["corpus_bleu4_graphrag"])
            delta = float(row["delta_corpus_bleu4"])
            label = task_display_label(task)
            rag_y = y_mid - bar_h - gap / 2
            graph_y = y_mid + gap / 2

            parts.append(f'<text x="{left - 12}" y="{y_mid + 3.5:.1f}" font-family="Arial, sans-serif" font-size="9.5" text-anchor="end" fill="#333">{svg_text(label)}</text>')
            parts.append(f'<rect x="{left}" y="{rag_y:.1f}" width="{x(rag) - left:.1f}" height="{bar_h}" fill="#4C78A8"/>')
            parts.append(f'<rect x="{left}" y="{graph_y:.1f}" width="{x(graph) - left:.1f}" height="{bar_h}" fill="#F58518"/>')
            parts.append(f'<text x="{x(rag) + 4:.1f}" y="{rag_y + 9:.1f}" font-family="Arial, sans-serif" font-size="8" fill="#333">{rag:.3f}</text>')
            parts.append(f'<text x="{x(graph) + 4:.1f}" y="{graph_y + 9:.1f}" font-family="Arial, sans-serif" font-size="8" fill="#333">{graph:.3f}</text>')
            delta_color = "#2f7f5f" if delta > 0 else "#b84040" if delta < 0 else "#555"
            parts.append(f'<text x="{box_x + box_w - 22}" y="{y_mid + 3.5:.1f}" font-family="Arial, sans-serif" font-size="8.5" text-anchor="end" fill="{delta_color}">Δ {delta:+.3f}</text>')

        parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{box_y + box_h - 10}" font-family="Arial, sans-serif" font-size="9.5" text-anchor="middle" fill="#555">BLEU-4</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_dataset_task_visuals(pairwise: pd.DataFrame, output_dir: Path) -> List[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    score_df = dataset_task_score_table(pairwise)
    files = [
        plot_dir / "rag_vs_graphrag_total_score_by_dataset_task_zero_shot.svg",
        plot_dir / "rag_vs_graphrag_total_score_by_dataset_task_fine_tuned.svg",
        plot_dir / "delta_score_by_dataset_task_training_state.svg",
    ]
    write_total_score_svg(score_df, "zero-shot", files[0])
    write_total_score_svg(score_df, "fine-tuned", files[1])
    write_delta_score_svg(score_df, files[2])
    return [str(path) for path in files if path.exists()]


def write_plots(pairwise: pd.DataFrame, output_dir: Path, skip_plots: bool) -> List[str]:
    if skip_plots:
        return ["Plots skipped by --skip-plots."]
    notes: List[str] = []
    svg_files = write_svg_dataset_task_visuals(pairwise, output_dir)
    if svg_files:
        notes.append("SVG plots written: " + ", ".join(svg_files))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        notes.append(f"PNG plots skipped because matplotlib is unavailable: {type(exc).__name__}: {exc}")
        return notes

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("question_intent", "delta_meteor", "delta_meteor_by_question_intent.png"),
        ("question_intent", "delta_rouge_l", "delta_rouge_l_by_question_intent.png"),
        ("question_intent", "delta_cosine", "delta_cosine_by_question_intent.png"),
        ("reference_type", "delta_meteor", "delta_meteor_by_reference_type.png"),
    ]
    for group_col, delta_col, filename in plot_specs:
        data = pairwise.groupby(group_col, dropna=False)[delta_col].mean().sort_values()
        fig_height = max(4, min(10, 0.45 * len(data) + 2))
        plt.figure(figsize=(9, fig_height))
        colors = ["#b84040" if value < 0 else "#2f7f5f" for value in data.values]
        plt.barh(data.index.astype(str), data.values, color=colors)
        plt.axvline(0, color="#222222", linewidth=0.8)
        plt.xlabel(f"Mean {delta_col} (GraphRAG - RAG)")
        plt.ylabel(group_col)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()
    return notes


def distribution_table(dataset_df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = dataset_df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    out["percentage"] = (out["count"] / len(dataset_df) * 100).round(2)
    return out


def ordered_distribution_table(dataset_df: pd.DataFrame, col: str, order: Sequence[str]) -> pd.DataFrame:
    counts = dataset_df[col].value_counts(dropna=False).to_dict()
    rows = []
    seen = set()
    for value in order:
        count = int(counts.get(value, 0))
        if count > 0 or value != "unknown_task":
            rows.append({col: value, "count": count, "percentage": round(count / len(dataset_df) * 100, 2)})
        seen.add(value)
    for value, count in sorted(counts.items()):
        if value not in seen:
            rows.append({col: value, "count": int(count), "percentage": round(count / len(dataset_df) * 100, 2)})
    return pd.DataFrame(rows)


def dataset_task_distribution(dataset_df: pd.DataFrame) -> pd.DataFrame:
    return ordered_distribution_table(dataset_df, "dataset_task", TASK_ORDER)


def dataset_task_distribution_for_report(dataset_df: pd.DataFrame) -> pd.DataFrame:
    dist = dataset_task_distribution(dataset_df)
    full_total = sum(FULL_DATASET_TASK_COUNTS.values())
    dist["full_dataset_count"] = dist["dataset_task"].map(FULL_DATASET_TASK_COUNTS).fillna(0).astype(int)
    dist["full_dataset_percentage"] = (dist["full_dataset_count"] / full_total * 100).round(2)
    dist["test_minus_full_percentage"] = (dist["percentage"] - dist["full_dataset_percentage"]).round(2)
    return dist


def truncate_text(text: object, limit: int = 360) -> str:
    value = normalize_spaces(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def build_report(
    dataset_df: pd.DataFrame,
    pairwise: pd.DataFrame,
    group_outputs: Dict[str, pd.DataFrame],
    best_examples: pd.DataFrame,
    rag_examples: pd.DataFrame,
    diagnostics: List[Dict[str, object]],
    plot_notes: List[str],
    output_dir: Path,
) -> str:
    prefix_dist = distribution_table(dataset_df, "label_prefix")
    task_dist = dataset_task_distribution_for_report(dataset_df)
    qword_dist = distribution_table(dataset_df, "question_word")
    qintent_dist = distribution_table(dataset_df, "question_intent")
    reftype_dist = distribution_table(dataset_df, "reference_type")

    task_groups = group_outputs["dataset_task"].sort_values("delta_score", ascending=False)
    graph_task_groups = task_groups[task_groups["delta_score"] > 0].sort_values("delta_score", ascending=False)
    rag_task_groups = task_groups[task_groups["delta_score"] < 0].sort_values("delta_score", ascending=True)
    if graph_task_groups.empty:
        graph_task_groups = task_groups.sort_values("delta_score", ascending=False)
    if rag_task_groups.empty:
        rag_task_groups = task_groups.sort_values("delta_score", ascending=True)

    training_summary = aggregate_by_dimension(pairwise, "training_state")
    model_summary = aggregate_by_dimension(pairwise, "model")
    task_overall = aggregate_by_dimension(pairwise, "dataset_task")
    fine_tuned_bleu_task = group_bleu4_metrics(pairwise, "dataset_task", training_state="fine-tuned")
    fine_tuned_bleu_task_overall = (
        fine_tuned_bleu_task.groupby("group_value", as_index=False)
        .agg(
            count=("count", "sum"),
            mean_bleu4_rag=("mean_bleu4_rag", "mean"),
            mean_bleu4_graphrag=("mean_bleu4_graphrag", "mean"),
            delta_bleu=("delta_bleu", "mean"),
            positive_delta_percentage=("positive_delta_percentage", "mean"),
            negative_delta_percentage=("negative_delta_percentage", "mean"),
        )
        .sort_values("delta_bleu", ascending=False)
    )
    fine_tuned_corpus_bleu_task_overall = group_corpus_bleu4_metrics(
        pairwise,
        "dataset_task",
        training_state="fine-tuned",
    )
    fine_tuned_corpus_bleu_task_overall = fine_tuned_corpus_bleu_task_overall[
        fine_tuned_corpus_bleu_task_overall["model"] == "all"
    ].sort_values("delta_corpus_bleu4", ascending=False)
    intent_overall = aggregate_by_dimension(pairwise, "question_intent")
    ref_overall = aggregate_by_dimension(pairwise, "reference_type")

    strong_task_groups = task_overall[task_overall["count"] >= 30].copy()
    max_abs_delta = float(strong_task_groups["delta_score"].abs().max()) if not strong_task_groups.empty else 0.0
    if max_abs_delta >= 0.05:
        paper_signal = (
            "Sinyal dataset_task cukup jelas untuk dipertimbangkan sebagai tabel kecil atau paragraf diskusi, "
            "terutama pada task dengan count memadai dan delta_score terbesar."
        )
    else:
        paper_signal = (
            "Sinyal dataset_task belum terlalu kuat; hasil lebih aman dipakai sebagai analisis tambahan, "
            "bahan diskusi internal, atau jawaban saat tanya-jawab juri."
        )

    detected_rows = pd.DataFrame(diagnostics)
    full_runs = int((detected_rows["rows"] == len(dataset_df)).sum()) if not detected_rows.empty else 0
    unknown_prefixes = prefix_dist[prefix_dist["label_prefix"].astype(str).map(map_prefix_to_dataset_task) == "unknown_task"]
    unknown_task_count = int(dataset_df["dataset_task"].eq("unknown_task").sum())
    present_main_tasks = set(dataset_df["dataset_task"]) & set(FULL_DATASET_TASK_COUNTS)
    missing_main_tasks = [task for task in FULL_DATASET_TASK_COUNTS if task not in present_main_tasks]

    insights = [
        f"Terdeteksi {len(diagnostics)} konfigurasi evaluasi; {full_runs} konfigurasi memiliki {len(dataset_df)} row sesuai test split penuh.",
        "metadata.label dipertahankan sebagai sample_id, sedangkan prefix label dipakai untuk memetakan sample ke 6 task asli dataset.",
        f"Dataset task terbesar pada test split adalah {task_dist.sort_values('count', ascending=False).iloc[0]['dataset_task']} ({task_dist.sort_values('count', ascending=False).iloc[0]['count']} sample).",
        f"Task paling positif untuk GraphRAG secara agregat adalah {task_overall.iloc[0]['dataset_task']} (delta_score={task_overall.iloc[0]['delta_score']}).",
        f"Task paling positif untuk RAG secara agregat adalah {task_overall.sort_values('delta_score').iloc[0]['dataset_task']} (delta_score={task_overall.sort_values('delta_score').iloc[0]['delta_score']}).",
        paper_signal,
    ]
    if unknown_task_count:
        insights.append(f"Ada {unknown_task_count} sample unknown_task; cek mapping prefix sebelum memakai hasil untuk paper.")

    lines: List[str] = []
    lines.append("# Label Analysis Report")
    lines.append("")
    lines.append("## Cara Menjalankan")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/analysis/analyze_label_performance.py")
    lines.append("```")
    lines.append("")
    lines.append("## Tujuan Analisis")
    lines.append("")
    lines.append(
        "Analisis ini membandingkan RAG dan GraphRAG pada output evaluasi yang sudah ada, "
        "lalu mengelompokkan delta performa berdasarkan task asli dataset dan kategori heuristic tambahan."
    )
    lines.append("")
    lines.append("## Catatan Label Dataset")
    lines.append("")
    lines.append(
        "Kolom metadata.label berbentuk granular seperti qna-1082, draft-345, revision-..., dan sejenisnya. "
        "Full label digunakan sebagai sample_id. Prefix label digunakan untuk memetakan sample ke task asli dataset. "
        "Analisis utama menggunakan dataset_task; question_intent, question_word, reference_type, dan answer_length_bin "
        "hanya analisis tambahan berbasis heuristic."
    )
    if unknown_task_count:
        lines.append("")
        lines.append(
            f"Warning: ditemukan {unknown_task_count} sample dengan dataset_task=unknown_task. "
            "Prefix yang belum terpetakan perlu dicek sebelum hasil dipakai untuk paper."
        )
    if missing_main_tasks:
        lines.append("")
        lines.append(f"Warning: task utama berikut tidak muncul di test split: {', '.join(missing_main_tasks)}.")
    lines.append("")
    lines.append("## Konfigurasi Terdeteksi")
    lines.append("")
    lines.append(markdown_table(detected_rows, ["model", "training_state", "method", "rows", "source_kind", "source_file"], 20))
    lines.append("")
    lines.append("## Distribusi Label Prefix")
    lines.append("")
    lines.append(markdown_table(prefix_dist, ["label_prefix", "count", "percentage"], 20))
    lines.append("")
    lines.append("## Distribusi Dataset Task")
    lines.append("")
    lines.append(
        markdown_table(
            task_dist,
            ["dataset_task", "count", "percentage", "full_dataset_count", "full_dataset_percentage", "test_minus_full_percentage"],
            20,
        )
    )
    lines.append("")
    lines.append("## Group Paling Diuntungkan GraphRAG Berdasarkan Dataset Task")
    lines.append("")
    lines.append(
        markdown_table(
            graph_task_groups,
            ["group_value", "model", "training_state", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"],
            18,
        )
    )
    lines.append("")
    lines.append("## Group Yang Lebih Cocok Dengan RAG Berdasarkan Dataset Task")
    lines.append("")
    lines.append(
        markdown_table(
            rag_task_groups,
            ["group_value", "model", "training_state", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"],
            18,
        )
    )
    lines.append("")
    lines.append("## Analisis Tambahan: Question Word")
    lines.append("")
    lines.append(markdown_table(qword_dist, ["question_word", "count", "percentage"], 20))
    lines.append("")
    lines.append("## Analisis Tambahan: Question Intent")
    lines.append("")
    lines.append(markdown_table(qintent_dist, ["question_intent", "count", "percentage"], 20))
    lines.append("")
    lines.append("## Analisis Tambahan: Reference Type")
    lines.append("")
    lines.append(markdown_table(reftype_dist, ["reference_type", "count", "percentage"], 20))
    lines.append("")
    lines.append("## Heuristic Group Paling Diuntungkan GraphRAG")
    lines.append("")
    heuristic_groups = pd.concat(
        [group_outputs[col] for col in ["question_word", "question_intent", "reference_type", "answer_length_bin"]],
        ignore_index=True,
    )
    lines.append(
        markdown_table(
            heuristic_groups.sort_values("delta_score", ascending=False),
            ["group_column", "group_value", "model", "training_state", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"],
            12,
        )
    )
    lines.append("")
    lines.append("## Heuristic Group Yang Lebih Cocok Dengan RAG")
    lines.append("")
    lines.append(
        markdown_table(
            heuristic_groups.sort_values("delta_score", ascending=True),
            ["group_column", "group_value", "model", "training_state", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"],
            12,
        )
    )
    lines.append("")
    lines.append("## Zero-Shot Vs Fine-Tuned")
    lines.append("")
    lines.append(markdown_table(training_summary, ["training_state", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"], 10))
    lines.append("")
    lines.append("## LLaMA 2 7B Vs WizardLM 13B")
    lines.append("")
    lines.append(markdown_table(model_summary, ["model", "count", "delta_meteor", "delta_rouge_l", "delta_cosine", "delta_score", "interpretation"], 10))
    lines.append("")
    lines.append("## Analisis Khusus BLEU-4 Fine-Tuned")
    lines.append("")
    lines.append(
        "Bagian ini menggunakan BLEU-4 agregat yang konsisten dengan metrik utama paper. "
        "Nilai delta positif berarti GraphRAG lebih tinggi daripada RAG; nilai negatif berarti RAG lebih tinggi."
    )
    lines.append("")
    lines.append("### BLEU-4 Per Dataset Task")
    lines.append("")
    fine_tuned_corpus_bleu_display = fine_tuned_corpus_bleu_task_overall.rename(
        columns={
            "corpus_bleu4_rag": "bleu4_rag",
            "corpus_bleu4_graphrag": "bleu4_graphrag",
            "delta_corpus_bleu4": "delta_bleu4",
        }
    )
    lines.append(
        markdown_table(
            fine_tuned_corpus_bleu_display,
            ["group_value", "count", "bleu4_rag", "bleu4_graphrag", "delta_bleu4", "bleu4_winner"],
            10,
        )
    )
    lines.append("")
    lines.append("### Mean Sentence BLEU-4 Per Dataset Task")
    lines.append("")
    lines.append(
        markdown_table(
            fine_tuned_bleu_task_overall,
            ["group_value", "count", "mean_bleu4_rag", "mean_bleu4_graphrag", "delta_bleu", "positive_delta_percentage", "negative_delta_percentage"],
            10,
        )
    )
    lines.append("")
    lines.append("## Insight Utama")
    lines.append("")
    for item in insights:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Contoh Kualitatif GraphRAG Unggul")
    lines.append("")
    for _, row in best_examples.head(2).iterrows():
        lines.append(
            f"- `{row.get('model')}` `{row.get('training_state')}` sample `{row.get('sample_id')}` "
            f"task `{row.get('dataset_task')}` ({row.get('question_intent')}, {row.get('reference_type')}), "
            f"delta_score={row.get('delta_score'):.4f}."
        )
        lines.append(f"  Pertanyaan: {truncate_text(row.get('question'))}")
        lines.append(f"  RAG: {truncate_text(row.get('rag_prediction'))}")
        lines.append(f"  GraphRAG: {truncate_text(row.get('graphrag_prediction'))}")
    lines.append("")
    lines.append("## Contoh Kualitatif RAG Unggul")
    lines.append("")
    for _, row in rag_examples.head(2).iterrows():
        lines.append(
            f"- `{row.get('model')}` `{row.get('training_state')}` sample `{row.get('sample_id')}` "
            f"task `{row.get('dataset_task')}` ({row.get('question_intent')}, {row.get('reference_type')}), "
            f"delta_score={row.get('delta_score'):.4f}."
        )
        lines.append(f"  Pertanyaan: {truncate_text(row.get('question'))}")
        lines.append(f"  RAG: {truncate_text(row.get('rag_prediction'))}")
        lines.append(f"  GraphRAG: {truncate_text(row.get('graphrag_prediction'))}")
    lines.append("")
    lines.append("## Keterbatasan Analisis")
    lines.append("")
    lines.append("- Kategori question_intent dibuat dengan heuristic rule-based, bukan anotasi manual.")
    lines.append("- Dataset task dipetakan dari prefix metadata.label; jika ada prefix baru, mapping perlu diperbarui.")
    lines.append("- Delta dihitung dari metrik otomatis, sehingga belum menggantikan penilaian kualitatif hukum oleh evaluator manusia.")
    lines.append("- Analisis ini memakai output evaluasi yang sudah ada dan tidak menjalankan ulang inference model.")
    lines.append("- Jika sebuah kategori memiliki count kecil, interpretasinya harus diperlakukan sebagai indikasi awal.")
    if plot_notes:
        lines.append("")
        lines.append("## Catatan Plot")
        lines.append("")
        for note in plot_notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Sinyal Untuk Paper")
    lines.append("")
    lines.append(paper_signal)

    report = "\n".join(lines) + "\n"
    (output_dir / "label_analysis_report.md").write_text(report, encoding="utf-8")
    return report


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_test_data(args.test_data)
    write_csv(
        dataset_df[
            [
                "index",
                "sample_id",
                "label_prefix",
                "dataset_task",
                "reference",
                "reference_from_input",
                "question",
                "reference_answer",
                "question_word",
                "question_intent",
                "reference_type",
                "article_number",
                "answer_length_bin",
            ]
        ],
        args.output_dir / "dataset_parsed_with_categories.csv",
    )
    write_csv(distribution_table(dataset_df, "label_prefix"), args.output_dir / "label_prefix_distribution.csv")
    write_csv(dataset_task_distribution(dataset_df), args.output_dir / "dataset_task_distribution.csv")

    eval_configs = discover_eval_configs(args.reports_dir, args.include_only)
    prediction_configs = discover_prediction_configs(args.predictions_dir, args.include_only)
    all_metrics, diagnostics = load_all_metrics(
        eval_configs=eval_configs,
        prediction_configs=prediction_configs,
        dataset_df=dataset_df,
        recompute_missing=args.recompute_missing_metrics,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    pairwise = build_pairwise_delta(all_metrics)
    write_csv(pairwise, args.output_dir / "per_item_rag_vs_graphrag_delta.csv")

    group_outputs: Dict[str, pd.DataFrame] = {}
    for group_col in GROUP_COLUMNS:
        grouped = group_metrics(pairwise, group_col)
        group_outputs[group_col] = grouped
        write_csv(grouped, args.output_dir / f"group_metrics_by_{group_col}.csv")

    bleu_outputs = []
    fine_tuned_bleu_outputs = []
    corpus_bleu_outputs = []
    fine_tuned_corpus_bleu_outputs = []
    for group_col in GROUP_COLUMNS:
        bleu_outputs.append(group_bleu4_metrics(pairwise, group_col))
        fine_tuned_bleu_outputs.append(group_bleu4_metrics(pairwise, group_col, training_state="fine-tuned"))
        corpus_bleu_outputs.append(group_corpus_bleu4_metrics(pairwise, group_col))
        fine_tuned_corpus_bleu_outputs.append(group_corpus_bleu4_metrics(pairwise, group_col, training_state="fine-tuned"))
    bleu_by_classification = pd.concat(bleu_outputs, ignore_index=True)
    fine_tuned_bleu_by_classification = pd.concat(fine_tuned_bleu_outputs, ignore_index=True)
    corpus_bleu_by_classification = pd.concat(corpus_bleu_outputs, ignore_index=True)
    fine_tuned_corpus_bleu_by_classification = pd.concat(fine_tuned_corpus_bleu_outputs, ignore_index=True)
    write_csv(bleu_by_classification, args.output_dir / "bleu4_metrics_by_classification.csv")
    write_csv(fine_tuned_bleu_by_classification, args.output_dir / "bleu4_metrics_by_classification_fine_tuned.csv")
    write_csv(
        fine_tuned_bleu_by_classification[fine_tuned_bleu_by_classification["group_column"] == "dataset_task"],
        args.output_dir / "bleu4_metrics_by_dataset_task_fine_tuned.csv",
    )
    write_csv(corpus_bleu_by_classification, args.output_dir / "corpus_bleu4_metrics_by_classification.csv")
    write_csv(fine_tuned_corpus_bleu_by_classification, args.output_dir / "corpus_bleu4_metrics_by_classification_fine_tuned.csv")
    write_csv(
        fine_tuned_corpus_bleu_by_classification[
            (fine_tuned_corpus_bleu_by_classification["group_column"] == "dataset_task")
            & (fine_tuned_corpus_bleu_by_classification["model"] == "all")
        ],
        args.output_dir / "corpus_bleu4_metrics_by_dataset_task_fine_tuned.csv",
    )

    best_examples = select_example_columns(pairwise.sort_values("delta_score", ascending=False).head(10))
    rag_examples = select_example_columns(pairwise.sort_values("delta_score", ascending=True).head(10))
    write_csv(best_examples, args.output_dir / "graphrag_best_examples.csv")
    write_csv(rag_examples, args.output_dir / "rag_better_examples.csv")

    plot_notes = write_plots(pairwise, args.output_dir, args.skip_plots)
    if not args.skip_plots:
        bleu_plot = args.output_dir / "plots" / "delta_bleu4_by_dataset_task_fine_tuned.svg"
        write_bleu4_delta_svg(fine_tuned_bleu_by_classification, bleu_plot)
        if bleu_plot.exists():
            plot_notes.append(f"BLEU-4 SVG plot written: {bleu_plot}")
        for model in ["llama2-7b", "wizardlm-13b"]:
            model_suffix = model.replace("-", "_")
            for state, state_suffix in [("zero-shot", "zero_shot"), ("fine-tuned", "fine_tuned")]:
                bleu_rag_vs_graph = (
                    args.output_dir
                    / "plots"
                    / f"rag_vs_graphrag_bleu4_by_dataset_task_{model_suffix}_{state_suffix}.svg"
                )
                write_bleu4_rag_vs_graphrag_svg(corpus_bleu_by_classification, state, bleu_rag_vs_graph, model=model)
                if bleu_rag_vs_graph.exists():
                    plot_notes.append(f"BLEU-4 RAG vs GraphRAG SVG plot written: {bleu_rag_vs_graph}")
        combined_bleu = args.output_dir / "plots" / "rag_vs_graphrag_bleu4_by_dataset_task_combined.svg"
        write_combined_bleu4_rag_vs_graphrag_svg(corpus_bleu_by_classification, combined_bleu)
        if combined_bleu.exists():
            plot_notes.append(f"Combined BLEU-4 RAG vs GraphRAG SVG plot written: {combined_bleu}")
        fine_tuned_two_model_bleu = (
            args.output_dir
            / "plots"
            / "rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_llama_wizard.svg"
        )
        write_finetuned_bleu4_two_model_svg(corpus_bleu_by_classification, fine_tuned_two_model_bleu)
        if fine_tuned_two_model_bleu.exists():
            plot_notes.append(f"Fine-tuned two-model BLEU-4 SVG plot written: {fine_tuned_two_model_bleu}")
        for model in ["llama2-7b", "wizardlm-13b"]:
            model_suffix = model.replace("-", "_")
            single_panel = (
                args.output_dir
                / "plots"
                / f"rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_{model_suffix}_paper.svg"
            )
            write_single_model_paper_bleu4_svg(corpus_bleu_by_classification, model, single_panel)
            if single_panel.exists():
                plot_notes.append(f"Paper-style single-model BLEU-4 SVG plot written: {single_panel}")
        two_model_single_paper = (
            args.output_dir
            / "plots"
            / "rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_two_models_paper.svg"
        )
        write_two_model_single_paper_bleu4_svg(corpus_bleu_by_classification, two_model_single_paper)
        if two_model_single_paper.exists():
            plot_notes.append(f"Paper-style two-model BLEU-4 SVG plot written: {two_model_single_paper}")
    build_report(
        dataset_df=dataset_df,
        pairwise=pairwise,
        group_outputs=group_outputs,
        best_examples=best_examples,
        rag_examples=rag_examples,
        diagnostics=diagnostics,
        plot_notes=plot_notes,
        output_dir=args.output_dir,
    )

    task_top = aggregate_by_dimension(pairwise, "dataset_task")
    graph_top = task_top.sort_values("delta_score", ascending=False).head(3)
    rag_top = task_top.sort_values("delta_score", ascending=True).head(3)

    print(f"Detected configurations: {len(diagnostics)}")
    for row in diagnostics:
        print(f"- {row['model']} {row['training_state']} {row['method']}: {row['rows']} rows ({row['source_kind']})")
    print(f"Parsed dataset rows: {len(dataset_df)}")
    print(f"Pairwise RAG vs GraphRAG rows: {len(pairwise)}")
    print(f"Output directory: {args.output_dir}")
    print("Top GraphRAG-favored dataset_task groups:")
    for _, row in graph_top.iterrows():
        label = row.get("dataset_task", "")
        print(f"- {label}: delta_score={row['delta_score']}, count={row['count']}, {row['interpretation']}")
    print("Top RAG-favored dataset_task groups:")
    for _, row in rag_top.iterrows():
        label = row.get("dataset_task", "")
        print(f"- {label}: delta_score={row['delta_score']}, count={row['count']}, {row['interpretation']}")
    if plot_notes:
        for note in plot_notes:
            print(note)


if __name__ == "__main__":
    main()
