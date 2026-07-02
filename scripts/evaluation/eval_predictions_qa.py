#!/usr/bin/env python3
"""
Evaluate QA-style prediction CSV files for Part 1 baseline validation.

This unified evaluator keeps the convenience of one script while using
library-based metrics that are appropriate for paper reporting:
- BLEU via NLTK
- METEOR via NLTK
- ROUGE-L via rouge_score
- Embedding cosine via sentence-transformers embeddings

Supported outputs:
- scripts/training/infer_lora.py
- scripts/training/infer_lora_rag_v2.py
- scripts/training/infer_lora_graphrag_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph.graphrag_retriever_v2 import load_sentence_transformer


def normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_text(text: str) -> str:
    text = normalize_spaces(text).lower()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sum(float(x) * float(x) for x in a) ** 0.5
    norm_b = sum(float(y) * float(y) for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def ensure_meteor_resources() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK resource 'wordnet' belum tersedia. "
            "Install dulu dengan: .venv/bin/python -c \"import nltk; nltk.download('wordnet')\""
        ) from exc


def safe_meteor(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    try:
        return float(meteor_score([ref_tokens], pred_tokens))
    except LookupError as exc:
        raise RuntimeError(
            "METEOR gagal karena resource NLTK belum lengkap. "
            "Install minimal 'wordnet' di environment ini."
        ) from exc


def evaluate_rows(
    rows: List[Dict[str, str]],
    encoder_model=None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    per_row: List[Dict[str, object]] = []
    by_label = defaultdict(list)
    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    corpus_references = []
    corpus_predictions = []

    bert_pairs: List[Tuple[int, str, str]] = []
    for idx, row in enumerate(rows):
        bert_pairs.append((idx, row.get("ground_truth", ""), row.get("prediction", "")))

    bert_scores: Dict[int, float] = {}
    if encoder_model is not None and bert_pairs:
        texts = []
        for _, reference, prediction in bert_pairs:
            texts.append(reference)
            texts.append(prediction)
        vectors = encoder_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for pair_index, (row_index, _, _) in enumerate(bert_pairs):
            bert_scores[row_index] = float(cosine_similarity(vectors[2 * pair_index], vectors[(2 * pair_index) + 1]))

    for idx, row in enumerate(rows):
        ground_truth = row.get("ground_truth", "")
        prediction = row.get("prediction", "")
        label = row.get("label", "")
        ref_tokens = tokenize(ground_truth)
        pred_tokens = tokenize(prediction)

        corpus_references.append([ref_tokens])
        corpus_predictions.append(pred_tokens)

        if ref_tokens and pred_tokens:
            bleu = float(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing,
                )
            )
        else:
            bleu = 0.0

        meteor = safe_meteor(ground_truth, prediction)
        rouge_score = rouge.score(normalize_spaces(ground_truth), normalize_spaces(prediction))["rougeL"]
        embedding_cos = bert_scores.get(idx, 0.0)

        result = {
            "index": safe_int(row.get("index", str(idx))),
            "label": label,
            "reference": row.get("reference", ""),
            "bleu": round(bleu, 6),
            "meteor": round(meteor, 6),
            "rouge_l": round(float(rouge_score.fmeasure), 6),
            "rouge_l_precision": round(float(rouge_score.precision), 6),
            "rouge_l_recall": round(float(rouge_score.recall), 6),
            "embedding_cosine": round(embedding_cos, 6),
            "prompt_token_count": safe_int(row.get("prompt_token_count", "0")),
            "prediction_chars": len(prediction),
            "ground_truth_chars": len(ground_truth),
            "prediction": prediction,
            "ground_truth": ground_truth,
        }
        per_row.append(result)
        by_label[label].append(result)

    if corpus_references and corpus_predictions:
        corpus_bleu_1 = float(
            corpus_bleu(
                corpus_references,
                corpus_predictions,
                weights=(1.0, 0.0, 0.0, 0.0),
                smoothing_function=smoothing,
            )
        )
        corpus_bleu_2 = float(
            corpus_bleu(
                corpus_references,
                corpus_predictions,
                weights=(0.5, 0.5, 0.0, 0.0),
                smoothing_function=smoothing,
            )
        )
        corpus_bleu_4 = float(
            corpus_bleu(
                corpus_references,
                corpus_predictions,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )
    else:
        corpus_bleu_1 = 0.0
        corpus_bleu_2 = 0.0
        corpus_bleu_4 = 0.0

    summary = {
        "total_rows": len(per_row),
        "bleu_1": round(corpus_bleu_1, 6),
        "bleu_2": round(corpus_bleu_2, 6),
        "bleu_4": round(corpus_bleu_4, 6),
        "avg_sentence_bleu": round(mean_or_zero(row["bleu"] for row in per_row), 6),
        "avg_meteor": round(mean_or_zero(row["meteor"] for row in per_row), 6),
        "avg_rouge_l": round(mean_or_zero(row["rouge_l"] for row in per_row), 6),
        "avg_rouge_l_precision": round(mean_or_zero(row["rouge_l_precision"] for row in per_row), 6),
        "avg_rouge_l_recall": round(mean_or_zero(row["rouge_l_recall"] for row in per_row), 6),
        "avg_embedding_cosine": round(mean_or_zero(row["embedding_cosine"] for row in per_row), 6),
        "avg_prompt_token_count": round(mean_or_zero(row["prompt_token_count"] for row in per_row), 2),
        "by_label": {},
        "metric_impl": {
            "bleu": "nltk.corpus_bleu + nltk.sentence_bleu",
            "meteor": "nltk.translate.meteor_score",
            "rouge_l": "rouge_score.RougeScorer(rougeL)",
            "embedding_cosine": "sentence-transformers cosine similarity",
        },
    }

    for label, label_rows in sorted(by_label.items()):
        summary["by_label"][label] = {
            "count": len(label_rows),
            "avg_sentence_bleu": round(mean_or_zero(row["bleu"] for row in label_rows), 6),
            "avg_meteor": round(mean_or_zero(row["meteor"] for row in label_rows), 6),
            "avg_rouge_l": round(mean_or_zero(row["rouge_l"] for row in label_rows), 6),
            "avg_embedding_cosine": round(mean_or_zero(row["embedding_cosine"] for row in label_rows), 6),
        }

    return per_row, summary


def write_per_row_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "label",
                "reference",
                "bleu",
                "meteor",
                "rouge_l",
                "rouge_l_precision",
                "rouge_l_recall",
                "embedding_cosine",
                "prompt_token_count",
                "prediction_chars",
                "ground_truth_chars",
                "prediction",
                "ground_truth",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA prediction CSV files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/part1_baseline_eval.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/reports/part1_baseline_eval_rows.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-embedding-cosine",
        "--skip-bert-cosine",
        action="store_true",
        dest="skip_embedding_cosine",
    )
    parser.add_argument(
        "--embedding-model",
        "--bert-model",
        default="intfloat/multilingual-e5-base",
        dest="embedding_model",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_meteor_resources()
    rows = load_rows(args.predictions)
    if args.limit:
        rows = rows[: args.limit]

    encoder_model = None
    if not args.skip_embedding_cosine:
        print(f"Loading semantic similarity model: {args.embedding_model}", flush=True)
        encoder_model = load_sentence_transformer(
            model_name=args.embedding_model,
            device=args.device,
            local_files_only=True,
        )

    per_row, summary = evaluate_rows(rows, encoder_model=encoder_model)
    report = {
        "run_name": args.name or args.predictions.stem,
        "predictions_file": str(args.predictions),
        "rows": summary["total_rows"],
        "embedding_model": args.embedding_model if not args.skip_embedding_cosine else "",
        "metrics": summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_per_row_csv(args.out_csv, per_row)

    print(f"rows: {summary['total_rows']}")
    print(f"bleu_1: {summary['bleu_1']}")
    print(f"bleu_2: {summary['bleu_2']}")
    print(f"bleu_4: {summary['bleu_4']}")
    print(f"avg_sentence_bleu: {summary['avg_sentence_bleu']}")
    print(f"avg_meteor: {summary['avg_meteor']}")
    print(f"avg_rouge_l: {summary['avg_rouge_l']}")
    print(f"avg_embedding_cosine: {summary['avg_embedding_cosine']}")
    print(f"avg_prompt_token_count: {summary['avg_prompt_token_count']}")
    print(f"report_json: {args.out_json}")
    print(f"report_rows_csv: {args.out_csv}")


if __name__ == "__main__":
    main()
