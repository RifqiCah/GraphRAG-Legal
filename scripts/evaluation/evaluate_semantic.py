import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_rouge_l(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rouge_l_precision = []
    rouge_l_recall = []
    rouge_l_fmeasure = []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)["rougeL"]
        rouge_l_precision.append(score.precision)
        rouge_l_recall.append(score.recall)
        rouge_l_fmeasure.append(score.fmeasure)

    return {
        "rouge_l_precision": float(np.mean(rouge_l_precision)) if rouge_l_precision else 0.0,
        "rouge_l_recall": float(np.mean(rouge_l_recall)) if rouge_l_recall else 0.0,
        "rouge_l_f1": float(np.mean(rouge_l_fmeasure)) if rouge_l_fmeasure else 0.0,
    }


def compute_bert_cosine(references, predictions, model_name, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embedding model: {model_name}", flush=True)
    print(f"Device: {device}", flush=True)

    model = SentenceTransformer(model_name, device=device)

    ref_embeddings = model.encode(
        references,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    pred_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    cosine_scores = np.sum(ref_embeddings * pred_embeddings, axis=1)

    return {
        "bert_cosine_mean": float(np.mean(cosine_scores)),
        "bert_cosine_median": float(np.median(cosine_scores)),
        "bert_cosine_std": float(np.std(cosine_scores)),
        "bert_cosine_min": float(np.min(cosine_scores)),
        "bert_cosine_max": float(np.max(cosine_scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_bert", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = ["ground_truth", "prediction"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    references = [normalize_text(x) for x in df["ground_truth"].fillna("").tolist()]
    predictions = [normalize_text(x) for x in df["prediction"].fillna("").tolist()]

    rouge_l_result = compute_rouge_l(references, predictions)

    bert_result = {}
    if not args.skip_bert:
        bert_result = compute_bert_cosine(
            references=references,
            predictions=predictions,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
        )

    result = {
        "evaluation_level": "Level 2 - Semantik",
        "input_file": str(input_path),
        "num_samples": int(len(df)),
        "embedding_model": None if args.skip_bert else args.embedding_model,
        "metrics": {
            **rouge_l_result,
            **bert_result,
        },
        "description": {
            "rouge_l": "Mengukur overlap longest common subsequence antara prediksi dan ground truth.",
            "bert_cosine": "Mengukur kemiripan semantik embedding antara prediksi dan ground truth.",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Evaluasi semantik selesai.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
