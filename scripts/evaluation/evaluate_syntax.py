import argparse
import json
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    text = normalize_text(text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def safe_meteor(reference_tokens, prediction_tokens):
    try:
        return meteor_score([reference_tokens], prediction_tokens)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = ["ground_truth", "prediction"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    references = []
    predictions = []
    meteor_scores = []

    for _, row in df.iterrows():
        ref_tokens = tokenize(row["ground_truth"])
        pred_tokens = tokenize(row["prediction"])

        references.append([ref_tokens])
        predictions.append(pred_tokens)

        meteor_scores.append(safe_meteor(ref_tokens, pred_tokens))

    smoothing = SmoothingFunction().method1

    bleu_1 = corpus_bleu(
        references,
        predictions,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smoothing,
    )

    bleu_2 = corpus_bleu(
        references,
        predictions,
        weights=(0.5, 0.5, 0.0, 0.0),
        smoothing_function=smoothing,
    )

    bleu_4 = corpus_bleu(
        references,
        predictions,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    result = {
        "evaluation_level": "Level 1 - Sintaksis",
        "input_file": str(input_path),
        "num_samples": int(len(df)),
        "metrics": {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_4": bleu_4,
            "meteor": meteor_avg,
        },
        "description": {
            "bleu": "Mengukur kemiripan n-gram antara prediksi dan ground truth.",
            "meteor": "Mengukur kemiripan token dengan alignment yang lebih fleksibel dari BLEU.",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Evaluasi sintaksis selesai.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
