import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_input(row):
    reference = clean_text(row.get("reference", ""))
    instruction = clean_text(row.get("instruction", ""))

    if reference:
        return f"Referensi: {reference}\nPertanyaan: {instruction}"
    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = ["instruction", "output"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    if "reference" not in df.columns:
        df["reference"] = ""

    if "label" not in df.columns:
        df["label"] = ""

    df["input"] = df.apply(build_input, axis=1)
    df["output"] = df["output"].apply(clean_text)
    df["label"] = df["label"].apply(clean_text)
    df["reference"] = df["reference"].apply(clean_text)

    df = df[df["input"].str.len() > 0]
    df = df[df["output"].str.len() > 0]
    df = df.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    def save_jsonl(dataframe, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in dataframe.iterrows():
                item = {
                    "input": row["input"],
                    "output": row["output"],
                    "metadata": {
                        "reference": row["reference"],
                        "label": row["label"],
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_df, out_dir / "train.jsonl")
    save_jsonl(test_df, out_dir / "test.jsonl")

    summary = {
        "input_file": str(input_path),
        "total_after_cleaning": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "columns": list(df.columns),
        "sample": {
            "input": train_df.iloc[0]["input"] if len(train_df) else "",
            "output": train_df.iloc[0]["output"] if len(train_df) else "",
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Dataset preparation selesai.")
    print(f"Total data: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
