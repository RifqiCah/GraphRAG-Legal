import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value):
    return " ".join(clean_text(value).split()).lower()


def build_input(row):
    reference = clean_text(row.get("reference", ""))
    instruction = clean_text(row.get("instruction", ""))

    if reference:
        return f"Referensi: {reference}\nPertanyaan: {instruction}"
    return instruction


def parse_reference(reference):
    ref = clean_text(reference)
    regulasi_id = ""

    if re.search(r"\bUUD\b.*1945", ref, flags=re.IGNORECASE):
        regulasi_id = "REG-UUD-1945"
    else:
        match = re.search(
            r"\b(PP|UU|Perpres|Permendikbud)\s*(?:No\.?|Nomor)?\s*([A-Za-z0-9]+)\s*(?:Tahun)?\s*(\d{4})",
            ref,
            flags=re.IGNORECASE,
        )
        if match:
            regulasi_id = f"REG-{match.group(1).upper()}-{match.group(2)}-{match.group(3)}"

    pasal_match = re.search(r"\bPasal\s+([0-9A-Za-z]+)", ref, flags=re.IGNORECASE)
    ayat_match = re.search(r"\bayat\s*\(?([0-9A-Za-z]+)\)?", ref, flags=re.IGNORECASE)

    return (
        regulasi_id,
        clean_text(pasal_match.group(1) if pasal_match else ""),
        clean_text(ayat_match.group(1) if ayat_match else ""),
    )


def build_group_key(reference):
    reference = clean_text(reference)
    regulasi_id, pasal, ayat = parse_reference(reference)

    if regulasi_id:
        return f"parsed::{regulasi_id}::{pasal or 'NOPASAL'}::{ayat or 'NOAYAT'}"

    normalized_reference = normalize_text(reference)
    if normalized_reference:
        return f"fallback::{normalized_reference}"

    return "fallback::__empty_reference__"


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            return
        if self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
            return
        self.parent[right_root] = left_root
        self.rank[left_root] += 1


def union_rows_by_key(union_find, series):
    buckets = collections.defaultdict(list)
    for index, value in enumerate(series):
        buckets[value].append(index)

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        first = indices[0]
        for other in indices[1:]:
            union_find.union(first, other)


def build_components(df):
    union_find = UnionFind(len(df))
    union_rows_by_key(union_find, [row["group_key"] for row in df])
    union_rows_by_key(union_find, [row["output_key"] for row in df])

    components = collections.defaultdict(list)
    for index in range(len(df)):
        components[union_find.find(index)].append(index)
    return list(components.values())


def assign_components_to_splits(components, total_rows, validation_size, test_size, seed):
    train_size = 1.0 - validation_size - test_size
    if train_size <= 0:
        raise ValueError("Gabungan validation_size dan test_size harus kurang dari 1.0")

    targets = {
        "train": int(round(total_rows * train_size)),
        "validation": int(round(total_rows * validation_size)),
        "test": int(round(total_rows * test_size)),
    }
    targets["train"] = total_rows - targets["validation"] - targets["test"]

    randomizer = random.Random(seed)
    randomized_components = list(components)
    randomizer.shuffle(randomized_components)
    randomized_components.sort(key=len, reverse=True)

    assignments = {"train": [], "validation": [], "test": []}
    counts = {name: 0 for name in assignments}
    split_order = ["test", "validation", "train"]

    def choose_split(component_size):
        best_split = None
        best_score = None
        for split_name in split_order:
            current = counts[split_name]
            target = targets[split_name]
            remaining_need = target - current
            overshoot = max(0, current + component_size - target)
            underfill_after = max(0, target - (current + component_size))
            score = (
                overshoot,
                -min(component_size, max(remaining_need, 0)),
                underfill_after,
                current / max(target, 1),
                current,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        return best_split

    for component in randomized_components:
        split_name = choose_split(len(component))
        assignments[split_name].extend(component)
        counts[split_name] += len(component)

    return assignments, counts, targets


def split_dataframe(df, validation_size, test_size, seed):
    components = build_components(df)
    assignments, counts, targets = assign_components_to_splits(
        components=components,
        total_rows=len(df),
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    split_frames = {}
    for split_name, indices in assignments.items():
        split_frames[split_name] = [df[index] for index in sorted(indices)]

    return split_frames, counts, targets, components


def compute_split_overlap(frames, column):
    split_names = ["train", "validation", "test"]
    overlap = {}
    value_sets = {
        split_name: {row[column] for row in frame}
        for split_name, frame in frames.items()
    }

    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap[f"{left}__{right}"] = len(value_sets[left] & value_sets[right])
    return overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_columns = list(reader.fieldnames or [])
        rows = list(reader)

    required_cols = ["instruction", "output"]
    for col in required_cols:
        if col not in input_columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    prepared_rows = []
    seen_input_output = set()
    for row in rows:
        normalized_row = {
            "reference": clean_text(row.get("reference", "")),
            "instruction": clean_text(row.get("instruction", "")),
            "output": clean_text(row.get("output", "")),
            "label": clean_text(row.get("label", "")),
        }
        normalized_row["input"] = build_input(normalized_row)

        if not normalized_row["input"] or not normalized_row["output"]:
            continue

        dedupe_key = (normalized_row["input"], normalized_row["output"])
        if dedupe_key in seen_input_output:
            continue
        seen_input_output.add(dedupe_key)

        normalized_row["group_key"] = build_group_key(normalized_row["reference"])
        normalized_row["output_key"] = normalize_text(normalized_row["output"])
        prepared_rows.append(normalized_row)

    split_frames, counts, targets, components = split_dataframe(
        df=prepared_rows,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_df = split_frames["train"]
    validation_df = split_frames["validation"]
    test_df = split_frames["test"]

    def save_jsonl(dataframe, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in dataframe:
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
    save_jsonl(validation_df, out_dir / "validation.jsonl")
    save_jsonl(test_df, out_dir / "test.jsonl")

    summary = {
        "input_file": str(input_path),
        "total_after_cleaning": len(prepared_rows),
        "train_size": len(train_df),
        "validation_size": len(validation_df),
        "test_size": len(test_df),
        "split_method": "grouped_components_by_reference_and_output",
        "group_key_basis": "regulasi+pasal+ayat, fallback ke reference mentah jika tidak ter-parse",
        "target_sizes": targets,
        "actual_sizes": counts,
        "total_components": len(components),
        "component_size_stats": {
            "max": max((len(component) for component in components), default=0),
            "mean": round(sum(len(component) for component in components) / max(len(components), 1), 2),
            "median": (
                sorted(len(component) for component in components)[len(components) // 2]
                if components else 0
            ),
        },
        "overlap_checks": {
            "group_key": compute_split_overlap(split_frames, "group_key"),
            "output_key": compute_split_overlap(split_frames, "output_key"),
        },
        "columns": input_columns + ["input", "group_key", "output_key"],
        "sample": {
            "input": train_df[0]["input"] if train_df else "",
            "output": train_df[0]["output"] if train_df else "",
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Dataset preparation selesai.")
    print(f"Total data: {len(prepared_rows)}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
