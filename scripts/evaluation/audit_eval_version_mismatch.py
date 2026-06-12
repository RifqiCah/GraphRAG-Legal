#!/usr/bin/env python3
"""
Audit evaluation set for possible version mismatch between referenced regulation
and ground-truth answer text.

Focus: data/sft/test.jsonl only.
Method: deterministic rules (no AI).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Known amendment pairs in this project corpus: old -> new
AMENDMENT_MAP = {
    "REG-PP-17-2010": "REG-PP-66-2010",
    "REG-PP-74-2008": "REG-PP-19-2017",
    "REG-PP-57-2021": "REG-PP-4-2022",
    "REG-PP-48-2008": "REG-PP-18-2022",
}


@dataclass
class RefInfo:
    regulasi_id: Optional[str]
    pasal: Optional[str]
    ayat: Optional[str]


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", normalize_text(s))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def parse_reference(reference: str) -> RefInfo:
    ref = str(reference or "")
    reg_id = None

    if re.search(r"\bUUD\b.*1945", ref, flags=re.IGNORECASE):
        reg_id = "REG-UUD-1945"
    else:
        m = re.search(
            r"\b(PP|UU|Perpres|Permendikbud)\s*(?:No\.?|Nomor)?\s*([A-Za-z0-9]+)\s*(?:Tahun)?\s*(\d{4})",
            ref,
            flags=re.IGNORECASE,
        )
        if m:
            reg_id = f"REG-{m.group(1).upper()}-{m.group(2)}-{m.group(3)}"

    pasal_m = re.search(r"\bPasal\s+([0-9A-Za-z]+)", ref, flags=re.IGNORECASE)
    ayat_m = re.search(r"\bayat\s*\(?([0-9A-Za-z]+)\)?", ref, flags=re.IGNORECASE)

    return RefInfo(
        regulasi_id=reg_id,
        pasal=pasal_m.group(1) if pasal_m else None,
        ayat=ayat_m.group(1) if ayat_m else None,
    )


def load_processed_texts(processed_dir: Path) -> Dict[str, Dict[Tuple[str, str], str]]:
    """
    Returns:
      reg_id -> {(pasal, ayat): text}
    """
    out: Dict[str, Dict[Tuple[str, str], str]] = {}
    for p in sorted(processed_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        reg_id = data.get("regulasi_id") or p.stem
        pasal_map: Dict[Tuple[str, str], str] = {}
        for pasal in data.get("pasal", []):
            no_pasal = str(pasal.get("nomor_pasal", "")).strip()
            for ayat in pasal.get("ayat", []):
                no_ayat = str(ayat.get("nomor_ayat", "")).strip()
                text = str(ayat.get("teks_isi", "")).strip()
                if no_pasal and no_ayat and text:
                    pasal_map[(no_pasal, no_ayat)] = text
        out[reg_id] = pasal_map
    return out


def load_test_jsonl(test_path: Path) -> List[Dict]:
    rows = []
    with test_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_row_index"] = idx
            rows.append(obj)
    return rows


def audit_rows(
    rows: List[Dict],
    pasal_texts: Dict[str, Dict[Tuple[str, str], str]],
    similarity_delta: float = 0.08,
    min_new_similarity: float = 0.25,
) -> List[Dict]:
    findings: List[Dict] = []

    for row in rows:
        ref = row.get("metadata", {}).get("reference", "")
        gt = row.get("output", "")
        label = row.get("metadata", {}).get("label", "")
        info = parse_reference(ref)

        if not info.regulasi_id:
            continue
        if info.regulasi_id not in AMENDMENT_MAP:
            continue
        if not (info.pasal and info.ayat):
            continue

        old_reg = info.regulasi_id
        new_reg = AMENDMENT_MAP[old_reg]
        key = (str(info.pasal), str(info.ayat))

        old_text = pasal_texts.get(old_reg, {}).get(key, "")
        new_text = pasal_texts.get(new_reg, {}).get(key, "")
        if not old_text or not new_text:
            continue

        gt_norm = normalize_text(gt)
        old_sim = jaccard(gt_norm, old_text)
        new_sim = jaccard(gt_norm, new_text)

        # Lexical cue unique to new text
        old_tokens = set(tokenize(old_text))
        new_tokens = set(tokenize(new_text))
        gt_tokens = set(tokenize(gt_norm))
        unique_new_tokens = sorted((new_tokens - old_tokens) & gt_tokens)

        suspicious = (
            new_sim >= min_new_similarity
            and new_sim >= old_sim + similarity_delta
        ) or bool(unique_new_tokens)

        if suspicious:
            findings.append(
                {
                    "row_index": row["_row_index"],
                    "label": label,
                    "reference": ref,
                    "old_reg": old_reg,
                    "new_reg": new_reg,
                    "pasal": info.pasal,
                    "ayat": info.ayat,
                    "old_similarity": round(old_sim, 4),
                    "new_similarity": round(new_sim, 4),
                    "new_minus_old": round(new_sim - old_sim, 4),
                    "unique_new_tokens_in_gt": unique_new_tokens[:20],
                    "ground_truth": gt,
                    "old_text_excerpt": old_text[:400],
                    "new_text_excerpt": new_text[:400],
                    "reason": "ground_truth appears closer to amended version than referenced old regulation",
                }
            )

    findings.sort(key=lambda x: x["new_minus_old"], reverse=True)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/sft/test.jsonl")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_json", default="outputs/reports/eval_version_mismatch_report.json")
    parser.add_argument("--out_csv", default="outputs/reports/eval_version_mismatch_report.csv")
    args = parser.parse_args()

    rows = load_test_jsonl(Path(args.test_path))
    pasal_texts = load_processed_texts(Path(args.processed_dir))
    findings = audit_rows(rows, pasal_texts)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_path": args.test_path,
                "processed_dir": args.processed_dir,
                "total_rows": len(rows),
                "flagged_rows": len(findings),
                "findings": findings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Simple CSV export
    csv_cols = [
        "row_index",
        "label",
        "reference",
        "old_reg",
        "new_reg",
        "pasal",
        "ayat",
        "old_similarity",
        "new_similarity",
        "new_minus_old",
        "unique_new_tokens_in_gt",
        "reason",
    ]
    out_csv = Path(args.out_csv)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for r in findings:
            vals = []
            for c in csv_cols:
                v = r.get(c, "")
                if isinstance(v, list):
                    v = "|".join(map(str, v))
                v = str(v).replace('"', '""')
                vals.append(f'"{v}"')
            f.write(",".join(vals) + "\n")

    print(f"Total rows: {len(rows)}")
    print(f"Flagged rows: {len(findings)}")
    print(f"JSON report: {out_json}")
    print(f"CSV report : {out_csv}")


if __name__ == "__main__":
    main()
