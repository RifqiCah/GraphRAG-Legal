#!/usr/bin/env python3
"""
Audit reference coverage/consistency in evaluation set (test.jsonl).

Outputs:
- summary counts
- detailed flagged rows
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_reference(reference: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    ref = str(reference or "")
    regulasi_id = None

    if re.search(r"\bUUD\b.*1945", ref, flags=re.IGNORECASE):
        regulasi_id = "REG-UUD-1945"
    else:
        reg_match = re.search(
            r"\b(PP|UU|Perpres|Permendikbud)\s*(?:No\.?|Nomor)?\s*([A-Za-z0-9]+)\s*(?:Tahun)?\s*(\d{4})",
            ref,
            flags=re.IGNORECASE,
        )
        if reg_match:
            regulasi_id = f"REG-{reg_match.group(1).upper()}-{reg_match.group(2)}-{reg_match.group(3)}"

    pasal_match = re.search(r"\bPasal\s+([0-9A-Za-z]+)", ref, flags=re.IGNORECASE)
    ayat_match = re.search(r"\bayat\s*\(?([0-9A-Za-z]+)\)?", ref, flags=re.IGNORECASE)

    return (
        regulasi_id,
        pasal_match.group(1) if pasal_match else None,
        ayat_match.group(1) if ayat_match else None,
    )


def load_processed_index(processed_dir: Path) -> Dict[str, Dict[str, Set]]:
    index: Dict[str, Dict[str, Set]] = {}
    for path in sorted(glob.glob(str(processed_dir / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reg_id = data.get("regulasi_id") or Path(path).stem
        pasal_only: Set[str] = set()
        pasal_ayat: Set[Tuple[str, str]] = set()
        for pasal in data.get("pasal", []):
            nomor_pasal = str(pasal.get("nomor_pasal", "")).strip()
            if nomor_pasal:
                pasal_only.add(nomor_pasal)
            for ayat in pasal.get("ayat", []):
                nomor_ayat = str(ayat.get("nomor_ayat", "")).strip()
                if nomor_pasal and nomor_ayat:
                    pasal_ayat.add((nomor_pasal, nomor_ayat))
        index[reg_id] = {"pasal_only": pasal_only, "pasal_ayat": pasal_ayat}
    return index


def audit(test_path: Path, processed_index: Dict[str, Dict[str, Set]]) -> Tuple[Counter, List[Dict]]:
    counts: Counter = Counter()
    findings: List[Dict] = []

    with open(test_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("metadata", {}).get("reference", "")
            label = obj.get("metadata", {}).get("label", "")
            rid, pasal, ayat = parse_reference(ref)

            counts["total_rows"] += 1
            issue = None
            issue_detail = ""

            if not rid:
                issue = "ref_no_reg_parse"
                issue_detail = "reference does not contain parseable regulation id (PP/UU/UUD + year)"
            elif rid not in processed_index:
                issue = "ref_reg_not_in_corpus"
                issue_detail = "regulation id referenced in test set not found in current corpus"
            elif pasal and ayat:
                if (str(pasal), str(ayat)) not in processed_index[rid]["pasal_ayat"]:
                    issue = "ref_pasal_ayat_not_found"
                    issue_detail = "pasal+ayat referenced not found in processed document structure"
            elif pasal and not ayat:
                if str(pasal) not in processed_index[rid]["pasal_only"]:
                    issue = "ref_pasal_not_found"
                    issue_detail = "pasal referenced not found in processed document structure"

            if issue:
                counts[issue] += 1
                findings.append(
                    {
                        "row_index": row_idx,
                        "label": label,
                        "reference": ref,
                        "parsed_regulasi_id": rid or "",
                        "parsed_pasal": pasal or "",
                        "parsed_ayat": ayat or "",
                        "issue_type": issue,
                        "issue_detail": issue_detail,
                    }
                )

    return counts, findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/sft/test.jsonl")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_json", default="outputs/reports/eval_reference_coverage_report.json")
    parser.add_argument("--out_csv", default="outputs/reports/eval_reference_coverage_report.csv")
    args = parser.parse_args()

    processed_index = load_processed_index(Path(args.processed_dir))
    counts, findings = audit(Path(args.test_path), processed_index)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_path": args.test_path,
                "processed_dir": args.processed_dir,
                "summary": dict(counts),
                "flagged_rows": len(findings),
                "findings": findings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    out_csv = Path(args.out_csv)
    cols = [
        "row_index",
        "label",
        "reference",
        "parsed_regulasi_id",
        "parsed_pasal",
        "parsed_ayat",
        "issue_type",
        "issue_detail",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in findings:
            writer.writerow(row)

    print("Summary:")
    for k in [
        "total_rows",
        "ref_no_reg_parse",
        "ref_reg_not_in_corpus",
        "ref_pasal_not_found",
        "ref_pasal_ayat_not_found",
    ]:
        print(f"- {k}: {counts.get(k, 0)}")
    print(f"Flagged rows: {len(findings)}")
    print(f"JSON report: {out_json}")
    print(f"CSV report : {out_csv}")


if __name__ == "__main__":
    main()
