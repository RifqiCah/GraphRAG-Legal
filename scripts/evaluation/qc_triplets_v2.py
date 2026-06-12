#!/usr/bin/env python3
"""
Rule-based QC for PSKP KG triplets v2.

This script does not call an LLM and does not modify triplet files. It audits
triplets_v2 for technical consistency before graph build.

Outputs:
- outputs/reports/v2/triplets_v2_qc_report.json
- outputs/reports/v2/triplets_v2_qc_issues.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SCHEMA_VERSION = "2.0.0"
DEFAULT_EXCLUDE = {"REG-UUD-1945"}
VALID_METHODS = {"structural", "regex", "llm", "metadata_status", "manual_review", "candidate_generation"}
SEMANTIC_PREDICATES = {
    "MENGATUR_TOPIK",
    "MENGATUR_SUBJEK",
    "MEMBERIKAN_HAK",
    "MEMBEBANKAN_KEWAJIBAN",
    "MELARANG",
    "MENDEFINISIKAN",
}
REQUIRED_FIELDS = [
    "schema_version",
    "source_id",
    "source_type",
    "predicate",
    "target_id",
    "target_type",
    "confidence",
    "extraction_method",
    "evidence",
    "properties",
]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_schema_predicates(schema_path: Path) -> Set[str]:
    schema = load_json(schema_path)
    labels: Set[str] = set()
    for group in schema.get("edge_definitions", {}).values():
        if not isinstance(group, list):
            continue
        for item in group:
            label = item.get("label")
            if label:
                labels.add(label)
    return labels


def as_set(value) -> Set[str]:
    if isinstance(value, list):
        return set(value)
    if isinstance(value, str):
        return {value}
    return set()


def load_edge_type_constraints(schema_path: Path) -> Dict[str, Dict[str, Set[str]]]:
    schema = load_json(schema_path)
    constraints: Dict[str, Dict[str, Set[str]]] = {}
    for group in schema.get("edge_definitions", {}).values():
        if not isinstance(group, list):
            continue
        for item in group:
            label = item.get("label")
            if not label:
                continue
            constraints[label] = {
                "source_type": as_set(item.get("source_type")),
                "target_type": as_set(item.get("target_type")),
            }
    return constraints


def load_chunk_ids(chunks_dir: Path, excluded: Set[str]) -> Set[str]:
    chunk_ids = set()
    for path in chunks_dir.glob("chunks_REG-*.jsonl"):
        rid = path.stem.replace("chunks_", "")
        if rid in excluded:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("chunk_id"):
                    chunk_ids.add(obj["chunk_id"])
    return chunk_ids


def issue(
    regulasi_id: str,
    file_name: str,
    triplet_index: int,
    severity: str,
    issue_type: str,
    detail: str,
    triplet: Dict | None = None,
) -> Dict:
    triplet = triplet or {}
    return {
        "regulasi_id": regulasi_id,
        "file": file_name,
        "triplet_index": triplet_index,
        "severity": severity,
        "issue_type": issue_type,
        "detail": detail,
        "source_id": triplet.get("source_id", ""),
        "predicate": triplet.get("predicate", ""),
        "target_id": triplet.get("target_id", ""),
        "extraction_method": triplet.get("extraction_method", ""),
    }


def triplet_key(row: Dict) -> Tuple:
    return (
        row.get("source_id", ""),
        row.get("source_type", ""),
        row.get("predicate", ""),
        row.get("target_id", ""),
        row.get("target_type", ""),
        row.get("extraction_method", ""),
        json.dumps(row.get("properties", {}), ensure_ascii=False, sort_keys=True),
    )


def looks_like_empty_node_id(value: str) -> bool:
    value = str(value or "").strip()
    return value in {"", "TOPIK-", "SBJ-", "PSL-", "AYT-", "CHK-", "REG-"}


def is_semantic_target_too_generic(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    generic = {
        "pendidikan",
        "regulasi",
        "peraturan",
        "ketentuan",
        "pemerintah",
        "pasal",
        "ayat",
        "hal",
        "umum",
        "lainnya",
    }
    return normalized in generic


def validate_triplet(
    row: Dict,
    regulasi_id: str,
    file_name: str,
    idx: int,
    valid_predicates: Set[str],
    edge_type_constraints: Dict[str, Dict[str, Set[str]]],
    chunk_ids: Set[str],
) -> List[Dict]:
    issues: List[Dict] = []

    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(issue(regulasi_id, file_name, idx, "error", "missing_required_field", field, row))

    if row.get("schema_version") != SCHEMA_VERSION:
        issues.append(issue(regulasi_id, file_name, idx, "error", "invalid_schema_version", str(row.get("schema_version")), row))

    if looks_like_empty_node_id(row.get("source_id", "")):
        issues.append(issue(regulasi_id, file_name, idx, "error", "empty_source_id", "source_id is empty or prefix-only", row))
    if looks_like_empty_node_id(row.get("target_id", "")):
        issues.append(issue(regulasi_id, file_name, idx, "error", "empty_target_id", "target_id is empty or prefix-only", row))

    if not row.get("source_type"):
        issues.append(issue(regulasi_id, file_name, idx, "error", "empty_source_type", "source_type is empty", row))
    if not row.get("target_type"):
        issues.append(issue(regulasi_id, file_name, idx, "error", "empty_target_type", "target_type is empty", row))

    predicate = row.get("predicate", "")
    if predicate not in valid_predicates:
        issues.append(issue(regulasi_id, file_name, idx, "error", "unknown_predicate", predicate, row))
    else:
        constraints = edge_type_constraints.get(predicate, {})
        allowed_source_types = constraints.get("source_type", set())
        allowed_target_types = constraints.get("target_type", set())
        if allowed_source_types and row.get("source_type") not in allowed_source_types:
            issues.append(issue(
                regulasi_id,
                file_name,
                idx,
                "error",
                "source_type_not_allowed_for_predicate",
                f"{row.get('source_type')} not in {sorted(allowed_source_types)}",
                row,
            ))
        if allowed_target_types and row.get("target_type") not in allowed_target_types:
            issues.append(issue(
                regulasi_id,
                file_name,
                idx,
                "error",
                "target_type_not_allowed_for_predicate",
                f"{row.get('target_type')} not in {sorted(allowed_target_types)}",
                row,
            ))

    method = row.get("extraction_method", "")
    if method not in VALID_METHODS:
        issues.append(issue(regulasi_id, file_name, idx, "error", "unknown_extraction_method", method, row))

    confidence = row.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        issues.append(issue(regulasi_id, file_name, idx, "error", "invalid_confidence", str(confidence), row))

    evidence = row.get("evidence", {})
    if not isinstance(evidence, dict):
        issues.append(issue(regulasi_id, file_name, idx, "error", "invalid_evidence", "evidence must be object", row))
    else:
        if not str(evidence.get("text", "")).strip():
            issues.append(issue(regulasi_id, file_name, idx, "warning", "empty_evidence_text", "evidence.text is empty", row))
        for chunk_id in evidence.get("chunk_ids", []) or []:
            if chunk_id and chunk_id not in chunk_ids:
                issues.append(issue(regulasi_id, file_name, idx, "warning", "unknown_chunk_id", chunk_id, row))

    if method == "llm":
        if predicate not in SEMANTIC_PREDICATES:
            issues.append(issue(regulasi_id, file_name, idx, "warning", "llm_non_semantic_predicate", predicate, row))
        target_name = row.get("properties", {}).get("target_name", "") if isinstance(row.get("properties"), dict) else ""
        if not str(target_name).strip():
            issues.append(issue(regulasi_id, file_name, idx, "warning", "semantic_empty_target_name", "properties.target_name is empty", row))
        elif len(str(target_name)) < 3:
            issues.append(issue(regulasi_id, file_name, idx, "warning", "semantic_target_too_short", str(target_name), row))
        elif len(str(target_name)) > 140:
            issues.append(issue(regulasi_id, file_name, idx, "warning", "semantic_target_too_long", str(target_name)[:180], row))
        elif is_semantic_target_too_generic(str(target_name)):
            issues.append(issue(regulasi_id, file_name, idx, "warning", "semantic_target_too_generic", str(target_name), row))
        if confidence is not None and isinstance(confidence, (int, float)) and float(confidence) < 0.5:
            issues.append(issue(regulasi_id, file_name, idx, "warning", "low_semantic_confidence", str(confidence), row))

    if predicate == "MEMILIKI_CHUNK" and not str(row.get("target_id", "")).startswith("CHK-"):
        issues.append(issue(regulasi_id, file_name, idx, "error", "memiliki_chunk_target_not_chunk", row.get("target_id", ""), row))

    return issues


def audit_triplets(
    triplets_dir: Path,
    chunks_dir: Path,
    schema_path: Path,
    excluded: Set[str],
) -> Dict:
    valid_predicates = load_schema_predicates(schema_path)
    edge_type_constraints = load_edge_type_constraints(schema_path)
    chunk_ids = load_chunk_ids(chunks_dir, excluded)
    all_issues: List[Dict] = []
    method_counts: Counter = Counter()
    predicate_counts: Counter = Counter()
    document_rows: List[Dict] = []
    duplicate_count = 0

    for path in sorted(triplets_dir.glob("triplets_REG-*.json")):
        payload = load_json(path)
        regulasi_id = payload.get("regulasi_id") or path.stem.replace("triplets_", "")
        if regulasi_id in excluded:
            document_rows.append({
                "regulasi_id": regulasi_id,
                "file": str(path),
                "status": "excluded",
                "total_triplets": len(payload.get("triplets", [])),
                "issues": 0,
            })
            continue

        rows = payload.get("triplets", [])
        seen = {}
        doc_issues: List[Dict] = []
        doc_methods: Counter = Counter()
        doc_predicates: Counter = Counter()

        for idx, row in enumerate(rows):
            key = triplet_key(row)
            if key in seen:
                duplicate_count += 1
                doc_issues.append(issue(
                    regulasi_id,
                    path.name,
                    idx,
                    "warning",
                    "duplicate_triplet",
                    f"duplicate of triplet_index {seen[key]}",
                    row,
                ))
            else:
                seen[key] = idx

            doc_methods[row.get("extraction_method", "")] += 1
            doc_predicates[row.get("predicate", "")] += 1
            doc_issues.extend(validate_triplet(
                row,
                regulasi_id,
                path.name,
                idx,
                valid_predicates,
                edge_type_constraints,
                chunk_ids,
            ))

        all_issues.extend(doc_issues)
        method_counts.update(doc_methods)
        predicate_counts.update(doc_predicates)
        document_rows.append({
            "regulasi_id": regulasi_id,
            "file": str(path),
            "status": "checked",
            "total_triplets": len(rows),
            "issues": len(doc_issues),
            "errors": sum(1 for item in doc_issues if item["severity"] == "error"),
            "warnings": sum(1 for item in doc_issues if item["severity"] == "warning"),
            "method_counts": dict(doc_methods),
        })

    issue_counts = Counter(item["issue_type"] for item in all_issues)
    severity_counts = Counter(item["severity"] for item in all_issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "excluded_regulasi_ids": sorted(excluded),
        "summary": {
            "documents_checked": sum(1 for row in document_rows if row["status"] == "checked"),
            "documents_excluded": sum(1 for row in document_rows if row["status"] == "excluded"),
            "total_triplets_checked": sum(row["total_triplets"] for row in document_rows if row["status"] == "checked"),
            "total_issues": len(all_issues),
            "total_errors": severity_counts.get("error", 0),
            "total_warnings": severity_counts.get("warning", 0),
            "duplicate_triplets": duplicate_count,
            "method_counts": dict(method_counts),
            "predicate_counts": dict(predicate_counts),
            "issue_counts": dict(issue_counts),
        },
        "documents": document_rows,
        "issues": all_issues[:1000],
        "all_issues": all_issues,
        "issues_truncated": len(all_issues) > 1000,
    }


def write_issues_csv(path: Path, issues: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regulasi_id",
        "file",
        "triplet_index",
        "severity",
        "issue_type",
        "detail",
        "source_id",
        "predicate",
        "target_id",
        "extraction_method",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in issues:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets-dir", default="data/triplets_v2")
    parser.add_argument("--chunks-dir", default="data/chunks_v2")
    parser.add_argument("--schema", default="ontology/v2/schema.json")
    parser.add_argument("--report-json", default="outputs/reports/v2/triplets_v2_qc_report.json")
    parser.add_argument("--issues-csv", default="outputs/reports/v2/triplets_v2_qc_issues.csv")
    parser.add_argument("--include-uud", action="store_true", help="Do not exclude REG-UUD-1945")
    args = parser.parse_args()

    excluded = set() if args.include_uud else set(DEFAULT_EXCLUDE)
    report = audit_triplets(
        triplets_dir=Path(args.triplets_dir),
        chunks_dir=Path(args.chunks_dir),
        schema_path=Path(args.schema),
        excluded=excluded,
    )
    write_issues_csv(Path(args.issues_csv), report["all_issues"])
    report.pop("all_issues", None)
    write_json(Path(args.report_json), report)

    summary = report["summary"]
    print("documents_checked:", summary["documents_checked"])
    print("documents_excluded:", summary["documents_excluded"])
    print("total_triplets_checked:", summary["total_triplets_checked"])
    print("total_errors:", summary["total_errors"])
    print("total_warnings:", summary["total_warnings"])
    print("duplicate_triplets:", summary["duplicate_triplets"])
    print("report_json:", args.report_json)
    print("issues_csv:", args.issues_csv)


if __name__ == "__main__":
    main()
