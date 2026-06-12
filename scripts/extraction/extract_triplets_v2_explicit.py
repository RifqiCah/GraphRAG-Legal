#!/usr/bin/env python3
"""
Generate rule-based explicit legal triplets for PSKP KG v2.

This script does not call an LLM. It reads processed_v2 and chunks_v2, then
adds explicit legal relations to the triplets_v2 files produced by the
structural extractor.

Usage:
    python scripts/extraction/extract_triplets_v2_explicit.py --file REG-PP-47-2008
    python scripts/extraction/extract_triplets_v2_explicit.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "2.0.0"
PROCESSED_V2_DIR = Path("data/processed_v2")
CHUNKS_V2_DIR = Path("data/chunks_v2")
TRIPLETS_V2_DIR = Path("data/triplets_v2")
REPORT_PATH = Path("outputs/reports/v2/triplets_v2_explicit_report.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGULATION_RE = re.compile(
    r"\b(Undang-Undang(?:\s+Republik\s+Indonesia)?|UU|Peraturan\s+Pemerintah|PP)\s*"
    r"(?:Nomor|No\.?)?\s*([A-Za-z0-9/.-]+)\s*(?:Tahun\s*([12][0-9OIl]{3}))?",
    re.IGNORECASE,
)
PASAL_REF_RE = re.compile(
    r"\bPasal\s+([0-9]+[A-Z]?|[IVXLC]+)(?:\s+ayat\s*\(?([0-9]+[A-Za-z]?)\)?)?",
    re.IGNORECASE,
)
OCR_DIGIT_TRANS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})
INVALID_REFERENCE_NUMBERS = {"INI", "REPUBLIK", "INDONESIA"}
REFERENCE_SECTION_TYPES = {
    "pasal",
    "ayat",
    "subitem",
    "menimbang",
    "mengingat",
    "diktum",
    "penjelasan_pasal",
    "lampiran",
}


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").split())


def slug_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip().upper())
    return re.sub(r"-{2,}", "-", value).strip("-")


def normalize_ocr_year(value: str) -> str:
    return str(value or "").translate(OCR_DIGIT_TRANS)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_chunks(regulasi_id: str) -> List[Dict]:
    path = CHUNKS_V2_DIR / f"chunks_{regulasi_id}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_regulasi_id(kind_raw: str, number: str, year: Optional[str]) -> str:
    kind_raw_lower = str(kind_raw or "").lower()
    kind = "UU" if "undang" in kind_raw_lower or kind_raw_lower == "uu" else "PP"
    number = slug_token(number)
    year = slug_token(normalize_ocr_year(year or ""))
    return f"REG-{kind}-{number}-{year}" if year else f"REG-{kind}-{number}"


def find_regulation_refs(text: str) -> List[str]:
    refs = []
    if re.search(r"\b(UUD|Undang-Undang\s+Dasar)\b.*1945", text, flags=re.IGNORECASE):
        refs.append("REG-UUD-1945")
    for match in REGULATION_RE.finditer(text or ""):
        if slug_token(match.group(2)) in INVALID_REFERENCE_NUMBERS:
            continue
        refs.append(build_regulasi_id(match.group(1), match.group(2), match.group(3)))
    return list(dict.fromkeys(refs))


def triplet(
    source_id: str,
    source_type: str,
    predicate: str,
    target_id: str,
    target_type: str,
    evidence_text: str,
    properties: Dict | None = None,
    chunk_ids: List[str] | None = None,
    source_file: str = "",
    confidence: float = 0.95,
) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "source_type": source_type,
        "predicate": predicate,
        "target_id": target_id,
        "target_type": target_type,
        "confidence": confidence,
        "extraction_method": "regex",
        "evidence": {
            "text": normalize_space(evidence_text),
            "chunk_ids": chunk_ids or [],
            "source_file": source_file,
        },
        "properties": properties or {},
        "validation": {
            "status": "valid",
            "errors": [],
            "warnings": [],
        },
    }


def event_triplet(
    regulasi_id: str,
    predicate: str,
    event_id: str,
    event_kind: str,
    date_text: str,
    evidence_text: str,
    source_file: str,
) -> Dict:
    return triplet(
        regulasi_id,
        "Regulasi",
        predicate,
        event_id,
        "Peristiwa_Hukum",
        evidence_text,
        {"jenis": event_kind, "tanggal": date_text},
        source_file=source_file,
        confidence=1.0,
    )


def dedupe_triplets(rows: Iterable[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for row in rows:
        key = (
            row.get("source_id"),
            row.get("source_type"),
            row.get("predicate"),
            row.get("target_id"),
            row.get("target_type"),
            row.get("extraction_method"),
            json.dumps(row.get("properties", {}), sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def existing_payload(regulasi_id: str) -> Dict:
    path = TRIPLETS_V2_DIR / f"triplets_{regulasi_id}.json"
    if path.exists():
        return load_json(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "regulasi_id": regulasi_id,
        "extraction_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extractor": "extract_triplets_v2_explicit.py",
            "methods": [],
        },
        "triplets": [],
    }


def add_berdasarkan(processed: Dict, rows: List[Dict]) -> None:
    source_file = processed["sources"].get("main_pdf", "")
    for item in processed.get("preamble", {}).get("mengingat", []):
        target = item.get("target_regulasi_id")
        if not target:
            continue
        rows.append(triplet(
            item["id"],
            "Dasar_Hukum",
            "BERDASARKAN",
            target,
            "Regulasi",
            item.get("teks_isi", ""),
            {"nomor": item.get("nomor", ""), "urutan": item.get("urutan", 0)},
            source_file=source_file,
            confidence=1.0,
        ))


def add_closing_events(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")
    closing = processed.get("closing", {})

    if closing.get("tanggal_diundangkan"):
        rows.append(event_triplet(
            regulasi_id,
            "DIUNDANGKAN_PADA",
            f"EVT-{regulasi_id}-DIUNDANGKAN",
            "diundangkan",
            closing.get("tanggal_diundangkan", ""),
            closing.get("teks_isi", ""),
            source_file,
        ))

    if closing.get("tanggal_ditetapkan"):
        rows.append(event_triplet(
            regulasi_id,
            "MULAI_BERLAKU",
            f"EVT-{regulasi_id}-MULAI-BERLAKU",
            "mulai_berlaku",
            closing.get("tanggal_ditetapkan", ""),
            closing.get("teks_isi", ""),
            source_file,
        ))

    # Prefer the explicit effective-date sentence if present.
    for pasal in processed.get("body", {}).get("pasal", []):
        text = collect_node_text(pasal)
        match = re.search(r"mulai\s+berlaku\s+pada\s+tanggal\s+([^.;\n]+)", text, flags=re.IGNORECASE)
        if match:
            rows.append(event_triplet(
                regulasi_id,
                "MULAI_BERLAKU",
                f"EVT-{regulasi_id}-MULAI-BERLAKU",
                "mulai_berlaku",
                normalize_space(match.group(1)),
                text,
                source_file,
            ))
            break


def add_amendment_from_title(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")
    title = " ".join([
        processed.get("metadata", {}).get("judul", ""),
        processed.get("metadata", {}).get("tentang", ""),
        processed.get("preamble", {}).get("menetapkan", ""),
    ])
    if not re.search(r"perubahan\s+(?:atas|kesatu|kedua|ketiga|keempat|kelima|keenam|ketujuh|kedelapan|kesembilan|kesepuluh|kesebelas)", title, flags=re.IGNORECASE):
        return
    refs = [ref for ref in find_regulation_refs(title) if ref != regulasi_id]
    for ref in refs:
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MENGAMANDEMEN",
            ref,
            "Regulasi",
            title,
            {"source": "title_or_menetapkan"},
            source_file=source_file,
            confidence=0.9,
        ))


def collect_node_text(node: Dict) -> str:
    parts = [node.get("teks_isi", "")]
    for ayat in node.get("ayat", []):
        parts.append(ayat.get("teks_isi", ""))
    for sub in node.get("sub_items", []):
        parts.append(sub.get("teks_isi", ""))
    return normalize_space(" ".join(parts))


def add_revocation_relations(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")
    for pasal in processed.get("body", {}).get("pasal", []):
        text = collect_node_text(pasal)
        if not re.search(r"\bdicabut\b", text, flags=re.IGNORECASE):
            continue

        refs = [ref for ref in find_regulation_refs(text) if ref != regulasi_id]
        for ref in refs:
            rows.append(triplet(
                regulasi_id,
                "Regulasi",
                "MENCABUT",
                ref,
                "Regulasi",
                text,
                {"source_pasal_id": pasal.get("id", ""), "nomor_pasal": pasal.get("nomor_pasal", "")},
                source_file=source_file,
                confidence=0.9,
            ))

        if re.search(r"\bPasal\s+", text, flags=re.IGNORECASE):
            for pasal_match in PASAL_REF_RE.finditer(text):
                nomor_pasal = pasal_match.group(1)
                nomor_ayat = pasal_match.group(2) or ""
                if refs:
                    for ref in refs:
                        target_id = f"PSL-{ref}-{slug_token(nomor_pasal)}"
                        rows.append(triplet(
                            regulasi_id,
                            "Regulasi",
                            "MENCABUT_PASAL",
                            target_id,
                            "Pasal",
                            text,
                            {
                                "target_regulasi_id": ref,
                                "nomor_pasal": nomor_pasal,
                                "nomor_ayat": nomor_ayat,
                                "source_pasal_id": pasal.get("id", ""),
                            },
                            source_file=source_file,
                            confidence=0.85,
                        ))


def chunk_source_type(section_type: str) -> str:
    mapping = {
        "pasal": "Pasal",
        "ayat": "Ayat",
        "subitem": "SubItem",
        "menimbang": "Konsiderans",
        "mengingat": "Dasar_Hukum",
        "penjelasan_pasal": "Penjelasan_Pasal",
        "lampiran": "Lampiran",
        "penutup": "Penutup",
    }
    return mapping.get(section_type, "Chunk")


def add_reference_relations(processed: Dict, chunks: List[Dict], rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    for chunk in chunks:
        section_type = chunk.get("section_type", "")
        if section_type not in REFERENCE_SECTION_TYPES:
            continue
        text = chunk.get("text", "")
        refs = [ref for ref in find_regulation_refs(text) if ref != regulasi_id]
        pasal_refs = list(PASAL_REF_RE.finditer(text))
        if not refs and not pasal_refs:
            continue

        source_id = chunk.get("source_node_id") or chunk.get("chunk_id")
        source_type = chunk_source_type(section_type)
        source_file = chunk.get("metadata", {}).get("source_file", "")

        for ref in refs:
            rows.append(triplet(
                source_id,
                source_type,
                "MENGACU_PADA",
                ref,
                "Regulasi",
                text,
                {"reference_type": "regulasi"},
                chunk_ids=[chunk.get("chunk_id", "")],
                source_file=source_file,
                confidence=0.9,
            ))

        for match in pasal_refs:
            nomor_pasal = match.group(1)
            nomor_ayat = match.group(2) or ""
            target_regs = refs or [regulasi_id]
            for target_reg in target_regs:
                target_id = f"PSL-{target_reg}-{slug_token(nomor_pasal)}"
                rows.append(triplet(
                    source_id,
                    source_type,
                    "MENGACU_PADA",
                    target_id,
                    "Pasal",
                    text,
                    {
                        "reference_type": "pasal",
                        "target_regulasi_id": target_reg,
                        "nomor_pasal": nomor_pasal,
                        "nomor_ayat": nomor_ayat,
                    },
                    chunk_ids=[chunk.get("chunk_id", "")],
                    source_file=source_file,
                    confidence=0.85,
                ))


def generate_explicit_triplets(regulasi_id: str) -> Dict:
    processed_path = PROCESSED_V2_DIR / f"{regulasi_id}.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed v2 file not found: {processed_path}")

    processed = load_json(processed_path)
    chunks = load_chunks(regulasi_id)
    payload = existing_payload(regulasi_id)
    structural_rows = [row for row in payload.get("triplets", []) if row.get("extraction_method") != "regex"]
    explicit_rows: List[Dict] = []

    add_berdasarkan(processed, explicit_rows)
    add_closing_events(processed, explicit_rows)
    add_amendment_from_title(processed, explicit_rows)
    add_revocation_relations(processed, explicit_rows)
    add_reference_relations(processed, chunks, explicit_rows)

    all_rows = dedupe_triplets(structural_rows + explicit_rows)
    methods = sorted(set(row.get("extraction_method", "") for row in all_rows if row.get("extraction_method")))
    payload["schema_version"] = SCHEMA_VERSION
    payload["regulasi_id"] = regulasi_id
    payload["extraction_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extractor": "extract_triplets_v2_explicit.py",
        "methods": methods,
        "processed_path": str(processed_path),
        "chunks_path": str(CHUNKS_V2_DIR / f"chunks_{regulasi_id}.jsonl"),
        "note": "Explicit triplets are rule-based and do not use an LLM.",
    }
    payload["triplets"] = all_rows
    return payload


def discover_regulasi_ids() -> List[str]:
    return sorted(path.stem for path in PROCESSED_V2_DIR.glob("REG-*.json"))


def process_file(regulasi_id: str) -> Dict:
    payload = generate_explicit_triplets(regulasi_id)
    output_path = TRIPLETS_V2_DIR / f"triplets_{regulasi_id}.json"
    write_json(output_path, payload)
    counts: Dict[str, int] = {}
    for row in payload["triplets"]:
        counts[row.get("predicate", "")] = counts.get(row.get("predicate", ""), 0) + 1
    return {
        "regulasi_id": regulasi_id,
        "output_path": str(output_path),
        "total_triplets": len(payload["triplets"]),
        "predicate_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Regulasi ID, e.g. REG-PP-47-2008")
    parser.add_argument("--all", action="store_true", help="Process all processed_v2 REG-*.json files")
    args = parser.parse_args()

    if not args.file and not args.all:
        parser.error("Use --file REG-ID or --all")

    targets = [args.file] if args.file else discover_regulasi_ids()
    results = []
    for regulasi_id in targets:
        try:
            logger.info("Generating explicit triplets for %s", regulasi_id)
            result = process_file(regulasi_id)
            results.append(result)
            logger.info("%s: %s total triplets after explicit pass", regulasi_id, result["total_triplets"])
        except Exception as exc:
            logger.error("%s failed: %s", regulasi_id, exc)
            results.append({"regulasi_id": regulasi_id, "error": str(exc)})

    write_json(REPORT_PATH, {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "results": results,
    })


if __name__ == "__main__":
    main()
