#!/usr/bin/env python3
"""
Generate deterministic structural triplets for PSKP KG v2.

This script does not call an LLM. It only converts parsed v2 documents and
chunk JSONL files into structural graph edges.

Usage:
    python scripts/extraction/extract_triplets_v2_structural.py --file REG-PP-47-2008
    python scripts/extraction/extract_triplets_v2_structural.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "2.0.0"
PROCESSED_V2_DIR = Path("data/processed_v2")
CHUNKS_V2_DIR = Path("data/chunks_v2")
TRIPLETS_V2_DIR = Path("data/triplets_v2")
REPORT_PATH = Path("outputs/reports/v2/triplets_v2_structural_report.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").split())


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "source_type": source_type,
        "predicate": predicate,
        "target_id": target_id,
        "target_type": target_type,
        "confidence": 1.0,
        "extraction_method": "structural",
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
            json.dumps(row.get("properties", {}), sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def add_preamble_triplets(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")
    preamble = processed.get("preamble", {})

    for item in preamble.get("menimbang", []):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_KONSIDERANS",
            item["id"],
            "Konsiderans",
            item.get("teks_isi", ""),
            {"urutan": item.get("urutan", 0), "nomor": item.get("nomor", "")},
            source_file=source_file,
        ))

    for item in preamble.get("mengingat", []):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_DASAR_HUKUM",
            item["id"],
            "Dasar_Hukum",
            item.get("teks_isi", ""),
            {"urutan": item.get("urutan", 0), "nomor": item.get("nomor", "")},
            source_file=source_file,
        ))

    if preamble.get("memutuskan"):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_DIKTUM",
            f"DIKTUM-{regulasi_id}-MEMUTUSKAN",
            "Diktum",
            preamble.get("memutuskan", ""),
            {"urutan": 1, "jenis": "memutuskan"},
            source_file=source_file,
        ))

    if preamble.get("menetapkan"):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_DIKTUM",
            f"DIKTUM-{regulasi_id}-MENETAPKAN",
            "Diktum",
            preamble.get("menetapkan", ""),
            {"urutan": 2, "jenis": "menetapkan"},
            source_file=source_file,
        ))


def add_body_triplets(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")
    body = processed.get("body", {})

    for bab in body.get("bab", []):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_BAB",
            bab["id"],
            "Bab",
            f"BAB {bab.get('nomor_bab', '')} {bab.get('judul', '')}",
            {"urutan": bab.get("urutan", 0), "nomor_bab": bab.get("nomor_bab", "")},
            source_file=source_file,
        ))

        for bagian in bab.get("bagian", []):
            rows.append(triplet(
                bab["id"],
                "Bab",
                "MEMILIKI_BAGIAN",
                bagian["id"],
                "Bagian",
                f"Bagian {bagian.get('nomor_bagian', '')} {bagian.get('judul', '')}",
                {"urutan": bagian.get("urutan", 0), "nomor_bagian": bagian.get("nomor_bagian", "")},
                source_file=source_file,
            ))

            for paragraf in bagian.get("paragraf", []):
                rows.append(triplet(
                    bagian["id"],
                    "Bagian",
                    "MEMILIKI_PARAGRAF",
                    paragraf["id"],
                    "Paragraf",
                    f"Paragraf {paragraf.get('nomor_paragraf', '')} {paragraf.get('judul', '')}",
                    {"urutan": paragraf.get("urutan", 0), "nomor_paragraf": paragraf.get("nomor_paragraf", "")},
                    source_file=source_file,
                ))

    for pasal in body.get("pasal", []):
        parent_id = pasal.get("parent_paragraf_id") or pasal.get("parent_bagian_id") or pasal.get("parent_bab_id") or regulasi_id
        parent_type = "Regulasi"
        if pasal.get("parent_paragraf_id"):
            parent_type = "Paragraf"
        elif pasal.get("parent_bagian_id"):
            parent_type = "Bagian"
        elif pasal.get("parent_bab_id"):
            parent_type = "Bab"

        rows.append(triplet(
            parent_id,
            parent_type,
            "MEMILIKI_PASAL",
            pasal["id"],
            "Pasal",
            f"Pasal {pasal.get('nomor_pasal', '')}",
            {"urutan": pasal.get("urutan", 0), "nomor_pasal": pasal.get("nomor_pasal", "")},
            source_file=source_file,
        ))

        for ayat in pasal.get("ayat", []):
            rows.append(triplet(
                pasal["id"],
                "Pasal",
                "MEMILIKI_AYAT",
                ayat["id"],
                "Ayat",
                ayat.get("teks_isi", ""),
                {"urutan": ayat.get("urutan", 0), "nomor_ayat": ayat.get("nomor_ayat", "")},
                source_file=source_file,
            ))
            add_subitem_triplets(ayat.get("id", ""), "Ayat", ayat.get("sub_items", []), rows, source_file)

        add_subitem_triplets(pasal.get("id", ""), "Pasal", pasal.get("sub_items", []), rows, source_file)


def add_subitem_triplets(parent_id: str, parent_type: str, sub_items: List[Dict], rows: List[Dict], source_file: str) -> None:
    for sub in sub_items:
        rows.append(triplet(
            parent_id,
            parent_type,
            "MEMILIKI_SUBITEM",
            sub["id"],
            "SubItem",
            sub.get("teks_isi", ""),
            {"urutan": sub.get("urutan", 0), "jenis": sub.get("jenis", ""), "nomor": sub.get("nomor", "")},
            source_file=source_file,
        ))
        add_subitem_triplets(sub["id"], "SubItem", sub.get("children", []), rows, source_file)


def add_explanation_triplets(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    explanation = processed.get("explanation", {})
    if not explanation.get("available"):
        return

    source_file = explanation.get("source_file") or processed["sources"].get("main_pdf", "")
    penjelasan_id = f"PJL-{regulasi_id}"
    rows.append(triplet(
        regulasi_id,
        "Regulasi",
        "MEMILIKI_PENJELASAN",
        penjelasan_id,
        "Penjelasan",
        "PENJELASAN",
        source_file=source_file,
    ))

    if explanation.get("umum"):
        umum_id = f"PJL-{regulasi_id}-UMUM"
        rows.append(triplet(
            penjelasan_id,
            "Penjelasan",
            "MEMILIKI_PENJELASAN_UMUM",
            umum_id,
            "Penjelasan_Umum",
            explanation.get("umum", ""),
            source_file=source_file,
        ))

    for idx, item in enumerate(explanation.get("pasal_demi_pasal", []), start=1):
        rows.append(triplet(
            penjelasan_id,
            "Penjelasan",
            "MEMILIKI_PENJELASAN_PASAL",
            item["id"],
            "Penjelasan_Pasal",
            item.get("teks_isi", ""),
            {"urutan": idx, "nomor_pasal": item.get("nomor_pasal", "")},
            source_file=source_file,
        ))
        if item.get("target_pasal_id"):
            rows.append(triplet(
                item["id"],
                "Penjelasan_Pasal",
                "MENJELASKAN",
                item["target_pasal_id"],
                "Pasal",
                item.get("teks_isi", ""),
                {"nomor_pasal": item.get("nomor_pasal", "")},
                source_file=source_file,
            ))


def add_attachment_and_closing_triplets(processed: Dict, rows: List[Dict]) -> None:
    regulasi_id = processed["regulasi_id"]
    source_file = processed["sources"].get("main_pdf", "")

    for idx, item in enumerate(processed.get("attachments", []), start=1):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_LAMPIRAN",
            item["id"],
            "Lampiran",
            item.get("teks_isi", ""),
            {"urutan": idx, "nomor_lampiran": item.get("nomor_lampiran", "")},
            source_file=item.get("source_file", source_file),
        ))

    closing = processed.get("closing", {})
    if closing.get("teks_isi") or closing.get("tanggal_diundangkan") or closing.get("tanggal_ditetapkan"):
        rows.append(triplet(
            regulasi_id,
            "Regulasi",
            "MEMILIKI_PENUTUP",
            f"PENUTUP-{regulasi_id}",
            "Penutup",
            closing.get("teks_isi", ""),
            {
                "tanggal_ditetapkan": closing.get("tanggal_ditetapkan", ""),
                "tanggal_diundangkan": closing.get("tanggal_diundangkan", ""),
            },
            source_file=source_file,
        ))


def node_type_from_chunk_section(section_type: str) -> str:
    mapping = {
        "menimbang": "Konsiderans",
        "mengingat": "Dasar_Hukum",
        "diktum": "Diktum",
        "bab": "Bab",
        "bagian": "Bagian",
        "paragraf": "Paragraf",
        "pasal": "Pasal",
        "ayat": "Ayat",
        "subitem": "SubItem",
        "penjelasan_umum": "Penjelasan_Umum",
        "penjelasan_pasal": "Penjelasan_Pasal",
        "lampiran": "Lampiran",
        "penutup": "Penutup",
    }
    return mapping.get(section_type, "Regulasi")


def add_chunk_triplets(processed: Dict, chunks: List[Dict], rows: List[Dict]) -> None:
    for idx, chunk in enumerate(chunks, start=1):
        source_node_id = chunk.get("source_node_id") or processed["regulasi_id"]
        section_type = chunk.get("section_type", "")
        rows.append(triplet(
            source_node_id,
            node_type_from_chunk_section(section_type),
            "MEMILIKI_CHUNK",
            chunk["chunk_id"],
            "Chunk",
            chunk.get("text", ""),
            {"urutan": idx, "section_type": section_type, "reference_label": chunk.get("reference_label", "")},
            chunk_ids=[chunk["chunk_id"]],
            source_file=chunk.get("metadata", {}).get("source_file", ""),
        ))


def generate_structural_triplets(regulasi_id: str) -> Dict:
    processed_path = PROCESSED_V2_DIR / f"{regulasi_id}.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed v2 file not found: {processed_path}")

    processed = load_json(processed_path)
    chunks = load_chunks(regulasi_id)
    rows: List[Dict] = []

    add_preamble_triplets(processed, rows)
    add_body_triplets(processed, rows)
    add_explanation_triplets(processed, rows)
    add_attachment_and_closing_triplets(processed, rows)
    add_chunk_triplets(processed, chunks, rows)

    rows = dedupe_triplets(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "regulasi_id": regulasi_id,
        "extraction_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extractor": "extract_triplets_v2_structural.py",
            "methods": ["structural"],
            "processed_path": str(processed_path),
            "chunks_path": str(CHUNKS_V2_DIR / f"chunks_{regulasi_id}.jsonl"),
        },
        "triplets": rows,
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def discover_regulasi_ids() -> List[str]:
    return sorted(path.stem for path in PROCESSED_V2_DIR.glob("REG-*.json"))


def process_file(regulasi_id: str) -> Dict:
    payload = generate_structural_triplets(regulasi_id)
    output_path = TRIPLETS_V2_DIR / f"triplets_{regulasi_id}.json"
    write_json(output_path, payload)
    return {
        "regulasi_id": regulasi_id,
        "output_path": str(output_path),
        "total_triplets": len(payload["triplets"]),
        "methods": ["structural"],
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
            logger.info("Generating structural triplets for %s", regulasi_id)
            result = process_file(regulasi_id)
            results.append(result)
            logger.info("%s: %s structural triplets", regulasi_id, result["total_triplets"])
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
