#!/usr/bin/env python3
"""
Generate LLM-assisted semantic triplets for PSKP KG v2.

This script reads chunks_v2 and appends semantic triplets to triplets_v2 files.
It is intentionally separated from deterministic structural/explicit extractors.

Semantic predicates produced here:
- MENGATUR_TOPIK
- MENGATUR_SUBJEK
- MEMBERIKAN_HAK
- MEMBEBANKAN_KEWAJIBAN
- MELARANG
- MENDEFINISIKAN

Usage:
    # Estimate workload without calling the LLM
    python scripts/extraction/extract_triplets_v2_semantic.py --all --dry-run

    # Process one document
    python scripts/extraction/extract_triplets_v2_semantic.py --file REG-PP-47-2008

    # Process all documents
    python scripts/extraction/extract_triplets_v2_semantic.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

SCHEMA_VERSION = "2.0.0"
PROCESSED_V2_DIR = Path("data/processed_v2")
CHUNKS_V2_DIR = Path("data/chunks_v2")
TRIPLETS_V2_DIR = Path("data/triplets_v2")
REPORT_PATH = Path("outputs/reports/v2/triplets_v2_semantic_report.json")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_SLEEP_SECONDS = float(os.getenv("LLM_RETRY_SLEEP_SECONDS", "2"))
SEMANTIC_MAX_CHARS = int(os.getenv("SEMANTIC_MAX_CHARS", "2500"))

SEMANTIC_SECTION_TYPES = {
    "pasal",
    "ayat",
    "subitem",
}
SEMANTIC_PREDICATES = {
    "MENGATUR_TOPIK",
    "MENGATUR_SUBJEK",
    "MEMBERIKAN_HAK",
    "MEMBEBANKAN_KEWAJIBAN",
    "MELARANG",
    "MENDEFINISIKAN",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)


def normalize_space(value: str) -> str:
    return " ".join(str(value or "").split())


def slug_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip().lower())
    return re.sub(r"-{2,}", "-", value).strip("-")


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


def existing_payload(regulasi_id: str) -> Dict:
    path = TRIPLETS_V2_DIR / f"triplets_{regulasi_id}.json"
    if path.exists():
        return load_json(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "regulasi_id": regulasi_id,
        "extraction_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extractor": "extract_triplets_v2_semantic.py",
            "methods": [],
        },
        "triplets": [],
    }


def target_type_for_predicate(predicate: str) -> str:
    if predicate in {"MENGATUR_SUBJEK", "MEMBERIKAN_HAK", "MEMBEBANKAN_KEWAJIBAN"}:
        return "Subjek_Hukum"
    if predicate == "MELARANG":
        return "Topik"
    return "Topik"


def target_id_for(predicate: str, target_name: str) -> str:
    slug = slug_token(target_name)[:80]
    if target_type_for_predicate(predicate) == "Subjek_Hukum":
        return f"SBJ-{slug}"
    return f"TOPIK-{slug}"


def source_type_from_section(section_type: str) -> str:
    mapping = {
        "pasal": "Pasal",
        "ayat": "Ayat",
        "subitem": "SubItem",
    }
    return mapping.get(section_type, "Chunk")


def triplet_from_semantic_item(chunk: Dict, item: Dict) -> Optional[Dict]:
    predicate = str(item.get("predicate", "")).strip().upper()
    target_name = normalize_space(item.get("target_name", ""))
    if predicate not in SEMANTIC_PREDICATES or not target_name:
        return None

    confidence = item.get("confidence", 0.75)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.75
    confidence = max(0.0, min(1.0, confidence))

    source_id = chunk.get("source_node_id") or chunk.get("chunk_id")
    section_type = chunk.get("section_type", "")
    source_file = chunk.get("metadata", {}).get("source_file", "")
    target_type = target_type_for_predicate(predicate)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "source_type": source_type_from_section(section_type),
        "predicate": predicate,
        "target_id": target_id_for(predicate, target_name),
        "target_type": target_type,
        "confidence": confidence,
        "extraction_method": "llm",
        "evidence": {
            "text": normalize_space(item.get("evidence", "")) or normalize_space(chunk.get("text", "")),
            "chunk_ids": [chunk.get("chunk_id", "")],
            "source_file": source_file,
        },
        "properties": {
            "target_name": target_name,
            "category": normalize_space(item.get("category", "")),
            "reason": normalize_space(item.get("reason", "")),
            "section_type": section_type,
            "reference_label": chunk.get("reference_label", ""),
        },
        "validation": {
            "status": "valid",
            "errors": [],
            "warnings": [],
        },
    }


def build_prompt(chunk: Dict) -> List[Dict[str, str]]:
    text = normalize_space(chunk.get("text", ""))[:SEMANTIC_MAX_CHARS]
    reference_label = chunk.get("reference_label", "")
    system = (
        "Kamu adalah extractor semantic triplets untuk Knowledge Graph hukum Indonesia. "
        "Ekstrak hanya relasi yang eksplisit atau sangat kuat didukung teks. "
        "Jangan membuat asumsi di luar teks. Jawab hanya JSON valid."
    )
    user = f"""
Ekstrak semantic triplets dari potongan regulasi berikut.

Reference: {reference_label}
Text: {text}

Predicate yang boleh dipakai hanya:
- MENGATUR_TOPIK: target_name adalah topik/konsep yang diatur.
- MENGATUR_SUBJEK: target_name adalah subjek hukum/lembaga/pihak yang diatur.
- MEMBERIKAN_HAK: target_name adalah subjek yang diberi hak.
- MEMBEBANKAN_KEWAJIBAN: target_name adalah subjek yang diberi kewajiban.
- MELARANG: target_name adalah tindakan/topik yang dilarang.
- MENDEFINISIKAN: target_name adalah istilah/konsep yang didefinisikan.

Return format JSON:
{{
  "triplets": [
    {{
      "predicate": "MENGATUR_TOPIK",
      "target_name": "...",
      "category": "...",
      "evidence": "kutipan pendek dari teks",
      "reason": "alasan singkat",
      "confidence": 0.0
    }}
  ]
}}

Batasan:
- Maksimal 8 triplets.
- Jika tidak ada relasi semantic yang jelas, return {{"triplets": []}}.
- Jangan ekstrak relasi struktural seperti Pasal/Ayat.
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_response(content: str) -> Dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def call_llm(chunk: Dict) -> Dict:
    messages = build_prompt(chunk)
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return parse_json_response(content)
        except Exception as exc:  # noqa: BLE001 - keep CLI extraction resilient.
            last_error = exc
            logger.warning(
                "LLM extraction failed for %s attempt %s/%s: %s",
                chunk.get("chunk_id"),
                attempt,
                LLM_MAX_RETRIES + 1,
                exc,
            )
            if attempt <= LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"LLM extraction failed for {chunk.get('chunk_id')}: {last_error}")


def should_process_chunk(chunk: Dict) -> bool:
    if chunk.get("section_type") not in SEMANTIC_SECTION_TYPES:
        return False
    text = normalize_space(chunk.get("text", ""))
    if len(text) < 30:
        return False
    if re.fullmatch(r"(?i)cukup\s+jelas\.?", text):
        return False
    return True


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


def generate_semantic_triplets(regulasi_id: str, dry_run: bool = False, limit: Optional[int] = None) -> Dict:
    chunks = [chunk for chunk in load_chunks(regulasi_id) if should_process_chunk(chunk)]
    if limit is not None:
        chunks = chunks[:limit]

    if dry_run:
        return {
            "regulasi_id": regulasi_id,
            "candidate_chunks": len(chunks),
            "dry_run": True,
        }

    payload = existing_payload(regulasi_id)
    non_semantic_rows = [row for row in payload.get("triplets", []) if row.get("extraction_method") != "llm"]
    semantic_rows: List[Dict] = []
    failures = []

    for idx, chunk in enumerate(chunks, start=1):
        logger.info("%s semantic chunk %s/%s: %s", regulasi_id, idx, len(chunks), chunk.get("chunk_id"))
        try:
            result = call_llm(chunk)
            for item in result.get("triplets", []):
                row = triplet_from_semantic_item(chunk, item)
                if row:
                    semantic_rows.append(row)
        except Exception as exc:  # noqa: BLE001 - collect failures in report.
            failures.append({"chunk_id": chunk.get("chunk_id"), "error": str(exc)})

    all_rows = dedupe_triplets(non_semantic_rows + semantic_rows)
    methods = sorted(set(row.get("extraction_method", "") for row in all_rows if row.get("extraction_method")))
    payload["schema_version"] = SCHEMA_VERSION
    payload["regulasi_id"] = regulasi_id
    payload["extraction_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extractor": "extract_triplets_v2_semantic.py",
        "methods": methods,
        "model": LLM_MODEL,
        "candidate_chunks": len(chunks),
        "semantic_triplets_added": len(semantic_rows),
        "failures": failures,
    }
    payload["triplets"] = all_rows
    return payload


def discover_regulasi_ids() -> List[str]:
    return sorted(path.stem.replace("chunks_", "") for path in CHUNKS_V2_DIR.glob("chunks_REG-*.jsonl"))


def process_file(regulasi_id: str, dry_run: bool = False, limit: Optional[int] = None) -> Dict:
    payload = generate_semantic_triplets(regulasi_id, dry_run=dry_run, limit=limit)
    if dry_run:
        return payload

    output_path = TRIPLETS_V2_DIR / f"triplets_{regulasi_id}.json"
    write_json(output_path, payload)
    semantic_count = sum(1 for row in payload["triplets"] if row.get("extraction_method") == "llm")
    return {
        "regulasi_id": regulasi_id,
        "output_path": str(output_path),
        "total_triplets": len(payload["triplets"]),
        "semantic_triplets": semantic_count,
        "failures": payload.get("extraction_metadata", {}).get("failures", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Regulasi ID, e.g. REG-PP-47-2008")
    parser.add_argument("--all", action="store_true", help="Process all chunks_v2 files")
    parser.add_argument("--dry-run", action="store_true", help="Count candidate chunks without calling the LLM")
    parser.add_argument("--limit", type=int, help="Limit chunks per document, useful for pilot runs")
    args = parser.parse_args()

    if not args.file and not args.all:
        parser.error("Use --file REG-ID or --all")

    targets = [args.file] if args.file else discover_regulasi_ids()
    results = []
    for regulasi_id in targets:
        try:
            logger.info("Generating semantic triplets for %s", regulasi_id)
            result = process_file(regulasi_id, dry_run=args.dry_run, limit=args.limit)
            results.append(result)
            if args.dry_run:
                logger.info("%s: %s candidate chunks", regulasi_id, result["candidate_chunks"])
            else:
                logger.info("%s: %s semantic triplets", regulasi_id, result["semantic_triplets"])
        except Exception as exc:
            logger.error("%s failed: %s", regulasi_id, exc)
            results.append({"regulasi_id": regulasi_id, "error": str(exc)})

    write_json(REPORT_PATH, {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": LLM_MODEL,
        "dry_run": args.dry_run,
        "total": len(results),
        "results": results,
    })


if __name__ == "__main__":
    main()
