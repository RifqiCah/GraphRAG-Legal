#!/usr/bin/env python3
"""
PSKP Legal KG v2 — document parser and chunk generator.

This script is intentionally separate from extract.py (v1). It produces:
- data/processed_v2/{regulasi_id}.json
- data/chunks_v2/chunks_{regulasi_id}.jsonl

Usage:
    python scripts/extraction/extract_v2.py --file REG-PP-47-2008
    python scripts/extraction/extract_v2.py --all
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

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extraction.rebuild_corpus_manifest import TITLE_MAP, parse_reg_id


RAW_DIR = Path("data/raw")
PROCESSED_V2_DIR = Path("data/processed_v2")
CHUNKS_V2_DIR = Path("data/chunks_v2")
SCHEMA_VERSION = "2.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


PASAL_RE = re.compile(r"(?im)^Pasal[ \t]+([0-9]+[A-Z]?|[IVXLC]+)[ \t]*$")
PASAL_LINE_RE = re.compile(
    r"(?i)^\s*(?:-\s*\d+\s*)?Pasal\s*,?\s*([0-9OIlT ]+[A-Z]?|[IVXLC]+)?\s*(?:\.\s*)*$"
)
PASAL_NUMBER_LINE_RE = re.compile(r"(?i)^\s*([0-9OIlT ]+[A-Z]?|[IVXLC]+)\s*$")
AYAT_RE = re.compile(r"(?m)^\((\d+[a-zA-Z]?)\)\s+")
BAB_RE = re.compile(r"(?im)^BAB[ \t]+([IVXLC]+|[0-9]+)[ \t]*(?:\n([^\n]+))?")
BAB_LINE_RE = re.compile(r"(?i)^\s*BAB\s+([IVXLC]+|[0-9]+)\s*$")
BAGIAN_LINE_RE = re.compile(r"(?i)^\s*Bagian\s+(.+?)\s*$")
PARAGRAF_LINE_RE = re.compile(r"(?i)^\s*Paragraf\s+(.+?)\s*$")
LAMPIRAN_LINE_RE = re.compile(r"(?i)^\s*LAMPIRAN(?:\s+([IVXLC]+|[0-9]+))?.*$")
NUMBERED_ITEM_RE = re.compile(
    r"(?ms)^\s*([a-z]|\d+)\.\s+(.+?)(?=^\s*(?:[a-z]|\d+)\.\s+|\Z)"
)
OCR_DIGIT_TRANS = str.maketrans({
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "T": "7",
})


def slug_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip().upper())
    return re.sub(r"-{2,}", "-", value).strip("-")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_ocr_number(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or "").strip())
    if not compact:
        return ""
    if re.fullmatch(r"[IVXLC]+", compact):
        return compact
    return compact.translate(OCR_DIGIT_TRANS)


def normalize_ocr_years(text: str) -> str:
    def replace(match: re.Match) -> str:
        return match.group(0).translate(OCR_DIGIT_TRANS)

    return re.sub(r"\b[12][0-9OIl]{3}\b", replace, str(text or ""))


def extract_pdf_text(pdf_path: Path) -> Tuple[str, bool, int]:
    text_parts: List[str] = []
    scan_detected = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < 100:
                scan_detected = True
            text_parts.append(page_text)
        page_count = len(pdf.pages)

    return "\n".join(text_parts), scan_detected, page_count


def clean_text(text: str) -> str:
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?i)M\s+E\s+M\s+U\s+T\s+U\s+S\s+K\s+A\s+N", "MEMUTUSKAN", text)
    text = re.sub(r"(?i)M\s+E\s+N\s+E\s+T\s+A\s+P\s+K\s+A\s+N", "Menetapkan", text)
    text = normalize_ocr_years(text)
    text = re.sub(r"(?m)^\s*-?\s*\d+\s*-?\s*$", "", text)
    text = re.sub(r"(?m)^\s*(PRESIDEN|REPUBLIK INDONESIA)\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_pasal_headings(text: str) -> str:
    lines = str(text or "").splitlines()
    normalized: List[str] = []
    seen_heading = False
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        match = PASAL_LINE_RE.match(stripped)

        if match:
            if re.search(r"\.\s*\.", stripped):
                idx += 1
                continue
            nomor = normalize_ocr_number(match.group(1) or "")
            if not nomor:
                lookahead_idx = idx + 1
                while lookahead_idx < len(lines) and not lines[lookahead_idx].strip():
                    lookahead_idx += 1
                next_line = lines[lookahead_idx].strip() if lookahead_idx < len(lines) else ""
                next_number = PASAL_NUMBER_LINE_RE.match(next_line)
                if next_number:
                    nomor = normalize_ocr_number(next_number.group(1))
                    idx = lookahead_idx
                elif not seen_heading:
                    nomor = "1"

            if nomor:
                normalized.append(f"Pasal {nomor}")
                seen_heading = True
                idx += 1
                continue

        normalized.append(line)
        idx += 1

    return "\n".join(normalized)


def section_between(text: str, start_pattern: str, end_patterns: Iterable[str]) -> str:
    start = re.search(start_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not start:
        return ""

    start_idx = start.end()
    end_idx = len(text)
    for end_pattern in end_patterns:
        end = re.search(end_pattern, text[start_idx:], flags=re.IGNORECASE | re.MULTILINE)
        if end:
            end_idx = min(end_idx, start_idx + end.start())

    return text[start_idx:end_idx].strip()


def body_start_index(text: str) -> int:
    decision = re.search(r"(?im)^\s*MEMUTUSKAN\s*:?", text)
    search_start = decision.end() if decision else 0

    bab_i = re.search(r"(?im)^BAB[ \t]+I[ \t]*$", text[search_start:])
    if bab_i:
        return search_start + bab_i.start()

    pasal = PASAL_RE.search(text[search_start:])
    if pasal:
        return search_start + pasal.start()

    return -1


def body_text(text: str) -> str:
    start = body_start_index(text)
    return text[start:].strip() if start >= 0 else ""


def text_before_body(text: str) -> str:
    start = body_start_index(text)
    return text[:start].strip() if start >= 0 else text


def text_from_first_pasal(text: str) -> str:
    return body_text(text)


def strip_after_explanation(text: str) -> str:
    match = re.search(r"(?im)^PENJELASAN\s*$", text)
    return text[: match.start()].strip() if match else text.strip()


def strip_before_attachment(text: str) -> str:
    match = re.search(r"(?m)^\s*LAMPIRAN(?:\s+([IVXLC]+|[0-9]+))?\b.*$", text)
    return text[: match.start()].strip() if match else text.strip()


def parse_numbered_items(section_text: str, node_prefix: str, regulasi_id: str) -> List[Dict]:
    items: List[Dict] = []
    section_text = section_text.strip()
    if not section_text:
        return items

    matches = list(NUMBERED_ITEM_RE.finditer(section_text))
    if not matches:
        return [{
            "id": f"{node_prefix}-{regulasi_id}-1",
            "nomor": "1",
            "teks_isi": normalize_space(section_text),
            "urutan": 1
        }]

    for idx, match in enumerate(matches, start=1):
        nomor = match.group(1)
        content = normalize_space(match.group(2))
        items.append({
            "id": f"{node_prefix}-{regulasi_id}-{slug_token(nomor)}",
            "nomor": nomor,
            "teks_isi": content,
            "urutan": idx
        })

    return items


def build_regulasi_id(kind: str, number: str, year: Optional[str]) -> str:
    kind = slug_token(kind)
    number = slug_token(number)
    year = slug_token(year or "")
    return f"REG-{kind}-{number}-{year}" if year else f"REG-{kind}-{number}"


def resolve_regulation_reference(text: str) -> str:
    text = normalize_ocr_years(text)
    if re.search(r"\b(UUD|Undang-Undang\s+Dasar)\b.*1945", text, flags=re.IGNORECASE):
        return "REG-UUD-1945"

    match = re.search(
        r"\b(Undang-Undang(?:\s+Republik\s+Indonesia)?|UU|Peraturan Pemerintah|PP)\s*(?:Nomor|No\.?)?\s*([A-Za-z0-9/.-]+)\s*(?:Tahun\s*(\d{4}))?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    kind_raw = match.group(1).lower()
    kind = "UU" if "undang" in kind_raw or kind_raw == "uu" else "PP"
    return build_regulasi_id(kind, match.group(2), match.group(3))


def parse_preamble(text: str, regulasi_id: str) -> Dict:
    preamble_text = text_before_body(text)

    menimbang_text = section_between(
        preamble_text,
        r"(?im)^\s*Menimbang\s*:?",
        [r"(?im)^\s*Mengingat\s*:?", r"(?im)^\s*MEMUTUSKAN\s*:?", r"(?im)^\s*Menetapkan\s*:?"],
    )
    mengingat_text = section_between(
        preamble_text,
        r"(?im)^\s*Mengingat\s*:?",
        [r"(?im)^\s*MEMUTUSKAN\s*:?", r"(?im)^\s*Menetapkan\s*:?"],
    )
    menetapkan_text = section_between(
        preamble_text,
        r"(?im)^\s*Menetapkan\s*:?",
        [r"(?im)^\s*Pasal\s+"],
    )

    menimbang = parse_numbered_items(menimbang_text, "KONS", regulasi_id)
    for item in menimbang:
        item["id"] = f"KONS-{regulasi_id}-MENIMBANG-{slug_token(item['nomor'])}"

    mengingat = parse_numbered_items(mengingat_text, "DASAR", regulasi_id)
    for item in mengingat:
        item["id"] = f"DASAR-{regulasi_id}-MENGINGAT-{slug_token(item['nomor'])}"
        item["target_regulasi_id"] = resolve_regulation_reference(item["teks_isi"])
        item["target_pasal_id"] = ""

    opening_match = re.search(
        r"(?im)DENGAN\s+RAHMAT\s+TUHAN\s+YANG\s+MAHA\s+ESA",
        preamble_text,
    )
    opening_phrase = opening_match.group(0).upper() if opening_match else ""

    return {
        "opening_phrase": opening_phrase,
        "menimbang": menimbang,
        "mengingat": mengingat,
        "memutuskan": "MEMUTUSKAN:" if re.search(r"(?im)^\s*MEMUTUSKAN\s*:?", preamble_text) else "",
        "menetapkan": normalize_space(menetapkan_text),
    }


def is_structural_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return bool(
        BAB_LINE_RE.match(stripped)
        or BAGIAN_LINE_RE.match(stripped)
        or PARAGRAF_LINE_RE.match(stripped)
        or PASAL_RE.match(stripped)
        or LAMPIRAN_LINE_RE.match(stripped)
        or re.match(r"(?im)^PENJELASAN$", stripped)
    )


def next_title_line(lines: List[str], start_idx: int) -> Tuple[str, int]:
    idx = start_idx + 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return "", start_idx
    title = lines[idx].strip()
    if is_structural_heading(title) or ". . ." in title:
        return "", start_idx
    return normalize_space(title), idx


def parse_body_structure(text: str, regulasi_id: str, pasal_items: List[Dict]) -> List[Dict]:
    body_text_value = normalize_pasal_headings(strip_before_attachment(strip_after_explanation(body_text(text))))
    lines = body_text_value.splitlines()
    pasal_by_nomor = {str(item["nomor_pasal"]): item for item in pasal_items}
    assigned_pasal_ids = set()
    bab_items: List[Dict] = []
    bab_by_id: Dict[str, Dict] = {}
    current_bab: Optional[Dict] = None
    current_bagian: Optional[Dict] = None
    current_paragraf: Optional[Dict] = None
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()

        bab_match = BAB_LINE_RE.match(stripped)
        if bab_match:
            nomor = normalize_ocr_number(bab_match.group(1))
            title, title_idx = next_title_line(lines, idx)
            node_id = f"BAB-{regulasi_id}-{slug_token(nomor)}"
            if node_id in bab_by_id:
                current_bab = bab_by_id[node_id]
                if title and not current_bab["judul"]:
                    current_bab["judul"] = title
            else:
                current_bab = {
                    "id": node_id,
                    "nomor_bab": nomor,
                    "judul": title,
                    "urutan": len(bab_items) + 1,
                    "bagian": [],
                    "pasal_ids": []
                }
                bab_by_id[node_id] = current_bab
                bab_items.append(current_bab)
            current_bagian = None
            current_paragraf = None
            idx = title_idx + 1 if title else idx + 1
            continue

        bagian_match = BAGIAN_LINE_RE.match(stripped)
        if bagian_match and current_bab:
            nomor = normalize_space(bagian_match.group(1))
            title, title_idx = next_title_line(lines, idx)
            bagian_id = f"BAGIAN-{regulasi_id}-{slug_token(current_bab['nomor_bab'])}-{slug_token(nomor)}"
            existing = next((item for item in current_bab["bagian"] if item["id"] == bagian_id), None)
            if existing:
                current_bagian = existing
                if title and not current_bagian["judul"]:
                    current_bagian["judul"] = title
            else:
                current_bagian = {
                    "id": bagian_id,
                    "nomor_bagian": nomor,
                    "judul": title,
                    "urutan": len(current_bab["bagian"]) + 1,
                    "paragraf": [],
                    "pasal_ids": []
                }
                current_bab["bagian"].append(current_bagian)
            current_paragraf = None
            idx = title_idx + 1 if title else idx + 1
            continue

        paragraf_match = PARAGRAF_LINE_RE.match(stripped)
        if paragraf_match:
            nomor = normalize_space(paragraf_match.group(1))
            title, title_idx = next_title_line(lines, idx)
            if current_bab and not current_bagian:
                current_bagian = {
                    "id": f"BAGIAN-{regulasi_id}-{slug_token(current_bab['nomor_bab'])}-UNSPECIFIED",
                    "nomor_bagian": "",
                    "judul": "",
                    "urutan": len(current_bab["bagian"]) + 1,
                    "paragraf": [],
                    "pasal_ids": []
                }
                current_bab["bagian"].append(current_bagian)
            if current_bagian:
                paragraf_id = (
                    f"PARAGRAF-{regulasi_id}-{slug_token(current_bab['nomor_bab'])}-"
                    f"{slug_token(current_bagian['nomor_bagian'] or 'UNSPECIFIED')}-{slug_token(nomor)}"
                )
                existing = next((item for item in current_bagian["paragraf"] if item["id"] == paragraf_id), None)
                if existing:
                    current_paragraf = existing
                    if title and not current_paragraf["judul"]:
                        current_paragraf["judul"] = title
                else:
                    current_paragraf = {
                        "id": paragraf_id,
                        "nomor_paragraf": nomor,
                        "judul": title,
                        "urutan": len(current_bagian["paragraf"]) + 1,
                        "pasal_ids": []
                    }
                    current_bagian["paragraf"].append(current_paragraf)
            idx = title_idx + 1 if title else idx + 1
            continue

        pasal_match = PASAL_RE.match(stripped)
        if pasal_match:
            nomor = normalize_ocr_number(pasal_match.group(1))
            pasal_item = pasal_by_nomor.get(nomor)
            if pasal_item and pasal_item["id"] not in assigned_pasal_ids:
                pasal_id = pasal_item["id"]
                assigned_pasal_ids.add(pasal_id)
                if current_bab and pasal_id not in current_bab["pasal_ids"]:
                    current_bab["pasal_ids"].append(pasal_id)
                    pasal_item["parent_bab_id"] = current_bab["id"]
                if current_bagian and pasal_id not in current_bagian["pasal_ids"]:
                    current_bagian["pasal_ids"].append(pasal_id)
                    pasal_item["parent_bagian_id"] = current_bagian["id"]
                if current_paragraf and pasal_id not in current_paragraf["pasal_ids"]:
                    current_paragraf["pasal_ids"].append(pasal_id)
                    pasal_item["parent_paragraf_id"] = current_paragraf["id"]
            idx += 1
            continue

        idx += 1

    return bab_items


def trim_structural_noise(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.match(r"(?i)^BAB\s+([IVXLC]+|[0-9]+)$", stripped):
            continue
        if re.match(r"(?i)^(BAGIAN|PARAGRAF)\s+", stripped):
            continue
        if stripped.isupper() and len(stripped) <= 80 and not re.search(r"\d", stripped):
            continue
        if stripped in {"Agar . . .", "BAB III . . ."}:
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_page_artifact(text: str) -> bool:
    normalized = normalize_space(text)
    if not normalized:
        return False
    return bool(re.fullmatch(r"(?i)SK\s+No\s+[A-Za-z0-9 ]+", normalized))


def parse_sub_items(text: str, regulasi_id: str, parent_id: str, prefix: str) -> List[Dict]:
    items: List[Dict] = []
    pattern = re.compile(r"(?ms)^\s*([a-z]|\d+)\.\s+(.+?)(?=^\s*(?:[a-z]|\d+)\.\s+|\Z)")

    for idx, match in enumerate(pattern.finditer(text), start=1):
        nomor = match.group(1)
        content = normalize_space(match.group(2))
        jenis = "huruf" if nomor.isalpha() else "angka"
        items.append({
            "id": f"SUB-{regulasi_id}-{prefix}-{jenis}-{slug_token(nomor)}",
            "jenis": jenis,
            "nomor": nomor,
            "teks_isi": content,
            "urutan": idx,
            "children": []
        })

    return items


def parse_pasal(text: str, regulasi_id: str) -> List[Dict]:
    body_text_value = normalize_pasal_headings(strip_before_attachment(strip_after_explanation(body_text(text))))
    matches = list(PASAL_RE.finditer(body_text_value))
    pasal_items: List[Dict] = []

    for idx, match in enumerate(matches, start=1):
        nomor_pasal = match.group(1)
        content_start = match.end()
        content_end = matches[idx].start() if idx < len(matches) else len(body_text_value)
        raw_content = trim_structural_noise(body_text_value[content_start:content_end])
        if not normalize_space(raw_content) or is_page_artifact(raw_content):
            continue
        pasal_id = f"PSL-{regulasi_id}-{slug_token(nomor_pasal)}"

        ayat_matches = list(AYAT_RE.finditer(raw_content))
        ayat_items = []
        pasal_sub_items = []
        pasal_text = ""

        if ayat_matches:
            pasal_intro = raw_content[:ayat_matches[0].start()].strip()
            pasal_text = normalize_space(pasal_intro)
            for ayat_idx, ayat_match in enumerate(ayat_matches, start=1):
                nomor_ayat = ayat_match.group(1)
                ayat_start = ayat_match.end()
                ayat_end = ayat_matches[ayat_idx].start() if ayat_idx < len(ayat_matches) else len(raw_content)
                ayat_text = raw_content[ayat_start:ayat_end].strip()
                ayat_id = f"AYT-{regulasi_id}-{slug_token(nomor_pasal)}-{slug_token(nomor_ayat)}"
                sub_items = parse_sub_items(
                    ayat_text,
                    regulasi_id,
                    ayat_id,
                    f"{slug_token(nomor_pasal)}-{slug_token(nomor_ayat)}",
                )
                ayat_items.append({
                    "id": ayat_id,
                    "nomor_ayat": nomor_ayat,
                    "teks_isi": normalize_space(ayat_text),
                    "status": "Aktif",
                    "urutan": ayat_idx,
                    "sub_items": sub_items
                })
        else:
            pasal_text = normalize_space(raw_content)
            pasal_sub_items = parse_sub_items(
                raw_content,
                regulasi_id,
                pasal_id,
                slug_token(nomor_pasal),
            )

        pasal_items.append({
            "id": pasal_id,
            "nomor_pasal": nomor_pasal,
            "teks_isi": pasal_text,
            "status": "Aktif",
            "urutan": len(pasal_items) + 1,
            "ayat": ayat_items,
            "sub_items": pasal_sub_items
        })

    return pasal_items


def parse_explanation_text(text: str, regulasi_id: str, source_file: str) -> Dict:
    cleaned = normalize_pasal_headings(clean_text(text))
    available = bool(cleaned)
    umum = ""
    pasal_demi_pasal = []

    if available:
        umum_match = re.search(r"(?is)\bI\.?\s*UMUM\s*(.+?)(?=\bII\.?\s*PASAL\s+DEMI\s+PASAL\b|\Z)", cleaned)
        if umum_match:
            umum = normalize_space(umum_match.group(1))

        pasal_section = section_between(
            cleaned,
            r"(?im)^\s*II\.?\s*PASAL\s+DEMI\s+PASAL\s*$",
            [],
        )
        pasal_matches = list(PASAL_RE.finditer(pasal_section))
        for idx, match in enumerate(pasal_matches, start=1):
            nomor = match.group(1)
            start = match.end()
            end = pasal_matches[idx].start() if idx < len(pasal_matches) else len(pasal_section)
            content = normalize_space(pasal_section[start:end])
            if not content or is_page_artifact(content):
                continue
            pasal_demi_pasal.append({
                "id": f"PJL-{regulasi_id}-PASAL-{slug_token(nomor)}",
                "nomor_pasal": nomor,
                "target_pasal_id": f"PSL-{regulasi_id}-{slug_token(nomor)}",
                "teks_isi": content
            })

    return {
        "available": available,
        "source_file": source_file,
        "umum": umum,
        "pasal_demi_pasal": pasal_demi_pasal
    }


def find_missing_numeric_pasal(pasal_items: List[Dict]) -> List[int]:
    numbers = []
    for item in pasal_items:
        nomor = str(item.get("nomor_pasal", ""))
        if re.fullmatch(r"\d+", nomor):
            numbers.append(int(nomor))

    if len(numbers) < 3:
        return []

    unique_numbers = sorted(set(numbers))
    return [
        number
        for number in range(unique_numbers[0], unique_numbers[-1] + 1)
        if number not in unique_numbers
    ]


def find_explanation_pdf(regulasi_id: str) -> Optional[Path]:
    candidate = RAW_DIR / f"PJL-{regulasi_id}.pdf"
    return candidate if candidate.exists() else None


def parse_attachments(text: str, regulasi_id: str, source_file: str) -> List[Dict]:
    attachment_matches = list(re.finditer(r"(?m)^\s*LAMPIRAN(?:\s+([IVXLC]+|[0-9]+))?.*$", text))
    attachments = []

    for idx, match in enumerate(attachment_matches, start=1):
        header = normalize_space(match.group(0))
        nomor = match.group(1) or str(idx)
        start = match.end()
        next_start = attachment_matches[idx].start() if idx < len(attachment_matches) else len(text)
        explanation = re.search(r"(?im)^\s*PENJELASAN\s*$", text[start:next_start])
        end = start + explanation.start() if explanation else next_start
        content = normalize_space(text[start:end])
        if not content:
            continue

        title = header
        first_content_line = next((line.strip() for line in text[start:end].splitlines() if line.strip()), "")
        if first_content_line and not is_structural_heading(first_content_line):
            title = normalize_space(f"{header} {first_content_line}")

        attachments.append({
            "id": f"LAMP-{regulasi_id}-{slug_token(nomor)}",
            "nomor_lampiran": str(nomor),
            "judul": title,
            "teks_isi": content,
            "source_file": source_file
        })

    return attachments


def cleaned_official_lines(section: str) -> List[str]:
    lines = []
    for line in str(section or "").splitlines():
        stripped = normalize_space(line).strip(" ,;")
        if not stripped:
            continue
        if re.match(r"(?i)^(ttd\.?|ditetapkan|diundangkan|pada tanggal|lembaran negara|tambahan lembaran|salinan|sk no)", stripped):
            continue
        if re.match(r"(?i)^agar\b", stripped):
            continue
        lines.append(stripped)
    return lines


def extract_official(section: str) -> str:
    lines = cleaned_official_lines(section)
    if not lines:
        return ""

    office_lines = []
    name = ""
    for line in lines:
        if line.isupper() and not re.search(r"\d", line):
            office_lines.append(line)
            continue
        name = line
        break

    if office_lines and name:
        return normalize_space(" ".join(office_lines + [name]))
    if office_lines:
        return normalize_space(" ".join(office_lines))
    return name


def append_following_year_if_needed(text: str, date_text: str, end_pos: int) -> str:
    date_text = normalize_space(date_text)
    if re.search(r"\b\d{4}\b", date_text):
        return date_text
    following_lines = text[end_pos:].splitlines()
    for line in following_lines[:3]:
        year_match = re.search(r"\b(19|20)\d{2}\b", line)
        if year_match:
            return normalize_space(f"{date_text} {year_match.group(0)}")
    return date_text


def parse_closing(text: str) -> Dict:
    main_text = strip_after_explanation(text)
    first_closing = re.search(r"Agar\s+setiap\s+orang", main_text)
    if not first_closing:
        first_closing = re.search(r"(?m)^\s*Agar\s+", main_text)
    closing_text = main_text[first_closing.start():].strip() if first_closing else ""
    tanggal_ditetapkan = ""
    tanggal_diundangkan = ""
    pejabat_penetapan = ""
    pejabat_pengundangan = ""

    ditetapkan = re.search(r"(?is)Ditetapkan\s+di\s+.+?\s+pada\s+tanggal\s+([^\n]+)", main_text)
    if ditetapkan:
        tanggal_ditetapkan = append_following_year_if_needed(main_text, ditetapkan.group(1), ditetapkan.end())

    diundangkan = re.search(r"(?is)Diundangkan\s+di\s+.+?\s+pada\s+tanggal\s+([^\n]+)", main_text)
    if diundangkan:
        tanggal_diundangkan = append_following_year_if_needed(main_text, diundangkan.group(1), diundangkan.end())
    if tanggal_ditetapkan and tanggal_diundangkan and not re.search(r"\b\d{4}\b", tanggal_ditetapkan):
        year_match = re.search(r"\b\d{4}\b", tanggal_diundangkan)
        if year_match:
            tanggal_ditetapkan = normalize_space(f"{tanggal_ditetapkan} {year_match.group(0)}")

    if ditetapkan:
        section_end = diundangkan.start() if diundangkan else len(main_text)
        pejabat_penetapan = extract_official(main_text[ditetapkan.end():section_end])
    if diundangkan:
        pengundangan_end = re.search(
            r"(?im)^\s*(LEMBARAN\s+NEGARA|TAMBAHAN\s+LEMBARAN|Salinan\s+sesuai|PENJELASAN)\b",
            main_text[diundangkan.end():],
        )
        end = diundangkan.end() + pengundangan_end.start() if pengundangan_end else len(main_text)
        pejabat_pengundangan = extract_official(main_text[diundangkan.end():end])

    return {
        "teks_isi": normalize_space(closing_text),
        "tanggal_ditetapkan": tanggal_ditetapkan,
        "tanggal_diundangkan": tanggal_diundangkan,
        "pejabat_penetapan": pejabat_penetapan,
        "pejabat_pengundangan": pejabat_pengundangan
    }


def build_metadata(regulasi_id: str, scan_detected: bool) -> Dict:
    jenis, nomor, tahun = parse_reg_id(regulasi_id)
    return {
        "jenis": jenis.upper() if jenis else "Unknown",
        "nomor": str(nomor),
        "tahun": int(tahun or 0),
        "judul": TITLE_MAP.get(regulasi_id, regulasi_id),
        "tentang": TITLE_MAP.get(regulasi_id, regulasi_id),
        "status": "Tidak_Diketahui",
        "tanggal_penetapan": "",
        "tanggal_pengundangan": "",
        "tanggal_berlaku": "",
        "sumber": "local_pdf",
        "url_sumber": "",
        "has_full_text": True,
        "scan_detected": scan_detected
    }


def build_processed_document(regulasi_id: str, raw_pdf: Path) -> Dict:
    raw_text, scan_detected, _ = extract_pdf_text(raw_pdf)
    cleaned = clean_text(raw_text)
    explanation_pdf = find_explanation_pdf(regulasi_id)
    explanation = {
        "available": False,
        "source_file": "",
        "umum": "",
        "pasal_demi_pasal": []
    }

    if explanation_pdf:
        explanation_text, _, _ = extract_pdf_text(explanation_pdf)
        explanation = parse_explanation_text(
            explanation_text,
            regulasi_id,
            str(explanation_pdf),
        )
    else:
        inline_explanation = section_between(cleaned, r"(?im)^\s*PENJELASAN\s*$", [])
        if inline_explanation:
            explanation = parse_explanation_text(inline_explanation, regulasi_id, "")

    preamble = parse_preamble(cleaned, regulasi_id)
    pasal = parse_pasal(cleaned, regulasi_id)
    bab = parse_body_structure(cleaned, regulasi_id, pasal)
    attachments = parse_attachments(cleaned, regulasi_id, str(raw_pdf))
    closing = parse_closing(cleaned)

    warnings = []
    if "Pasal" in cleaned and not pasal:
        warnings.append("Text contains Pasal but no pasal nodes were parsed.")
    if re.search(r"(?i)\bMenimbang\b|\bMengingat\b", cleaned) and not (preamble["menimbang"] or preamble["mengingat"]):
        warnings.append("Text contains preamble labels but no preamble items were parsed.")
    if re.search(r"(?im)^\s*PENJELASAN\s*$", cleaned) and not explanation["available"]:
        warnings.append("Text contains PENJELASAN heading but explanation was not parsed.")
    missing_pasal = find_missing_numeric_pasal(pasal)
    if missing_pasal:
        missing_preview = ", ".join(str(number) for number in missing_pasal[:20])
        if len(missing_pasal) > 20:
            missing_preview += ", ..."
        warnings.append(f"Numeric pasal sequence has gaps: {missing_preview}.")

    total_ayat = sum(len(item["ayat"]) for item in pasal)
    quality_status = "PASS" if not warnings and pasal else "WARN"

    return {
        "schema_version": SCHEMA_VERSION,
        "regulasi_id": regulasi_id,
        "metadata": build_metadata(regulasi_id, scan_detected),
        "sources": {
            "main_pdf": str(raw_pdf),
            "explanation_pdf": str(explanation_pdf) if explanation_pdf else "",
            "attachment_files": []
        },
        "preamble": preamble,
        "body": {
            "bab": bab,
            "pasal": pasal
        },
        "explanation": explanation,
        "attachments": attachments,
        "closing": closing,
        "parse_quality": {
            "status": quality_status,
            "scan_detected": scan_detected,
            "total_pasal": len(pasal),
            "total_ayat": total_ayat,
            "total_bab": len(bab),
            "total_bagian": sum(len(item.get("bagian", [])) for item in bab),
            "total_paragraf": sum(
                len(bagian.get("paragraf", []))
                for item in bab
                for bagian in item.get("bagian", [])
            ),
            "total_attachments": len(attachments),
            "has_preamble": bool(preamble["menimbang"] or preamble["mengingat"]),
            "has_explanation": explanation["available"],
            "warnings": warnings
        }
    }


def chunk_id(regulasi_id: str, index: int) -> str:
    return f"CHK-{regulasi_id}-{index:04d}"


def base_chunk(regulasi_id: str, metadata: Dict, source_node_id: str, section_type: str, reference_label: str, text: str, index: int, source_file: str = "") -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id(regulasi_id, index),
        "regulasi_id": regulasi_id,
        "source_node_id": source_node_id,
        "section_type": section_type,
        "reference_label": reference_label,
        "text": normalize_space(text),
        "metadata": {
            "jenis": metadata.get("jenis", ""),
            "nomor": str(metadata.get("nomor", "")),
            "tahun": metadata.get("tahun", 0),
            "judul": metadata.get("judul", ""),
            "source_file": source_file
        }
    }


def generate_chunks(processed: Dict) -> List[Dict]:
    chunks: List[Dict] = []
    regulasi_id = processed["regulasi_id"]
    metadata = processed["metadata"]
    main_pdf = processed["sources"]["main_pdf"]

    def add(source_node_id: str, section_type: str, reference_label: str, text: str, source_file: str = "") -> None:
        text_norm = normalize_space(text)
        if not text_norm:
            return
        chunks.append(base_chunk(
            regulasi_id,
            metadata,
            source_node_id,
            section_type,
            reference_label,
            text_norm,
            len(chunks) + 1,
            source_file or main_pdf,
        ))

    for item in processed["preamble"]["menimbang"]:
        add(item["id"], "menimbang", f"{regulasi_id} Menimbang {item['nomor']}", item["teks_isi"])

    for item in processed["preamble"]["mengingat"]:
        add(item["id"], "mengingat", f"{regulasi_id} Mengingat {item['nomor']}", item["teks_isi"])

    for pasal in processed["body"]["pasal"]:
        pasal_ref = f"{regulasi_id} Pasal {pasal['nomor_pasal']}"
        if pasal["ayat"]:
            if pasal["teks_isi"]:
                add(pasal["id"], "pasal", pasal_ref, pasal["teks_isi"])
            for ayat in pasal["ayat"]:
                add(
                    ayat["id"],
                    "ayat",
                    f"{pasal_ref} ayat ({ayat['nomor_ayat']})",
                    ayat["teks_isi"],
                )
                for sub in ayat.get("sub_items", []):
                    add(
                        sub["id"],
                        "subitem",
                        f"{pasal_ref} ayat ({ayat['nomor_ayat']}) {sub['jenis']} {sub['nomor']}",
                        sub["teks_isi"],
                    )
        else:
            add(pasal["id"], "pasal", pasal_ref, pasal["teks_isi"])
            for sub in pasal.get("sub_items", []):
                add(sub["id"], "subitem", f"{pasal_ref} {sub['jenis']} {sub['nomor']}", sub["teks_isi"])

    explanation = processed["explanation"]
    if explanation["available"]:
        source_file = explanation.get("source_file", "")
        add(
            f"PJL-{regulasi_id}-UMUM",
            "penjelasan_umum",
            f"{regulasi_id} Penjelasan Umum",
            explanation["umum"],
            source_file,
        )
        for item in explanation["pasal_demi_pasal"]:
            add(
                item["id"],
                "penjelasan_pasal",
                f"{regulasi_id} Penjelasan Pasal {item['nomor_pasal']}",
                item["teks_isi"],
                source_file,
            )

    for item in processed.get("attachments", []):
        add(
            item["id"],
            "lampiran",
            f"{regulasi_id} Lampiran {item['nomor_lampiran']}",
            item["teks_isi"],
            item.get("source_file", ""),
        )

    add(f"PENUTUP-{regulasi_id}", "penutup", f"{regulasi_id} Penutup", processed["closing"]["teks_isi"])

    return chunks


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(regulasi_id: str) -> Dict:
    raw_pdf = RAW_DIR / f"{regulasi_id}.pdf"
    if not raw_pdf.exists():
        raise FileNotFoundError(f"Raw PDF not found: {raw_pdf}")

    logger.info("Processing %s", regulasi_id)
    processed = build_processed_document(regulasi_id, raw_pdf)
    chunks = generate_chunks(processed)

    processed_path = PROCESSED_V2_DIR / f"{regulasi_id}.json"
    chunks_path = CHUNKS_V2_DIR / f"chunks_{regulasi_id}.jsonl"
    write_json(processed_path, processed)
    write_jsonl(chunks_path, chunks)

    return {
        "regulasi_id": regulasi_id,
        "processed_path": str(processed_path),
        "chunks_path": str(chunks_path),
        "total_pasal": processed["parse_quality"]["total_pasal"],
        "total_ayat": processed["parse_quality"]["total_ayat"],
        "total_chunks": len(chunks),
        "warnings": processed["parse_quality"]["warnings"],
    }


def discover_regulasi_ids() -> List[str]:
    return sorted(path.stem for path in RAW_DIR.glob("REG-*.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Regulasi ID, e.g. REG-PP-47-2008")
    parser.add_argument("--all", action="store_true", help="Process all REG-*.pdf files in data/raw")
    args = parser.parse_args()

    if not args.file and not args.all:
        parser.error("Use --file REG-ID or --all")

    targets = [args.file] if args.file else discover_regulasi_ids()
    results = []
    for regulasi_id in targets:
        try:
            result = process_file(regulasi_id)
            results.append(result)
            logger.info(
                "%s: %s pasal, %s ayat, %s chunks",
                regulasi_id,
                result["total_pasal"],
                result["total_ayat"],
                result["total_chunks"],
            )
        except Exception as exc:
            logger.error("%s failed: %s", regulasi_id, exc)
            results.append({"regulasi_id": regulasi_id, "error": str(exc)})

    report = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "results": results
    }
    write_json(Path("outputs/reports/v2/extract_v2_report.json"), report)


if __name__ == "__main__":
    main()
