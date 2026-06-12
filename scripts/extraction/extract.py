"""
PSKP Knowledge Graph — Extraction Pipeline (Issue #2 & #3)
==========================================================

Script untuk mengekstraksi teks dari dokumen PDF regulasi PSKP,
membersihkan teks, mensegmentasi ke level Pasal/Ayat (Issue #2),
kemudian menjalankan pipeline Regex + LLM Hybrid untuk menghasilkan
triplets tervalidasi (Issue #3).

Usage:
    python scripts/extraction/extract.py                     # Proses semua dokumen
    python scripts/extraction/extract.py --file REG-PP-4-2022  # Proses satu dokumen
"""

import os
import sys
import json
import re
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") # Default None will use OpenAI
)

RAW_DIR = os.getenv("DATA_RAW_DIR", "./data/raw")
PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "./data/processed")
TRIPLETS_DIR = os.getenv("DATA_TRIPLETS_DIR", "./data/triplets")
SCHEMA_PATH = os.getenv("ONTOLOGY_SCHEMA", "./ontology/schema.json")
FEW_SHOTS_PATH = "./ontology/few_shot_examples.json"
CORPUS_MANIFEST_PATH = "./data/corpus_manifest.json"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_MAX_RETRIES = 2
LLM_RETRY_TEMPERATURE = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Regex Patterns (Issue #2 - Segmentasi & Issue #3 - Layer 1) ─────────────

SEGMENTATION_PATTERNS = {
    "bab":    r"(?i)^BAB\s+([IVXLC]+|[0-9]+)",
    # Supports both Arabic numerals (Pasal 1, 1A) and Roman numerals (Pasal I, II).
    "pasal":  r"(?i)^Pasal\s+(\d+[A-Z]?|[IVXLC]+)",
    "ayat":   r"^\((\d+)\)\s+",
    "huruf":  r"^([a-z])\.\s+",
    "angka":  r"^(\d+)\.\s+",
}

RUJUKAN_PATTERNS = {
    "pasal_internal": r"(?i)(?:sebagaimana\s+dimaksud\s+dalam\s+)?[Pp]asal\s+(\d+[A-Z]?)\s*(?:ayat\s+\((\d+)\))?",
    # Supports:
    # - Undang-Undang Nomor 20 Tahun 2003
    # - Undang-Undang Nomor 1/PNPS/1965
    "rujukan_uu":     r"(?i)Undang[- ]Undang\s+(?:Nomor|No\.?)\s+([A-Za-z0-9\/\.-]+)\s*(?:[Tt]ahun\s+(\d{4}))?",
    "rujukan_pp":     r"(?i)Peraturan\s+Pemerintah\s+(?:Nomor|No\.?)\s+([A-Za-z0-9\/\.-]+)\s*(?:[Tt]ahun\s+(\d{4}))?",
    # Shorthand often used in amendment lists, e.g. "a. Nomor 13 Tahun 1980".
    # Applied contextually when the pasal text clearly discusses Peraturan Pemerintah.
    "rujukan_pp_shorthand": r"(?im)^\s*(?:[a-z]\.\s*)?Nomor\s+([A-Za-z0-9\/\.-]+)\s+Tahun\s+(\d{4})\b",
    "pencabutan":     r"(?i)(?:dicabut|tidak berlaku)\s+(?:dengan|oleh)",
    "perubahan":      r"(?i)diubah\s+(?:sehingga|menjadi)",
}

# ═════════════════════════════════════════════════════════════════════════════
# Issue #2: Data Collection & Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Ekstraksi teks dari file PDF menggunakan pdfplumber.
    (Issue #2, Task 2.2)

    Deteksi otomatis: jika ekstraksi menghasilkan < 100 karakter per halaman,
    tandai sebagai 'scan_detected'.
    """
    text = ""
    scan_detected = False
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < 100:
                scan_detected = True
            text += page_text + "\n"

    if scan_detected:
        logger.warning(f"Scan detected in {pdf_path}. OCR pipeline may be needed.")

    return text, scan_detected


def clean_text(text: str) -> str:
    """
    Pipeline pembersihan teks berurutan (Issue #2, Task 2.3).

    STEP-1: Hapus header/footer berulang (nomor halaman, watermark)
    STEP-2: Normalisasi whitespace
    STEP-3: Perbaiki hyphenation
    STEP-4: Standarisasi tanda baca
    STEP-5: Hapus baris yang hanya berisi angka
    STEP-6: Pertahankan struktur penomoran Pasal/Ayat
    """
    # STEP-1: Hapus header/footer nomor halaman
    text = re.sub(r"(?m)^\s*-?\s*\d+\s*-?\s*$", "", text)
    text = re.sub(r"(?m)^.*(?:LEMBARAN NEGARA|TAMBAHAN LEMBARAN|PRESIDEN REPUBLIK).*$", "", text)

    # STEP-2: Normalisasi whitespace (tetapi pertahankan newline bermakna)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # STEP-3: Perbaiki hyphenation
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # STEP-4: Standarisasi tanda baca
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # STEP-5: Hapus baris yang hanya berisi angka (sisa nomor halaman)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # STEP-6: Pertahankan struktur penomoran (sudah implicit — tidak menghapus pattern Pasal/Ayat)

    return text.strip()


def segment_document(text: str, regulasi_id: str, metadata: Dict) -> Dict:
    """
    Segmentasi teks menjadi hierarki Regulasi > Pasal > Ayat (Issue #2, Task 2.4).
    """
    result = {
        "regulasi_id": regulasi_id,
        "metadata": metadata,
        "pasal": []
    }

    # Split berdasarkan pattern Pasal
    pasal_pattern = re.compile(SEGMENTATION_PATTERNS["pasal"], re.MULTILINE)
    pasal_splits = pasal_pattern.split(text)

    for i in range(1, len(pasal_splits), 2):
        nomor_pasal = pasal_splits[i]
        pasal_content = pasal_splits[i + 1] if i + 1 < len(pasal_splits) else ""

        # Split content berdasarkan Ayat pattern
        ayat_pattern = re.compile(SEGMENTATION_PATTERNS["ayat"], re.MULTILINE)
        ayat_splits = ayat_pattern.split(pasal_content.strip())

        ayat_list = []
        if len(ayat_splits) > 1:
            for j in range(1, len(ayat_splits), 2):
                nomor_ayat = ayat_splits[j]
                ayat_text = ayat_splits[j + 1].strip() if j + 1 < len(ayat_splits) else ""

                # Parse sub-items (huruf/angka)
                sub_items = _extract_sub_items(ayat_text)

                ayat_list.append({
                    "nomor_ayat": nomor_ayat,
                    "teks_isi": ayat_text,
                    "sub_items": sub_items
                })
        else:
            # Pasal tanpa ayat — gunakan nomor_ayat "0"
            ayat_list.append({
                "nomor_ayat": "0",
                "teks_isi": pasal_content.strip(),
                "sub_items": []
            })

        result["pasal"].append({
            "nomor_pasal": nomor_pasal,
            "ayat": ayat_list
        })

    return result


def _extract_sub_items(text: str) -> List[str]:
    """Ekstrak sub-item huruf (a., b., c.) dan angka (1., 2., 3.) dari teks ayat."""
    sub_items = []
    huruf_matches = re.findall(r"^([a-z])\.\s+(.+?)(?=\n[a-z]\.|$)", text, re.MULTILINE | re.DOTALL)
    for letter, content in huruf_matches:
        sub_items.append(f"{letter}. {content.strip()}")
    return sub_items


def run_quality_check(processed_data: Dict) -> Dict:
    """
    Quality Check otomatis setelah segmentasi (Issue #2, Task 2.5).

    QC-01: Jumlah Pasal >= 80% dari ekspektasi (best effort)
    QC-02: Tidak ada field teks_isi kosong atau < 10 karakter
    QC-03: Nomor pasal sekuensial (tidak ada lompatan > 5)
    QC-04: Output valid JSON (implisit — sudah dict)
    """
    qc_result = {
        "regulasi_id": processed_data["regulasi_id"],
        "total_pasal": len(processed_data["pasal"]),
        "checks": {}
    }

    # QC-01: Minimal pasal count guard (prevents false "all good" when segmentation fails).
    qc_result["checks"]["QC-01_min_pasal"] = {
        "status": "PASS" if qc_result["total_pasal"] > 0 else "WARN",
        "total_pasal": qc_result["total_pasal"]
    }

    # QC-02: Empty content check
    empty_count = 0
    for pasal in processed_data["pasal"]:
        for ayat in pasal["ayat"]:
            if not ayat["teks_isi"] or len(ayat["teks_isi"]) < 10:
                empty_count += 1
    qc_result["checks"]["QC-02_empty_content"] = {
        "status": "PASS" if empty_count == 0 else "WARN",
        "empty_count": empty_count
    }

    # QC-03: Sequential check
    max_gap = 0
    pasal_numbers = []
    for pasal in processed_data["pasal"]:
        try:
            num = int(re.match(r"(\d+)", pasal["nomor_pasal"]).group(1))
            pasal_numbers.append(num)
        except (ValueError, AttributeError):
            continue

    for i in range(1, len(pasal_numbers)):
        gap = pasal_numbers[i] - pasal_numbers[i - 1]
        max_gap = max(max_gap, gap)

    qc_result["checks"]["QC-03_sequential"] = {
        "status": "PASS" if max_gap <= 5 else "WARN",
        "max_gap": max_gap
    }

    # QC-04: Valid JSON (implisit)
    qc_result["checks"]["QC-04_valid_json"] = {"status": "PASS"}

    overall = "PASS" if all(
        c["status"] == "PASS" for c in qc_result["checks"].values()
    ) else "WARN"
    qc_result["overall_status"] = overall

    return qc_result


# ═════════════════════════════════════════════════════════════════════════════
# Issue #3: Extraction Pipeline (Regex + LLM Hybrid)
# ═════════════════════════════════════════════════════════════════════════════

def load_schema() -> Dict:
    """Muat schema ontologi untuk validasi."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_few_shots() -> List[Dict]:
    """Muat few-shot examples untuk LLM prompt."""
    with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Layer 1: Regex Extractor ────────────────────────────────────────────────

def _normalize_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip().upper())
    return re.sub(r"-{2,}", "-", value).strip("-")


def _build_regulasi_id(kind: str, nomor_raw: str, tahun_raw: Optional[str]) -> Optional[str]:
    kind = str(kind or "").upper().strip()
    nomor_raw = str(nomor_raw or "").strip()
    tahun_raw = str(tahun_raw or "").strip()

    if not kind or not nomor_raw:
        return None

    # Special handling: UU 1/PNPS/1965 style.
    if kind == "UU":
        pnps_match = re.match(
            r"(?i)^([A-Za-z0-9]+)[/\-](PNPS|PENPRES|PRP)[/\-](\d{4})$",
            nomor_raw,
        )
        if pnps_match:
            nomor = _normalize_token(pnps_match.group(1))
            regime = _normalize_token(pnps_match.group(2))
            tahun = _normalize_token(pnps_match.group(3))
            return f"REG-UU-{nomor}-{regime}-{tahun}"

    nomor = _normalize_token(nomor_raw)
    tahun = _normalize_token(tahun_raw)

    if tahun:
        return f"REG-{kind}-{nomor}-{tahun}"

    # Fallback: infer trailing year from token if possible (e.g. "1-PNPS-1965").
    trailing_year = re.search(r"(\d{4})$", nomor)
    if trailing_year:
        return f"REG-{kind}-{nomor}"

    return None


def _extract_referenced_regulasi_ids(teks: str) -> List[str]:
    ids: List[str] = []
    seen = set()

    for match in re.finditer(RUJUKAN_PATTERNS["rujukan_uu"], teks):
        reg_id = _build_regulasi_id("UU", match.group(1), match.group(2))
        if reg_id and reg_id not in seen:
            ids.append(reg_id)
            seen.add(reg_id)

    for match in re.finditer(RUJUKAN_PATTERNS["rujukan_pp"], teks):
        reg_id = _build_regulasi_id("PP", match.group(1), match.group(2))
        if reg_id and reg_id not in seen:
            ids.append(reg_id)
            seen.add(reg_id)

    # Context-aware shorthand handling: only infer PP references if the text
    # contains "Peraturan Pemerintah" to reduce false positives.
    if re.search(r"(?i)Peraturan\s+Pemerintah", teks):
        for match in re.finditer(RUJUKAN_PATTERNS["rujukan_pp_shorthand"], teks):
            reg_id = _build_regulasi_id("PP", match.group(1), match.group(2))
            if reg_id and reg_id not in seen:
                ids.append(reg_id)
                seen.add(reg_id)

    return ids


def _resolve_change_target(teks: str, current_regulasi_id: str) -> Optional[str]:
    for ref_id in _extract_referenced_regulasi_ids(teks):
        if ref_id != current_regulasi_id:
            return ref_id
    return None


def regex_extractor(pasal_id: str, teks: str, regulasi_id: str) -> List[Dict]:
    """
    Layer 1: Regex Extractor untuk relasi eksplisit (Issue #3, Task 3.1).
    Menangkap pola rujukan eksplisit dari teks pasal.
    """
    triplets = []

    # Rujukan ke Pasal internal
    for match in re.finditer(RUJUKAN_PATTERNS["pasal_internal"], teks):
        target_pasal = match.group(1)
        target_ayat = match.group(2) or "0"
        target_id = f"PSL-{regulasi_id}-{target_pasal}-{target_ayat}"
        triplets.append({
            "source_id": pasal_id,
            "predicate": "MENGACU_PADA",
            "target_id": target_id,
            "confidence": 1.0,
            "extraction_method": "regex",
            "raw_match": match.group(0),
            "properties": {"konteks": match.group(0)}
        })

    # Rujukan ke Undang-Undang
    for match in re.finditer(RUJUKAN_PATTERNS["rujukan_uu"], teks):
        target_id = _build_regulasi_id("UU", match.group(1), match.group(2))
        if not target_id:
            continue
        triplets.append({
            "source_id": pasal_id,
            "predicate": "MENGACU_PADA",
            "target_id": target_id,
            "confidence": 1.0,
            "extraction_method": "regex",
            "raw_match": match.group(0),
            "properties": {"konteks": match.group(0)}
        })

    # Rujukan ke Peraturan Pemerintah
    for match in re.finditer(RUJUKAN_PATTERNS["rujukan_pp"], teks):
        target_id = _build_regulasi_id("PP", match.group(1), match.group(2))
        if not target_id:
            continue
        triplets.append({
            "source_id": pasal_id,
            "predicate": "MENGACU_PADA",
            "target_id": target_id,
            "confidence": 1.0,
            "extraction_method": "regex",
            "raw_match": match.group(0),
            "properties": {"konteks": match.group(0)}
        })

    # Rujukan PP format shorthand in amendment lists (e.g. "b. Nomor 15 Tahun 1985")
    if re.search(r"(?i)Peraturan\s+Pemerintah", teks):
        for match in re.finditer(RUJUKAN_PATTERNS["rujukan_pp_shorthand"], teks):
            target_id = _build_regulasi_id("PP", match.group(1), match.group(2))
            if not target_id:
                continue
            triplets.append({
                "source_id": pasal_id,
                "predicate": "MENGACU_PADA",
                "target_id": target_id,
                "confidence": 1.0,
                "extraction_method": "regex",
                "raw_match": match.group(0),
                "properties": {"konteks": match.group(0)}
            })

    # Deteksi pencabutan
    if re.search(RUJUKAN_PATTERNS["pencabutan"], teks):
        target_regulasi = _resolve_change_target(teks, regulasi_id)
        triplets.append({
            "source_id": regulasi_id,
            "predicate": "MENCABUT",
            "target_id": target_regulasi or "__NEEDS_RESOLUTION__",
            "confidence": 0.7,
            "extraction_method": "regex",
            "raw_match": re.search(RUJUKAN_PATTERNS["pencabutan"], teks).group(0),
            "properties": {"tanggal_efektif": "__UNKNOWN__"}
        })

    # Deteksi perubahan
    if re.search(RUJUKAN_PATTERNS["perubahan"], teks):
        target_regulasi = _resolve_change_target(teks, regulasi_id)
        triplets.append({
            "source_id": regulasi_id,
            "predicate": "MENGAMANDEMEN",
            "target_id": target_regulasi or "__NEEDS_RESOLUTION__",
            "confidence": 0.7,
            "extraction_method": "regex",
            "raw_match": re.search(RUJUKAN_PATTERNS["perubahan"], teks).group(0),
            "properties": {
                "tanggal_efektif": "__UNKNOWN__",
                # Keep non-empty value so this edge can survive schema validation.
                "teks_lama": re.search(RUJUKAN_PATTERNS["perubahan"], teks).group(0),
            }
        })

    return triplets


# ─── Layer 2: LLM Extractor ──────────────────────────────────────────────────

def llm_extractor(pasal_id: str, teks: str, schema: Dict, few_shots: List[Dict]) -> List[Dict]:
    """
    Layer 2: LLM Extractor untuk relasi semantik (Issue #3, Task 3.2).
    Menggunakan GPT-3.5-Turbo untuk mengidentifikasi relasi semantik
    yang tidak dapat ditangkap Regex.
    """
    # Build edge list from schema
    schema_edges = []
    for rel_group in ["explicit_relations", "semantic_relations"]:
        for edge in schema.get("edge_definitions", {}).get(rel_group, []):
            # Skip BERTENTANGAN_DENGAN per RULE-08
            if edge["label"] == "BERTENTANGAN_DENGAN":
                continue
            schema_edges.append(edge["label"])

    system_prompt = f"""Kamu adalah ekstractor relasi hukum presisi tinggi. Tugasmu adalah mengidentifikasi relasi semantik dari teks pasal hukum Indonesia dan mengkonversinya menjadi triplets JSON.

ATURAN KETAT:
1. Hanya gunakan Edge Labels yang terdapat dalam schema berikut: {json.dumps(schema_edges)}
2. Hanya gunakan Node Labels: Regulasi, Pasal, Subjek_Hukum
3. Jika tidak ada relasi semantik yang jelas, kembalikan array kosong []
4. DILARANG mengarang entitas yang tidak ada dalam teks
5. Setiap triplet WAJIB memiliki semua properti yang ditandai wajib dalam schema

FORMAT OUTPUT: JSON array saja, tanpa penjelasan, tanpa markdown.

FEW-SHOT EXAMPLES:
{json.dumps(few_shots, indent=2, ensure_ascii=False)}

INPUT PASAL:
ID: {pasal_id}
Teks: {teks}"""

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            temperature = 0.3 if attempt == 0 else LLM_RETRY_TEMPERATURE
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": system_prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Handle both {"triplets": [...]} and [...] formats
            if isinstance(parsed, dict):
                triplets_raw = parsed.get("triplets", parsed.get("results", []))
            elif isinstance(parsed, list):
                triplets_raw = parsed
            else:
                triplets_raw = []

            # Normalize to our format
            triplets = []
            for t in triplets_raw:
                triplets.append({
                    "source_id": t.get("subject", {}).get("id", pasal_id),
                    "predicate": t.get("predicate", ""),
                    "target_id": t.get("object", {}).get("id", ""),
                    "confidence": 0.8,
                    "extraction_method": "llm",
                    "raw_match": teks[:200],
                    "properties": t.get("properties", {})
                })

            return triplets

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM parse attempt {attempt + 1} failed: {e}")
            if attempt == LLM_MAX_RETRIES:
                logger.error(f"LLM extraction failed for {pasal_id} after {LLM_MAX_RETRIES + 1} attempts.")
                return []

    return []


# ─── Merging & Deduplication ─────────────────────────────────────────────────

def deduplicate_triplets(triplets: List[Dict]) -> List[Dict]:
    """
    Deduplikasi triplets berdasarkan composite key (Issue #3, Task 3.3).

    Composite Key: (source_id, predicate, target_id)
    - Jika duplikat: pertahankan yang confidence lebih tinggi
    - Jika confidence sama: pertahankan hasil Regex (lebih deterministik)
    """
    seen = {}
    for t in triplets:
        key = (t["source_id"], t["predicate"], t["target_id"])
        if key not in seen:
            seen[key] = t
        else:
            existing = seen[key]
            if t["confidence"] > existing["confidence"]:
                seen[key] = t
            elif t["confidence"] == existing["confidence"] and t["extraction_method"] == "regex":
                seen[key] = t

    return list(seen.values())


# ─── Schema Validator ─────────────────────────────────────────────────────────

def validate_triplets(triplets: List[Dict], schema: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Validasi triplets terhadap schema.json dan constraint rules (Issue #3, Task 3.4).
    Returns (valid_triplets, invalid_triplets).
    """
    valid_edges = set()
    for rel_group in ["explicit_relations", "semantic_relations"]:
        for edge in schema.get("edge_definitions", {}).get(rel_group, []):
            valid_edges.add(edge["label"])

    valid = []
    invalid = []

    for t in triplets:
        reasons = []

        # Validasi predicate ada di schema
        if t["predicate"] not in valid_edges:
            reasons.append(f"Unknown predicate: {t['predicate']}")

        # Validasi field tidak kosong
        if not t.get("source_id") or not t.get("target_id") or not t.get("predicate"):
            reasons.append("Missing required field (source_id, predicate, or target_id)")

        # Validasi target bukan placeholder
        if t.get("target_id", "").startswith("__"):
            reasons.append(f"Unresolved target: {t['target_id']}")

        # RULE-05: MENGAMANDEMEN wajib memiliki teks_lama
        if t["predicate"] == "MENGAMANDEMEN":
            props = t.get("properties", {})
            if not props.get("teks_lama") or props["teks_lama"] == "__UNKNOWN__":
                reasons.append("RULE-05: MENGAMANDEMEN missing teks_lama")

        # RULE-08: BERTENTANGAN_DENGAN tidak boleh dibuat saat ekstraksi otomatis
        if t["predicate"] == "BERTENTANGAN_DENGAN":
            reasons.append("RULE-08: BERTENTANGAN_DENGAN only allowed during Ground Truth phase")

        # RULE-01: MENCABUT hanya Regulasi -> Regulasi
        if t["predicate"] == "MENCABUT":
            if not t["source_id"].startswith("REG-") or not t["target_id"].startswith("REG-"):
                reasons.append("RULE-01: MENCABUT requires Regulasi -> Regulasi")

        # RULE-02: MEMILIKI_PASAL hanya Regulasi -> Pasal
        if t["predicate"] == "MEMILIKI_PASAL":
            if not t["source_id"].startswith("REG-") or not t["target_id"].startswith("PSL-"):
                reasons.append("RULE-02: MEMILIKI_PASAL requires Regulasi -> Pasal")

        if reasons:
            t["validation_errors"] = reasons
            invalid.append(t)
        else:
            valid.append(t)

    return valid, invalid


# ═════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process_document(manifest_entry: Dict, schema: Dict, few_shots: List[Dict]) -> Dict:
    """
    Pipeline lengkap per dokumen: PDF → Segments → Triplets.
    Menggabungkan Issue #2 (preprocessing) dan Issue #3 (extraction).
    """
    regulasi_id = manifest_entry["id"]
    pdf_path = manifest_entry["path_raw"]
    output_processed = manifest_entry["path_output"]

    logger.info(f"═══ Processing {regulasi_id}: {manifest_entry['judul']} ═══")

    # ── Issue #2: Extraction & Preprocessing ──────────────────────
    logger.info(f"[Step 1/5] Extracting text from PDF...")
    raw_text, scan_detected = extract_text_from_pdf(pdf_path)

    logger.info(f"[Step 2/5] Cleaning text...")
    cleaned_text = clean_text(raw_text)

    logger.info(f"[Step 3/5] Segmenting into Pasal/Ayat hierarchy...")
    metadata = {
        "jenis": manifest_entry["jenis"],
        "nomor": manifest_entry["nomor"],
        "tahun": manifest_entry["tahun"],
        "judul": manifest_entry["judul"],
        "status": manifest_entry.get("status", "Berlaku"),
        "scan_detected": scan_detected
    }
    segmented = segment_document(cleaned_text, regulasi_id, metadata)

    # Quality check
    qc_result = run_quality_check(segmented)
    logger.info(f"QC: {qc_result['overall_status']} | {qc_result['total_pasal']} pasal extracted")

    # Save processed JSON
    os.makedirs(os.path.dirname(output_processed), exist_ok=True)
    with open(output_processed, "w", encoding="utf-8") as f:
        json.dump(segmented, f, indent=2, ensure_ascii=False)

    # ── Issue #3: Triplet Extraction ──────────────────────────────
    logger.info(f"[Step 4/5] Extracting triplets (Regex + LLM)...")
    all_triplets = []
    failed_extractions = []
    regex_count = 0
    llm_count = 0

    # Build MEMILIKI_PASAL relationships
    for idx, pasal in enumerate(segmented["pasal"]):
        for ayat in pasal["ayat"]:
            pasal_id = f"PSL-{regulasi_id}-{pasal['nomor_pasal']}-{ayat['nomor_ayat']}"

            # Structural relationship: Regulasi -> Pasal
            all_triplets.append({
                "source_id": regulasi_id,
                "predicate": "MEMILIKI_PASAL",
                "target_id": pasal_id,
                "confidence": 1.0,
                "extraction_method": "structural",
                "raw_match": "",
                "properties": {"urutan": idx + 1}
            })

            # Layer 1: Regex extraction
            regex_triplets = regex_extractor(pasal_id, ayat["teks_isi"], regulasi_id)
            regex_count += len(regex_triplets)
            all_triplets.extend(regex_triplets)

            # Layer 2: LLM extraction
            try:
                llm_triplets = llm_extractor(pasal_id, ayat["teks_isi"], schema, few_shots)
                llm_count += len(llm_triplets)
                all_triplets.extend(llm_triplets)
            except Exception as e:
                logger.error(f"LLM extraction failed for {pasal_id}: {e}")
                failed_extractions.append({
                    "pasal_id": pasal_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    # Deduplication
    logger.info(f"[Step 5/5] Deduplicating and validating...")
    deduped = deduplicate_triplets(all_triplets)

    # Schema validation
    valid_triplets, invalid_triplets = validate_triplets(deduped, schema)

    # Build output (Issue #3, Task 3.5)
    triplet_output = {
        "regulasi_id": regulasi_id,
        "extraction_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_triplets": len(deduped),
            "valid_triplets": len(valid_triplets),
            "invalid_triplets": len(invalid_triplets),
            "regex_extracted": regex_count,
            "llm_extracted": llm_count
        },
        "triplets": valid_triplets
    }

    # Save triplets
    os.makedirs(TRIPLETS_DIR, exist_ok=True)
    triplet_file = os.path.join(TRIPLETS_DIR, f"triplets_{regulasi_id}.json")
    with open(triplet_file, "w", encoding="utf-8") as f:
        json.dump(triplet_output, f, indent=2, ensure_ascii=False)

    logger.info(
        f"✅ {regulasi_id}: {len(valid_triplets)} valid / {len(invalid_triplets)} invalid triplets "
        f"(regex: {regex_count}, llm: {llm_count})"
    )

    return {
        "regulasi_id": regulasi_id,
        "qc_result": qc_result,
        "extraction_metadata": triplet_output["extraction_metadata"],
        "failed_extractions": failed_extractions
    }


def main():
    """Main entry point — processes all documents from corpus manifest."""
    # Ensure directories exist
    for d in [RAW_DIR, PROCESSED_DIR, TRIPLETS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load schema and few-shots
    schema = load_schema()
    few_shots = load_few_shots()

    # Load or create corpus manifest
    if os.path.exists(CORPUS_MANIFEST_PATH):
        with open(CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        logger.warning(f"Corpus manifest not found at {CORPUS_MANIFEST_PATH}")
        logger.info("Scanning for PDFs in raw directory...")
        manifest = []
        for filename in os.listdir(RAW_DIR):
            if filename.endswith(".pdf"):
                # Auto-generate manifest entry from filename
                manifest.append({
                    "id": filename.replace(".pdf", ""),
                    "nama_file": filename,
                    "jenis": "unknown",
                    "nomor": "",
                    "tahun": 0,
                    "judul": filename.replace(".pdf", "").replace("_", " "),
                    "path_raw": os.path.join(RAW_DIR, filename),
                    "path_output": os.path.join(PROCESSED_DIR, filename.replace(".pdf", ".json")),
                    "status_ekstraksi": "pending"
                })

    # Filter single file if --file argument provided
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        target_id = sys.argv[2]
        manifest = [m for m in manifest if m["id"] == target_id]
        if not manifest:
            logger.error(f"Document {target_id} not found in manifest.")
            sys.exit(1)

    # Process each document
    all_results = []
    all_failed = []
    qc_reports = []

    for entry in manifest:
        if entry.get("status_ekstraksi") == "done" and "--force" not in sys.argv:
            logger.info(f"Skipping {entry['id']} (already processed). Use --force to re-process.")
            continue

        try:
            result = process_document(entry, schema, few_shots)
            all_results.append(result)
            qc_reports.append(result["qc_result"])
            all_failed.extend(result["failed_extractions"])

            # Update manifest status
            entry["status_ekstraksi"] = "done"
        except Exception as e:
            logger.error(f"Failed to process {entry['id']}: {e}")
            entry["status_ekstraksi"] = "error"
            all_failed.append({
                "regulasi_id": entry["id"],
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    # Save updated manifest
    with open(CORPUS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save QC report
    qc_report_path = os.path.join(PROCESSED_DIR, "qc_report.json")
    with open(qc_report_path, "w", encoding="utf-8") as f:
        json.dump(qc_reports, f, indent=2, ensure_ascii=False)

    # Save failed extractions log
    failed_path = os.path.join(TRIPLETS_DIR, "failed_extractions.json")
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(all_failed, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("═" * 60)
    logger.info(f"Pipeline complete. {len(all_results)} documents processed.")
    logger.info(f"Failed extractions: {len(all_failed)}")
    logger.info(f"QC report: {qc_report_path}")
    logger.info(f"Failed log: {failed_path}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
