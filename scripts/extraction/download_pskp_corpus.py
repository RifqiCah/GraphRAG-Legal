#!/usr/bin/env python3
"""
Download raw PSKP legal corpus PDFs from peraturan.bpk.go.id.

This script:
1. Reads regulation IDs + metadata from data/processed/*.json
2. Finds the corresponding BPK detail page
3. Extracts the PDF download link
4. Downloads PDFs into data/raw/
5. Writes data/corpus_manifest.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://peraturan.bpk.go.id"
SEARCH_URL = f"{BASE_URL}/Search?keywords={{keywords}}"


@dataclass
class Regulation:
    regulasi_id: str
    jenis: str
    nomor: str
    tahun: str
    judul: str


def _http_get_text(url: str, timeout: int = 60) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _http_get_bytes(url: str, timeout: int = 120) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_regulations_from_processed(processed_dir: Path) -> List[Regulation]:
    regs: List[Regulation] = []
    for path in sorted(glob.glob(str(processed_dir / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("metadata", {})
        regs.append(
            Regulation(
                regulasi_id=payload.get("regulasi_id") or Path(path).stem,
                jenis=str(meta.get("jenis", "")).strip(),
                nomor=str(meta.get("nomor", "")).strip(),
                tahun=str(meta.get("tahun", "")).strip(),
                judul=str(meta.get("judul", "")).strip(),
            )
        )
    return regs


def build_keywords(reg: Regulation) -> str:
    if reg.jenis.upper() == "UUD":
        return "UUD 1945"
    return f"{reg.jenis} No {reg.nomor} Tahun {reg.tahun} {reg.judul}".strip()


def find_detail_url(reg: Regulation) -> Optional[str]:
    keywords = quote(build_keywords(reg))
    html = _http_get_text(SEARCH_URL.format(keywords=keywords))

    jenis = reg.jenis.lower()
    nomor = re.escape(reg.nomor.lower())
    tahun = re.escape(reg.tahun.lower())

    if jenis == "uud":
        pattern = re.compile(r"/Details/\d+/[^\"]*1945[^\"]*", re.IGNORECASE)
    else:
        # Prefer exact "jenis + nomor + tahun" in slug.
        pattern = re.compile(
            rf"/Details/\d+/[^\"]*{jenis}[^\"]*no-?{nomor}[^\"]*tahun-?{tahun}[^\"]*",
            re.IGNORECASE,
        )

    match = pattern.search(html)
    if match:
        return urljoin(BASE_URL, match.group(0))

    # Fallback: any details link that has both number and year tokens.
    candidates = re.findall(r"/Details/\d+/[^\"]+", html, flags=re.IGNORECASE)
    for c in candidates:
        c_lower = c.lower()
        if reg.jenis.lower() in c_lower and reg.nomor.lower() in c_lower and reg.tahun.lower() in c_lower:
            return urljoin(BASE_URL, c)
    return None


def find_pdf_url(detail_url: str) -> Optional[str]:
    html = _http_get_text(detail_url)

    # Prefer direct href /Download/{id}/{filename}.pdf
    match = re.search(r'href="(/Download/\d+/[^"]+\.pdf)"', html, flags=re.IGNORECASE)
    if match:
        return urljoin(BASE_URL, match.group(1))

    # Fallback for escaped URLs
    match = re.search(r"(/Download/\d+/[^'\" ]+\.pdf)", html, flags=re.IGNORECASE)
    if match:
        return urljoin(BASE_URL, match.group(1))
    return None


def sha256sum(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def download_one(reg: Regulation, raw_dir: Path, sleep_s: float = 0.4) -> Dict:
    result: Dict = {
        "regulasi_id": reg.regulasi_id,
        "jenis": reg.jenis,
        "nomor": reg.nomor,
        "tahun": reg.tahun,
        "judul": reg.judul,
        "keywords": build_keywords(reg),
        "detail_url": None,
        "pdf_url": None,
        "output_path": None,
        "sha256": None,
        "status": "pending",
        "error": None,
    }
    try:
        detail_url = find_detail_url(reg)
        result["detail_url"] = detail_url
        if not detail_url:
            result["status"] = "not_found_detail"
            return result

        pdf_url = find_pdf_url(detail_url)
        result["pdf_url"] = pdf_url
        if not pdf_url:
            result["status"] = "not_found_pdf"
            return result

        pdf_bytes = _http_get_bytes(pdf_url)
        if not pdf_bytes.startswith(b"%PDF"):
            result["status"] = "invalid_pdf"
            result["error"] = "Downloaded content is not a PDF."
            return result

        out_name = f"{reg.regulasi_id}.pdf"
        out_path = raw_dir / out_name
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)

        result["output_path"] = str(out_path)
        result["sha256"] = sha256sum(pdf_bytes)
        result["status"] = "downloaded"
        time.sleep(sleep_s)
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--manifest_path", default="data/corpus_manifest.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    manifest_path = Path(args.manifest_path)

    regulations = load_regulations_from_processed(processed_dir)
    if not regulations:
        raise SystemExit(f"Tidak ada file processed di: {processed_dir}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for reg in regulations:
        if args.dry_run:
            detail_url = find_detail_url(reg)
            pdf_url = find_pdf_url(detail_url) if detail_url else None
            results.append(
                {
                    "regulasi_id": reg.regulasi_id,
                    "jenis": reg.jenis,
                    "nomor": reg.nomor,
                    "tahun": reg.tahun,
                    "judul": reg.judul,
                    "keywords": build_keywords(reg),
                    "detail_url": detail_url,
                    "pdf_url": pdf_url,
                    "status": "ok" if (detail_url and pdf_url) else "needs_manual_check",
                }
            )
            print(f"[DRY-RUN] {reg.regulasi_id}: {results[-1]['status']}")
            continue

        result = download_one(reg, raw_dir)
        results.append(result)
        print(f"[{result['status']}] {reg.regulasi_id}")

    manifest = {
        "source": "peraturan.bpk.go.id",
        "generated_at_epoch": int(time.time()),
        "total": len(results),
        "downloaded": sum(1 for r in results if r.get("status") == "downloaded"),
        "items": results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest saved to: {manifest_path}")
    print(f"Downloaded: {manifest['downloaded']}/{manifest['total']}")


if __name__ == "__main__":
    main()
