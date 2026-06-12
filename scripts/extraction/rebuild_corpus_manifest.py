#!/usr/bin/env python3
"""
Rebuild extraction manifest (list format) from data/raw/*.pdf.

This is needed by scripts/extraction/extract.py and scripts/extraction/batch_extract.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


TITLE_MAP = {
    "REG-PP-55-2007": "Pendidikan Agama dan Pendidikan Keagamaan",
    "REG-PP-47-2008": "Wajib Belajar",
    "REG-PP-48-2008": "Pendanaan Pendidikan",
    "REG-PP-74-2008": "Guru",
    "REG-PP-41-2009": "Tunjangan Profesi Guru dan Dosen, Tunjangan Khusus Guru dan Dosen, serta Tunjangan Kehormatan Profesor",
    "REG-PP-37-2009": "Dosen",
    "REG-PP-17-2010": "Pengelolaan dan Penyelenggaraan Pendidikan",
    "REG-PP-66-2010": "Perubahan atas Peraturan Pemerintah No 17 Tahun 2010 tentang Pengelolaan dan Penyelenggaraan Pendidikan",
    "REG-PP-19-2017": "Perubahan atas Peraturan Pemerintah No 74 Tahun 2008 tentang Guru",
    "REG-PP-57-2021": "Standar Nasional Pendidikan",
    "REG-PP-4-2022": "Perubahan atas Peraturan Pemerintah Nomor 57 Tahun 2021 tentang Standar Nasional Pendidikan",
    "REG-PP-18-2022": "Perubahan atas Peraturan Pemerintah No 48 Tahun 2008 tentang Pendanaan Pendidikan",
    "REG-PP-19-2005": "Standar Nasional Pendidikan",
    "REG-PP-13-2015": "Perubahan Kedua atas Peraturan Pemerintah Nomor 19 Tahun 2005 tentang Standar Nasional Pendidikan",
    "REG-PP-7-1977": "Peraturan Gaji Pegawai Negeri Sipil",
    "REG-PP-8-2009": "Perubahan atas Peraturan Gaji Pegawai Negeri Sipil",
    "REG-UU-20-2003": "Sistem Pendidikan Nasional",
    "REG-UU-14-2005": "Guru dan Dosen",
    "REG-UU-12-2012": "Pendidikan Tinggi",
    "REG-UU-9-2009": "Badan Hukum Pendidikan",
    "REG-UU-1-PNPS-1965": "Pencegahan Penyalahgunaan dan/atau Penodaan Agama",
    "REG-UUD-1945": "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
}


def parse_reg_id(reg_id: str) -> Tuple[str, str, int]:
    if reg_id == "REG-UUD-1945":
        return "UUD", "", 1945

    m = re.match(r"^REG-([A-Z]+)-(.+)-(\d{4})$", reg_id)
    if not m:
        return "unknown", "", 0

    jenis = m.group(1)
    nomor = m.group(2)
    tahun = int(m.group(3))
    return jenis, nomor, tahun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--manifest_path", default="data/corpus_manifest.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    manifest_path = Path(args.manifest_path)

    entries: List[Dict] = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        reg_id = pdf_path.stem
        jenis, nomor, tahun = parse_reg_id(reg_id)
        out_json = processed_dir / f"{reg_id}.json"

        entry = {
            "id": reg_id,
            "nama_file": pdf_path.name,
            "jenis": jenis,
            "nomor": nomor,
            "tahun": tahun,
            "judul": TITLE_MAP.get(reg_id, reg_id),
            "path_raw": str(pdf_path),
            "path_output": str(out_json),
            "status_ekstraksi": "done" if out_json.exists() else "pending",
        }
        entries.append(entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    pending = sum(1 for e in entries if e["status_ekstraksi"] == "pending")
    done = sum(1 for e in entries if e["status_ekstraksi"] == "done")
    print(f"Manifest rebuilt: {manifest_path}")
    print(f"Total: {len(entries)} | done: {done} | pending: {pending}")


if __name__ == "__main__":
    main()
