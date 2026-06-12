# V2 Output Contract

Dokumen ini menjelaskan format output v2 yang akan dipakai parser, chunker, extractor, graph builder, embedding index, dan evaluator.

## Target Direktori

Output v2 dipisah dari output lama.

```text
data/processed_v2/
data/chunks_v2/
data/triplets_v2/
indexes/graph_v2/
indexes/vector_v2/
outputs/reports/v2/
```

Output lama di `data/processed`, `data/triplets`, dan `indexes/graph` tetap dibiarkan.

## Processed Document

Setiap dokumen utama menghasilkan satu JSON:

```text
data/processed_v2/{regulasi_id}.json
```

Schema:

```text
ontology/v2/processed_document.schema.json
```

Isi utama:

1. `metadata`: identitas regulasi dan status.
2. `sources`: path PDF utama, penjelasan, dan lampiran jika ada.
3. `preamble`: `Menimbang`, `Mengingat`, `Memutuskan`, `Menetapkan`.
4. `body`: `Bab`, `Bagian`, `Paragraf`, `Pasal`, `Ayat`, dan `SubItem`.
5. `explanation`: `Penjelasan_Umum` dan `Penjelasan_Pasal`.
6. `attachments`: lampiran.
7. `closing`: penutup, penetapan, dan pengundangan.
8. `parse_quality`: QC hasil parsing.

Contoh ringkas:

```json
{
  "schema_version": "2.0.0",
  "regulasi_id": "REG-PP-47-2008",
  "metadata": {
    "jenis": "PP",
    "nomor": "47",
    "tahun": 2008,
    "judul": "Wajib Belajar",
    "status": "Berlaku",
    "has_full_text": true
  },
  "sources": {
    "main_pdf": "data/raw/REG-PP-47-2008.pdf",
    "explanation_pdf": "data/raw/PJL-REG-PP-47-2008.pdf",
    "attachment_files": []
  },
  "preamble": {
    "opening_phrase": "DENGAN RAHMAT TUHAN YANG MAHA ESA",
    "menimbang": [],
    "mengingat": [],
    "memutuskan": "MEMUTUSKAN:",
    "menetapkan": ""
  },
  "body": {
    "bab": [],
    "pasal": []
  },
  "explanation": {
    "available": true,
    "source_file": "data/raw/PJL-REG-PP-47-2008.pdf",
    "umum": "",
    "pasal_demi_pasal": []
  },
  "attachments": [],
  "closing": {
    "teks_isi": ""
  },
  "parse_quality": {
    "status": "WARN",
    "scan_detected": false,
    "total_pasal": 0,
    "total_ayat": 0,
    "has_preamble": false,
    "has_explanation": true,
    "warnings": []
  }
}
```

## Chunk JSONL

Setiap regulasi menghasilkan JSONL:

```text
data/chunks_v2/chunks_{regulasi_id}.jsonl
```

Satu baris = satu chunk.

Schema:

```text
ontology/v2/chunk.schema.json
```

Unit chunk:

1. Satu item `Menimbang`.
2. Satu item `Mengingat`.
3. Satu `Pasal` tanpa ayat.
4. Satu `Ayat` jika pasal punya ayat.
5. Satu `SubItem` jika ayat panjang atau bernomor jelas.
6. Satu `Penjelasan_Umum`.
7. Satu `Penjelasan_Pasal`.
8. Satu bagian lampiran yang logis.
9. Satu penutup jika diperlukan untuk tanggal/status.

Contoh:

```json
{
  "schema_version": "2.0.0",
  "chunk_id": "CHK-REG-PP-47-2008-0001",
  "regulasi_id": "REG-PP-47-2008",
  "source_node_id": "AYT-REG-PP-47-2008-5-1",
  "section_type": "ayat",
  "reference_label": "PP 47/2008 Pasal 5 ayat (1)",
  "text": "Satuan pendidikan dasar penyelenggara program wajib belajar wajib menjaga keberlangsungan pelaksanaan program wajib belajar...",
  "metadata": {
    "jenis": "PP",
    "nomor": "47",
    "tahun": 2008,
    "judul": "Wajib Belajar",
    "nomor_pasal": "5",
    "nomor_ayat": "1",
    "source_file": "data/raw/REG-PP-47-2008.pdf"
  }
}
```

## Triplet JSON

Setiap regulasi menghasilkan satu JSON:

```text
data/triplets_v2/triplets_{regulasi_id}.json
```

Isi file:

```json
{
  "schema_version": "2.0.0",
  "regulasi_id": "REG-PP-47-2008",
  "extraction_metadata": {},
  "triplets": []
}
```

Setiap item `triplets[]` mengikuti:

```text
ontology/v2/triplet.schema.json
```

Contoh:

```json
{
  "schema_version": "2.0.0",
  "source_id": "REG-PP-47-2008",
  "source_type": "Regulasi",
  "predicate": "MEMILIKI_PASAL",
  "target_id": "PSL-REG-PP-47-2008-5",
  "target_type": "Pasal",
  "confidence": 1.0,
  "extraction_method": "structural",
  "evidence": {
    "text": "Pasal 5"
  },
  "properties": {
    "urutan": 5
  },
  "validation": {
    "status": "valid",
    "errors": [],
    "warnings": []
  }
}
```

## Hubungan Output

Relasi antar-output:

```text
processed_v2 node id
  -> menjadi source_node_id di chunk
  -> menjadi source_id/target_id di triplet
  -> menjadi node di graph_v2
  -> chunk_id menjadi key di vector_v2
```

Aturan traceability:

1. Semua chunk harus punya `source_node_id`.
2. Semua triplet harus punya evidence teks.
3. Semua dataset evaluasi harus punya `chunk_id` atau `evidence_chunk_ids`.
4. Jawaban GraphRAG harus bisa ditelusuri ke `regulasi_id`, `node_id`, dan `chunk_id`.
