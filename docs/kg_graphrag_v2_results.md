# KG dan GraphRAG V2 Results Log

Dokumen ini mencatat hasil eksekusi pipeline KG dan GraphRAG v2. Dokumen ini melengkapi rencana di `docs/kg_graphrag_rerun_plan.md`, checklist di `docs/kg_graphrag_execution_checklist.md`, dan rencana eksperimen dua bagian di `docs/experiment_part1_part2_execution.md`.

Tanggal status terakhir: 2026-05-23 Asia/Jakarta.

## 1. Ringkasan Status

Status pipeline saat ini:

| Tahap | Status | Output |
| --- | --- | --- |
| Raw PDF collection | Selesai | `data/raw/` |
| Parser v2 | Selesai | `data/processed_v2/` |
| Chunk v2 | Selesai | `data/chunks_v2/` |
| Structural triplets | Selesai | `data/triplets_v2/` |
| Explicit triplets | Selesai | `data/triplets_v2/` |
| Semantic triplets | Selesai | `data/triplets_v2/` |
| Graph v2 build + QC | Selesai | `indexes/graph_v2/` |
| Vector embedding/index | Selesai | `indexes/vector_v2_model/` |
| GraphRAG retrieval | Selesai awal | `scripts/graph/graphrag_retriever_v2.py` |
| Dataset konflik/evaluasi | Belum | dataset + report evaluasi |

Output utama saat ini adalah file master triplet per dokumen:

```text
data/triplets_v2/triplets_{regulasi_id}.json
```

Setiap file master berisi gabungan:

```text
structural + explicit(regex) + semantic(llm)
```

## 2. Raw Document Collection

Folder raw:

```text
data/raw/
```

Jumlah file raw:

| Jenis file | Jumlah |
| --- | ---: |
| Dokumen utama `REG-*.pdf` | 22 |
| Dokumen penjelasan `PJL-*.pdf` | 1 |
| Total PDF raw | 23 |

Daftar file raw:

| No | File | Catatan |
| ---: | --- | --- |
| 1 | `REG-PP-13-2015.pdf` | Dokumen utama |
| 2 | `REG-PP-17-2010.pdf` | Dokumen utama |
| 3 | `REG-PP-18-2022.pdf` | Dokumen utama |
| 4 | `REG-PP-19-2005.pdf` | Dokumen utama |
| 5 | `REG-PP-19-2017.pdf` | Dokumen utama |
| 6 | `REG-PP-37-2009.pdf` | Dokumen utama |
| 7 | `REG-PP-4-2022.pdf` | Dokumen utama |
| 8 | `REG-PP-41-2009.pdf` | Dokumen utama |
| 9 | `REG-PP-47-2008.pdf` | Dokumen utama |
| 10 | `REG-PP-48-2008.pdf` | Dokumen utama |
| 11 | `REG-PP-55-2007.pdf` | Dokumen utama |
| 12 | `REG-PP-57-2021.pdf` | Dokumen utama |
| 13 | `REG-PP-66-2010.pdf` | Dokumen utama |
| 14 | `REG-PP-7-1977.pdf` | Dokumen utama |
| 15 | `REG-PP-74-2008.pdf` | Dokumen utama |
| 16 | `REG-PP-8-2009.pdf` | Dokumen utama |
| 17 | `REG-UU-1-PNPS-1965.pdf` | Dokumen utama |
| 18 | `REG-UU-12-2012.pdf` | Dokumen utama |
| 19 | `REG-UU-14-2005.pdf` | Dokumen utama |
| 20 | `REG-UU-20-2003.pdf` | Dokumen utama |
| 21 | `REG-UU-9-2009.pdf` | Dokumen utama |
| 22 | `REG-UUD-1945.pdf` | Ada di raw, tetapi salah label: isi file terdeteksi sebagai `PERATURAN PEMERINTAH NO. 2 TAHUN 1945`, bukan naskah UUD 1945 |
| 23 | `PJL-REG-PP-47-2008.pdf` | Dokumen penjelasan terpisah untuk `REG-PP-47-2008` |

Catatan penting:

- `REG-UUD-1945.pdf` tidak dipakai untuk semantic/final GraphRAG karena salah label.
- File tersebut masih ada di raw dan sempat ikut structural/explicit karena proses awal memakai `--all`.
- Untuk tahap berikutnya, graph builder sebaiknya skip `REG-UUD-1945` sampai file UUD yang benar tersedia.

## 3. Ontology dan Output Contract

File yang dibuat untuk rancangan v2:

| File | Fungsi |
| --- | --- |
| `ontology/v2/schema.json` | Ontology node dan edge v2 |
| `ontology/v2/constraints.md` | Aturan validasi/constraint konseptual |
| `ontology/v2/few_shot_examples.json` | Contoh ekstraksi v2 |
| `ontology/v2/processed_document.schema.json` | Schema output parser v2 |
| `ontology/v2/chunk.schema.json` | Schema chunk v2 |
| `ontology/v2/triplet.schema.json` | Schema triplet v2 |
| `docs/v2_output_contract.md` | Kontrak output v2 |
| `docs/kg_graphrag_execution_checklist.md` | Checklist operasional |

Node dan edge yang sudah didukung mencakup:

```text
Regulasi, Bab, Bagian, Paragraf, Pasal, Ayat, SubItem,
Konsiderans, Dasar_Hukum, Diktum, Penjelasan,
Penjelasan_Umum, Penjelasan_Pasal, Lampiran, Penutup,
Peristiwa_Hukum, Chunk, Subjek_Hukum, Topik
```

Edge utama:

```text
MEMILIKI_*, BERDASARKAN, MENGACU_PADA, MENGAMANDEMEN,
MENCABUT, MENCABUT_PASAL, MULAI_BERLAKU, DIUNDANGKAN_PADA,
MENGATUR_TOPIK, MENGATUR_SUBJEK, MEMBERIKAN_HAK,
MEMBEBANKAN_KEWAJIBAN, MELARANG, MENDEFINISIKAN
```

## 4. Parser dan Chunker V2

Script:

```text
scripts/extraction/extract_v2.py
```

Output:

```text
data/processed_v2/{regulasi_id}.json
data/chunks_v2/chunks_{regulasi_id}.jsonl
outputs/reports/v2/extract_v2_report.json
```

Hasil eksekusi parser v2:

| Metric | Nilai |
| --- | ---: |
| Dokumen utama raw | 22 |
| Processed v2 JSON | 22 |
| Chunk v2 JSONL | 22 |
| Error parser | 0 |
| Warning parser | 4 |

Warning parser:

| Regulasi | Warning |
| --- | --- |
| `REG-PP-13-2015` | Numeric pasal sequence has gaps |
| `REG-PP-19-2017` | Numeric pasal sequence has gaps |
| `REG-PP-4-2022` | Numeric pasal sequence has gaps |
| `REG-PP-66-2010` | Numeric pasal sequence has gaps |

Interpretasi warning:

- Warning ini banyak muncul pada dokumen perubahan/amandemen.
- Dokumen perubahan sering hanya memuat pasal tertentu dari regulasi lama, sehingga nomor pasal terlihat loncat.
- Warning ini tidak otomatis berarti parser gagal.

## 5. Hasil Parser Per Dokumen

| Regulasi | Pasal | Ayat | Bab | Bagian | Paragraf | Lampiran | Preamble | Penjelasan | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `REG-PP-13-2015` | 16 | 59 | 0 | 0 | 0 | 0 | Ya | Ya | WARN |
| `REG-PP-17-2010` | 222 | 654 | 18 | 33 | 40 | 0 | Ya | Ya | PASS |
| `REG-PP-18-2022` | 11 | 7 | 0 | 0 | 0 | 0 | Ya | Tidak | PASS |
| `REG-PP-19-2005` | 97 | 320 | 17 | 15 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-19-2017` | 19 | 41 | 0 | 0 | 0 | 0 | Ya | Ya | WARN |
| `REG-PP-37-2009` | 46 | 149 | 8 | 13 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-4-2022` | 11 | 26 | 0 | 0 | 0 | 0 | Ya | Ya | WARN |
| `REG-PP-41-2009` | 25 | 24 | 7 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-47-2008` | 16 | 52 | 9 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-48-2008` | 89 | 217 | 9 | 18 | 18 | 0 | Ya | Tidak | PASS |
| `REG-PP-55-2007` | 51 | 167 | 6 | 7 | 3 | 0 | Ya | Ya | PASS |
| `REG-PP-57-2021` | 59 | 152 | 8 | 11 | 4 | 0 | Ya | Ya | PASS |
| `REG-PP-66-2010` | 33 | 112 | 1 | 0 | 0 | 0 | Ya | Ya | WARN |
| `REG-PP-7-1977` | 26 | 1 | 4 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-74-2008` | 68 | 219 | 9 | 20 | 0 | 0 | Ya | Ya | PASS |
| `REG-PP-8-2009` | 2 | 0 | 0 | 0 | 0 | 0 | Ya | Tidak | PASS |
| `REG-UU-1-PNPS-1965` | 5 | 0 | 0 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-UU-12-2012` | 100 | 356 | 12 | 30 | 24 | 0 | Ya | Ya | PASS |
| `REG-UU-14-2005` | 84 | 220 | 8 | 17 | 0 | 0 | Ya | Ya | PASS |
| `REG-UU-20-2003` | 77 | 204 | 4 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-UU-9-2009` | 69 | 229 | 14 | 0 | 0 | 0 | Ya | Ya | PASS |
| `REG-UUD-1945` | 2 | 0 | 0 | 0 | 0 | 0 | Tidak | Ya | PASS, tetapi salah label |

Catatan lampiran:

- Parser mendukung node `Lampiran`, tetapi hanya jika ada heading `LAMPIRAN` yang benar-benar muncul sebagai bagian dokumen.
- Kata `Lampiran` yang hanya berupa rujukan dalam pasal tidak dijadikan node lampiran agar isi pasal tidak terpotong.
- Pada raw saat ini, `total_attachments = 0` untuk semua dokumen.

## 6. Triplet Extraction V2

Triplet v2 dibagi menjadi tiga tahap:

| Tahap | Script | Metode | LLM |
| --- | --- | --- | --- |
| Structural | `scripts/extraction/extract_triplets_v2_structural.py` | `structural` | Tidak |
| Explicit | `scripts/extraction/extract_triplets_v2_explicit.py` | `regex` | Tidak |
| Semantic | `scripts/extraction/extract_triplets_v2_semantic.py` | `llm` | Ya |

Output utama:

```text
data/triplets_v2/triplets_{regulasi_id}.json
```

Report proses:

```text
outputs/reports/v2/triplets_v2_structural_report.json
outputs/reports/v2/triplets_v2_explicit_report.json
outputs/reports/v2/triplets_v2_semantic_report.json
```

Catatan report semantic:

- Semantic dijalankan per dokumen, sehingga `triplets_v2_semantic_report.json` hanya menyimpan report run terakhir.
- Rekap final harus dihitung dari semua file `data/triplets_v2/triplets_REG-*.json`.

## 7. Total Triplets Saat Ini

Rekap final tidak memasukkan `REG-UUD-1945` karena file raw salah label.

| Method | Makna | Jumlah |
| --- | --- | ---: |
| `structural` | Relasi struktur dokumen | 17,223 |
| `regex` | Relasi eksplisit hukum/rule-based | 1,454 |
| `llm` | Relasi semantik dari LLM | 12,199 |
| **Total** | Semua triplet final | **30,876** |

Catatan perubahan:

- Total final turun dari `31,906` menjadi `30,876` setelah explicit regex dibersihkan.
- Pembersihan ini mengurangi false positive rujukan dari bagian penutup dan token OCR/format yang tidak valid.
- Tidak ada duplicate triplets pada hasil QC terakhir.

Distribusi predicate utama:

| Predicate | Jumlah |
| --- | ---: |
| `MEMILIKI_CHUNK` | 7,537 |
| `MENGATUR_TOPIK` | 6,057 |
| `MEMILIKI_AYAT` | 3,209 |
| `MEMILIKI_SUBITEM` | 2,889 |
| `MENGATUR_SUBJEK` | 2,122 |
| `MEMBEBANKAN_KEWAJIBAN` | 1,485 |
| `MENGACU_PADA` | 1,298 |
| `MENDEFINISIKAN` | 1,233 |
| `MEMILIKI_PASAL` | 1,126 |
| `MEMBERIKAN_HAK` | 1,111 |
| `MEMILIKI_PENJELASAN_PASAL` | 935 |
| `MENJELASKAN` | 935 |
| `MELARANG` | 191 |
| `MEMILIKI_BAGIAN` | 164 |
| `MEMILIKI_BAB` | 134 |
| `MEMILIKI_PARAGRAF` | 89 |
| `MEMILIKI_DASAR_HUKUM` | 54 |
| `MEMILIKI_KONSIDERANS` | 53 |
| `BERDASARKAN` | 49 |
| `MEMILIKI_DIKTUM` | 42 |
| `MULAI_BERLAKU` | 36 |
| `MENGAMANDEMEN` | 27 |
| `MEMILIKI_PENUTUP` | 21 |
| `DIUNDANGKAN_PADA` | 20 |
| `MEMILIKI_PENJELASAN` | 18 |
| `MEMILIKI_PENJELASAN_UMUM` | 17 |
| `MENCABUT` | 16 |
| `MENCABUT_PASAL` | 8 |

## 8. QC Triplets V2

QC teknis terakhir dijalankan terhadap semua `triplets_v2` final dengan `REG-UUD-1945` dikecualikan.

| Metric | Nilai |
| --- | ---: |
| Documents checked | 21 |
| Documents excluded | 1 |
| Total triplets checked | 30,876 |
| Total errors | 0 |
| Total warnings | 321 |
| Duplicate triplets | 0 |

Output QC:

```text
outputs/reports/v2/triplets_v2_qc_report.json
outputs/reports/v2/triplets_v2_qc_issues.csv
```

Interpretasi:

- `total_errors = 0` berarti seluruh triplet final lolos validasi teknis predicate dan type berdasarkan ontology v2.
- `warnings = 321` bukan blocker; warning dipakai untuk audit kualitas seperti confidence rendah, target semantic yang perlu sampling, atau catatan traceability.
- QC ini belum menggantikan review hukum substantif. Untuk paper, sampling manual tetap perlu pada bagian semantic/conflict candidate.

## 9. Triplets Per Dokumen

Rekap berikut tidak memasukkan `REG-UUD-1945`.

| Regulasi | Total Triplets | Structural | Explicit/Regex | Semantic/LLM |
| --- | ---: | ---: | ---: | ---: |
| `REG-PP-13-2015` | 673 | 378 | 78 | 217 |
| `REG-PP-17-2010` | 7,035 | 4,021 | 277 | 2,737 |
| `REG-PP-18-2022` | 150 | 57 | 56 | 37 |
| `REG-PP-19-2005` | 2,740 | 1,520 | 51 | 1,169 |
| `REG-PP-19-2017` | 752 | 423 | 100 | 229 |
| `REG-PP-37-2009` | 1,327 | 723 | 35 | 569 |
| `REG-PP-4-2022` | 456 | 268 | 59 | 129 |
| `REG-PP-41-2009` | 395 | 231 | 27 | 137 |
| `REG-PP-47-2008` | 419 | 234 | 18 | 167 |
| `REG-PP-48-2008` | 2,087 | 1,141 | 45 | 901 |
| `REG-PP-55-2007` | 1,076 | 628 | 28 | 420 |
| `REG-PP-57-2021` | 1,601 | 996 | 90 | 515 |
| `REG-PP-66-2010` | 1,387 | 718 | 149 | 520 |
| `REG-PP-7-1977` | 215 | 91 | 28 | 96 |
| `REG-PP-74-2008` | 2,336 | 1,276 | 86 | 974 |
| `REG-PP-8-2009` | 73 | 46 | 22 | 5 |
| `REG-UU-1-PNPS-1965` | 100 | 44 | 35 | 21 |
| `REG-UU-12-2012` | 2,836 | 1,627 | 77 | 1,132 |
| `REG-UU-14-2005` | 1,914 | 1,063 | 94 | 757 |
| `REG-UU-20-2003` | 1,369 | 658 | 49 | 662 |
| `REG-UU-9-2009` | 1,935 | 1,080 | 50 | 805 |

## 10. Makna Setiap Jenis Triplet

Satu triplet adalah satu edge/relasi graph:

```json
{
  "source_id": "PSL-REG-PP-57-2021-3",
  "source_type": "Pasal",
  "predicate": "MENGATUR_TOPIK",
  "target_id": "TOPIK-standar-nasional-pendidikan",
  "target_type": "Topik",
  "extraction_method": "llm",
  "evidence": {
    "text": "...",
    "chunk_ids": ["CHK-REG-PP-57-2021-0005"]
  }
}
```

Interpretasi:

```text
Pasal REG-PP-57-2021 Pasal 3
MENGATUR_TOPIK
Standar Nasional Pendidikan
```

Jenis metode:

| `extraction_method` | Makna |
| --- | --- |
| `structural` | Relasi deterministik dari hasil parser, misalnya `MEMILIKI_PASAL`, `MEMILIKI_AYAT`, `MEMILIKI_CHUNK` |
| `regex` | Relasi eksplisit/rule-based, misalnya `BERDASARKAN`, `MENGACU_PADA`, `MENGAMANDEMEN`, `MENCABUT` |
| `llm` | Relasi semantik dari LLM, misalnya `MENGATUR_TOPIK`, `MENGATUR_SUBJEK`, `MEMBEBANKAN_KEWAJIBAN` |

## 11. Catatan Kualitas dan Risiko

Catatan yang perlu diingat sebelum build graph:

1. `REG-UUD-1945` harus di-skip atau diganti file yang benar.
2. Semantic triplets perlu QC sampling karena dihasilkan LLM.
3. Explicit `MENGACU_PADA` sudah dibersihkan dari noise besar, tetapi rujukan kompleks tetap perlu sampling.
4. Warning pasal sequence pada dokumen perubahan tidak otomatis error, tetapi perlu dicatat.
5. Lampiran belum muncul sebagai node aktif karena raw saat ini tidak punya heading lampiran yang jelas.
6. File report semantic hanya menunjukkan run terakhir karena semantic dijalankan per dokumen.

## 12. Graph V2 Build dan QC

Output graph v2:

```text
indexes/graph_v2/graph_index.pkl
indexes/graph_v2/nodes.jsonl
indexes/graph_v2/edges.jsonl
indexes/graph_v2/summary.json
```

Script:

```text
scripts/graph/build_graph_v2.py
scripts/evaluation/audit_graph_v2.py
```

Hasil build dan QC graph v2:

| Metric | Nilai |
| --- | ---: |
| Nodes | 21,757 |
| Edges | 30,876 |
| Stub nodes | 5,981 |
| Orphan chunks | 0 |
| QC errors | 0 |
| QC warnings | 0 |

Output QC:

```text
outputs/reports/v2/graph_v2_qc_report.json
outputs/reports/v2/graph_v2_qc_issues.csv
```

Catatan:

- `stub nodes` adalah node yang muncul dari triplet tetapi tidak ada sebagai struktur utama di `processed_v2`, misalnya `Topik`, `Subjek_Hukum`, `Peristiwa_Hukum`, regulasi rujukan di luar korpus, atau pasal rujukan eksternal.
- Stub bukan error; justru ini membuat edge hukum dan semantik tetap tersambung di graph.

## 13. Embedding dan Vector Index

Output embedding model:

```text
indexes/vector_v2_model/vector_index.pkl
indexes/vector_v2_model/embeddings.jsonl
indexes/vector_v2_model/metadata.jsonl
indexes/vector_v2_model/summary.json
```

Script:

```text
scripts/graph/build_embedding_index_v2.py
scripts/graph/search_embedding_index_v2.py
```

Hasil full embedding:

| Metric | Nilai |
| --- | ---: |
| Backend | `sentence_transformers` |
| Model | `intfloat/multilingual-e5-base` |
| Chunks embedded | 7,537 |
| Embedding dimension | 768 |
| Batch size | 32 |
| Excluded docs | 1 |
| Output `embeddings.jsonl` | 123 MB |
| Output `vector_index.pkl` | 57 MB |

Catatan:

- `chunks_v2` tetap menjadi source of truth untuk citation.
- Script embedding model membuat `embedding_text` yang diperkaya konteks: regulasi, judul, reference label, section type, dan teks chunk.
- Backend lokal memakai `sentence-transformers`; passage chunk diberi prefix `passage:` dan query diberi prefix `query:`.
- Baseline `scripts/graph/build_vector_index_v2.py` memakai hashing TF-IDF dan hanya dipakai untuk smoke test, bukan final riset.

Smoke test semantic search:

| Query | Hasil teratas |
| --- | --- |
| `standar nasional pendidikan kurikulum peserta didik` | Chunk definisi peserta didik dan penjelasan Standar Nasional Pendidikan |
| `wajib belajar pendidikan dasar` | `REG-PP-47-2008` Pasal 6 ayat (4), Pasal 5 ayat (1), dan definisi wajib belajar |

## 14. GraphRAG Retrieval V2

Script:

```text
scripts/graph/graphrag_retriever_v2.py
```

Input:

```text
indexes/graph_v2/graph_index.pkl
indexes/vector_v2_model/vector_index.pkl
```

Fungsi utama:

- Exact reference retrieval untuk query seperti `PP 47 Tahun 2008 Pasal 5 ayat (1)`.
- Semantic vector search memakai embedding `intfloat/multilingual-e5-base`.
- Graph expansion dari seed node/chunk ke parent struktur, relasi hukum eksplisit, dan relasi semantik.
- Context packing dengan `regulasi_id`, `source_node_id`, `chunk_id`, `reference_label`, dan teks.

Smoke test:

| Query | Hasil |
| --- | --- |
| `PP 47 Tahun 2008 Pasal 5 ayat (1)` | Resolve ke `AYT-REG-PP-47-2008-5-1`; graph expansion membawa parent pasal, topik, subjek, dan kewajiban |
| `wajib belajar pendidikan dasar` | Top result dari `REG-PP-47-2008`, termasuk Pasal 6 ayat (4), Pasal 5 ayat (1), dan definisi pendidikan dasar |

Contoh command GPU:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/extended-home/rifqicahyono/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/graph/graphrag_retriever_v2.py --device cuda --top-k 5 "wajib belajar pendidikan dasar"
```

Catatan:

- Di environment sandbox, smoke test dijalankan dengan `--device cpu`.
- Di tmux/server user, GPU bisa dipakai dengan `CUDA_VISIBLE_DEVICES=1`.
- Query cleaning dan explainable rerank awal sudah dibuat.
- Hasil query exact seperti `PP 47 Tahun 2008 Pasal 5 ayat (1)` sebelumnya menunjukkan kebutuhan query cleaning/rerank karena vector search bisa tertarik ke frasa angka pasal seperti `Pasal 5 ayat (2)` pada bagian `Mengingat`.

Perbaikan retrieval yang sudah aktif:

- Query cleaning memisahkan referensi formal dari intent substantif.
- Exact-only query otomatis skip vector search.
- Explainable feature-based reranking aktif secara default.
- Rerank ini bukan definisi wajib semua GraphRAG, tetapi strategi domain-specific untuk GraphRAG hukum.
- Rerank tidak memakai model tambahan; skor dihitung dari fitur yang sudah tersedia di index graph dan vector.
- Fitur awal: embedding similarity, exact legal reference, graph proximity, metadata regulasi, dan section priority.

Smoke test rerank:

| Query | Hasil |
| --- | --- |
| `PP 47 Tahun 2008 Pasal 5 ayat (1)` | `vector_skipped_reason = exact_reference_only`, exact match ke `AYT-REG-PP-47-2008-5-1` |
| `apa kewajiban satuan pendidikan menurut PP 47 Tahun 2008 Pasal 5 ayat (1)` | `vector_query = apa kewajiban satuan pendidikan`; top vector result menjadi `REG-PP-47-2008 Pasal 5 ayat (1)` |

Flag pembanding baseline:

```bash
--no-query-cleaning --no-rerank --no-auto-skip-vector-exact-only
```

Rationale paper:

```text
The retrieval layer uses legal-reference-aware graph retrieval. Exact legal references are resolved as graph anchors, while dense retrieval captures semantic similarity. Candidate contexts are reranked using explainable graph and metadata features so that formal citation tokens do not dominate semantic retrieval.
```

## 15. Next Step

Urutan kerja berikutnya yang disarankan:

1. Bekukan tabel hasil final `Bagian 1` di dokumen penelitian.
2. Mulai desain curated evaluation set `Bagian 2`.
3. Tentukan format final setiap item structural legal reasoning.
4. Mulai kurasi 50-100 item awal untuk prototype `Bagian 2`.

Catatan:

- Generation/explanation layer ditandai opsional dan belum dikerjakan.
- Layer tersebut tidak menjadi mekanisme utama conflict detection.
- `Bagian 1 - Baseline Validation` sudah selesai.
- Fokus berikutnya secara resmi adalah `Bagian 2 - Structural Legal Reasoning`.
- Script inference baru yang sudah tersedia: `scripts/training/infer_lora_rag_v2.py` dan `scripts/training/infer_lora_graphrag_v2.py`.
- Script zero-shot yang sudah disiapkan: `scripts/training/infer_zeroshot.py`, `scripts/training/infer_zeroshot_rag_v2.py`, dan `scripts/training/infer_zeroshot_graphrag_v2.py`.
- Script evaluasi yang sudah tersedia: `scripts/evaluation/eval_predictions_qa.py`.
- Smoke test `--limit 10` untuk `RAG` dan `GraphRAG` sudah berhasil dijalankan end-to-end.
- Hasil final `Bagian 1`:
  `Fine-tune only 512` tetap terbaik, `GraphRAG v2` konsisten di atas `RAG v2`, tetapi keduanya masih di bawah baseline lama pada dataset QA.
- Eksperimen lanjutan yang disarankan mentor adalah menambahkan pembanding `zero-shot`, `zero-shot + RAG`, dan `zero-shot + GraphRAG`.

## 16. Command Audit Cepat

Cek total triplets final:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from collections import Counter

methods = Counter()
total = 0
for p in Path("data/triplets_v2").glob("triplets_REG-*.json"):
    rid = p.stem.replace("triplets_", "")
    if rid == "REG-UUD-1945":
        continue
    d = json.load(open(p))
    total += len(d["triplets"])
    for x in d["triplets"]:
        methods[x["extraction_method"]] += 1

print("total_triplets:", total)
print("methods:", dict(methods))
PY
```

Cek predicate distribution:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from collections import Counter

predicates = Counter()
for p in Path("data/triplets_v2").glob("triplets_REG-*.json"):
    rid = p.stem.replace("triplets_", "")
    if rid == "REG-UUD-1945":
        continue
    d = json.load(open(p))
    for x in d["triplets"]:
        predicates[x["predicate"]] += 1

for pred, count in predicates.most_common():
    print(pred, count)
PY
```
