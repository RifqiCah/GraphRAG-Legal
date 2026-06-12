# PSKP Knowledge Graph dan GraphRAG Rerun Plan

Dokumen ini menjelaskan rencana rerun pipeline dari raw PDF sampai Knowledge Graph, GraphRAG, dataset, dan evaluasi. Targetnya bukan sekadar menghasilkan triplet, tetapi membangun graph hukum yang punya struktur dokumen, relasi hukum, chunk teks, embedding, retrieval, dan evaluasi yang jelas.

Catatan penting per 2026-05-28:

1. Struktur eksperimen resmi sekarang dipisah menjadi `Bagian 1 - Baseline Validation` dan `Bagian 2 - Main Contribution: Structural Legal Reasoning`.
2. Dokumen operasional paling detail untuk dua bagian tersebut ada di `docs/experiment_part1_part2_execution.md`.
3. Dokumen ini tetap dipakai sebagai plan teknis pipeline KG/GraphRAG v2.

## Ringkasan Eksekusi

Urutan kerja paling efektif:

1. Perbaiki schema v2: tambah node `Konsiderans`, `Dasar_Hukum`, `Diktum`, `Penjelasan`, `Lampiran`, `Chunk`, dan edge pendukung.
2. Perbaiki parser v2: ambil `Menimbang`, `Mengingat`, `Memutuskan`, `Menetapkan`, `Pasal`, `Ayat`, `Penjelasan`, `Lampiran`, dan `Penutup`.
3. Rerun semua PDF: hasilkan `processed` v2 dan QC report.
4. Generate chunk: satu chunk per unit hukum yang logis.
5. Extract triplets: structural, explicit regex, dan semantic LLM.
6. Build graph: local graph index.
7. Build embedding: simpan chunk embedding ke vector index.
8. Implement GraphRAG retrieval: exact reference + embedding search + graph expansion + rerank.
9. Buat dataset: extraction, retrieval, QA, conflict, dan regression.
10. Evaluasi: extraction, graph, retrieval, generation, dan conflict detection.

Status eksekusi terakhir per 2026-05-28:

| Tahap | Status | Catatan |
| --- | --- | --- |
| Schema/ontology v2 | Selesai | Node dan edge v2 sudah dibuat di `ontology/v2/` |
| Parser dan chunker v2 | Selesai | 22 dokumen utama diproses, 0 error parser, 4 warning non-fatal |
| Triplets structural | Selesai | Deterministic, tanpa LLM |
| Triplets explicit | Selesai | Regex/rule-based, tanpa LLM |
| Triplets semantic | Selesai | LLM-assisted, berbasis chunk |
| QC triplets v2 | Selesai | 21 dokumen checked, 1 excluded, 30,876 triplets, 0 error |
| Graph v2 | Selesai | 21,757 nodes, 30,876 edges, QC 0 error |
| Embedding/vector index | Selesai | `intfloat/multilingual-e5-base`, 7,537 chunks |
| GraphRAG retrieval | Selesai awal | Exact reference + vector search + graph expansion |
| Baseline validation Part 1 | Selesai | `Fine-tune only 512` > `GraphRAG v2` > `RAG v2` pada dataset QA lama |
| Curated eval set Part 2 | Belum | Dibuat setelah baseline validation Part 1 |

Catatan scope:

1. `REG-UUD-1945` dikecualikan dari final GraphRAG karena file raw salah label.
2. Total final saat ini tidak memasukkan `REG-UUD-1945`.
3. Detail hasil eksekusi ada di `docs/kg_graphrag_v2_results.md`.
4. Checklist operasional ada di `docs/kg_graphrag_execution_checklist.md`.
5. Rencana eksperimen part-by-part ada di `docs/experiment_part1_part2_execution.md`.

## 1. Tujuan Akhir

Output utama yang harus tersedia:

1. `data/processed_v2/*.json`: hasil parsing struktur dokumen hukum.
2. `data/chunks_v2/*.jsonl`: chunk teks siap embedding dan retrieval.
3. `data/triplets_v2/*.json`: relasi graph tervalidasi.
4. `indexes/graph_v2/graph_index.pkl`: graph index lokal.
5. `indexes/vector_v2/*`: vector index atau export untuk Chroma/Milvus/Qdrant.
6. Dataset evaluasi: QA, retrieval relevance, dan conflict/ground-truth cases.
7. Report evaluasi: extraction quality, graph quality, retrieval quality, dan generation quality.

Detail kontrak output v2 ada di `docs/v2_output_contract.md`.

## 2. Research Task dan Dataset Target

Masalah utama yang diselesaikan dalam project ini sekarang dibagi menjadi dua tingkat:

1. `Bagian 1 - Baseline Validation`: memvalidasi setup fine-tuning dan retrieval pada dataset QA lama.
2. `Bagian 2 - Main Contribution`: menguji GraphRAG pada `structural legal reasoning` untuk regulasi PSKP.

Dengan struktur baru ini, `conflict analysis` tetap penting, tetapi posisinya menjadi salah satu task di dalam `structural legal reasoning`, bukan satu-satunya fokus eksperimen.

Dalam paper, task keseluruhan bisa ditulis sebagai:

1. `Legal Knowledge Graph Construction`: membangun graph dari dokumen regulasi Indonesia.
2. `Baseline Legal QA Validation`: memvalidasi setup model dan retrieval terhadap dataset QA lama.
3. `Structural Legal Reasoning with GraphRAG`: menguji reasoning berbasis hirarki, rujukan, amendment, overlap, conflict, dan multi-hop.
4. `Legal Retrieval Evaluation`: menguji apakah sistem menemukan pasal/chunk/node yang relevan.

Hierarchy dataset yang lebih sesuai dengan struktur baru:

```text
Bagian 1:
Baseline validation on legacy QA dataset

Bagian 2:
Structural legal reasoning evaluation set

Task utama Bagian 2:
1. Hierarchy reasoning
2. Reference reasoning
3. Amendment reasoning
4. Overlap reasoning
5. Conflict analysis
6. Multi-hop reasoning

Subtask konflik:
1. Direct_Contradiction
2. Hierarchical_Conflict
3. Temporal_Inconsistency
4. Regulatory_Overlap

Dataset pendukung:
1. Retrieval dataset
2. QA dataset
3. Extraction dataset
4. Regression dataset
```

Untuk subtask `conflict analysis`, kategori yang dipakai tetap 4 konflik berikut:

| Kategori | Definisi | Contoh |
| --- | --- | --- |
| `Direct_Contradiction` | Dua pasal dari regulasi setara mengatur hal sama dengan ketentuan yang bertentangan. | Pasal A mewajibkan X, pasal B dari regulasi setara melarang X. |
| `Hierarchical_Conflict` | Pasal dari regulasi lebih rendah bertentangan dengan regulasi lebih tinggi. | Pasal PP bertentangan dengan UU induknya. |
| `Temporal_Inconsistency` | Regulasi baru mengatur ulang topik lama tanpa mencabut/mengubah ketentuan lama yang bertentangan. | PP 2022 mengatur ulang topik tanpa mencabut PP lama yang relevan. |
| `Regulatory_Overlap` | Dua atau lebih regulasi mengatur topik yang sama tanpa hierarki atau status perubahan yang jelas. | Dua PP berbeda mengatur standar kompetensi guru secara tumpang tindih. |

Konsekuensi ke schema v2:

1. Perlu node struktur hukum yang detail agar konflik bisa dilacak sampai `Pasal`, `Ayat`, atau `SubItem`.
2. Perlu node `Dasar_Hukum` agar `Hierarchical_Conflict` bisa diuji dari relasi dasar hukum dan hierarki regulasi.
3. Perlu edge `MENGAMANDEMEN`, `MENCABUT`, dan `MENCABUT_PASAL` agar `Temporal_Inconsistency` bisa dibedakan dari perubahan yang sah.
4. Perlu node `Topik` dan `Subjek_Hukum` agar `Direct_Contradiction` dan `Regulatory_Overlap` bisa dicari dari kesamaan objek pengaturan.
5. Perlu `Chunk` dan embedding agar kandidat konflik bisa ditemukan walau istilahnya tidak sama persis.

Dataset turunan yang perlu dibuat:

1. `conflict_candidates`: kandidat konflik otomatis dari graph.
2. `conflict_ground_truth`: konflik yang sudah direview manual.
3. `retrieval_ground_truth`: query dan chunk/node relevan.
4. `qa_ground_truth`: pertanyaan, jawaban, dan citation.
5. `extraction_gold`: sampel teks dan triplet benar untuk mengukur kualitas ekstraksi.

Field minimal untuk conflict dataset:

```json
{
  "id": "CONFLICT-0001",
  "category": "Hierarchical_Conflict",
  "source_node_id": "PSL-REG-PP-...",
  "target_node_id": "PSL-REG-UU-...",
  "shared_topic_id": "TOPIK-...",
  "conflict_summary": "...",
  "evidence_chunk_ids": ["CHK-..."],
  "gold_label": "conflict",
  "review_status": "reviewed"
}
```

## 3. Raw Document Layer

Input:

1. PDF regulasi di `data/raw/`.
2. Metadata dokumen di `data/corpus_manifest.json`.
3. Metadata minimal: `regulasi_id`, `jenis`, `nomor`, `tahun`, `judul`, `status`, `path_raw`, `url_sumber`.

Langkah:

1. Pastikan semua PDF ada dan bisa diekstrak teksnya.
2. Tandai dokumen hasil scan/OCR lemah.
3. Simpan log ekstraksi per dokumen.
4. Jangan langsung buang bagian sebelum `Pasal`, karena `Menimbang`, `Mengingat`, dan `Menetapkan` penting untuk graph.

Output:

1. Raw text per dokumen.
2. Status ekstraksi: `ok`, `scan_detected`, `file_missing`, `parse_warning`.

## 4. Document Parsing Layer

Parser perlu mengambil seluruh struktur hukum, bukan hanya pasal.

Struktur yang perlu diparse:

1. `judul`: jenis, nomor, tahun, tentang.
2. `pembukaan`: frasa pembuka seperti `Dengan Rahmat Tuhan Yang Maha Esa`.
3. `konsiderans_menimbang`: alasan pembentukan regulasi.
4. `dasar_hukum_mengingat`: daftar dasar hukum.
5. `diktum_memutuskan`: bagian `MEMUTUSKAN`.
6. `diktum_menetapkan`: judul regulasi yang ditetapkan.
7. `batang_tubuh`: `Bab`, `Bagian`, `Paragraf`, `Pasal`, `Ayat`, `Huruf`, `Angka`.
8. `penjelasan`: bagian penjelasan dokumen, biasanya berisi `Umum` dan `Pasal demi Pasal`.
9. `lampiran`: tabel, daftar, atau materi lampiran.
10. `penutup`: tanggal ditetapkan, tanggal diundangkan, pejabat, lembaran negara.

Output JSON v2 yang disarankan:

```json
{
  "regulasi_id": "REG-PP-8-2009",
  "metadata": {},
  "preamble": {
    "menimbang": [],
    "mengingat": [],
    "memutuskan": "",
    "menetapkan": ""
  },
  "body": {
    "bab": [],
    "pasal": []
  },
  "explanation": {
    "umum": "",
    "pasal_demi_pasal": []
  },
  "attachments": [],
  "closing": {}
}
```

## 5. Knowledge Graph Design

### Node

Node inti:

1. `Regulasi`: satu dokumen hukum.
2. `Bab`: bab dalam regulasi.
3. `Bagian`: bagian dalam bab.
4. `Pasal`: pasal atau ayat.
5. `SubItem`: huruf/angka dalam ayat.
6. `Konsiderans`: item `Menimbang`.
7. `Dasar_Hukum`: item `Mengingat`.
8. `Diktum`: `Memutuskan` dan `Menetapkan`.
9. `Penjelasan`: bagian penjelasan dokumen, bukan norma utama.
10. `Penjelasan_Umum`: uraian umum dalam bagian penjelasan.
11. `Penjelasan_Pasal`: penjelasan pasal demi pasal.
12. `Lampiran`: lampiran regulasi.
13. `Chunk`: potongan teks untuk retrieval.
14. `Subjek_Hukum`: entitas hukum yang diatur.
15. `Topik`: konsep/topik hukum.

Node ID yang disarankan:

1. `REG-PP-8-2009`
2. `BAB-REG-PP-8-2009-I`
3. `PSL-REG-PP-8-2009-I-0`
4. `KONS-REG-PP-8-2009-MENIMBANG-a`
5. `DASAR-REG-PP-8-2009-MENGINGAT-1`
6. `DIKTUM-REG-PP-8-2009-MENETAPKAN`
7. `PJL-REG-PP-8-2009`
8. `PJL-REG-PP-8-2009-UMUM`
9. `PJL-REG-PP-8-2009-PASAL-I`
10. `LAMP-REG-PP-8-2009-1`
11. `CHK-REG-PP-8-2009-0001`

### Edge

Relasi struktural:

1. `MEMILIKI_BAB`: `Regulasi -> Bab`
2. `MEMILIKI_BAGIAN`: `Bab -> Bagian`
3. `MEMILIKI_PASAL`: `Regulasi/Bab/Bagian -> Pasal`
4. `MEMILIKI_SUBITEM`: `Pasal -> SubItem`
5. `MEMILIKI_PENJELASAN`: `Regulasi -> Penjelasan`
6. `MEMILIKI_PENJELASAN_UMUM`: `Penjelasan -> Penjelasan_Umum`
7. `MEMILIKI_PENJELASAN_PASAL`: `Penjelasan -> Penjelasan_Pasal`
8. `MENJELASKAN`: `Penjelasan_Pasal -> Pasal`
9. `MEMILIKI_LAMPIRAN`: `Regulasi -> Lampiran`
10. `MEMILIKI_CHUNK`: `Regulasi/Pasal/Konsiderans/Dasar_Hukum/Diktum/Penjelasan/Lampiran -> Chunk`

Relasi hukum eksplisit:

1. `MENGACU_PADA`: rujukan eksplisit ke regulasi/pasal lain.
2. `BERDASARKAN`: `Dasar_Hukum -> Regulasi/Pasal`.
3. `MENGAMANDEMEN`: regulasi mengubah regulasi/pasal lain.
4. `MENCABUT`: regulasi mencabut regulasi lain.
5. `MENCABUT_PASAL`: regulasi mencabut pasal tertentu.
6. `MULAI_BERLAKU`: regulasi/pasal berlaku sejak tanggal tertentu.
7. `DIUNDANGKAN_PADA`: regulasi diundangkan pada tanggal tertentu.

Relasi semantik:

1. `MENGATUR_TOPIK`: pasal/chunk mengatur topik.
2. `MENGATUR_SUBJEK`: pasal/chunk mengatur subjek hukum.
3. `MEMBERIKAN_HAK`: pasal/subitem memberi hak kepada subjek.
4. `MEMBEBANKAN_KEWAJIBAN`: pasal/subitem memberi kewajiban.
5. `MELARANG`: pasal/subitem memuat larangan.
6. `MENDEFINISIKAN`: pasal/subitem mendefinisikan istilah.
7. `BERTENTANGAN_DENGAN`: hanya untuk ground truth/evaluasi konflik.

## 6. Extraction Pipeline

Urutan ekstraksi:

1. Extract text dari PDF.
2. Clean text tanpa menghapus struktur hukum.
3. Parse preamble: `Menimbang`, `Mengingat`, `Memutuskan`, `Menetapkan`.
4. Parse body: `Bab`, `Bagian`, `Pasal`, `Ayat`, `Huruf`, `Angka`.
5. Parse penjelasan dokumen, lampiran, dan penutup.
6. Buat node struktural.
7. Buat edge struktural.
8. Regex extraction untuk rujukan eksplisit.
9. LLM extraction untuk relasi semantik yang tidak stabil ditangkap regex.
10. Dedup triplets.
11. Validasi terhadap schema dan constraints.
12. Simpan processed JSON, chunk JSONL, triplet JSON, dan report QC.

Quality check wajib:

1. Jumlah pasal tidak boleh nol jika teks mengandung `Pasal`.
2. Bagian preamble harus terdeteksi jika teks mengandung `Menimbang` atau `Mengingat`.
3. Bagian `Penjelasan` harus terdeteksi jika dokumen mengandung heading `PENJELASAN`.
4. Rujukan `Nomor X Tahun Y` di konteks PP/UU harus dinormalisasi ke `REG-*`.
5. Node ID harus unik.
6. Edge harus sesuai tipe source-target.
7. Triplet invalid harus disimpan untuk audit, bukan diam-diam dibuang.

## 7. Chunking Strategy

Chunk tetap diperlukan walaupun memakai graph.

Unit chunk utama:

1. Satu item `Menimbang`.
2. Satu item `Mengingat`.
3. Satu `Pasal` tanpa ayat.
4. Satu `Ayat` jika pasal punya ayat.
5. Satu `Huruf` atau `Angka` jika ayat sangat panjang.
6. Satu bagian `Penjelasan_Umum`.
7. Satu item `Penjelasan_Pasal`.
8. Satu bagian lampiran yang logis.

Metadata chunk:

1. `chunk_id`
2. `regulasi_id`
3. `source_node_id`
4. `section_type`: `menimbang`, `mengingat`, `pasal`, `ayat`, `huruf`, `penjelasan_umum`, `penjelasan_pasal`, `lampiran`, `penutup`
5. `reference_label`: contoh `PP 8/2009 Pasal I`
6. `text`
7. `char_start`, `char_end`
8. `page_start`, `page_end` jika tersedia
9. `status`
10. `created_at`

Ukuran chunk:

1. Target: 300 sampai 900 token.
2. Maksimum: 1200 token.
3. Overlap kecil hanya untuk chunk berbasis paragraf panjang, sekitar 50 sampai 100 token.
4. Untuk pasal/ayat hukum, prioritaskan batas struktural dibanding overlap generik.

## 8. Embedding dan Vector Index

Tujuan:

1. Mendukung semantic search.
2. Menemukan chunk relevan walau query tidak menyebut referensi pasal tepat.
3. Melengkapi graph traversal dengan teks yang benar-benar mirip secara makna.

Langkah:

1. Baca `data/chunks_v2/*.jsonl`.
2. Buat retrieval text yang diperkaya konteks: regulasi, judul, `reference_label`, `section_type`, dan `text`.
3. Buat embedding model untuk retrieval text tersebut.
4. Simpan embedding ke vector DB atau index lokal.
5. Simpan metadata chunk agar hasil search bisa ditelusuri ke node graph.

Script saat ini:

1. `scripts/graph/build_embedding_index_v2.py`: embedding model via OpenAI-compatible API.
2. `scripts/graph/search_embedding_index_v2.py`: search top-k dengan model embedding yang sama.
3. `scripts/graph/build_vector_index_v2.py`: baseline hashing TF-IDF tanpa dependency, hanya untuk smoke test.
4. `scripts/graph/search_vector_index_v2.py`: search baseline.

Pilihan storage:

1. ChromaDB: cocok untuk local development.
2. Qdrant: bagus untuk service vector DB yang stabil.
3. Milvus: cocok jika data membesar.
4. FAISS: cocok untuk index lokal sederhana.

Rekomendasi awal:

1. Pakai ChromaDB atau FAISS dulu untuk eksperimen lokal.
2. Simpan `chunk_id` sebagai primary key.
3. Simpan `source_node_id` agar vector result bisa diekspansi via graph.

## 9. Graph Index Lokal

1. Build graph dari processed JSON dan triplets.
2. Simpan ke `indexes/graph_v2/graph_index.pkl`.
3. Gunakan untuk audit cepat dan retrieval lokal.

## 10. GraphRAG Retrieval

Retrieval di GraphRAG mengambil konteks dari graph dan teks pendukung. Embedding chunk, metadata, exact reference matching, dan graph expansion dipakai sebagai bagian dari proses retrieval dan augmentation.

Tahap retrieval:

1. Query understanding: ambil regulasi, pasal, tahun, topik, dan intent.
2. Exact reference search: jika user menyebut `PP 8 Tahun 2009 Pasal I`, langsung resolve ke node.
3. Embedding search: cari chunk semantik paling relevan.
4. Graph expansion: dari node/chunk hasil retrieval, ambil tetangga 1 sampai 2 hop.
5. Reranking opsional: gabungkan skor vector, exact match, relasi graph, dan metadata hukum jika hasil top-k perlu dipadatkan.
6. Context packing: pilih chunk terbaik agar prompt tidak terlalu panjang.
7. Citation building: sertakan `reference_label`, `chunk_id`, dan `source_node_id`.

Implementasi awal:

```text
scripts/graph/graphrag_retriever_v2.py
```

Status implementasi:

1. Exact reference search sudah aktif.
2. Embedding search sudah aktif memakai `indexes/vector_v2_model/`.
3. Graph expansion 1-hop sudah aktif dari exact node dan source node hasil vector search.
4. Context packing sudah aktif dengan citation internal.
5. Query cleaning dan explainable rerank awal sudah aktif; evaluasi kuantitatif dilakukan setelah dataset retrieval/konflik tersedia.

Catatan scope:

- Tidak semua GraphRAG harus memakai reranking seperti ini.
- Reranking di project ini dipakai karena domain hukum punya query yang sering mencampur referensi formal dan intent substantif.
- Contoh: `PP 47 Tahun 2008 Pasal 5 ayat (1)` adalah referensi formal, sedangkan `kewajiban satuan pendidikan` adalah intent substantif.
- Jika referensi formal langsung masuk ke vector search, hasil bisa bias ke angka dan frasa pasal, bukan substansi norma.
- Karena itu, strategi yang dipakai adalah `legal-reference-aware graph retrieval`, bukan perluasan tujuan di luar GraphRAG.

Reranking yang dipakai adalah explainable feature-based reranking, bukan neural reranker tambahan dan bukan model learning-to-rank yang dilatih.

Formula awal yang direncanakan:

```text
context_score =
  alpha * embedding_similarity +
  beta  * exact_reference_score +
  gamma * graph_proximity_score +
  delta * metadata_priority_score +
  eta   * section_priority_score
```

Makna fitur:

1. `embedding_similarity`: skor similarity dari embedding E5.
2. `exact_reference_score`: kecocokan dengan regulasi/pasal/ayat eksplisit.
3. `graph_proximity_score`: kedekatan node kandidat dengan node exact di graph.
4. `metadata_priority_score`: prior seperti regulasi sama, jenis regulasi, status, dan tahun.
5. `section_priority_score`: prior bagian dokumen, misalnya `pasal`/`ayat` lebih relevan daripada `mengingat` untuk pertanyaan norma.

Referensi konseptual:

1. Lewis et al. 2020, `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks`: dasar RAG.
2. Karpukhin et al. 2020, `Dense Passage Retrieval for Open-Domain Question Answering`: dasar dense/vector retrieval.
3. `Graph Retrieval-Augmented Generation: A Survey`: dasar graph-guided retrieval dan graph-enhanced generation.
4. Burges, `From RankNet to LambdaRank to LambdaMART`: inspirasi feature-based ranking/learning-to-rank.

Klaim paper yang aman:

```text
This study applies an explainable feature-based reranking strategy inspired by learning-to-rank. The reranker is not trained; it combines dense embedding similarity with legal-reference exactness, graph proximity, regulation metadata, and section-type priority.
```

Status implementasi:

1. Exact-only query otomatis skip vector search.
2. Query campuran memakai cleaned vector query.
3. Candidate vector diambil lebih banyak dari `top_k`, lalu direrank menjadi `top_k` final.
4. Output menyimpan `vector_score` dan `rerank_features` agar keputusan ranking bisa diaudit.
5. Baseline tanpa cleaning/rerank tetap bisa dijalankan dengan flag:

```bash
--no-query-cleaning --no-rerank --no-auto-skip-vector-exact-only
```

Context yang dikirim ke LLM:

1. Pertanyaan user.
2. Chunk teks relevan.
3. Metadata sumber.
4. Relasi hukum penting.
5. Instruksi agar jawaban hanya memakai konteks.

## 11. Optional Conflict Explanation Layer

Status: opsional dan belum dikerjakan.

Layer ini bukan mekanisme utama untuk deteksi konflik. Deteksi konflik tetap dilakukan lewat KG, retrieval, candidate generation, dataset, dan evaluasi. Layer ini hanya dipakai jika nanti kita ingin membuat penjelasan natural language dari evidence yang sudah ditemukan.

Jika dikerjakan, prompt generation harus memaksa jawaban grounded.

Aturan generation:

1. Jawab berdasarkan konteks.
2. Jika konteks tidak cukup, katakan konteks belum cukup.
3. Sertakan rujukan pasal/regulasi.
4. Jangan mengarang status hukum.
5. Jika ada relasi perubahan/cabut, jelaskan dampaknya.

Output jawaban ideal:

1. Jawaban ringkas.
2. Dasar hukum.
3. Kutipan/parafrasa pasal.
4. Catatan status jika ada amandemen/pencabutan.
5. Citation internal.

## 12. Dataset Construction

Dataset yang perlu dibuat:

1. Extraction dataset: teks pasal/preamble -> triplets benar.
2. Retrieval dataset: query -> chunk/node relevan.
3. QA dataset: pertanyaan hukum -> jawaban + sumber.
4. Conflict dataset: pasangan pasal/regulasi -> jenis konflik + penjelasan.
5. Regression dataset: kasus rawan seperti angka Romawi, lampiran, daftar `Nomor X Tahun Y`.

Format QA dataset:

```json
{
  "id": "QA-0001",
  "question": "Apa isi Pasal I PP 8 Tahun 2009?",
  "answer": "...",
  "references": [
    {
      "regulasi_id": "REG-PP-8-2009",
      "node_id": "PSL-REG-PP-8-2009-I-0",
      "chunk_id": "CHK-REG-PP-8-2009-0001"
    }
  ]
}
```

Format retrieval dataset:

```json
{
  "query": "dasar hukum perubahan gaji PNS tahun 2009",
  "relevant_chunk_ids": ["CHK-REG-PP-8-2009-0001"],
  "relevant_node_ids": ["PSL-REG-PP-8-2009-I-0"]
}
```

Format conflict dataset:

```json
{
  "id": "CONFLICT-0001",
  "category": "Direct_Contradiction",
  "regulation_level": "same_level",
  "source": {
    "node_id": "PSL-REG-PP-...",
    "chunk_id": "CHK-...",
    "claim_type": "Kewajiban"
  },
  "target": {
    "node_id": "PSL-REG-PP-...",
    "chunk_id": "CHK-...",
    "claim_type": "Larangan"
  },
  "shared_topic_id": "TOPIK-...",
  "explanation": "...",
  "label": "conflict"
}
```

## 13. Evaluation

Evaluasi extraction:

1. Pasal coverage: persentase pasal yang berhasil diparse.
2. Preamble coverage: `Menimbang/Mengingat/Menetapkan` berhasil diparse.
3. Triplet precision: sampel triplet valid secara manual.
4. Triplet schema validity: edge sesuai schema.
5. Reference normalization accuracy: rujukan jadi `REG-*`/`PSL-*` yang benar.

Evaluasi graph:

1. Node count by type.
2. Edge count by label.
3. Orphan node count.
4. Missing target node count.
5. Invalid edge by schema.
6. Connectivity untuk regulasi yang saling mengubah/mencabut.

Evaluasi retrieval:

1. Recall@k.
2. Precision@k.
3. MRR.
4. nDCG@k.
5. Citation correctness.

Evaluasi generation:

1. Exact citation match.
2. Faithfulness: jawaban didukung konteks.
3. Answer relevance.
4. Completeness.
5. Hallucination rate.

Evaluasi conflict detection:

1. Precision konflik.
2. Recall konflik.
3. F1 per kategori konflik.
4. False positive analysis.
5. False negative analysis.

## 14. Rerun Order yang Disarankan

Rerun dari awal sebaiknya dilakukan dengan urutan ini:

1. [x] Backup/pisahkan output lama: v2 memakai folder `*_v2`.
2. [x] Update schema v2 untuk node dan edge baru.
3. [x] Update parser dokumen agar menangkap preamble, body, lampiran, dan penutup.
4. [x] Generate `data/processed_v2/*.json`.
5. [x] Generate `data/chunks_v2/*.jsonl`.
6. [x] Extract triplets dari semua unit struktur.
7. [x] Validate triplets teknis dengan QC v2.
8. [x] Build graph index lokal.
9. [x] Build embedding dan vector index.
10. [ ] Jalankan graph audit.
11. [ ] Jalankan retrieval evaluation.
12. [ ] Jalankan generation evaluation.
13. [ ] Simpan report final di `outputs/reports/v2/`.

## 15. Implementasi Bertahap

Tahap 1: Parser v2

1. [x] Tambah parsing `Menimbang`, `Mengingat`, `Memutuskan`, `Menetapkan`.
2. [x] Tambah parsing `Penjelasan`, lampiran, dan penutup.
3. [x] Tambah QC coverage.

Tahap 2: Schema v2

1. [x] Tambah node `Konsiderans`, `Dasar_Hukum`, `Diktum`, `Penjelasan`, `Lampiran`, `Chunk`, `Topik`.
2. [x] Tambah edge struktural dan hukum baru.
3. [x] Update validator/QC teknis.

Tahap 3: Chunk dan embedding

1. [x] Generate chunks dari processed JSON.
2. [x] Build embedding.
3. [x] Simpan vector index.

Tahap 4: GraphRAG Retrieval

1. Exact reference retrieval.
2. Vector retrieval.
3. Graph expansion.
4. Reranking.
5. Context packing dan citation.

Tahap 5: Dataset dan evaluasi

1. Buat retrieval dataset.
2. Buat QA dataset.
3. Buat conflict dataset.
4. Jalankan evaluasi rutin.

## 16. Catatan Penting

1. Graph tanpa chunk akan lemah untuk menjawab isi pasal secara detail.
2. Chunk tanpa graph akan lemah untuk memahami rujukan, amandemen, dan status hukum.
3. `Menimbang` dan `Mengingat` penting untuk dasar hukum dan alasan pembentukan regulasi.
4. `Penjelasan` penting untuk interpretasi resmi, tetapi harus dibedakan dari pasal normatif.
5. Lampiran penting karena banyak regulasi memindahkan substansi utama ke lampiran.
6. Semua hasil retrieval harus bisa ditelusuri balik ke PDF, regulasi, pasal, penjelasan/lampiran jika relevan, dan chunk.
7. Rerun keseluruhan masuk akal setelah parser v2 dan schema v2 siap, karena output lama belum menyimpan bagian atas dokumen dan penjelasan dokumen.
