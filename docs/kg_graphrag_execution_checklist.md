# KG dan GraphRAG V2 Execution Checklist

Checklist ini adalah versi operasional dari `docs/kg_graphrag_rerun_plan.md`. Gunakan file ini untuk menandai apa yang sudah dilakukan, apa yang sedang dijalankan, dan apa yang harus dikerjakan berikutnya.

Legend:

- `[x]` selesai
- `[ ]` belum selesai
- `[~]` sedang berjalan atau sebagian selesai

## 0. Keputusan Scope

- [x] Pakai istilah `GraphRAG`, bukan `Hybrid GraphRAG`.
- [x] GraphRAG berarti retrieval mengambil konteks dari graph, chunk text, metadata, exact reference, dan graph expansion sebagai bagian dari augmentation.
- [x] Tools seperti Neo4j, Chroma, Milvus, atau Qdrant hanya dipakai untuk storage, query, visualisasi, atau indexing.
- [x] Proses utama tetap dibuat by code di repository ini.
- [x] V1 tidak dihapus; output v2 dipisahkan.
- [x] Untuk sementara pakai dokumen yang sudah ada di `data/raw/`.
- [x] Untuk sementara tidak pakai `corpus_role`.
- [x] Struktur eksperimen dibagi menjadi `Bagian 1 - Baseline Validation` dan `Bagian 2 - Main Contribution: Structural Legal Reasoning`.
- [x] `Conflict Analysis` tetap penting, tetapi diposisikan sebagai salah satu task dalam `Structural Legal Reasoning`.
- [x] Dokumen operasional eksperimen ada di `docs/experiment_part1_part2_execution.md`.

## 1. Git dan Repo Hygiene

- [x] Folder script dirapikan ke kategori seperti `extraction`, `evaluation`, `graph`, `ingestion`, dan `training`.
- [x] Import dasar diperbaiki setelah pemindahan folder script.
- [x] `.gitignore` dibuat agar data besar dan output eksperimen tidak ikut push.
- [x] Folder yang di-ignore: `data/`, `logs/`, `outputs/`, `indexes/`, `models/`, `checkpoints/`, `runs/`, `wandb/`, `mlruns/`.
- [ ] Commit perubahan kode, docs, ontology, dan `.gitignore`.
- [ ] Push repo tanpa data mentah/output besar.

Command cek sebelum commit:

```bash
git status --short
git diff --cached --stat
```

## 2. Ontology V2

- [x] Buat folder `ontology/v2/`.
- [x] Buat schema utama ontology v2 di `ontology/v2/schema.json`.
- [x] Tambahkan node struktur dokumen: `Regulasi`, `Bab`, `Bagian`, `Paragraf`, `Pasal`, `Ayat`, `SubItem`.
- [x] Tambahkan node preamble: `Konsiderans`, `Dasar_Hukum`, `Diktum`.
- [x] Tambahkan node dokumen pendukung: `Penjelasan`, `Penjelasan_Umum`, `Penjelasan_Pasal`, `Lampiran`, `Penutup`.
- [x] Tambahkan node retrieval/semantic: `Chunk`, `Subjek_Hukum`, `Topik`.
- [x] Tambahkan node event: `Peristiwa_Hukum`.
- [x] Tambahkan edge structural seperti `MEMILIKI_PASAL`, `MEMILIKI_AYAT`, `MEMILIKI_CHUNK`.
- [x] Tambahkan edge explicit legal seperti `MENGACU_PADA`, `BERDASARKAN`, `MENGAMANDEMEN`, `MENCABUT`, `MENCABUT_PASAL`.
- [x] Tambahkan edge semantic seperti `MENGATUR_TOPIK`, `MENGATUR_SUBJEK`, `MEMBERIKAN_HAK`, `MEMBEBANKAN_KEWAJIBAN`, `MELARANG`, `MENDEFINISIKAN`.
- [x] Tambahkan edge conflict candidate/evaluation: `BERTENTANGAN_DENGAN`.
- [x] Buat constraints ontology di `ontology/v2/constraints.md`.
- [x] Buat few-shot examples di `ontology/v2/few_shot_examples.json`.
- [x] Buat schema output `processed_document`, `chunk`, dan `triplet`.
- [ ] Aktifkan validasi penuh dengan `jsonschema` setelah dependency tersedia.

Definition of done:

- Semua file ontology valid JSON/Markdown.
- Node dan edge cukup untuk membangun graph dokumen, graph hukum eksplisit, graph semantic, dan dataset konflik.

## 3. Output Contract V2

- [x] Buat kontrak output v2 di `docs/v2_output_contract.md`.
- [x] Tetapkan output parser ke `data/processed_v2/`.
- [x] Tetapkan output chunk ke `data/chunks_v2/`.
- [x] Tetapkan rencana output triplet ke `data/triplets_v2/`.
- [x] Tetapkan rencana graph index ke `indexes/graph_v2/`.
- [x] Tetapkan rencana vector index ke `indexes/vector_v2/`.
- [x] Tetapkan report v2 ke `outputs/reports/v2/`.

Definition of done:

- Setiap output punya lokasi jelas.
- V1 dan v2 tidak saling menimpa.

## 4. Parser dan Chunker V2

- [x] Buat script parser v2 di `scripts/extraction/extract_v2.py`.
- [x] Parser membaca PDF dari `data/raw/`.
- [x] Parser menulis `processed_v2` dan `chunks_v2`.
- [x] Parser mengambil metadata dasar dari `regulasi_id` dan `TITLE_MAP`.
- [x] Parser mengambil `Menimbang` sebagai `Konsiderans`.
- [x] Parser mengambil `Mengingat` sebagai `Dasar_Hukum`.
- [x] Parser mengambil `Memutuskan` dan `Menetapkan`.
- [x] Parser mengambil `Bab`.
- [x] Parser mengambil `Pasal`.
- [x] Parser mengambil `Ayat`.
- [x] Parser mengambil `SubItem` huruf/angka.
- [x] Parser mengambil `Penjelasan` inline jika ada.
- [x] Parser mengambil `Penjelasan` dari PDF terpisah dengan format `PJL-{regulasi_id}.pdf` jika ada.
- [x] Parser mengambil `Penutup` dasar.
- [x] Chunk dibuat dari preamble, pasal, ayat, subitem, penjelasan, dan penutup.
- [x] Tambah normalisasi OCR untuk heading pasal.
- [x] Tambah normalisasi OCR untuk tahun rujukan seperti `2OO3` menjadi `2003`.
- [x] Tambah QC warning untuk pasal numerik yang loncat.
- [x] Parser lampiran menangkap heading `LAMPIRAN` sebagai node `Lampiran` dan chunk `lampiran` jika teks lampiran tersedia di PDF.
- [x] Parser `Bagian` dan `Paragraf` membentuk struktur relasional `Bab -> Bagian -> Paragraf -> Pasal`.
- [x] Parser pasal menyimpan `parent_bab_id`, `parent_bagian_id`, dan `parent_paragraf_id` jika struktur parent tersedia.
- [x] Parser tanggal/pejabat penutup mengambil tanggal ditetapkan, tanggal diundangkan, pejabat penetapan, dan pejabat pengundangan.
- [ ] Parser tabel lampiran kompleks/OCR tabel belum dinormalisasi menjadi baris-kolom.

Definition of done:

- `extract_v2.py --file REG-ID` menghasilkan processed JSON dan chunk JSONL.
- `extract_v2.py --all` bisa jalan untuk semua dokumen raw.
- QC report memberi warning jika parsing mencurigakan.

## 5. Pilot Parser V2

Pilot dipakai untuk memastikan parser tahan terhadap dokumen dengan karakter berbeda.

- [x] Pilot `REG-PP-8-2009`: dokumen pendek, bentuk amandemen, hanya 2 pasal.
- [x] Pilot `REG-PP-17-2010`: dokumen panjang, banyak pasal dan ayat.
- [x] Pilot `REG-PP-47-2008`: punya dokumen penjelasan terpisah.
- [x] Pilot `REG-PP-57-2021`: OCR cukup kotor, banyak heading pasal sulit.

Hasil pilot terakhir:

| Dokumen | Pasal | Ayat | Penjelasan Pasal | Status |
| --- | ---: | ---: | ---: | --- |
| `REG-PP-8-2009` | 2 | 0 | 0 | PASS |
| `REG-PP-17-2010` | 222 | 654 | 222 | PASS |
| `REG-PP-47-2008` | 16 | 52 | 16 | PASS |
| `REG-PP-57-2021` | 59 | 152 | 59 | PASS |

Definition of done:

- Tidak ada duplicate pasal pada pilot.
- Tidak ada missing pasal sequence pada pilot yang numeriknya jelas.
- Penjelasan pasal demi pasal terbaca pada dokumen yang punya penjelasan.

## 6. Rerun Parser V2 Semua Dokumen

Status: selesai.

Command yang dipakai:

```bash
cd /mnt/extended-home/rifqicahyono/gemastik/pskp-legal-graph
mkdir -p logs
.venv/bin/python scripts/extraction/extract_v2.py --all 2>&1 | tee logs/extract_v2_all.log
```

Output yang sudah muncul:

- [x] `data/processed_v2/{regulasi_id}.json`
- [x] `data/chunks_v2/chunks_{regulasi_id}.jsonl`
- [x] `outputs/reports/v2/extract_v2_report.json`

Hasil ringkas:

- Dokumen utama raw: 22
- Processed v2 JSON: 22
- Chunk v2 JSONL: 22
- Error parser: 0
- Warning parser: 4
- `REG-UUD-1945` tetap dikecualikan untuk tahap final karena file salah label.

Definition of done:

- [x] Semua dokumen `REG-*.pdf` punya output `processed_v2`.
- [x] Semua dokumen `REG-*.pdf` punya output `chunks_v2`.
- [x] Report tidak punya error fatal.
- [x] Warning parser sudah dipahami sebagai kasus dokumen perubahan/amandemen.

## 7. QC Setelah Rerun Parser V2

Status: selesai untuk QC teknis parser.

- [x] Hitung jumlah dokumen raw vs jumlah `processed_v2`.
- [x] Hitung jumlah dokumen raw vs jumlah `chunks_v2`.
- [x] Review dokumen dengan `parse_quality.status = WARN`.
- [x] Review dokumen dengan `total_pasal = 0`.
- [x] Review dokumen dengan `has_preamble = false` padahal dokumen normal.
- [x] Review dokumen dengan `has_explanation = false` jika dokumen sebenarnya punya penjelasan.
- [x] Review missing pasal sequence warning.
- [x] Pilih sampel parser untuk dibandingkan dengan PDF asli.

Catatan QC:

- `REG-PP-8-2009` valid hanya 2 pasal; ini sesuai dokumen.
- `REG-PP-17-2010`, `REG-PP-47-2008`, dan `REG-PP-57-2021` dipakai sebagai sampel berbeda karakter.
- `REG-UUD-1945` salah label dan tidak dipakai untuk final GraphRAG.
- Lampiran belum aktif karena raw saat ini tidak punya heading lampiran yang jelas.

Definition of done:

- [x] Tidak ada error fatal.
- [x] Warning yang tersisa sudah dipahami dan dicatat.
- [x] Parser dianggap cukup aman untuk menjadi basis triplet v2.

## 8. Triplets V2

Status: selesai untuk structural, explicit, semantic, dan QC teknis.

Script yang dibuat:

- [x] `scripts/extraction/extract_triplets_v2_structural.py`
- [x] `scripts/extraction/extract_triplets_v2_explicit.py`
- [x] `scripts/extraction/extract_triplets_v2_semantic.py`
- [x] `scripts/evaluation/qc_triplets_v2.py`

Output:

- [x] `data/triplets_v2/triplets_{regulasi_id}.json`
- [x] Structural triplets dari hasil parser, tanpa LLM.
- [x] Explicit legal triplets dari regex/reference parser, tanpa LLM.
- [x] Semantic triplets dari LLM dengan evidence chunk/source node.
- [x] Validasi teknis predicate dan type terhadap ontology v2.
- [x] Deduplication triplets.
- [x] Confidence/source tracking untuk tiap triplet.

Rekap final, tidak memasukkan `REG-UUD-1945`:

| Method | Jumlah |
| --- | ---: |
| `structural` | 17,223 |
| `regex` | 1,454 |
| `llm` | 12,199 |
| **Total** | **30,876** |

QC triplets terakhir:

| Metric | Nilai |
| --- | ---: |
| Documents checked | 21 |
| Documents excluded | 1 |
| Total errors | 0 |
| Total warnings | 321 |
| Duplicate triplets | 0 |

Triplet structural minimal:

- [x] `Regulasi -> MEMILIKI_BAB -> Bab`
- [x] `Regulasi/Bab -> MEMILIKI_PASAL -> Pasal`
- [x] `Pasal -> MEMILIKI_AYAT -> Ayat`
- [x] `Pasal/Ayat -> MEMILIKI_SUBITEM -> SubItem`
- [x] `Regulasi -> MEMILIKI_KONSIDERANS -> Konsiderans`
- [x] `Regulasi -> MEMILIKI_DASAR_HUKUM -> Dasar_Hukum`
- [x] `Regulasi -> MEMILIKI_DIKTUM -> Diktum`
- [x] `Regulasi -> MEMILIKI_PENJELASAN -> Penjelasan`
- [x] `Penjelasan -> MEMILIKI_PENJELASAN_UMUM -> Penjelasan_Umum`
- [x] `Penjelasan -> MEMILIKI_PENJELASAN_PASAL -> Penjelasan_Pasal`
- [x] `Penjelasan_Pasal -> MENJELASKAN -> Pasal`
- [x] `Node -> MEMILIKI_CHUNK -> Chunk`

Triplet explicit minimal:

- [x] `Dasar_Hukum -> BERDASARKAN -> Regulasi`
- [x] `Pasal/Chunk -> MENGACU_PADA -> Regulasi/Pasal`
- [x] `Regulasi -> MENGAMANDEMEN -> Regulasi`
- [x] `Regulasi -> MENCABUT -> Regulasi`
- [x] `Regulasi -> MENCABUT_PASAL -> Pasal`
- [x] `Regulasi -> MULAI_BERLAKU -> Peristiwa_Hukum`
- [x] `Regulasi -> DIUNDANGKAN_PADA -> Peristiwa_Hukum`

Triplet semantic minimal:

- [x] `Pasal/Chunk -> MENGATUR_TOPIK -> Topik`
- [x] `Pasal/Chunk -> MENGATUR_SUBJEK -> Subjek_Hukum`
- [x] `Pasal/SubItem -> MEMBERIKAN_HAK -> Subjek_Hukum`
- [x] `Pasal/SubItem -> MEMBEBANKAN_KEWAJIBAN -> Subjek_Hukum`
- [x] `Pasal/SubItem -> MELARANG -> Subjek_Hukum/Topik`
- [x] `Pasal/SubItem -> MENDEFINISIKAN -> Topik`

Definition of done:

- [x] Semua dokumen final punya `triplets_v2`.
- [x] Triplets tervalidasi teknis terhadap ontology v2.
- [x] Structural triplets deterministic tidak bergantung LLM.
- [x] Semantic triplets punya evidence chunk/source node.

## 9. Graph V2

Status: selesai untuk build lokal dan QC teknis.

Yang sudah dibuat:

- [x] Builder graph lokal dari `processed_v2`, `chunks_v2`, dan `triplets_v2`.
- [x] Script `scripts/graph/build_graph_v2.py`.
- [x] Script QC `scripts/evaluation/audit_graph_v2.py`.
- [x] Default exclude `REG-UUD-1945` karena file raw salah label.
- [x] Graph v2 berhasil dengan `30,876` edges dan `0` orphan chunks.
- [x] Output graph lokal di `indexes/graph_v2/`.
- [x] QC graph v2 PASS: `21,757` nodes, `30,876` edges, `0` errors, `0` warnings.

Yang masih opsional:

- [ ] Export/import ke Neo4j jika dibutuhkan.
- [ ] Constraint Neo4j untuk node ID unik.
- [ ] Query audit graph.

QC graph:

- [ ] Semua `Regulasi` punya minimal satu `Pasal`, kecuali dokumen khusus yang valid.
- [ ] Semua `Pasal` punya edge ke `Regulasi` atau parent struktur.
- [ ] Semua `Chunk` punya `source_node_id`.
- [ ] Semua `Dasar_Hukum` yang resolve punya target regulasi.
- [ ] Tidak ada edge dengan node kosong.

Definition of done:

- [x] Graph bisa ditelusuri dari regulasi ke pasal, ayat, chunk, dan dasar hukum.
- [x] Graph bisa dipakai untuk expansion pada retrieval.

## 10. Embedding dan Vector Index

Status: selesai untuk full local E5 embedding dan smoke test search.

Catatan keputusan:

- `chunks_v2` tetap menjadi source of truth untuk citation.
- Embedding memakai retrieval text yang diperkaya konteks: regulasi, judul, reference label, section type, dan teks chunk.
- Baseline dependency-free `hashing_tfidf` boleh dipakai untuk smoke test, tetapi bukan final riset.

Yang sudah dibuat:

- [x] Script baseline `scripts/graph/build_vector_index_v2.py`.
- [x] Script baseline search `scripts/graph/search_vector_index_v2.py`.
- [x] Script model embedding `scripts/graph/build_embedding_index_v2.py`.
- [x] Script model search `scripts/graph/search_embedding_index_v2.py`.
- [x] Backend local `sentence_transformers` untuk `intfloat/multilingual-e5-base`.
- [x] E5 prefix didukung: `passage:` untuk chunk dan `query:` untuk query.
- [x] Dry-run local E5 berhasil.
- [x] Input dari `data/chunks_v2/*.jsonl`.
- [x] Metadata chunk disimpan bersama embedding.
- [x] Filter metadata seperti `regulasi_id` dan `section_type` didukung pada search.
- [x] Pilih embedding model final awal: `intfloat/multilingual-e5-base`.
- [x] Jalankan pilot model embedding dengan `--limit`.
- [x] Jalankan full model embedding ke `indexes/vector_v2_model/`.
- [x] Uji semantic search top-k dengan beberapa query.

Hasil full embedding:

| Metric | Nilai |
| --- | ---: |
| Chunks embedded | 7,537 |
| Embedding dimension | 768 |
| Batch size | 32 |
| Excluded docs | 1 |

Definition of done:

- [x] Query teks bisa mencari chunk relevan.
- [x] Hasil search mengembalikan `chunk_id`, score, text, dan metadata.
- [x] Vector index bisa direbuild dari code.

## 11. GraphRAG Retrieval

Status: selesai untuk retriever v2 awal dan smoke test.

Retrieval GraphRAG yang diinginkan:

- [x] Exact reference retrieval: menangkap query seperti `Pasal 74 ayat (4) PP 17 Tahun 2010`.
- [x] Semantic retrieval: cari chunk relevan via embedding.
- [x] Graph expansion: dari chunk/pasal ke parent regulasi, dasar hukum, perubahan, pencabutan, topik, dan penjelasan.
- [x] Query cleaning untuk memisahkan referensi formal dari intent substantif.
- [x] Explainable feature-based rerank hasil retrieval.
- [x] Context packer: susun konteks untuk LLM dengan citation.
- [x] Traceability: jawaban harus bisa menunjuk `regulasi_id`, pasal, ayat, chunk.

Script:

```text
scripts/graph/graphrag_retriever_v2.py
```

Smoke test:

- [x] Exact reference: `PP 47 Tahun 2008 Pasal 5 ayat (1)` resolve ke `AYT-REG-PP-47-2008-5-1`.
- [x] Semantic query: `wajib belajar pendidikan dasar` mengambil chunk teratas dari `REG-PP-47-2008`.
- [x] Graph expansion menambahkan relasi seperti `MEMBEBANKAN_KEWAJIBAN`, `MENGATUR_SUBJEK`, dan `MENGATUR_TOPIK`.
- [x] Exact-only query otomatis skip vector search dengan alasan `exact_reference_only`.
- [x] Query campuran `apa kewajiban satuan pendidikan menurut PP 47 Tahun 2008 Pasal 5 ayat (1)` dibersihkan menjadi vector query `apa kewajiban satuan pendidikan`.
- [x] Rerank menaikkan chunk exact `REG-PP-47-2008 Pasal 5 ayat (1)` ke ranking pertama pada query campuran.

Catatan desain rerank:

- Rerank bukan syarat wajib semua GraphRAG.
- Rerank dipakai di project ini untuk mengurangi noise saat query mencampur referensi hukum formal dan intent substantif.
- Strategi yang dipakai adalah rule-based/explainable feature scoring, bukan neural cross-encoder dan bukan learning-to-rank terlatih.
- Fitur awal: embedding similarity, exact legal reference, graph proximity, metadata regulasi, dan section priority.
- Untuk query yang hanya berisi referensi pasal, retriever otomatis mengandalkan exact reference + graph expansion tanpa vector search.
- Untuk query campuran, referensi formal dibersihkan dari query embedding, tetapi tetap dipakai sebagai anchor graph.

Flag pembanding:

```bash
--no-query-cleaning --no-rerank --no-auto-skip-vector-exact-only
```

Definition of done:

- [x] Query hukum menghasilkan konteks yang relevan dan bisa dilacak.
- [x] Retrieval tidak hanya vector search, tetapi juga memanfaatkan graph.

## 12. Optional Conflict Explanation Layer

Status: opsional dan belum dibuat untuk v2.

Catatan scope:

- Layer ini bukan mekanisme utama untuk deteksi konflik.
- Layer ini hanya menjelaskan evidence dan kandidat konflik dalam bahasa natural.
- Pipeline utama tetap: retrieval -> conflict candidate generation -> dataset -> evaluation.
- Tidak wajib untuk klaim utama paper.

Jika nanti dibutuhkan:

- [ ] Prompt answer generation berbasis konteks GraphRAG.
- [ ] Format jawaban dengan citation.
- [ ] Guardrail agar model tidak menjawab di luar konteks.
- [ ] Format khusus untuk konflik: ringkasan konflik, evidence, kategori, reasoning, confidence.

Definition of done:

- Jawaban final grounded pada chunk/node yang retrieved.
- Jawaban menyebut sumber pasal/chunk.

## 13. Dataset

Status: belum dibuat.

Dataset utama:

- [ ] `conflict_candidates`
- [ ] `conflict_ground_truth`

Kategori konflik:

- [ ] `Direct_Contradiction`
- [ ] `Hierarchical_Conflict`
- [ ] `Temporal_Inconsistency`
- [ ] `Regulatory_Overlap`

Dataset pendukung:

- [ ] `retrieval_ground_truth`
- [ ] `qa_ground_truth`
- [ ] `extraction_gold`
- [ ] `regression_dataset`

Definition of done:

- Dataset konflik punya evidence node/chunk.
- Dataset bisa dipakai untuk evaluasi conflict detection dan explanation.
- Dataset pendukung bisa mengukur retrieval, QA, dan ekstraksi.

## 14. Evaluation

Status: belum dibuat untuk v2.

Evaluasi extraction:

- [ ] Precision/recall parser pada sampel gold.
- [ ] Triplet validity rate.
- [ ] Reference resolution accuracy.

Evaluasi graph:

- [ ] Node/edge coverage.
- [ ] Orphan node rate.
- [ ] Invalid edge rate.
- [ ] Citation traceability.

Evaluasi retrieval:

- [ ] Recall@k.
- [ ] MRR.
- [ ] nDCG.
- [ ] Exact reference hit rate.

Evaluasi generation:

- [ ] Citation correctness.
- [ ] Faithfulness terhadap konteks.
- [ ] Answer relevance.

Evaluasi konflik:

- [ ] Candidate recall.
- [ ] Label accuracy.
- [ ] Evidence correctness.
- [ ] Explanation quality.

Definition of done:

- Ada report angka evaluasi.
- Ada sampel kualitatif untuk paper.
- Pipeline bisa dibandingkan dengan baseline.

## 15. Urutan Kerja Terdekat

Urutan paling aman dari posisi sekarang:

1. [x] Jalankan `extract_v2.py --all` di tmux.
2. [x] Review `outputs/reports/v2/extract_v2_report.json`.
3. [x] Perbaiki parser jika ada warning fatal.
4. [x] Buat script `triplets_v2` structural.
5. [x] Tambahkan explicit legal triplets.
6. [x] Tambahkan semantic triplets.
7. [x] Jalankan QC teknis triplets v2.
8. [x] Build graph v2 lokal dan QC teknis.
9. [x] Build embedding/vector index dari `chunks_v2`.
10. [x] Implement GraphRAG retrieval v2.
11. [x] Buat baseline `Fine-tune + RAG` untuk `Bagian 1`.
12. [x] Buat inference `Fine-tune + GraphRAG` berbasis retriever v2 untuk `Bagian 1`.
13. [x] Buat evaluator hasil prediksi QA lama untuk `Bagian 1`.
14. [x] Jalankan smoke test `RAG` dan `GraphRAG` pada subset kecil dataset QA lama.
15. [x] Jalankan full baseline validation `Bagian 1`.
16. [ ] Setelah `Bagian 1` stabil, mulai curated eval set `Bagian 2`.
17. [ ] Tambahkan baseline pembanding baru: `zero-shot`, `zero-shot + RAG`, dan `zero-shot + GraphRAG`.
18. [ ] Pisahkan output zero-shot ke subfolder khusus agar tidak menimpa hasil `fine-tune`.

## 16. Catatan Penting

- Jangan push `data/`, `logs/`, `outputs/`, atau `indexes/`.
- Jangan menimpa output v1.
- Jangan menjadikan `BERTENTANGAN_DENGAN` sebagai hasil final otomatis tanpa review.
- `Penjelasan` membantu interpretasi, tetapi bukan norma utama.
- Triplet structural sebaiknya deterministic dari parser.
- Triplet semantic boleh dibantu LLM, tapi wajib punya evidence chunk.
- Dataset konflik dibuat berdasarkan kandidat masalah hukum, bukan sekadar semua pasangan pasal.
