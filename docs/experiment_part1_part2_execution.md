# PSKP Experiment Execution Plan

Dokumen ini merapikan struktur eksperimen penelitian menjadi dua bagian:

1. `Bagian 1 - Baseline Validation`
2. `Bagian 2 - Main Contribution: GraphRAG for Structural Legal Reasoning`

Dokumen ini bersifat operasional. Tujuannya agar langkah berikutnya bisa langsung dieksekusi tanpa mengubah arah metodologi di tengah jalan.

## 1. Prinsip Umum

Prinsip eksperimen yang harus dijaga:

1. `Bagian 1` dipakai untuk memvalidasi setup terhadap dataset QA lama.
2. `Bagian 2` dipakai untuk menunjukkan kontribusi utama GraphRAG pada task yang memang membutuhkan struktur regulasi.
3. `Evaluation set` tidak boleh dipakai sebagai data training.
4. Untuk perbandingan retrieval, model inti harus sama; yang dibedakan adalah konteks yang diberikan saat inference.
5. `RAG` tidak memakai knowledge graph sebagai mekanisme reasoning.
6. `GraphRAG` memakai chunk yang sama dengan RAG, tetapi menambahkan exact reference resolution, graph expansion, dan relation/path context.

Kalimat ringkas yang dipakai sebagai pegangan:

```text
Same model, same corpus, same embedding backbone, different retrieval structure.
```

Maknanya:

1. `Fine-tune + RAG` dan `Fine-tune + GraphRAG` menggunakan checkpoint model yang sama.
2. Keduanya menggunakan korpus regulasi PSKP yang sama.
3. Keduanya boleh menggunakan embedding model yang sama.
4. Yang berbeda adalah bentuk retrieval dan bentuk context.

## 2. Status Repo Saat Ini

Komponen yang sudah tersedia:

1. Parser v2: `scripts/extraction/extract_v2.py`
2. Chunk v2: `data/chunks_v2/`
3. Triplets v2: `data/triplets_v2/`
4. Graph v2: `indexes/graph_v2/`
5. Embedding index v2: `indexes/vector_v2_model/`
6. Retriever GraphRAG v2: `scripts/graph/graphrag_retriever_v2.py`
7. Inference fine-tune only: `scripts/training/infer_lora.py`

Gap yang masih ada:

1. Script inference `fine-tune + GraphRAG` yang lama masih memakai retriever lama:
   `scripts/training/infer_lora_graphrag.py` -> `scripts/graph/graphrag_retriever.py`
2. Belum ada curated evaluation set untuk `Bagian 2`.

Konsekuensinya:

1. `Bagian 1` sudah siap dan sudah selesai dieksekusi end-to-end.
2. Fokus berikutnya berpindah ke desain dan kurasi `Bagian 2`.
3. Setelah hasil `fine-tune` pada `Bagian 1` tidak jauh berubah dari baseline lama, eksperimen tambahan `zero-shot`, `zero-shot + RAG`, dan `zero-shot + GraphRAG` layak ditambahkan sebagai baseline pembanding baru.

## 3. Bagian 1 - Baseline Validation

## 3.1 Tujuan

Tujuan `Bagian 1` adalah memvalidasi bahwa pipeline model + retrieval berjalan baik pada dataset QA lama dan tetap comparable dengan eksperimen sebelumnya.

Bagian ini bukan tempat utama untuk membuktikan keunggulan GraphRAG.

## 3.2 Skenario yang Dipakai Sekarang

Untuk kerja praktis saat ini, fokus yang dipakai adalah:

1. `Fine-tune + RAG`
2. `Fine-tune + GraphRAG`

`Fine-tune only` tidak perlu direrun sekarang jika:

1. Checkpoint model sama dengan eksperimen sebelumnya.
2. Dataset QA lama sama.
3. Hasil lama masih bisa dijadikan baseline referensi.

Catatan:

1. `Fine-tune only` tetap boleh muncul di paper sebagai baseline pembanding.
2. Tetapi untuk implementasi sekarang, kita fokus dulu pada retrieval-based methods.

## 3.3 Definisi Fair Comparison

Agar perbandingan adil:

1. Base model sama.
2. Adapter/checkpoint LoRA sama.
3. Test file sama.
4. Prompt instruction sebisa mungkin sama.
5. Perbedaan utama hanya ada pada bentuk context retrieval.

Definisi masing-masing skenario:

### Fine-tune + RAG

Alur:

```text
query
-> embedding search ke chunk index
-> top-k chunk
-> context linear
-> prompt ke model
-> jawaban
```

Karakteristik:

1. Retrieval berbasis chunk.
2. Context berbentuk linear text blocks.
3. Tidak ada graph expansion.
4. Tidak ada relation/path reasoning.
5. Tidak ada graph proximity rerank.

### Fine-tune + GraphRAG

Alur:

```text
query
-> exact reference resolution
-> embedding search
-> graph expansion
-> rerank / context packing
-> prompt ke model
-> jawaban
```

Karakteristik:

1. Retrieval berbasis chunk + graph.
2. Context boleh memuat relasi hukum, seed node, atau jalur graph yang dirapikan untuk prompt.
3. Exact legal reference menjadi anchor.
4. Dapat memakai query cleaning dan graph-aware rerank.

## 3.4 Komponen yang Harus Dibuat atau Diperbarui

Sebelum `Bagian 1` bisa dijalankan, yang perlu disiapkan adalah:

1. Script `Fine-tune + RAG`
2. Script `Fine-tune + GraphRAG` berbasis retriever v2
3. Script evaluasi output prediksi pada dataset QA lama

Rencana file:

1. `scripts/training/infer_lora_rag_v2.py`
2. `scripts/training/infer_lora_graphrag_v2.py`
3. `scripts/evaluation/eval_predictions_qa.py`

Status file:

1. `infer_lora_rag_v2.py`: sudah dibuat
2. `infer_lora_graphrag_v2.py`: sudah dibuat
3. `eval_predictions_qa.py`: sudah dibuat

Catatan penting:

1. `scripts/training/infer_lora_graphrag.py` yang sekarang masih memakai retriever lama.
2. Untuk eksperimen baru, lebih aman membuat script `*_v2.py` baru daripada menimpa script lama.

## 3.5 Rencana Implementasi Bagian 1

Urutan implementasi yang disarankan:

1. Buat `infer_lora_rag_v2.py`
2. Buat `infer_lora_graphrag_v2.py`
3. Samakan template prompt antara RAG dan GraphRAG sebisa mungkin
4. Jalankan smoke test kecil dengan `--limit`
5. Jalankan full inference pada dataset QA lama
6. Evaluasi metrik `BLEU`, `METEOR`, `ROUGE-L`, dan `BERT-Cosine`
7. Tulis ringkasan hasil `Bagian 1`

## 3.6 Hasil Final Bagian 1

Eksperimen `Bagian 1` sudah selesai dijalankan pada `1702` sampel `data/sft/test.jsonl`.

Tabel hasil final:

| Skenario | BLEU-4 | METEOR | ROUGE-L | BERT-Cosine | Avg Prompt Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Fine-tune only 512` | `0.575582` | `0.673393` | `0.693756` | `0.955259` | `0` |
| `Fine-tune + RAG v2` | `0.510550` | `0.646809` | `0.659949` | `0.949213` | `1054.64` |
| `Fine-tune + GraphRAG v2` | `0.517019` | `0.650537` | `0.662720` | `0.949791` | `1673.92` |

Interpretasi resmi `Bagian 1`:

1. `Fine-tune only` tetap menjadi baseline terbaik pada dataset QA lama.
2. `GraphRAG v2` konsisten mengungguli `RAG v2` pada semua metrik utama, tetapi selisihnya kecil.
3. Kedua retrieval-based settings masih berada di bawah `fine-tune only`, sehingga dataset QA lama tidak ideal untuk menunjukkan manfaat penuh dari retrieval hukum berbasis graph.
4. Hasil ini mendukung revisi metodologi bahwa kontribusi utama GraphRAG sebaiknya dibuktikan pada `Bagian 2 - Structural Legal Reasoning`, bukan pada benchmark QA lama.

Narasi aman untuk paper:

```text
On the legacy QA benchmark, the fine-tuned model without external retrieval remains the strongest baseline. Both retrieval-based settings underperform the fine-tune-only setup, indicating that the old QA dataset is highly aligned with the fine-tuning distribution and is not an ideal benchmark for demonstrating the benefit of structured retrieval. However, GraphRAG consistently outperforms plain RAG across all metrics, suggesting that graph-structured legal context is more useful than unstructured chunk retrieval when external context is introduced.
```

## 3.6.1 Ekstensi Eksperimen Part 1: Zero-shot Baselines

Sebagai tindak lanjut setelah hasil `fine-tune` pada `Bagian 1` tetap mirip dengan baseline lama, eksperimen tambahan akan dibuka untuk tiga skenario baru:

1. `zero-shot only`
2. `zero-shot + RAG`
3. `zero-shot + GraphRAG`

Prinsip implementasi:

1. Model dasar tetap sama, tetapi tanpa adapter LoRA.
2. Corpus, graph, dan embedding index tetap shared dengan pipeline v2 yang sudah ada.
3. Folder hasil dipisahkan agar tidak menimpa eksperimen `fine-tune`.

Folder output yang direkomendasikan:

```text
outputs/predictions/part1_zeroshot/
outputs/reports/part1_zeroshot/
logs/part1_zeroshot/
```

Script yang disiapkan:

1. `scripts/training/infer_zeroshot.py`
2. `scripts/training/infer_zeroshot_rag_v2.py`
3. `scripts/training/infer_zeroshot_graphrag_v2.py`

## 3.7 Format Input dan Output Bagian 1

Input utama:

1. `data/sft/test.jsonl`
2. Checkpoint LoRA yang sama untuk semua skenario
3. `indexes/vector_v2_model/` untuk RAG
4. `indexes/vector_v2_model/` + `indexes/graph_v2/` untuk GraphRAG

Output yang harus tersedia:

1. `outputs/predictions/lora_rag_v2_predictions.csv`
2. `outputs/predictions/lora_graphrag_v2_predictions.csv`
3. `outputs/reports/part1_baseline_eval.json`
4. `outputs/reports/part1_baseline_eval.csv`

Kolom minimum output prediksi RAG:

1. `index`
2. `input`
3. `reference`
4. `label`
5. `ground_truth`
6. `retrieved_chunk_ids`
7. `retrieved_context`
8. `prompt_token_count`
9. `prediction`

Kolom minimum output prediksi GraphRAG:

1. `index`
2. `input`
3. `reference`
4. `label`
5. `ground_truth`
6. `graph_seed_nodes`
7. `graph_context`
8. `prompt_token_count`
9. `prediction`

## 3.8 Prompt Fairness Rules

Agar hasil tidak bias:

1. Instruksi model harus semirip mungkin antara RAG dan GraphRAG.
2. Gaya jawaban harus sama.
3. Perbedaan utama hanya pada bentuk context yang diberikan.
4. Jangan memberi aturan reasoning tambahan ke GraphRAG yang tidak diberikan ke RAG, kecuali yang memang berasal dari struktur context.

Aturan praktis:

1. `System prompt` boleh sama.
2. `Context section` beda isi.
3. `Instruction section` sama.
4. `Answer section` sama.

## 3.8 Query Processing untuk RAG

RAG baseline boleh memakai:

1. Query cleaning ringan
2. Top-k retrieval
3. Metadata chunk dasar seperti `reference_label` dan `regulasi_id`
4. Dedup context jika ada chunk berulang

RAG baseline tidak boleh memakai:

1. Graph traversal
2. Graph expansion
3. Graph path
4. Exact relation following seperti `MENGAMANDEMEN`, `MENCABUT`, `BERDASARKAN`
5. Graph distance rerank

## 3.9 Metrik Bagian 1

Metrik utama:

1. `BLEU`
2. `METEOR`
3. `ROUGE-L`
4. `BERT-Cosine`

Metrik tambahan yang boleh dicatat:

1. `prompt_token_count`
2. `retrieval_coverage`
3. `reference_hit_rate`

Interpretasi:

1. Hasil `Bagian 1` dipakai untuk validasi setup, bukan klaim utama efektivitas GraphRAG.
2. Jika GraphRAG unggul di QA lama, itu bonus.
3. Jika GraphRAG tidak unggul di QA lama, itu tidak otomatis melemahkan kontribusi `Bagian 2`.

## 4. Bagian 2 - Main Contribution: Structural Legal Reasoning

## 4.1 Tujuan

`Bagian 2` adalah kontribusi utama penelitian.

Fokusnya bukan lagi QA biasa, tetapi task yang memang membutuhkan struktur regulasi:

1. `Hierarchy Reasoning`
2. `Reference Reasoning`
3. `Amendment Reasoning`
4. `Overlap Reasoning`
5. `Conflict Analysis`
6. `Multi-hop Reasoning`

## 4.2 Posisi Conflict Analysis

`Conflict Analysis` tetap penting, tetapi bukan satu-satunya task.

Posisinya sekarang:

1. Salah satu task dalam `structural legal reasoning`
2. Bukan satu-satunya pusat pipeline
3. Dapat memanfaatkan `conflict candidate generation` sebagai submodule

## 4.3 Dataset Bagian 2

Untuk `Bagian 2`, yang dibuat terlebih dahulu adalah:

```text
curated evaluation set
```

Bukan training set baru.

Ukuran bertahap:

1. `50-100`: prototype/debugging
2. `200-300`: evaluasi awal
3. `500+`: evaluasi stabil
4. `1000`: evaluasi utama jika waktu memungkinkan

## 4.4 Struktur Item Eval Set

Field minimum yang disarankan:

1. `question`
2. `answer`
3. `task_type`
4. `answer_type`
5. `required_evidence`
6. `required_node_ids`
7. `required_relations`
8. `required_path`
9. `required_context`

Contoh:

```json
{
  "question": "Apakah PP 57 Tahun 2021 mengalami perubahan? Jika iya, regulasi apa yang mengubahnya?",
  "answer": "Ya. PP 57 Tahun 2021 diubah oleh PP 4 Tahun 2022.",
  "task_type": "amendment_reasoning",
  "answer_type": "short_explanation",
  "required_evidence": [
    "REG-PP-57-2021",
    "REG-PP-4-2022"
  ],
  "required_node_ids": [
    "REG-PP-57-2021",
    "REG-PP-4-2022"
  ],
  "required_relations": [
    "MENGAMANDEMEN"
  ],
  "required_path": [
    "REG-PP-4-2022 -> MENGAMANDEMEN -> REG-PP-57-2021"
  ],
  "required_context": [
    "PP 4 Tahun 2022 tentang Perubahan atas PP 57 Tahun 2021"
  ]
}
```

## 4.5 Apa yang Tidak Dilakukan Sekarang

Untuk tahap sekarang:

1. Tidak membuat training set structural reasoning dulu.
2. Tidak fine-tune model pada curated eval set.
3. Tidak menjadikan generation layer sebagai milestone wajib.

Jika nanti dibutuhkan:

1. Buat train/dev/test split structural reasoning yang terpisah
2. Fine-tune ulang satu checkpoint bersama
3. Rerun `model only vs RAG vs GraphRAG`

## 5. Urutan Eksekusi Nyata

Urutan kerja yang sekarang paling masuk akal:

1. Bekukan struktur eksperimen `Bagian 1` dan `Bagian 2`
2. Implement `infer_lora_rag_v2.py`
3. Implement `infer_lora_graphrag_v2.py` berbasis retriever v2
4. Implement evaluator `Bagian 1`
5. Jalankan smoke test kecil di dataset QA lama
6. Jalankan full `Bagian 1`
7. Simpan hasil baseline validation
8. Baru mulai desain curated eval set `Bagian 2`

## 6. Checklist Eksekusi Dekat

Checklist paling dekat yang bisa langsung dieksekusi:

- [x] Tulis script `scripts/training/infer_lora_rag_v2.py`
- [x] Tulis script `scripts/training/infer_lora_graphrag_v2.py`
- [x] Tulis script `scripts/evaluation/eval_predictions_qa.py`
- [x] Jalankan smoke test `--limit 10` untuk RAG
- [x] Jalankan smoke test `--limit 10` untuk GraphRAG
- [x] Jalankan full inference pada `data/sft/test.jsonl`
- [x] Hitung metrik `BLEU`, `METEOR`, `ROUGE-L`, `BERT-Cosine`
- [x] Rekap hasil `Bagian 1`
- [ ] Mulai desain curated eval set `Bagian 2`

## 7. Decision Log

Keputusan yang sudah dibekukan:

1. `RAG` tidak memakai knowledge graph sebagai mekanisme retrieval/reasoning.
2. `GraphRAG` memakai chunk yang sama, tetapi ditambah struktur graph.
3. `Fine-tune only` tidak direrun sekarang; hasil lama cukup sebagai baseline referensi sementara.
4. `Bagian 1` dikerjakan lebih dulu pada dataset QA lama.
5. `Bagian 2` memakai curated evaluation set baru.
6. `Generation layer` hanya opsional, bukan pusat kontribusi.
