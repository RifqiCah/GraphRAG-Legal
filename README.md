# PSKP Legal GraphRAG

Implementasi riset Knowledge Graph dan GraphRAG untuk korpus regulasi
pendidikan Indonesia. Repository ini berfokus pada eksperimen yang digunakan
dalam paper:

1. **Part 1 - Baseline Validation** pada dataset legal QA.
2. **Part 2 - Structural Legal Reasoning** dengan exact reference resolution,
   vector retrieval, graph expansion, dan context reranking.

Graph disimpan dan diproses sebagai index lokal. Pipeline ini tidak
memerlukan database graph eksternal.

## Scope Repository

File yang dilacak Git:

- `dataset.csv`: dataset QA sumber untuk menyiapkan split Part 1.
- `ontology/`: schema dan constraint Knowledge Graph.
- `scripts/`: ekstraksi, graph construction, retrieval, training, dan evaluasi.
- `docs/`: rencana eksperimen, checklist, serta ringkasan hasil.

File yang sengaja tidak dilacak Git:

- `.env` dan credential lokal.
- `data/raw/`: PDF sumber yang diperoleh dari portal regulasi resmi.
- `data/processed*`, `data/chunks_v2`, `data/triplets_v2`, dan `data/sft`.
- `indexes/`, `outputs/`, `logs/`, model, checkpoint, dan artefak eksperimen.

Raw PDF tidak perlu dipush untuk review mentor. Simpan PDF secara lokal dan
cantumkan sumber dokumen dalam paper. Artefak intermediate dan hasil eksperimen
dapat diregenerasi dari script yang tersedia.

## Struktur Utama

```text
pskp-legal-graph/
|-- dataset.csv
|-- docs/
|-- ontology/
|   `-- v2/
|-- scripts/
|   |-- evaluation/
|   |-- extraction/
|   |-- graph/
|   `-- training/
|-- .env.example
|-- .gitignore
`-- requirements.txt
```

Direktori berikut akan dibuat secara lokal ketika pipeline dijalankan:

```text
data/raw/
data/processed_v2/
data/chunks_v2/
data/triplets_v2/
data/sft/
indexes/graph_v2/
indexes/vector_v2_model/
outputs/
logs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Isi `OPENAI_API_KEY` hanya jika menjalankan semantic triplet extraction atau
embedding melalui endpoint OpenAI-compatible. Embedding lokal dengan
`sentence-transformers` tidak membutuhkan API key.

## Menyiapkan Dataset Part 1

`dataset.csv` adalah satu-satunya data penelitian yang dilacak langsung.
Split train/test dibuat secara deterministik dengan seed `42`:

```bash
python scripts/training/prepare_sft_dataset.py \
  --input dataset.csv \
  --out_dir data/sft \
  --seed 42
```

## Pipeline Knowledge Graph V2

Letakkan PDF regulasi yang diperoleh secara legal di `data/raw/`, kemudian
jalankan:

```bash
# Parse dokumen dan buat chunks
python scripts/extraction/extract_v2.py --all

# Ekstrak relasi structural, explicit, dan semantic
python scripts/extraction/extract_triplets_v2_structural.py --all
python scripts/extraction/extract_triplets_v2_explicit.py --all
python scripts/extraction/extract_triplets_v2_semantic.py --all

# QC dan build graph index lokal
python scripts/evaluation/qc_triplets_v2.py
python scripts/graph/build_graph_v2.py

# Build embedding index lokal
python scripts/graph/build_embedding_index_v2.py \
  --backend sentence_transformers \
  --model intfloat/multilingual-e5-base
```

Smoke test retriever:

```bash
python scripts/graph/graphrag_retriever_v2.py \
  "Apa ketentuan mengenai standar nasional pendidikan?" \
  --allow-download
```

## Eksperimen dan Evaluasi

Implementasi skenario Part 1 tersedia untuk:

- Fine-tune only
- Fine-tune + RAG v2
- Fine-tune + GraphRAG v2
- Zero-shot only
- Zero-shot + RAG v2
- Zero-shot + GraphRAG v2

Detail konfigurasi, urutan eksekusi, dan hasil eksperimen tersedia di:

- `docs/experiment_part1_part2_execution.md`
- `docs/kg_graphrag_rerun_plan.md`
- `docs/kg_graphrag_execution_checklist.md`
- `docs/kg_graphrag_v2_results.md`

Evaluasi prediksi QA dapat dijalankan dengan:

```bash
python scripts/evaluation/eval_predictions_qa.py \
  --predictions outputs/predictions/lora_graphrag_v2_predictions.csv \
  --name lora_graphrag_v2
```

## Reproducibility dan Keamanan

- Jangan commit `.env`, token, model weights, checkpoint, atau output eksperimen.
- Simpan versi model, seed, dan parameter eksperimen pada laporan.
- Pastikan lisensi dan sumber PDF regulasi dicantumkan sebelum mendistribusikan
  salinan raw corpus.
