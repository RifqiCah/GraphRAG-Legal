# PSKP Legal Knowledge Graph ⚖️🕸️

Proyek ini bertujuan untuk membangun Knowledge Graph dari korpus kebijakan PSKP (Peraturan Pemerintah terkait Standar Nasional Pendidikan, Guru, Dosen, serta UU Sisdiknas dan UUD 1945). Knowledge Graph ini akan menggantikan mekanisme Vector RAG pada model ASVRI-Legal untuk mendeteksi konflik hukum dan traversal aturan yang saling merujuk.

Pipeline ini dirancang sesuai dengan **[Detailed Issues Document](PSKP_KG_Detailed_Issues.md)**, mengimplementasikan ekstraksi berbasis Regex + LLM Hybrid, serta ingestion ke dalam Neo4j Property Graph.

---

## 📁 Struktur Proyek Akhir

```text
pskp-legal-graph/
├── data/
│   ├── raw/                        # File PDF asli (14 dokumen)
│   ├── processed/                  # Hasil segmentasi JSON (hierarki Regulasi > Pasal > Ayat)
│   ├── triplets/                   # Triplet hasil ekstraksi (Regex + LLM)
│   └── corpus_manifest.json        # Inventaris dan status ekstraksi seluruh dokumen
├── ontology/
│   ├── schema.json                 # Skema graph ketat (node, property, dan relasi yang diizinkan)
│   ├── constraints.md              # 8 Constraint Rules validasi
│   └── few_shot_examples.json      # Contoh ekstraksi sebagai few-shot prompt LLM
├── scripts/
│   ├── extract.py                  # Core script: Ekstraksi, pembersihan, Regex + LLM
│   ├── batch_extract.py            # Batch processing untuk seluruh manifest
│   ├── ingest_neo4j.py             # Script ingestion ke Neo4j via Cypher (Idempotent MERGE)
│   ├── init_constraints.cypher     # Inisialisasi DB Constraints dan Index
│   └── validate_graph.py           # Verifikasi integritas Graph (Issue #5)
├── evaluation/                     # Direktori untuk Ground Truth Dataset (Issue #6)
├── docker-compose.yml              # Konfigurasi container Neo4j
├── requirements.txt                # Dependensi Python
├── .env                            # Env vars untuk API keys (OpenAI, Neo4j)
└── README.md                       # Panduan ini
```

---

## 🚀 Instalasi & Setup Database (Docker)

1. **Instal Library Python:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Konfigurasi Environment:**
   Salin `.env.example` (jika ada) ke `.env`, lalu isi:
   ```env
   OPENAI_API_KEY=your_key
   NEO4J_PASSWORD=pskp_admin_2026
   ```

3. **Jalankan Database Neo4j via Docker-compose:**
   ```bash
   docker-compose up -d
   ```
   > Dashboard Neo4j dapat diakses di `http://localhost:7474` dengan kredensial `neo4j` / `pskp_admin_2026`.

4. **Inisialisasi Database Constraints:**
   ```bash
   python scripts/ingest_neo4j.py --init
   ```

---

## 🛠️ Step-by-Step Penggunaan Pipeline

Proses ekstraksi ke ingestion dilakukan secara berurutan sesuai issue yang ada. Pastikan file PDF dari korpus (11 PP, UU 20/2003, UU 14/2005, UUD 1945) telah berada di dalam folder `data/raw/` dan metadata terdaftar di `data/corpus_manifest.json`.

### 1. Ekstraksi Batch (Issue #2 & #3)
Menjalankan ekstraksi teks, segmentasi struktural (ke format Hierarki JSON), dilanjutkan dengan ekstraksi Triplet menggunakan Regex (relasi eksplisit) dan LLM Hybrid (relasi semantik).

```bash
python scripts/batch_extract.py
```
*Gunakan `python scripts/batch_extract.py --force` jika ingin memproses ulang seluruh file.*
*Gunakan `python scripts/extract.py --file REG-PP-4-2022` untuk ekstraksi spesifik.*

### 2. Memasukkan Triplet ke Neo4j (Issue #4)
Membaca seluruh triplet hasil dari pipeline sebelumnya dan mengingestinya menggunakan teknik idempotent (MERGE).

```bash
python scripts/ingest_neo4j.py
```
> Script ini otomatis akan memastikan node dibuat sebelum edge dan melaporkan keseluruhan proses di direktori `/reports/ingestion_report.json`.

### 3. Validasi Graph (Issue #5)
Mendeteksi apakah terdapat *orphaned nodes* serta menguji jalur rujukan logika hukum untuk memastikan graph layak dan bisa merespon pertanyaan hierarki.

```bash
python scripts/validate_graph.py
```

---

## 🚧 Dataset Evaluasi Ground Truth (Issue #6)

Fase berikutnya (setelah graph valid) adalah pembuatan dataset evaluasi konflik hukum. Silakan lihat `PSKP_KG_Detailed_Issues.md` bagian Issue #6 untuk instruksi manual review dari kandidat konflik dan menyusun set lengkap Ground Truth.

## 📜 Lisensi
MIT
