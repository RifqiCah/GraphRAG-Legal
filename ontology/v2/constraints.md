# Constraint Rules — PSKP Legal Knowledge Graph v2

Dokumen ini mendefinisikan aturan validasi untuk ontology v2. Aturan ini menjadi kontrak untuk parser, extractor, graph builder lokal, retrieval, dan evaluasi.

## RULE-V2-01: ID Unik Global

Semua node wajib memiliki `id` unik secara global.

## RULE-V2-02: Regulasi Adalah Root Struktur Dokumen

Node `Bab`, `Bagian`, `Paragraf`, `Pasal`, `Ayat`, `SubItem`, `Konsiderans`, `Dasar_Hukum`, `Diktum`, `Penjelasan`, `Lampiran`, `Penutup`, dan `Chunk` wajib dapat ditelusuri balik ke satu `Regulasi`.

## RULE-V2-03: Batang Tubuh Harus Menjaga Urutan Dokumen

Edge struktural seperti `MEMILIKI_BAB`, `MEMILIKI_PASAL`, `MEMILIKI_AYAT`, dan `MEMILIKI_SUBITEM` wajib memiliki properti `urutan` jika source dapat memiliki lebih dari satu child.

## RULE-V2-04: Pasal dan Ayat Dipisahkan

Node `Pasal` merepresentasikan pasal normatif. Node `Ayat` merepresentasikan ayat di dalam pasal. Pasal tanpa ayat boleh memiliki `teks_isi`, tetapi pasal dengan ayat sebaiknya menyimpan substansi utama di node `Ayat`.

## RULE-V2-05: SubItem Selalu Punya Parent

Node `SubItem` wajib memiliki `parent_id` yang mengarah ke `Pasal`, `Ayat`, atau `SubItem` lain.

## RULE-V2-06: Penjelasan Bukan Norma Utama

Node `Penjelasan`, `Penjelasan_Umum`, dan `Penjelasan_Pasal` dipakai sebagai interpretasi resmi/pendukung. Node ini tidak boleh menjadi source utama untuk edge konflik final `BERTENTANGAN_DENGAN`.

## RULE-V2-07: Penjelasan Pasal Harus Menjelaskan Pasal Jika Target Tersedia

Jika `Penjelasan_Pasal.target_pasal_id` tersedia, harus ada edge `MENJELASKAN` dari `Penjelasan_Pasal` ke `Pasal`.

## RULE-V2-08: Chunk Wajib Traceable

Setiap `Chunk` wajib memiliki `source_node_id`, `regulasi_id`, `section_type`, `reference_label`, dan `text`. Chunk harus dapat ditelusuri balik ke node sumber dan dokumen asal.

## RULE-V2-09: Embedding Tidak Menggantikan Teks

Embedding hanya indeks retrieval. Node `Chunk` tetap wajib menyimpan `text` dan metadata sumber.

## RULE-V2-10: Rujukan Eksplisit Wajib Menyimpan Konteks

Edge `MENGACU_PADA`, `BERDASARKAN`, `MENGAMANDEMEN`, `MENCABUT`, dan `MENCABUT_PASAL` wajib menyimpan properti `konteks` jika berasal dari teks dokumen atau metadata status.

## RULE-V2-11: Mencabut Sepenuhnya vs Mencabut Sebagian Dipisahkan

Gunakan `MENCABUT` untuk `Regulasi -> Regulasi`. Gunakan `MENCABUT_PASAL` untuk pencabutan sebagian pada `Pasal`, `Ayat`, atau `SubItem`.

## RULE-V2-12: Amandemen Boleh Menargetkan Regulasi atau Unit Norma

Edge `MENGAMANDEMEN` boleh mengarah ke `Regulasi`, `Pasal`, `Ayat`, atau `SubItem`. Jika target spesifik belum bisa di-resolve, gunakan target `Regulasi` dan simpan konteksnya.

## RULE-V2-13: Dasar Hukum Tidak Sama Dengan Konflik

Edge `BERDASARKAN` menunjukkan dasar hukum atau hierarki rujukan. Konflik hierarkis tetap harus dibuktikan dengan isi norma yang bertentangan.

## RULE-V2-14: Topik dan Subjek Dipakai Untuk Candidate Generation

Node `Topik` dan `Subjek_Hukum` dipakai untuk menemukan kandidat konflik atau overlap. Kesamaan topik/subjek belum otomatis berarti konflik.

## RULE-V2-15: Edge Norma Semantik Wajib Berbasis Teks

Edge `MEMBERIKAN_HAK`, `MEMBEBANKAN_KEWAJIBAN`, `MELARANG`, dan `MENDEFINISIKAN` wajib menyimpan konteks/definisi dari teks sumber.

## RULE-V2-16: BERTENTANGAN_DENGAN Bukan Output Ekstraksi Otomatis Final

Edge `BERTENTANGAN_DENGAN` hanya boleh dibuat sebagai:

1. Kandidat konflik dengan `review_status = "candidate"`, atau
2. Ground truth/review final dengan `review_status = "reviewed"`.

Extractor otomatis tidak boleh menganggap konflik sebagai fakta final tanpa review.

## RULE-V2-17: Kategori Konflik Harus Berasal Dari Empat Label

Nilai `jenis_konflik` pada `BERTENTANGAN_DENGAN` wajib salah satu dari:

1. `Direct_Contradiction`
2. `Hierarchical_Conflict`
3. `Temporal_Inconsistency`
4. `Regulatory_Overlap`

## RULE-V2-18: Referenced-Only Node Tetap Sah

Regulasi yang disebut dalam status/rujukan tetapi belum memiliki PDF boleh dibuat sebagai node `Regulasi` dengan `has_full_text = false`.

## RULE-V2-19: Tidak Semua Rujukan Harus Diunduh

Download full-text hanya diperlukan jika dokumen dibutuhkan untuk substansi, kandidat konflik, evaluasi, atau jawaban GraphRAG. Relasi status tetap boleh dibuat dari metadata resmi.

## RULE-V2-20: Semua Evidence Evaluasi Harus Punya Chunk

Dataset konflik, QA, dan retrieval harus menyertakan `chunk_id` atau daftar `evidence_chunk_ids` agar hasil evaluasi dapat ditelusuri ke teks sumber.

## RULE-V2-21: Tanggal Penting Direpresentasikan Sebagai Peristiwa Hukum

Edge `MULAI_BERLAKU` dan `DIUNDANGKAN_PADA` harus mengarah ke node `Peristiwa_Hukum`, bukan ke node regulasi lain.
