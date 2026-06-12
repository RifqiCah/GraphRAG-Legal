# Constraint Rules — PSKP Legal Knowledge Graph

Dokumen ini mendokumentasikan 8 aturan validasi yang **wajib** ditegakkan secara programatik sebelum graph index dibangun.

---

## RULE-01: Validasi Edge MENCABUT
Edge `MENCABUT` **hanya boleh** antara dua Node berlabel `Regulasi`.
- Source: `Regulasi`
- Target: `Regulasi`

## RULE-02: Validasi Edge MEMILIKI_PASAL
Edge `MEMILIKI_PASAL` **hanya boleh** dari Node `Regulasi` ke Node `Pasal`.
- Source: `Regulasi`
- Target: `Pasal`

## RULE-03: Uniqueness ID Global
Properti `id` pada **semua Node** harus unik secara global. Tidak boleh ada dua Node (dari label mana pun) yang memiliki `id` sama.

## RULE-04: Pasal Tanpa Ayat
Node `Pasal` dengan `nomor_ayat = "0"` berarti pasal tersebut **tidak memiliki ayat** — ini bukan error, melainkan representasi valid.

## RULE-05: Edge MENGAMANDEMEN Wajib Memiliki `teks_lama`
Edge `MENGAMANDEMEN` **wajib** menyertakan properti `teks_lama` yang berisi teks pasal sebelum diubah. Tanpa properti ini, triplet ditolak.

## RULE-06: Regulasi Dicabut Tidak Boleh Menjadi Source MENGACU_PADA
Node `Regulasi` dengan `status = "Dicabut"` **tidak boleh** menjadi source pada Edge `MENGACU_PADA`. Logikanya: regulasi yang sudah tidak berlaku tidak seharusnya mengacu ke regulasi lain.

## RULE-07: Enum validation pada `jenis` Regulasi
Nilai field `jenis` pada Node `Regulasi` harus berasal dari Enum: `UUD`, `UU`, `PP`, `Perpres`, `Permendikbud`. Nilai di luar daftar ini ditolak.

## RULE-08: Edge BERTENTANGAN_DENGAN Hanya untuk Ground Truth
Edge `BERTENTANGAN_DENGAN` **hanya dibuat** pada fase Issue #6 (Ground Truth construction), **bukan** pada fase ekstraksi otomatis (Issue #3). Pipeline ekstraksi otomatis harus menolak triplet dengan predicate ini.
