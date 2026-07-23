# Label Analysis Report

## Cara Menjalankan

```bash
.venv/bin/python scripts/analysis/analyze_label_performance.py
```

## Tujuan Analisis

Analisis ini membandingkan RAG dan GraphRAG pada output evaluasi yang sudah ada, lalu mengelompokkan delta performa berdasarkan task asli dataset dan kategori heuristic tambahan.

## Catatan Label Dataset

Kolom metadata.label berbentuk granular seperti qna-1082, draft-345, revision-..., dan sejenisnya. Full label digunakan sebagai sample_id. Prefix label digunakan untuk memetakan sample ke task asli dataset. Analisis utama menggunakan dataset_task; question_intent, question_word, reference_type, dan answer_length_bin hanya analisis tambahan berbasis heuristic.

## Konfigurasi Terdeteksi

| model | training_state | method | rows | source_kind | source_file |
| --- | --- | --- | --- | --- | --- |
| llama2-7b | fine-tuned | graphrag | 1701 | eval_rows | outputs/reports/part1_llama2_finetune_graphrag_newsplit_eval_rows.csv |
| llama2-7b | fine-tuned | only | 1701 | eval_rows | outputs/reports/part1_llama2_finetune_only_newsplit_eval_rows.csv |
| llama2-7b | fine-tuned | rag | 1701 | eval_rows | outputs/reports/part1_llama2_finetune_rag_newsplit_eval_rows.csv |
| llama2-7b | zero-shot | graphrag | 1701 | eval_rows | outputs/reports/part1_llama2_zeroshot_graphrag_newsplit_eval_rows.csv |
| llama2-7b | zero-shot | only | 1701 | eval_rows | outputs/reports/part1_llama2_zeroshot_newsplit_eval_rows.csv |
| llama2-7b | zero-shot | rag | 1701 | eval_rows | outputs/reports/part1_llama2_zeroshot_rag_newsplit_eval_rows.csv |
| wizardlm-13b | fine-tuned | graphrag | 1701 | eval_rows | outputs/reports/part1_wizardlm_finetune_graphrag_newsplit_eval_rows.csv |
| wizardlm-13b | fine-tuned | only | 1701 | eval_rows | outputs/reports/part1_wizardlm_finetune_only_newsplit_eval_rows.csv |
| wizardlm-13b | fine-tuned | rag | 1701 | eval_rows | outputs/reports/part1_wizardlm_finetune_rag_newsplit_eval_rows.csv |
| wizardlm-13b | zero-shot | graphrag | 1701 | eval_rows | outputs/reports/part1_wizardlm_zeroshot_graphrag_newsplit_eval_rows.csv |
| wizardlm-13b | zero-shot | only | 1701 | eval_rows | outputs/reports/part1_wizardlm_zeroshot_newsplit_eval_rows.csv |
| wizardlm-13b | zero-shot | rag | 1701 | eval_rows | outputs/reports/part1_wizardlm_zeroshot_rag_newsplit_eval_rows.csv |

## Distribusi Label Prefix

| label_prefix | count | percentage |
| --- | --- | --- |
| qna | 324 | 19.0500 |
| summarization | 314 | 18.4600 |
| draft | 292 | 17.1700 |
| revision | 289 | 16.9900 |
| similarity | 251 | 14.7600 |
| element | 231 | 13.5800 |

## Distribusi Dataset Task

| dataset_task | count | percentage | full_dataset_count | full_dataset_percentage | test_minus_full_percentage |
| --- | --- | --- | --- | --- | --- |
| legal_overlapping_analysis | 251 | 14.7600 | 1087 | 12.7800 | 1.9800 |
| element_extraction | 231 | 13.5800 | 890 | 10.4600 | 3.1200 |
| legal_qna | 324 | 19.0500 | 1794 | 21.0900 | -2.0400 |
| legal_summarization | 314 | 18.4600 | 1415 | 16.6300 | 1.8300 |
| drafting_revisions | 289 | 16.9900 | 1571 | 18.4700 | -1.4800 |
| drafting_provisions | 292 | 17.1700 | 1750 | 20.5700 | -3.4000 |

## Group Paling Diuntungkan GraphRAG Berdasarkan Dataset Task

| group_value | model | training_state | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drafting_provisions | wizardlm-13b | fine-tuned | 292 | 0.0432 | 0.0364 | 0.0071 | 0.0866 | GraphRAG unggul |
| drafting_provisions | llama2-7b | fine-tuned | 292 | 0.0447 | 0.0341 | 0.0063 | 0.0851 | GraphRAG unggul |
| element_extraction | llama2-7b | fine-tuned | 231 | 0.0261 | 0.0308 | 0.0060 | 0.0629 | GraphRAG unggul |
| legal_summarization | wizardlm-13b | fine-tuned | 314 | 0.0270 | 0.0234 | 0.0044 | 0.0549 | GraphRAG unggul |
| drafting_revisions | llama2-7b | zero-shot | 289 | 0.0287 | 0.0090 | 0.0088 | 0.0464 | GraphRAG unggul |
| element_extraction | wizardlm-13b | fine-tuned | 231 | 0.0168 | 0.0247 | 0.0041 | 0.0455 | GraphRAG unggul |
| drafting_revisions | llama2-7b | fine-tuned | 289 | 0.0148 | 0.0224 | 0.0014 | 0.0386 | GraphRAG unggul |
| legal_summarization | llama2-7b | zero-shot | 314 | 0.0230 | 0.0077 | 0.0070 | 0.0377 | GraphRAG unggul |
| legal_qna | llama2-7b | fine-tuned | 324 | 0.0200 | 0.0152 | 0.0014 | 0.0367 | GraphRAG unggul |
| drafting_revisions | wizardlm-13b | fine-tuned | 289 | 0.0174 | 0.0157 | -0.0006 | 0.0326 | GraphRAG unggul |
| legal_overlapping_analysis | llama2-7b | fine-tuned | 251 | 0.0126 | 0.0085 | 0.0023 | 0.0234 | GraphRAG unggul |
| legal_qna | llama2-7b | zero-shot | 324 | 0.0062 | 0.0036 | 0.0034 | 0.0132 | GraphRAG unggul |
| legal_summarization | llama2-7b | fine-tuned | 314 | 0.0030 | 0.0095 | 0.0002 | 0.0127 | GraphRAG unggul |
| element_extraction | llama2-7b | zero-shot | 231 | 0.0046 | 0.0008 | 0.0016 | 0.0070 | GraphRAG unggul |
| legal_qna | wizardlm-13b | fine-tuned | 324 | 0.0025 | 0.0052 | -0.0011 | 0.0066 | GraphRAG unggul |

## Group Yang Lebih Cocok Dengan RAG Berdasarkan Dataset Task

| group_value | model | training_state | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| element_extraction | wizardlm-13b | zero-shot | 231 | -0.0666 | -0.0987 | -0.0095 | -0.1748 | RAG unggul |
| legal_overlapping_analysis | wizardlm-13b | zero-shot | 251 | -0.0533 | -0.0958 | -0.0142 | -0.1633 | RAG unggul |
| legal_qna | wizardlm-13b | zero-shot | 324 | -0.0440 | -0.0980 | -0.0132 | -0.1551 | RAG unggul |
| legal_summarization | wizardlm-13b | zero-shot | 314 | -0.0215 | -0.0621 | -0.0069 | -0.0905 | RAG unggul |
| drafting_provisions | wizardlm-13b | zero-shot | 292 | -0.0292 | -0.0293 | -0.0021 | -0.0605 | RAG unggul |
| drafting_revisions | wizardlm-13b | zero-shot | 289 | -0.0146 | -0.0384 | -0.0071 | -0.0601 | RAG unggul |
| drafting_provisions | llama2-7b | zero-shot | 292 | -0.0197 | -0.0116 | -0.0191 | -0.0504 | RAG unggul |
| legal_overlapping_analysis | llama2-7b | zero-shot | 251 | -0.0172 | -0.0034 | -0.0039 | -0.0245 | RAG unggul |
| legal_overlapping_analysis | wizardlm-13b | fine-tuned | 251 | -0.0111 | 0.0006 | -0.0013 | -0.0118 | Campuran |

## Analisis Tambahan: Question Word

| question_word | count | percentage |
| --- | --- | --- |
| apa | 842 | 49.5000 |
| lainnya | 613 | 36.0400 |
| siapa | 114 | 6.7000 |
| bagaimana | 78 | 4.5900 |
| berapa | 23 | 1.3500 |
| apakah | 21 | 1.2300 |
| kapan | 4 | 0.2400 |
| di mana | 3 | 0.1800 |
| mengapa | 3 | 0.1800 |

## Analisis Tambahan: Question Intent

| question_intent | count | percentage |
| --- | --- | --- |
| general_qna | 968 | 56.9100 |
| obligation | 223 | 13.1100 |
| authority | 208 | 12.2300 |
| definition | 129 | 7.5800 |
| procedure | 99 | 5.8200 |
| sanction | 25 | 1.4700 |
| amount | 24 | 1.4100 |
| requirement | 20 | 1.1800 |
| time | 3 | 0.1800 |
| permission | 2 | 0.1200 |

## Analisis Tambahan: Reference Type

| reference_type | count | percentage |
| --- | --- | --- |
| multi_reference | 871 | 51.2100 |
| pasal_ayat | 360 | 21.1600 |
| pasal_only | 333 | 19.5800 |
| other_reference | 111 | 6.5300 |
| ayat_only | 26 | 1.5300 |

## Heuristic Group Paling Diuntungkan GraphRAG

| group_column | group_value | model | training_state | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question_intent | permission | wizardlm-13b | zero-shot | 2 | 0.1731 | 0.0857 | 0.0151 | 0.2740 | GraphRAG unggul |
| question_intent | time | wizardlm-13b | fine-tuned | 3 | 0.0755 | 0.0885 | 0.0020 | 0.1660 | GraphRAG unggul |
| answer_length_bin | long | wizardlm-13b | fine-tuned | 20 | 0.0654 | 0.0707 | 0.0217 | 0.1579 | GraphRAG unggul |
| question_intent | time | llama2-7b | fine-tuned | 3 | 0.0416 | 0.1078 | -0.0056 | 0.1438 | GraphRAG unggul |
| answer_length_bin | long | llama2-7b | zero-shot | 20 | 0.0865 | 0.0343 | 0.0218 | 0.1425 | GraphRAG unggul |
| answer_length_bin | long | llama2-7b | fine-tuned | 20 | 0.0709 | 0.0554 | 0.0147 | 0.1410 | GraphRAG unggul |
| question_word | bagaimana | wizardlm-13b | fine-tuned | 78 | 0.0603 | 0.0676 | 0.0102 | 0.1382 | GraphRAG unggul |
| question_intent | procedure | wizardlm-13b | fine-tuned | 99 | 0.0610 | 0.0643 | 0.0097 | 0.1350 | GraphRAG unggul |
| question_word | mengapa | llama2-7b | zero-shot | 3 | 0.0601 | 0.0324 | 0.0364 | 0.1288 | GraphRAG unggul |
| question_word | kapan | wizardlm-13b | fine-tuned | 4 | 0.0566 | 0.0664 | 0.0015 | 0.1245 | GraphRAG unggul |
| question_word | kapan | llama2-7b | fine-tuned | 4 | 0.0312 | 0.0808 | -0.0042 | 0.1078 | GraphRAG unggul |
| question_word | lainnya | wizardlm-13b | fine-tuned | 613 | 0.0409 | 0.0390 | 0.0063 | 0.0862 | GraphRAG unggul |

## Heuristic Group Yang Lebih Cocok Dengan RAG

| group_column | group_value | model | training_state | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question_word | di mana | wizardlm-13b | zero-shot | 3 | -0.2350 | -0.2688 | -0.0329 | -0.5367 | RAG unggul |
| question_word | kapan | wizardlm-13b | zero-shot | 4 | -0.1900 | -0.3002 | -0.0420 | -0.5322 | RAG unggul |
| question_intent | requirement | wizardlm-13b | zero-shot | 20 | -0.2427 | -0.2155 | -0.0316 | -0.4897 | RAG unggul |
| question_intent | time | wizardlm-13b | zero-shot | 3 | -0.1832 | -0.2678 | -0.0352 | -0.4862 | RAG unggul |
| question_intent | sanction | wizardlm-13b | zero-shot | 25 | -0.1443 | -0.1608 | -0.0199 | -0.3250 | RAG unggul |
| question_word | apakah | wizardlm-13b | zero-shot | 21 | -0.1148 | -0.1625 | -0.0153 | -0.2926 | RAG unggul |
| question_intent | definition | wizardlm-13b | zero-shot | 129 | -0.1033 | -0.1643 | -0.0195 | -0.2872 | RAG unggul |
| question_word | di mana | llama2-7b | zero-shot | 3 | -0.1847 | -0.0496 | -0.0351 | -0.2695 | RAG unggul |
| question_word | bagaimana | wizardlm-13b | zero-shot | 78 | -0.0660 | -0.1370 | -0.0122 | -0.2152 | RAG unggul |
| question_word | siapa | wizardlm-13b | zero-shot | 114 | -0.0506 | -0.1282 | -0.0176 | -0.1965 | RAG unggul |
| question_intent | procedure | wizardlm-13b | zero-shot | 99 | -0.0579 | -0.1199 | -0.0116 | -0.1895 | RAG unggul |
| question_intent | sanction | wizardlm-13b | fine-tuned | 25 | -0.0930 | -0.0743 | -0.0086 | -0.1759 | RAG unggul |

## Zero-Shot Vs Fine-Tuned

| training_state | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| fine-tuned | 3402 | 0.0182 | 0.0187 | 0.0024 | 0.0394 | GraphRAG unggul |
| zero-shot | 3402 | -0.0159 | -0.0340 | -0.0045 | -0.0543 | RAG unggul |

## LLaMA 2 7B Vs WizardLM 13B

| model | count | delta_meteor | delta_rouge_l | delta_cosine | delta_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| llama2-7b | 3402 | 0.0125 | 0.0105 | 0.0013 | 0.0243 | GraphRAG unggul |
| wizardlm-13b | 3402 | -0.0101 | -0.0258 | -0.0033 | -0.0392 | RAG unggul |

## Analisis Khusus BLEU-4 Fine-Tuned

Bagian ini menggunakan BLEU-4 agregat yang konsisten dengan metrik utama paper. Nilai delta positif berarti GraphRAG lebih tinggi daripada RAG; nilai negatif berarti RAG lebih tinggi.

### BLEU-4 Per Dataset Task

| group_value | count | bleu4_rag | bleu4_graphrag | delta_bleu4 | bleu4_winner |
| --- | --- | --- | --- | --- | --- |
| drafting_provisions | 584 | 0.2902 | 0.3259 | 0.0357 | GraphRAG |
| element_extraction | 462 | 0.3951 | 0.4252 | 0.0301 | GraphRAG |
| legal_summarization | 628 | 0.3779 | 0.3843 | 0.0064 | GraphRAG |
| drafting_revisions | 578 | 0.4390 | 0.4283 | -0.0107 | RAG |
| legal_qna | 648 | 0.4212 | 0.3947 | -0.0266 | RAG |
| legal_overlapping_analysis | 502 | 0.4789 | 0.4468 | -0.0321 | RAG |

### Mean Sentence BLEU-4 Per Dataset Task

| group_value | count | mean_bleu4_rag | mean_bleu4_graphrag | delta_bleu | positive_delta_percentage | negative_delta_percentage |
| --- | --- | --- | --- | --- | --- | --- |
| drafting_provisions | 584 | 0.2634 | 0.2914 | 0.0280 | 36.4700 | 29.2800 |
| drafting_revisions | 578 | 0.4131 | 0.4342 | 0.0211 | 33.9100 | 33.5650 |
| element_extraction | 462 | 0.3479 | 0.3681 | 0.0202 | 33.5450 | 26.4050 |
| legal_summarization | 628 | 0.3359 | 0.3544 | 0.0185 | 34.7150 | 29.6150 |
| legal_qna | 648 | 0.3739 | 0.3908 | 0.0169 | 31.0200 | 27.1600 |
| legal_overlapping_analysis | 502 | 0.4661 | 0.4750 | 0.0089 | 34.4600 | 32.0700 |

## Insight Utama

- Terdeteksi 12 konfigurasi evaluasi; 12 konfigurasi memiliki 1701 row sesuai test split penuh.
- metadata.label dipertahankan sebagai sample_id, sedangkan prefix label dipakai untuk memetakan sample ke 6 task asli dataset.
- Dataset task terbesar pada test split adalah legal_qna (324 sample).
- Task paling positif untuk GraphRAG secara agregat adalah drafting_provisions (delta_score=0.015191).
- Task paling positif untuk RAG secara agregat adalah legal_overlapping_analysis (delta_score=-0.044048).
- Sinyal dataset_task belum terlalu kuat; hasil lebih aman dipakai sebagai analisis tambahan, bahan diskusi internal, atau jawaban saat tanya-jawab juri.

## Contoh Kualitatif GraphRAG Unggul

- `wizardlm-13b` `fine-tuned` sample `draft-345` task `drafting_provisions` (general_qna, multi_reference), delta_score=2.2006.
  Pertanyaan: Buat draft pasal baru berdasarkan Pasal 32 Ayat 2 PP 48 Tahun 2008
  RAG: Pasal [Nomor Pasal] [Tentang Isi Pasal]
  GraphRAG: Pendanaan biaya investasi untuk lahan satuan pendidikan, baik formal maupun nonformal, yang diselenggarakan masyarakat menjadi tanggung jawab penyelenggara atau satuan pendidikan yang bersangkutan.
- `llama2-7b` `fine-tuned` sample `draft-1387` task `drafting_provisions` (general_qna, pasal_only), delta_score=2.1987.
  Pertanyaan: Buatlah draft pasal baru berdasarkan ketentuan peralihan Pasal 217 PP 17 Tahun 2010.
  RAG: Pasal [Nomor Pasal] [Tentang Isi Pasal]
  GraphRAG: Satuan pendidikan yang dinyatakan oleh pendirinya sebagai sekolah internasional sebelum berlakunya Peraturan Pemerintah ini, paling lambat 3 (tiga) tahun sejak Peraturan Pemerintah ini berlaku, wajib menyesuaikan menjadi: a. satuan pendidikan kategori standar atau katagori mandiri sesuai dengan peraturan yang mengatur tentang standar nasional pendidikan; ...

## Contoh Kualitatif RAG Unggul

- `wizardlm-13b` `fine-tuned` sample `draft-860` task `drafting_provisions` (general_qna, multi_reference), delta_score=-2.0680.
  Pertanyaan: Buatlah draft pasal baru berdasarkan Pasal 28 Ayat (1) dari PP 57 Tahun 2021.
  RAG: Pasal [Nomor Pasal], [Tentang] [Isi Pasal 28 Ayat (1)]
  GraphRAG: Perencanaan kegiatan Pendidikan sebagaimana dimaksud dalam Pasal 27 bertujuan untuk peningkatan kualitas proses dan hasil belajar secara berkelanjutan berdasarkan evaluasi diri Satuan Pendidikan.
- `llama2-7b` `fine-tuned` sample `draft-2054` task `drafting_provisions` (authority, multi_reference), delta_score=-1.9003.
  Pertanyaan: Tuliskan siapa yang bertanggung jawab dalam melaksanakan pengembangan Standar Nasional Pendidikan serta pemantauan dan pelaporan pencapaiannya secara nasional.
  RAG: Badan yang menyelenggarakan tugas dan fungsi standardisasi, penjaminan, dan pengendalian mutu Pendidikan bertanggung jawab dalam melaksanakan pengembangan Standar Nasional Pendidikan serta pemantauan dan pelaporan pencapaiannya secara nasional.
  GraphRAG: Badan sebagaimana dimaksud pada ayat (1) bertanggung jawab kepada Menteri.

## Keterbatasan Analisis

- Kategori question_intent dibuat dengan heuristic rule-based, bukan anotasi manual.
- Dataset task dipetakan dari prefix metadata.label; jika ada prefix baru, mapping perlu diperbarui.
- Delta dihitung dari metrik otomatis, sehingga belum menggantikan penilaian kualitatif hukum oleh evaluator manusia.
- Analisis ini memakai output evaluasi yang sudah ada dan tidak menjalankan ulang inference model.
- Jika sebuah kategori memiliki count kecil, interpretasinya harus diperlakukan sebagai indikasi awal.

## Catatan Plot

- SVG plots written: reports/label_analysis/plots/rag_vs_graphrag_total_score_by_dataset_task_zero_shot.svg, reports/label_analysis/plots/rag_vs_graphrag_total_score_by_dataset_task_fine_tuned.svg, reports/label_analysis/plots/delta_score_by_dataset_task_training_state.svg
- PNG plots skipped because matplotlib is unavailable: ModuleNotFoundError: No module named 'matplotlib'
- BLEU-4 SVG plot written: reports/label_analysis/plots/delta_bleu4_by_dataset_task_fine_tuned.svg
- BLEU-4 RAG vs GraphRAG SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_llama2_7b_zero_shot.svg
- BLEU-4 RAG vs GraphRAG SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_llama2_7b_fine_tuned.svg
- BLEU-4 RAG vs GraphRAG SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_wizardlm_13b_zero_shot.svg
- BLEU-4 RAG vs GraphRAG SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_wizardlm_13b_fine_tuned.svg
- Combined BLEU-4 RAG vs GraphRAG SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_combined.svg
- Fine-tuned two-model BLEU-4 SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_llama_wizard.svg
- Paper-style single-model BLEU-4 SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_llama2_7b_paper.svg
- Paper-style single-model BLEU-4 SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_wizardlm_13b_paper.svg
- Paper-style two-model BLEU-4 SVG plot written: reports/label_analysis/plots/rag_vs_graphrag_bleu4_by_dataset_task_fine_tuned_two_models_paper.svg

## Sinyal Untuk Paper

Sinyal dataset_task belum terlalu kuat; hasil lebih aman dipakai sebagai analisis tambahan, bahan diskusi internal, atau jawaban saat tanya-jawab juri.
