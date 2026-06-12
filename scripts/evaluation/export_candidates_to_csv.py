import json
import csv
import os

input_path = "evaluation/candidates_for_review.json"
output_path = "evaluation/candidates_overview.csv"

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    exit(1)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Buat CSV
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Header yang ramah dibaca manusia
    writer.writerow([
        "ID Skenario", 
        "Kategori Konflik", 
        "Regulasi A", 
        "Pasal A", 
        "Isi Teks A", 
        "Regulasi B", 
        "Pasal B", 
        "Isi Teks B", 
        "Subjek / Topik Fokus",
        "STATUS PAKAR (Isi: Valid/Tidak)"
    ])
    
    for item in data:
        cat = item["kategori"]
        rec = item["data_record"]
        
        # Mengekstrak fields sesuai kategori
        if cat == "Direct_Contradiction":
            reg_a = rec.get("r1")
            pasal_a = rec.get("p1")
            teks_a = f"({rec.get('jenis1')}) {rec.get('t1')}"
            reg_b = rec.get("r2")
            pasal_b = rec.get("p2")
            teks_b = f"({rec.get('jenis2')}) {rec.get('t2')}"
            subjek = rec.get("subjek")
            
        elif cat == "Hierarchical_Conflict":
            reg_a = rec.get("req_uu")
            pasal_a = rec.get("p_high")
            teks_a = rec.get("t_high")
            reg_b = rec.get("req_pp")
            pasal_b = rec.get("p_low")
            teks_b = rec.get("t_low")
            subjek = rec.get("overlapping_subject")
            
        elif cat == "Temporal_Inconsistency":
            reg_a = rec.get("reg_lama")
            pasal_a = rec.get("p_lama")
            teks_a = rec.get("t_lama")
            reg_b = rec.get("reg_baru")
            pasal_b = rec.get("p_baru")
            teks_b = rec.get("t_baru")
            subjek = rec.get("subjek_diatur")
            
        elif cat == "Regulatory_Overlap":
            reg_a = rec.get("r1")
            pasal_a = rec.get("p1")
            teks_a = rec.get("t1")
            reg_b = rec.get("r2")
            pasal_b = rec.get("p2")
            teks_b = rec.get("t2")
            subjek = rec.get("overlapping_subject")
            
        else:
            continue
            
        writer.writerow([
            item["scenario_id"], 
            cat, 
            reg_a, 
            pasal_a, 
            teks_a.replace("\n", " ") if teks_a else "", 
            reg_b, 
            pasal_b, 
            teks_b.replace("\n", " ") if teks_b else "", 
            subjek,
            "" # Kolom kosong untuk diisi pakar
        ])

print(f"Sukses mengekstrak {len(data)} kandidat konflik.")
print(f"File siap dibuka di Excel/Spreadsheet: {output_path}")
