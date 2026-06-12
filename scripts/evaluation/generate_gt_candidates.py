"""
PSKP Knowledge Graph — Ground Truth Candidate Generator (Issue #6, Task 6.4)
=============================================================================

Skrip ini akan mengeksekusi beberapa query Cypher pada Neo4j untuk
mengidentifikasi *kandidat* konflik hukum (4 kategori). Sesuai pipeline,
output dari file ini (dalam folder evaluation/) harus di-review secara manual
oleh domain expert untuk dikurasi menjadi final "Ground Truth".

Kategori Konflik:
1. Direct_Contradiction
2. Hierarchical_Conflict
3. Temporal_Inconsistency
4. Regulatory_Overlap

Usage:
    python scripts/evaluation/generate_gt_candidates.py
"""

import os
import json
import logging
from datetime import datetime, timezone

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

URI = _require_env("NEO4J_URI")
AUTH = (_require_env("NEO4J_USERNAME"), _require_env("NEO4J_PASSWORD"))
EVAL_DIR = "./evaluation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_QUERIES = {
    "Direct_Contradiction": """
        // Kandidat Direct Contradiction: 2 Pasal mengatur Subjek Hukum yg sama dengan jenis_pengaturan berbeda (misal Hak vs Larangan)
        MATCH (r1:Regulasi)-[:MEMILIKI_PASAL]->(p1:Pasal)-[e1:MENGATUR_TOPIK]->(s:Subjek_Hukum)
        MATCH (r2:Regulasi)-[:MEMILIKI_PASAL]->(p2:Pasal)-[e2:MENGATUR_TOPIK]->(s)
        WHERE r1.id <= r2.id AND p1.id <> p2.id
          AND e1.jenis_pengaturan <> e2.jenis_pengaturan
        RETURN r1.id AS r1, r2.id AS r2, p1.id AS p1, p2.id AS p2, 
               p1.teks_isi AS t1, p2.teks_isi AS t2, s.id AS subjek, 
               e1.jenis_pengaturan AS jenis1, e2.jenis_pengaturan AS jenis2
        LIMIT 50
    """,
    "Hierarchical_Conflict": """
        // Kandidat Hierarchical Conflict: Pasal PP dan UU mengatur Subjek Hukum yang sama
        MATCH (pp_pasal:Pasal)<-[:MEMILIKI_PASAL]-(pp:Regulasi {jenis:"PP"})
        MATCH (uu_pasal:Pasal)<-[:MEMILIKI_PASAL]-(uu:Regulasi {jenis:"UU"})
        MATCH (pp_pasal)-[:MENGATUR_TOPIK]->(subjek:Subjek_Hukum)<-[:MENGATUR_TOPIK]-(uu_pasal)
        RETURN pp.id AS req_pp, uu.id AS req_uu, pp_pasal.id AS p_low, uu_pasal.id AS p_high,
               pp_pasal.teks_isi AS t_low, uu_pasal.teks_isi AS t_high, subjek.id AS overlapping_subject
        LIMIT 50
    """,
    "Temporal_Inconsistency": """
        // Kandidat Temporal Inconsistency: 2 PP tahun berbeda terkait hal dan subjek hukum yang sama, tanpa relasi MENCABUT
        MATCH (p_baru:Regulasi {jenis:"PP"})-[:MEMILIKI_PASAL]->(pa_baru:Pasal)-[:MENGATUR_TOPIK]->(subjek:Subjek_Hukum)
        MATCH (p_lama:Regulasi {jenis:"PP"})-[:MEMILIKI_PASAL]->(pa_lama:Pasal)-[:MENGATUR_TOPIK]->(subjek)
        WHERE p_baru.tahun > p_lama.tahun
          AND NOT (p_baru)-[:MENCABUT]->(p_lama)
        RETURN p_baru.id AS reg_baru, p_lama.id AS reg_lama, pa_baru.id AS p_baru, pa_lama.id AS p_lama,
               pa_baru.teks_isi AS t_baru, pa_lama.teks_isi AS t_lama, subjek.id AS subjek_diatur
        LIMIT 50
    """,
    "Regulatory_Overlap": """
        // Kandidat Regulatory Overlap: 2 PP berbeda yang mengatur subjek yang sama tanpa relasi hierarki
        MATCH (r1:Regulasi {jenis:"PP"})-[:MEMILIKI_PASAL]->(p1:Pasal)-[:MENGATUR_TOPIK]->(subjek:Subjek_Hukum)
        MATCH (r2:Regulasi {jenis:"PP"})-[:MEMILIKI_PASAL]->(p2:Pasal)-[:MENGATUR_TOPIK]->(subjek)
        WHERE r1.id < r2.id
        RETURN r1.id AS r1, r2.id AS r2, p1.id AS p1, p2.id AS p2,
               p1.teks_isi AS t1, p2.teks_isi AS t2, subjek.id AS overlapping_subject
        LIMIT 50
    """
}

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(EVAL_DIR, "cypher_queries"), exist_ok=True)

    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {URI}")
    except Exception as e:
        logger.error(f"Cannot connect to Neo4j: {e}")
        return

    all_candidates = []
    
    with driver.session() as session:
        for kategori, query in CANDIDATE_QUERIES.items():
            logger.info(f"Executing candidate query for: {kategori}...")
            try:
                result = session.run(query)
                records = [dict(r) for r in result]
                logger.info(f"  Found {len(records)} candidates.")
                
                for idx, record in enumerate(records):
                    candidate_id = f"GT-CAND-{kategori[:3].upper()}-{idx+1:03d}"
                    all_candidates.append({
                        "scenario_id": candidate_id,
                        "kategori": kategori,
                        "deskripsi_awal": "Auto-identified. Review needed.",
                        "data_record": record,
                        "gold_answer": None,
                        "status_review": "pending"
                    })
            except Exception as e:
                logger.error(f"  Query Failed: {e}")

    output_path = os.path.join(EVAL_DIR, "candidates_for_review.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    driver.close()
    
    logger.info("═" * 60)
    logger.info(f"Candidates generated: {len(all_candidates)}")
    logger.info(f"Please hand over {output_path} to the domain experts for manual review.")
    logger.info("Once reviewed, compile the valid conflicts into pskp_ground_truth.json (Issue #6).")
    logger.info("═" * 60)

if __name__ == "__main__":
    main()
