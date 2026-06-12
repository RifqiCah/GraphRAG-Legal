"""
PSKP Knowledge Graph — Neo4j Ingestion Script (Issue #4)
========================================================

Script untuk menginisialisasi Neo4j dan mengingesti triplets
tervalidasi dari pipeline ekstraksi ke dalam Knowledge Graph.

Features:
- Docker-ready setup
- Uniqueness constraints & indexes
- Idempotent ingestion (MERGE-based)
- Batch processing (500 triplets per batch)
- Ingestion report generation

Usage:
    python scripts/ingestion/ingest_neo4j.py                   # Ingest semua triplets
    python scripts/ingestion/ingest_neo4j.py --init             # Inisialisasi constraints & indexes saja
    python scripts/ingestion/ingest_neo4j.py --file REG-PP-4-2022  # Ingest satu dokumen
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

URI = _require_env("NEO4J_URI")
AUTH = (_require_env("NEO4J_USERNAME"), _require_env("NEO4J_PASSWORD"))
TRIPLETS_DIR = os.getenv("DATA_TRIPLETS_DIR", "./data/triplets")
PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "./data/processed")
REPORTS_DIR = "./reports"
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Issue #4, Task 4.2: Constraint & Index Initialization
# ═════════════════════════════════════════════════════════════════════════════

INIT_CYPHER = [
    # Uniqueness Constraints
    "CREATE CONSTRAINT regulasi_id_unique IF NOT EXISTS FOR (r:Regulasi) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT pasal_id_unique IF NOT EXISTS FOR (p:Pasal) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT subjek_id_unique IF NOT EXISTS FOR (s:Subjek_Hukum) REQUIRE s.id IS UNIQUE",

    # Indexes untuk query performa
    "CREATE INDEX pasal_topik_index IF NOT EXISTS FOR (p:Pasal) ON (p.topik)",
    "CREATE INDEX regulasi_tahun_index IF NOT EXISTS FOR (r:Regulasi) ON (r.tahun)",
    "CREATE INDEX regulasi_status_index IF NOT EXISTS FOR (r:Regulasi) ON (r.status)",
]


def init_constraints(driver):
    """
    Inisialisasi constraints dan indexes pada Neo4j (Issue #4, Task 4.2).
    """
    logger.info("Initializing constraints and indexes...")
    with driver.session() as session:
        for cypher in INIT_CYPHER:
            try:
                session.run(cypher)
                logger.info(f"  ✅ {cypher[:60]}...")
            except Exception as e:
                logger.warning(f"  ⚠️ {cypher[:60]}... -> {e}")

    logger.info("Constraints and indexes initialized.")


# ═════════════════════════════════════════════════════════════════════════════
# Issue #4, Task 4.3: Ingestion Functions
# ═════════════════════════════════════════════════════════════════════════════

def _determine_label(node_id: str) -> str:
    """Determine Neo4j node label from ID prefix."""
    if node_id.startswith("REG-"):
        return "Regulasi"
    elif node_id.startswith("PSL-"):
        return "Pasal"
    elif node_id.startswith("SBJ-"):
        return "Subjek_Hukum"
    else:
        return "Unknown"


def _build_node_properties(node_id: str, label: str, processed_data: Dict = None) -> Dict:
    """Build minimal node properties based on label and available data."""
    props = {"id": node_id}

    if label == "Regulasi" and processed_data:
        meta = processed_data.get("metadata", {})
        props.update({
            "jenis": meta.get("jenis", ""),
            "nomor": meta.get("nomor", ""),
            "tahun": meta.get("tahun", 0),
            "judul": meta.get("judul", ""),
            "status": meta.get("status", "Berlaku"),
        })
    elif label == "Pasal" and processed_data:
        # Find pasal in processed data
        for pasal in processed_data.get("pasal", []):
            for ayat in pasal.get("ayat", []):
                candidate_id = f"PSL-{processed_data['regulasi_id']}-{pasal['nomor_pasal']}-{ayat['nomor_ayat']}"
                if candidate_id == node_id:
                    props.update({
                        "nomor_pasal": pasal["nomor_pasal"],
                        "nomor_ayat": ayat["nomor_ayat"],
                        "teks_isi": ayat.get("teks_isi", ""),
                        "topik": [],
                        "status": "Aktif",
                    })
                    break

    return props


def ingest_triplet_batch(tx, batch: List[Dict], processed_data: Dict = None):
    """
    Ingest a batch of triplets using MERGE for idempotency (Issue #4, Task 4.3).
    Menggunakan MERGE (bukan CREATE) untuk mencegah duplikasi data.
    """
    for triplet in batch:
        source_id = triplet["source_id"]
        target_id = triplet["target_id"]
        predicate = triplet["predicate"]
        edge_props = triplet.get("properties", {})

        source_label = _determine_label(source_id)
        target_label = _determine_label(target_id)

        source_props = _build_node_properties(source_id, source_label, processed_data)
        target_props = _build_node_properties(target_id, target_label, processed_data)

        # Build dynamic Cypher query
        # Note: Neo4j doesn't support parameterized labels, so we build them dynamically
        # but still use parameterized properties for security
        query = f"""
        MERGE (s:{source_label} {{id: $source_id}})
        SET s += $source_props
        MERGE (t:{target_label} {{id: $target_id}})
        SET t += $target_props
        MERGE (s)-[r:{predicate}]->(t)
        SET r += $edge_props
        """

        try:
            tx.run(query,
                   source_id=source_id,
                   source_props=source_props,
                   target_id=target_id,
                   target_props=target_props,
                   edge_props=edge_props)
        except Exception as e:
            logger.error(f"Failed to ingest triplet {source_id}-[{predicate}]->{target_id}: {e}")
            raise


def ingest_document_nodes(tx, processed_data: Dict):
    """
    Ingest Regulasi node and all Pasal nodes from processed data.
    This ensures all nodes exist before edge ingestion.
    """
    regulasi_id = processed_data["regulasi_id"]
    meta = processed_data.get("metadata", {})

    # Create Regulasi node
    tx.run("""
        MERGE (r:Regulasi {id: $id})
        SET r.jenis = $jenis,
            r.nomor = $nomor,
            r.tahun = $tahun,
            r.judul = $judul,
            r.status = $status
    """,
           id=regulasi_id,
           jenis=meta.get("jenis", ""),
           nomor=meta.get("nomor", ""),
           tahun=meta.get("tahun", 0),
           judul=meta.get("judul", ""),
           status=meta.get("status", "Berlaku"))

    # Create all Pasal nodes
    for pasal in processed_data.get("pasal", []):
        for ayat in pasal.get("ayat", []):
            pasal_id = f"PSL-{regulasi_id}-{pasal['nomor_pasal']}-{ayat['nomor_ayat']}"
            tx.run("""
                MERGE (p:Pasal {id: $id})
                SET p.nomor_pasal = $nomor_pasal,
                    p.nomor_ayat = $nomor_ayat,
                    p.teks_isi = $teks_isi,
                    p.topik = $topik,
                    p.status = $status
            """,
                   id=pasal_id,
                   nomor_pasal=pasal["nomor_pasal"],
                   nomor_ayat=ayat["nomor_ayat"],
                   teks_isi=ayat.get("teks_isi", ""),
                   topik=ayat.get("topik", []),
                   status="Aktif")


# ═════════════════════════════════════════════════════════════════════════════
# Issue #4, Task 4.4: Ingestion Report
# ═════════════════════════════════════════════════════════════════════════════

def get_graph_stats(driver) -> Dict:
    """Query current graph statistics."""
    stats = {
        "nodes_by_label": {},
        "edges_by_type": {},
        "total_nodes": 0,
        "total_edges": 0
    }

    with driver.session() as session:
        # Count nodes by label
        for label in ["Regulasi", "Pasal", "Subjek_Hukum"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            count = result.single()["cnt"]
            stats["nodes_by_label"][label] = count
            stats["total_nodes"] += count

        # Count edges by type
        edge_types = [
            "MEMILIKI_PASAL", "MENGACU_PADA", "MENCABUT", "MENCABUT_PASAL",
            "MENGAMANDEMEN", "MENDAHULUI", "MENGATUR_TOPIK", "BERTENTANGAN_DENGAN"
        ]
        for etype in edge_types:
            result = session.run(f"MATCH ()-[r:{etype}]->() RETURN count(r) AS cnt")
            count = result.single()["cnt"]
            if count > 0:
                stats["edges_by_type"][etype] = count
                stats["total_edges"] += count

    return stats


def generate_report(stats: Dict, errors: List[str], duration: float) -> Dict:
    """Generate ingestion report (Issue #4, Task 4.4)."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_nodes_created": stats["total_nodes"],
        "total_edges_created": stats["total_edges"],
        "nodes_by_label": stats["nodes_by_label"],
        "edges_by_type": stats["edges_by_type"],
        "errors": errors,
        "duration_seconds": round(duration, 2)
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    driver = GraphDatabase.driver(URI, auth=AUTH)
    errors = []
    start_time = time.time()

    try:
        # Verify connectivity
        driver.verify_connectivity()
        logger.info(f"✅ Connected to Neo4j at {URI}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Neo4j at {URI}: {e}")
        logger.info("Make sure Neo4j is running. Use: docker compose up -d")
        sys.exit(1)

    # Initialize constraints (always run — idempotent)
    init_constraints(driver)

    if "--init" in sys.argv:
        logger.info("Init-only mode. Exiting.")
        driver.close()
        return

    # Find triplet files to ingest
    triplet_files = []
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        target_id = sys.argv[2]
        target_file = os.path.join(TRIPLETS_DIR, f"triplets_{target_id}.json")
        if os.path.exists(target_file):
            triplet_files.append(target_file)
        else:
            logger.error(f"Triplet file not found: {target_file}")
            sys.exit(1)
    else:
        if os.path.exists(TRIPLETS_DIR):
            triplet_files = [
                os.path.join(TRIPLETS_DIR, f)
                for f in os.listdir(TRIPLETS_DIR)
                if f.startswith("triplets_") and f.endswith(".json")
            ]

    if not triplet_files:
        logger.warning("No triplet files found. Run extract.py first.")
        driver.close()
        return

    # Ingest each document
    for filepath in triplet_files:
        filename = os.path.basename(filepath)
        logger.info(f"═══ Ingesting {filename} ═══")

        with open(filepath, "r", encoding="utf-8") as f:
            triplet_data = json.load(f)

        regulasi_id = triplet_data["regulasi_id"]
        triplets = triplet_data["triplets"]

        # Load corresponding processed data for node properties
        processed_path = os.path.join(PROCESSED_DIR, f"{regulasi_id}.json")
        processed_data = None
        if os.path.exists(processed_path):
            with open(processed_path, "r", encoding="utf-8") as f:
                processed_data = json.load(f)

        # Step 1: Ingest all nodes first
        if processed_data:
            with driver.session() as session:
                try:
                    session.execute_write(ingest_document_nodes, processed_data)
                    logger.info(f"  Nodes ingested for {regulasi_id}")
                except Exception as e:
                    logger.error(f"  Node ingestion failed: {e}")
                    errors.append(f"Node ingestion error for {regulasi_id}: {str(e)}")

        # Step 2: Ingest triplets in batches
        total = len(triplets)
        for batch_start in range(0, total, BATCH_SIZE):
            batch = triplets[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            logger.info(f"  Ingesting batch {batch_start + 1}-{batch_end} of {total}...")

            with driver.session() as session:
                try:
                    session.execute_write(ingest_triplet_batch, batch, processed_data)
                except Exception as e:
                    logger.error(f"  Batch ingestion failed: {e}")
                    errors.append(f"Batch error for {regulasi_id} ({batch_start}-{batch_end}): {str(e)}")

        logger.info(f"  ✅ {regulasi_id}: {total} triplets ingested")

    # Generate report
    duration = time.time() - start_time
    stats = get_graph_stats(driver)
    report = generate_report(stats, errors, duration)

    report_path = os.path.join(REPORTS_DIR, "ingestion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    driver.close()

    # Summary
    logger.info("═" * 60)
    logger.info(f"Ingestion complete in {report['duration_seconds']}s")
    logger.info(f"Nodes: {report['total_nodes_created']} | Edges: {report['total_edges_created']}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Report: {report_path}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
