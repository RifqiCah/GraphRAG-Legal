"""
PSKP Knowledge Graph — Graph Validation (Issue #5)
===================================================

Memverifikasi integritas graf setelah ingestion:
- Deteksi orphaned nodes
- Validasi jalur logika hukum
- Generate validation report

Usage:
    python scripts/ingestion/validate_graph.py
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

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
REPORTS_DIR = "./reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Issue #5, Task 5.1: Orphaned Node Detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_orphaned_nodes(driver) -> Dict:
    """Deteksi simpul tanpa relasi apa pun."""
    with driver.session() as session:
        # Node tanpa satupun relasi
        result = session.run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n) AS label, n.id AS id, n.judul AS judul
            ORDER BY label
        """)
        orphaned_all = [dict(r) for r in result]

        # Pasal yang tidak terhubung ke Regulasi manapun
        result = session.run("""
            MATCH (p:Pasal)
            WHERE NOT (:Regulasi)-[:MEMILIKI_PASAL]->(p)
            RETURN p.id AS orphaned_pasal
        """)
        orphaned_pasal = [r["orphaned_pasal"] for r in result]

    return {
        "orphaned_nodes": orphaned_all,
        "orphaned_nodes_count": len(orphaned_all),
        "orphaned_pasal": orphaned_pasal,
        "orphaned_pasal_count": len(orphaned_pasal)
    }


# ═════════════════════════════════════════════════════════════════════════════
# Issue #5, Task 5.2: Logic Path Validation
# ═════════════════════════════════════════════════════════════════════════════

QUERY_TESTS = [
    {
        "test_id": "TEST-01",
        "description": "Hierarki rujukan PP ke UUD",
        "query": """
            MATCH path = (pp:Regulasi {jenis: "PP"})-[:MENGACU_PADA*1..5]->(uud:Regulasi {jenis: "UUD"})
            RETURN pp.id, length(path) AS depth
            LIMIT 10
        """
    },
    {
        "test_id": "TEST-02",
        "description": "Seluruh Pasal yang mengatur Peserta Didik",
        "query": """
            MATCH (p:Pasal)-[:MENGATUR_TOPIK]->(s:Subjek_Hukum {id: "SBJ-peserta-didik"})
            RETURN p.id, p.teks_isi
            LIMIT 20
        """
    },
    {
        "test_id": "TEST-03",
        "description": "Regulasi yang telah dicabut",
        "query": """
            MATCH (baru:Regulasi)-[:MENCABUT]->(lama:Regulasi)
            RETURN baru.id AS pencabut, lama.id AS dicabut, lama.status
        """
    },
    {
        "test_id": "TEST-04",
        "description": "Rantai amandemen pasal",
        "query": """
            MATCH (r:Regulasi)-[:MENGAMANDEMEN]->(p:Pasal)
            RETURN r.id AS pengubah, p.id AS pasal_diubah, p.status
        """
    }
]


def run_query_tests(driver) -> List[Dict]:
    """Jalankan query traversal untuk validasi jalur logika hukum."""
    results = []

    with driver.session() as session:
        for test in QUERY_TESTS:
            logger.info(f"  Running {test['test_id']}: {test['description']}...")
            try:
                result = session.run(test["query"])
                records = list(result)
                count = len(records)
                status = "PASS" if count > 0 else "WARN_EMPTY"

                results.append({
                    "test_id": test["test_id"],
                    "description": test["description"],
                    "status": status,
                    "result_count": count
                })
                logger.info(f"    {status}: {count} results")

            except Exception as e:
                logger.error(f"    ERROR: {e}")
                results.append({
                    "test_id": test["test_id"],
                    "description": test["description"],
                    "status": "ERROR",
                    "result_count": 0,
                    "error": str(e)
                })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Issue #5, Task 5.4: Validation Report
# ═════════════════════════════════════════════════════════════════════════════

def get_graph_summary(driver) -> Dict:
    """Get overall graph statistics for the report."""
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        regulasi_count = session.run("MATCH (n:Regulasi) RETURN count(n) AS cnt").single()["cnt"]
        pasal_count = session.run("MATCH (n:Pasal) RETURN count(n) AS cnt").single()["cnt"]
        subjek_count = session.run("MATCH (n:Subjek_Hukum) RETURN count(n) AS cnt").single()["cnt"]

    return {
        "total_nodes": node_count,
        "total_edges": edge_count,
        "regulasi_count": regulasi_count,
        "pasal_count": pasal_count,
        "subjek_hukum_count": subjek_count
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        logger.info(f"✅ Connected to Neo4j at {URI}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Neo4j: {e}")
        return

    logger.info("═" * 60)
    logger.info("PSKP Knowledge Graph — Validation (Issue #5)")
    logger.info("═" * 60)

    # Graph summary
    logger.info("\n📊 Graph Summary:")
    summary = get_graph_summary(driver)
    logger.info(f"  Total Nodes: {summary['total_nodes']}")
    logger.info(f"  Total Edges: {summary['total_edges']}")
    logger.info(f"  Regulasi: {summary['regulasi_count']}")
    logger.info(f"  Pasal: {summary['pasal_count']}")
    logger.info(f"  Subjek Hukum: {summary['subjek_hukum_count']}")

    # Task 5.1: Orphaned nodes
    logger.info("\n🔍 Detecting orphaned nodes...")
    orphan_data = detect_orphaned_nodes(driver)
    logger.info(f"  Orphaned nodes (no relations): {orphan_data['orphaned_nodes_count']}")
    logger.info(f"  Orphaned pasal (no MEMILIKI_PASAL): {orphan_data['orphaned_pasal_count']}")

    if orphan_data["orphaned_nodes"]:
        for node in orphan_data["orphaned_nodes"][:10]:
            logger.warning(f"    ⚠️ {node['label']}: {node['id']}")

    # Task 5.2: Query tests
    logger.info("\n🧪 Running logic path tests...")
    query_results = run_query_tests(driver)

    # Overall status (Orphaned pasal is normal due to Dummy Nodes, so don't fail on it)
    has_unintended_orphans = orphan_data["orphaned_nodes_count"] > 0 
    has_test_errors = any(t["status"] == "ERROR" for t in query_results)
    overall = "FAIL" if (has_unintended_orphans or has_test_errors) else "PASS"

    # Generate report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "graph_summary": summary,
        "orphaned_nodes": orphan_data["orphaned_nodes_count"],
        "orphaned_pasal": orphan_data["orphaned_pasal_count"],
        "orphaned_details": orphan_data["orphaned_nodes"][:50],
        "query_tests": query_results,
        "overall_status": overall
    }

    report_path = os.path.join(REPORTS_DIR, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    driver.close()

    logger.info("\n" + "═" * 60)
    logger.info(f"Overall Status: {overall}")
    logger.info(f"Report saved to {report_path}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
