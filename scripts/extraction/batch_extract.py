"""
PSKP Knowledge Graph — Batch Extraction (Issue #3)
===================================================

Wrapper script to process all 14 documents sequentially.
Calls extract.py's pipeline for each document in corpus_manifest.json.

Usage:
    python scripts/extraction/batch_extract.py
    python scripts/extraction/batch_extract.py --force   # Re-process all, ignore status
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so absolute imports keep working when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extraction.extract import (
    load_schema, load_few_shots, process_document,
    RAW_DIR, PROCESSED_DIR, TRIPLETS_DIR, CORPUS_MANIFEST_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    force = "--force" in sys.argv
    start_time = time.time()

    # Ensure directories
    for d in [RAW_DIR, PROCESSED_DIR, TRIPLETS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load schema & few-shots
    schema = load_schema()
    few_shots = load_few_shots()

    # Load corpus manifest
    if not os.path.exists(CORPUS_MANIFEST_PATH):
        logger.error(f"Corpus manifest not found: {CORPUS_MANIFEST_PATH}")
        logger.info("Create data/corpus_manifest.json first (see Issue #2, Task 2.1).")
        sys.exit(1)

    with open(CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded {len(manifest)} documents from corpus manifest.")
    logger.info("=" * 70)

    # Track results
    results_summary = []
    total_valid = 0
    total_invalid = 0
    total_failed = 0

    for idx, entry in enumerate(manifest, 1):
        reg_id = entry["id"]
        status = entry.get("status_ekstraksi", "pending")

        if status == "done" and not force:
            logger.info(f"[{idx}/{len(manifest)}] Skipping {reg_id} (already done)")
            continue

        # Check if PDF exists
        if not os.path.exists(entry["path_raw"]):
            logger.warning(f"[{idx}/{len(manifest)}] PDF not found: {entry['path_raw']}")
            entry["status_ekstraksi"] = "file_missing"
            results_summary.append({
                "regulasi_id": reg_id,
                "status": "SKIP",
                "reason": "PDF file not found"
            })
            continue

        logger.info(f"\n[{idx}/{len(manifest)}] Processing {reg_id}...")
        doc_start = time.time()

        try:
            result = process_document(entry, schema, few_shots)
            doc_duration = time.time() - doc_start

            meta = result["extraction_metadata"]
            total_valid += meta["valid_triplets"]
            total_invalid += meta["invalid_triplets"]
            total_failed += len(result["failed_extractions"])

            entry["status_ekstraksi"] = "done"
            results_summary.append({
                "regulasi_id": reg_id,
                "status": "OK",
                "valid_triplets": meta["valid_triplets"],
                "invalid_triplets": meta["invalid_triplets"],
                "duration_seconds": round(doc_duration, 2)
            })

        except Exception as e:
            doc_duration = time.time() - doc_start
            logger.error(f"FAILED: {reg_id} — {e}")
            entry["status_ekstraksi"] = "error"
            results_summary.append({
                "regulasi_id": reg_id,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": round(doc_duration, 2)
            })

    # Save updated manifest
    with open(CORPUS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Calculate validation rate
    total_triplets = total_valid + total_invalid
    validation_rate = (total_valid / total_triplets * 100) if total_triplets > 0 else 0

    total_duration = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BATCH EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Documents processed: {len(results_summary)}")
    logger.info(f"Total valid triplets: {total_valid}")
    logger.info(f"Total invalid triplets: {total_invalid}")
    logger.info(f"Validation rate: {validation_rate:.1f}% (target: ≥95%)")
    logger.info(f"Failed LLM extractions: {total_failed}")
    logger.info(f"Total duration: {total_duration:.1f}s")
    logger.info("=" * 70)

    if validation_rate < 95 and total_triplets > 0:
        logger.warning(f"⚠️ Validation rate {validation_rate:.1f}% is below 95% target!")

    # Save batch report
    batch_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(manifest),
        "documents_processed": len(results_summary),
        "total_valid_triplets": total_valid,
        "total_invalid_triplets": total_invalid,
        "validation_rate_percent": round(validation_rate, 2),
        "total_failed_extractions": total_failed,
        "total_duration_seconds": round(total_duration, 2),
        "per_document": results_summary
    }

    report_path = os.path.join(TRIPLETS_DIR, "batch_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Batch report saved to {report_path}")


if __name__ == "__main__":
    main()
