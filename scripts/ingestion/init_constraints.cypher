// ═══════════════════════════════════════════════════════════════════
// PSKP Knowledge Graph — Neo4j Constraints & Indexes (Issue #4, Task 4.2)
// ═══════════════════════════════════════════════════════════════════
// Jalankan saat pertama kali setup Neo4j.
// Semua statement bersifat idempotent (IF NOT EXISTS).

// ─── Uniqueness Constraints ──────────────────────────────────────
CREATE CONSTRAINT regulasi_id_unique IF NOT EXISTS FOR (r:Regulasi) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT pasal_id_unique IF NOT EXISTS FOR (p:Pasal) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT subjek_id_unique IF NOT EXISTS FOR (s:Subjek_Hukum) REQUIRE s.id IS UNIQUE;

// ─── Indexes untuk Query Performa ────────────────────────────────
CREATE INDEX pasal_topik_index IF NOT EXISTS FOR (p:Pasal) ON (p.topik);
CREATE INDEX regulasi_tahun_index IF NOT EXISTS FOR (r:Regulasi) ON (r.tahun);
CREATE INDEX regulasi_status_index IF NOT EXISTS FOR (r:Regulasi) ON (r.status);
