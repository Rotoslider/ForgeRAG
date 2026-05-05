#!/usr/bin/env bash
# ForgeRAG Backup Script
#
# Creates a timestamped Neo4j database dump and a JSON document manifest.
# Keeps the last 5 backups and prunes older ones.
#
# Crontab example (daily at 3 AM):
#   0 3 * * * /home/nuc1/projects/ForgeRAG/scripts/backup.sh >> /home/nuc1/projects/ForgeRAG/data/backups/backup.log 2>&1
#
# Requirements:
#   - neo4j-admin on PATH
#   - sudo access for stopping/starting neo4j service (passwordless via sudoers recommended for cron)
#   - ForgeRAG API running on localhost:8200 (for manifest export)

set -euo pipefail

# ---- Configuration ----
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/data/backups"
NEO4J_DATABASE="neo4j"
API_BASE="http://localhost:8200"
KEEP_COUNT=5
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_SUBDIR="${BACKUP_DIR}/${TIMESTAMP}"

# ---- Helpers ----
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

# ---- Setup ----
mkdir -p "${BACKUP_SUBDIR}"
log "Starting ForgeRAG backup: ${TIMESTAMP}"
log "Backup directory: ${BACKUP_SUBDIR}"

# ---- Step 1: Export document manifest via API ----
log "Exporting document manifest from API..."
MANIFEST_FILE="${BACKUP_SUBDIR}/manifest.json"
if curl -sf "${API_BASE}/admin/backup/manifest" -o "${MANIFEST_FILE}"; then
    DOC_COUNT=$(python3 -c "import json; d=json.load(open('${MANIFEST_FILE}')); print(len(d.get('data',{}).get('documents',[])))" 2>/dev/null || echo "unknown")
    log "Manifest exported: ${DOC_COUNT} documents"
else
    log "WARNING: Could not reach API for manifest export (API may be down). Continuing with DB dump."
fi

# ---- Step 2: Stop Neo4j, create dump, restart ----
log "Stopping Neo4j service..."
sudo systemctl stop neo4j || die "Failed to stop Neo4j service"

DUMP_FILE="${BACKUP_SUBDIR}/${NEO4J_DATABASE}_${TIMESTAMP}.dump"
log "Creating Neo4j database dump..."
if sudo neo4j-admin database dump "${NEO4J_DATABASE}" --to-path="${BACKUP_SUBDIR}" 2>&1; then
    # neo4j-admin dump creates a file named <database>.dump in --to-path
    ACTUAL_DUMP="${BACKUP_SUBDIR}/${NEO4J_DATABASE}.dump"
    if [ -f "${ACTUAL_DUMP}" ]; then
        mv "${ACTUAL_DUMP}" "${DUMP_FILE}"
    fi
    log "Database dump created: ${DUMP_FILE}"
else
    log "ERROR: neo4j-admin dump failed"
    log "Restarting Neo4j service..."
    sudo systemctl start neo4j
    die "Backup aborted due to dump failure"
fi

log "Restarting Neo4j service..."
sudo systemctl start neo4j || die "Failed to restart Neo4j service"
log "Neo4j service restarted"

# ---- Step 3: Prune old backups (keep last N) ----
log "Pruning old backups (keeping last ${KEEP_COUNT})..."
BACKUP_DIRS=($(ls -dt "${BACKUP_DIR}"/[0-9]* 2>/dev/null || true))
PRUNED=0
if [ ${#BACKUP_DIRS[@]} -gt ${KEEP_COUNT} ]; then
    for OLD_DIR in "${BACKUP_DIRS[@]:${KEEP_COUNT}}"; do
        if [ -d "${OLD_DIR}" ]; then
            rm -rf "${OLD_DIR}"
            log "  Pruned: $(basename "${OLD_DIR}")"
            PRUNED=$((PRUNED + 1))
        fi
    done
fi
log "Pruned ${PRUNED} old backup(s)"

# ---- Step 4: Report total backup size ----
TOTAL_SIZE=$(du -sh "${BACKUP_SUBDIR}" 2>/dev/null | cut -f1)
TOTAL_BYTES=$(du -sb "${BACKUP_SUBDIR}" 2>/dev/null | cut -f1)
ALL_BACKUPS_SIZE=$(du -sh "${BACKUP_DIR}" 2>/dev/null | cut -f1)

log "--------------------------------------------"
log "Backup complete!"
log "  This backup:    ${TOTAL_SIZE} (${BACKUP_SUBDIR})"
log "  All backups:    ${ALL_BACKUPS_SIZE} (${BACKUP_DIR})"
log "  Dump file:      $(du -sh "${DUMP_FILE}" 2>/dev/null | cut -f1)"
if [ -f "${MANIFEST_FILE}" ]; then
    log "  Manifest:       $(du -sh "${MANIFEST_FILE}" 2>/dev/null | cut -f1) (${DOC_COUNT} docs)"
fi
log "--------------------------------------------"
