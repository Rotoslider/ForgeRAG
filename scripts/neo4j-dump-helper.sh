#!/usr/bin/env bash
# Helper script for Neo4j database dump.
#
# This script is meant to be called via sudo from the ForgeRAG backup
# process. It stops Neo4j, creates a dump, and restarts Neo4j.
#
# Usage: sudo /home/nuc1/projects/ForgeRAG/scripts/neo4j-dump-helper.sh <output_dir> <timestamp>
#
# Setup (run once):
#   sudo cp /home/nuc1/projects/ForgeRAG/scripts/neo4j-dump-helper.sh /usr/local/bin/forgerag-dump
#   sudo chmod 755 /usr/local/bin/forgerag-dump
#   echo 'nuc1 ALL=(ALL) NOPASSWD: /usr/local/bin/forgerag-dump' | sudo tee /etc/sudoers.d/forgerag-dump
#   sudo chmod 440 /etc/sudoers.d/forgerag-dump

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> <timestamp>}"
TIMESTAMP="${2:?Usage: $0 <output_dir> <timestamp>}"
DATABASE="neo4j"
DUMP_FILE="${OUTPUT_DIR}/neo4j_${TIMESTAMP}.dump"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Ensure output dir exists
mkdir -p "${OUTPUT_DIR}"

log "Stopping Neo4j..."
systemctl stop neo4j || { log "ERROR: Failed to stop Neo4j"; exit 1; }

log "Creating database dump..."
DUMP_OK=0
neo4j-admin database dump "${DATABASE}" --to-path="${OUTPUT_DIR}" --overwrite-destination=true 2>&1 && DUMP_OK=1

# Always restart Neo4j
log "Restarting Neo4j..."
systemctl start neo4j

if [ "${DUMP_OK}" -eq 1 ]; then
    # neo4j-admin creates neo4j.dump in --to-path
    ACTUAL="${OUTPUT_DIR}/${DATABASE}.dump"
    if [ -f "${ACTUAL}" ]; then
        mv "${ACTUAL}" "${DUMP_FILE}"
        # Make readable by nuc1
        chown nuc1:nuc1 "${DUMP_FILE}"
        SIZE=$(du -sh "${DUMP_FILE}" | cut -f1)
        log "Dump created: ${DUMP_FILE} (${SIZE})"
    else
        log "WARNING: neo4j-admin succeeded but dump file not found at ${ACTUAL}"
        exit 1
    fi
else
    log "ERROR: neo4j-admin dump failed"
    exit 1
fi
