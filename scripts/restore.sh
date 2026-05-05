#!/usr/bin/env bash
# ForgeRAG Restore Script
#
# Restores a Neo4j database from a neo4j-admin dump file.
# The dump can come from a local backup directory or be downloaded
# from Google Drive first.
#
# Usage:
#   ./scripts/restore.sh --from-local <backup_dir>
#   ./scripts/restore.sh --from-drive
#
# Requirements:
#   - neo4j-admin on PATH
#   - sudo access for stopping/starting neo4j service
#   - For --from-drive: Python venv with google-auth-oauthlib + google-api-python-client

set -euo pipefail

# ---- Configuration ----
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/data/backups"
NEO4J_DATABASE="neo4j"
NEO4J_DATA_DIR="/var/lib/neo4j/data"
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python3"
GDRIVE_RESTORE_SCRIPT="${PROJECT_ROOT}/scripts/gdrive_restore.py"
NEO4J_RESTARTED=false

# ---- Helpers ----
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*" >&2
    # If we stopped Neo4j but haven't restarted it, try to restart
    if [ "$NEO4J_RESTARTED" = false ] && systemctl is-active --quiet neo4j 2>/dev/null; then
        : # Neo4j is still running, nothing to do
    elif [ "$NEO4J_RESTARTED" = false ]; then
        log "Attempting to restart Neo4j after error..."
        sudo systemctl start neo4j 2>/dev/null || log "WARNING: Could not restart Neo4j"
    fi
    exit 1
}

usage() {
    cat <<'USAGE'
ForgeRAG Restore Script

Usage:
  ./scripts/restore.sh --from-local <backup_dir>    Restore from a local backup directory
  ./scripts/restore.sh --from-drive                  Download latest backup from Google Drive and restore

Options:
  --from-local <path>   Path to a backup directory containing a .dump file
                         (e.g. data/backups/20260505_143017/)
  --from-drive          Download the most recent .dump from the "ForgeRAG Backup"
                         Google Drive folder, then restore from it
  --help                Show this help message

Examples:
  ./scripts/restore.sh --from-local data/backups/20260505_143017/
  ./scripts/restore.sh --from-drive
USAGE
    exit 0
}

# ---- Prerequisite checks ----
check_prerequisites() {
    log "Checking prerequisites..."

    # neo4j-admin must be available
    if ! command -v neo4j-admin &>/dev/null; then
        die "neo4j-admin not found on PATH. Install Neo4j or add it to PATH."
    fi
    log "  neo4j-admin: $(which neo4j-admin)"

    # sudo access
    if ! sudo -n true 2>/dev/null; then
        log "  sudo: will prompt for password when needed"
    else
        log "  sudo: passwordless access confirmed"
    fi

    # Check Neo4j data dir exists
    if [ ! -d "${NEO4J_DATA_DIR}" ]; then
        log "  WARNING: Neo4j data directory ${NEO4J_DATA_DIR} not found."
        log "  This is expected on a fresh install — neo4j-admin load will create it."
    else
        log "  Neo4j data: ${NEO4J_DATA_DIR}"
    fi
}

# ---- Find dump file in a directory ----
find_dump_file() {
    local search_dir="$1"

    if [ ! -d "${search_dir}" ]; then
        die "Directory does not exist: ${search_dir}"
    fi

    # Look for .dump files
    local dump_files
    dump_files=($(find "${search_dir}" -maxdepth 1 -name "*.dump" -type f 2>/dev/null | sort -r))

    if [ ${#dump_files[@]} -eq 0 ]; then
        die "No .dump files found in ${search_dir}"
    fi

    echo "${dump_files[0]}"
}

# ---- Download from Google Drive ----
download_from_drive() {
    log "Downloading latest backup from Google Drive..."

    if [ ! -f "${VENV_PYTHON}" ]; then
        die "Python venv not found at ${VENV_PYTHON}. Run: python3 -m venv ${PROJECT_ROOT}/venv"
    fi

    if [ ! -f "${GDRIVE_RESTORE_SCRIPT}" ]; then
        die "Google Drive restore script not found at ${GDRIVE_RESTORE_SCRIPT}"
    fi

    # Create a download directory
    local download_dir="${BACKUP_DIR}/restore_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${download_dir}"

    log "Download directory: ${download_dir}"

    if "${VENV_PYTHON}" "${GDRIVE_RESTORE_SCRIPT}" --download-to "${download_dir}"; then
        log "Download complete"
        # Find the downloaded dump
        DUMP_FILE=$(find_dump_file "${download_dir}")
    else
        die "Google Drive download failed"
    fi
}

# ---- Confirm with user ----
confirm_restore() {
    local dump_file="$1"
    local dump_size
    dump_size=$(du -sh "${dump_file}" 2>/dev/null | cut -f1)

    echo ""
    log "============================================"
    log "  ForgeRAG Database Restore"
    log "============================================"
    log ""
    log "  Dump file:  ${dump_file}"
    log "  Dump size:  ${dump_size}"
    log "  Database:   ${NEO4J_DATABASE}"
    log "  Data dir:   ${NEO4J_DATA_DIR}"
    log ""
    log "  WARNING: This will OVERWRITE the existing Neo4j database."
    log "  The Neo4j service will be stopped during the restore."
    log ""
    log "============================================"
    echo ""

    read -p "Proceed with restore? (yes/no): " answer
    case "${answer}" in
        yes|YES|y|Y)
            log "User confirmed restore"
            ;;
        *)
            log "Restore cancelled by user"
            exit 0
            ;;
    esac
}

# ---- Perform the restore ----
do_restore() {
    local dump_file="$1"

    # Step 1: Stop Neo4j
    log "Stopping Neo4j service..."
    if systemctl is-active --quiet neo4j 2>/dev/null; then
        sudo systemctl stop neo4j || die "Failed to stop Neo4j service"
        log "Neo4j service stopped"
    else
        log "Neo4j service was not running"
    fi

    # Step 2: Load the dump
    # neo4j-admin database load replaces the existing database from a dump.
    # The --overwrite-destination flag is needed if the database already exists.
    log "Loading database from dump..."
    log "  Command: neo4j-admin database load ${NEO4J_DATABASE} --from-path=$(dirname "${dump_file}") --overwrite-destination=true"

    # neo4j-admin 5.x uses: neo4j-admin database load <db> --from-path=<dir> --overwrite-destination=true
    # The dump file must be named <database>.dump in the from-path directory.
    # We need to ensure the file is named correctly.
    local dump_dir
    dump_dir="$(dirname "${dump_file}")"
    local expected_name="${NEO4J_DATABASE}.dump"
    local actual_name
    actual_name="$(basename "${dump_file}")"

    # If the dump file isn't named <database>.dump, create a temporary symlink
    local needs_cleanup=false
    if [ "${actual_name}" != "${expected_name}" ]; then
        log "  Dump file is named '${actual_name}', creating symlink as '${expected_name}'"
        if [ -e "${dump_dir}/${expected_name}" ] && [ "$(readlink -f "${dump_dir}/${expected_name}")" != "$(readlink -f "${dump_file}")" ]; then
            # Back up existing file with that name
            mv "${dump_dir}/${expected_name}" "${dump_dir}/${expected_name}.bak"
        fi
        ln -sf "${actual_name}" "${dump_dir}/${expected_name}"
        needs_cleanup=true
    fi

    if sudo neo4j-admin database load "${NEO4J_DATABASE}" --from-path="${dump_dir}" --overwrite-destination=true 2>&1; then
        log "Database loaded successfully"
    else
        log "ERROR: neo4j-admin database load failed"
        # Clean up symlink if we created one
        if [ "${needs_cleanup}" = true ]; then
            rm -f "${dump_dir}/${expected_name}"
            [ -f "${dump_dir}/${expected_name}.bak" ] && mv "${dump_dir}/${expected_name}.bak" "${dump_dir}/${expected_name}"
        fi
        log "Restarting Neo4j service..."
        sudo systemctl start neo4j || log "WARNING: Failed to restart Neo4j"
        NEO4J_RESTARTED=true
        die "Restore aborted due to load failure"
    fi

    # Clean up symlink
    if [ "${needs_cleanup}" = true ]; then
        rm -f "${dump_dir}/${expected_name}"
        [ -f "${dump_dir}/${expected_name}.bak" ] && mv "${dump_dir}/${expected_name}.bak" "${dump_dir}/${expected_name}"
    fi

    # Step 3: Start Neo4j
    log "Starting Neo4j service..."
    sudo systemctl start neo4j || die "Failed to start Neo4j service"
    NEO4J_RESTARTED=true
    log "Neo4j service started"

    # Step 4: Wait for Neo4j to become ready
    log "Waiting for Neo4j to become ready..."
    local max_wait=60
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if cypher-shell -u neo4j -p "$(grep -oP 'password\s*=\s*"\K[^"]+' "${PROJECT_ROOT}/config/forgerag.toml" 2>/dev/null || echo 'neo4j')" "RETURN 1" &>/dev/null; then
            log "Neo4j is ready (waited ${waited}s)"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    if [ $waited -ge $max_wait ]; then
        log "WARNING: Neo4j did not respond within ${max_wait}s — it may still be starting up."
        log "Check: sudo systemctl status neo4j"
    fi
}

# ---- Post-restore report ----
report() {
    local dump_file="$1"
    echo ""
    log "============================================"
    log "  Restore Complete"
    log "============================================"
    log ""
    log "  Restored from:  ${dump_file}"
    log "  Database:       ${NEO4J_DATABASE}"
    log "  Neo4j status:   $(systemctl is-active neo4j 2>/dev/null || echo 'unknown')"
    log ""
    log "  What was restored:"
    log "    - Neo4j graph database (documents, pages, entities, relationships)"
    log ""
    log "  What may still need attention:"
    log "    - Page images (data/page_images/): regenerated on demand by the app."
    log "      If the directory is empty, images will be re-rendered when pages"
    log "      are viewed. To bulk regenerate, re-run ingestion with skip_extract."
    log "    - Reduced images (data/reduced_images/): generated on demand from"
    log "      page images."
    log "    - Source PDFs (data/uploads/): not backed up. Only needed if you"
    log "      want to re-ingest from scratch."
    log "    - Embeddings: stored in Neo4j, restored with the dump."
    log ""
    log "  Next steps:"
    log "    1. Start ForgeRAG:  cd ${PROJECT_ROOT} && ./venv/bin/python -m uvicorn backend.main:app --port 8200"
    log "    2. Verify health:   curl http://localhost:8200/health"
    log "    3. Check doc count: curl http://localhost:8200/health | python3 -m json.tool"
    log ""
    log "============================================"
}

# ---- Main ----
main() {
    if [ $# -eq 0 ]; then
        usage
    fi

    local mode=""
    local local_path=""
    DUMP_FILE=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --from-local)
                mode="local"
                if [ -z "${2:-}" ]; then
                    die "--from-local requires a path argument"
                fi
                local_path="$2"
                shift 2
                ;;
            --from-drive)
                mode="drive"
                shift
                ;;
            --help|-h)
                usage
                ;;
            *)
                die "Unknown argument: $1. Use --help for usage."
                ;;
        esac
    done

    if [ -z "${mode}" ]; then
        usage
    fi

    check_prerequisites

    case "${mode}" in
        local)
            # Resolve relative paths
            if [[ "${local_path}" != /* ]]; then
                local_path="${PWD}/${local_path}"
            fi
            DUMP_FILE=$(find_dump_file "${local_path}")
            log "Found dump file: ${DUMP_FILE}"
            ;;
        drive)
            download_from_drive
            log "Using downloaded dump: ${DUMP_FILE}"
            ;;
    esac

    confirm_restore "${DUMP_FILE}"
    do_restore "${DUMP_FILE}"
    report "${DUMP_FILE}"
}

main "$@"
