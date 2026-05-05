#!/usr/bin/env python3
"""Upload ForgeRAG backups to Google Drive.

Authenticates using the shared Choom OAuth credentials/token (with a fallback
to a local google_auth/ directory), creates a "ForgeRAG Backup" folder on
Drive if it doesn't exist, uploads the most recent graph backup JSON,
manifest, and neo4j dump file, and rotates to keep the last 5 backup sets
on Drive (grouped by date prefix).

Can be called standalone or from scripts/backup.sh.

Requirements (same packages Choom uses):
    pip install google-auth-oauthlib google-api-python-client
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"

# Primary: reuse Choom's already-authorized credentials
CHOOM_AUTH_DIR = Path("/home/nuc1/projects/Choom/nextjs-app/services/signal-bridge/google_auth")

# Fallback: local google_auth/ inside ForgeRAG's scripts directory
LOCAL_AUTH_DIR = Path(__file__).resolve().parent / "google_auth"

DRIVE_FOLDER_NAME = "ForgeRAG Backup"
KEEP_ON_DRIVE = 5

# Only need Drive scope for uploads
SCOPES = ["https://www.googleapis.com/auth/drive"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _find_auth_files() -> tuple[Path, Path]:
    """Locate credentials.json and token.json, preferring Choom's directory."""
    for auth_dir in (CHOOM_AUTH_DIR, LOCAL_AUTH_DIR):
        creds = auth_dir / "credentials.json"
        token = auth_dir / "token.json"
        if creds.exists() and token.exists():
            logger.info("Using auth files from %s", auth_dir)
            return creds, token
    # If only credentials exist (no token yet), return that location
    for auth_dir in (CHOOM_AUTH_DIR, LOCAL_AUTH_DIR):
        creds = auth_dir / "credentials.json"
        if creds.exists():
            return creds, auth_dir / "token.json"
    raise FileNotFoundError(
        f"No credentials.json found in {CHOOM_AUTH_DIR} or {LOCAL_AUTH_DIR}. "
        "Place OAuth credentials in one of those directories."
    )


def get_drive_service():
    """Build an authenticated Google Drive v3 service."""
    credentials_file, token_file = _find_auth_files()

    creds: Optional[Credentials] = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired Google credentials")
            creds.refresh(Request())
            # Save refreshed token back
            with open(token_file, "w") as f:
                f.write(creds.to_json())
            logger.info("Refreshed token saved to %s", token_file)
        else:
            raise RuntimeError(
                "Token is missing or invalid and has no refresh_token. "
                "Run an interactive OAuth flow first (e.g. via Choom's google_client.py) "
                "to generate token.json with Drive scope."
            )

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def find_or_create_folder(service, folder_name: str) -> str:
    """Find the first Drive folder with the given name, or create it.

    Returns the folder's Drive file ID.
    """
    query = (
        f"name = '{folder_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results = service.files().list(
        q=query, pageSize=1, fields="files(id, name)"
    ).execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
        logger.info("Found existing Drive folder '%s' (id=%s)", folder_name, folder_id)
        return folder_id

    # Create the folder
    body = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=body, fields="id, name").execute()
    folder_id = folder["id"]
    logger.info("Created Drive folder '%s' (id=%s)", folder_name, folder_id)
    return folder_id


def upload_file(
    service,
    local_path: Path,
    folder_id: str,
    mimetype: str = "application/json",
    display_name: str | None = None,
) -> dict:
    """Upload a single file to the specified Drive folder.

    Uses resumable upload, which is required for files over 5 MB and
    recommended for all uploads.  For large files (neo4j dumps) progress
    is logged every 10%.

    Returns metadata dict with id, name, size.
    """
    media = MediaFileUpload(
        str(local_path),
        mimetype=mimetype,
        resumable=True,
        chunksize=50 * 1024 * 1024,  # 50 MB chunks for large dumps
    )
    file_name = display_name or local_path.name
    body = {
        "name": file_name,
        "parents": [folder_id],
    }
    request = service.files().create(
        body=body, media_body=media, fields="id, name, size, webViewLink"
    )

    result = None
    while result is None:
        status, result = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            if pct % 10 == 0:
                logger.info("  Upload %s: %d%%", file_name, pct)

    return result


def list_folder_files(service, folder_id: str) -> list[dict]:
    """List all non-trashed files in a Drive folder, newest first."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=100,
        fields="files(id, name, createdTime, size)",
        orderBy="createdTime desc",
    ).execute()
    return results.get("files", [])


def _extract_date_prefix(name: str) -> str:
    """Extract the YYYYMMDD_HHMMSS date prefix from a backup filename.

    Expected patterns:
      graph_20260505_143017.json  -> 20260505_143017
      manifest.json               -> '' (no date, kept separately)
      neo4j_20260505_143017.dump  -> 20260505_143017
      manifest_20260505_143017.json -> 20260505_143017
    """
    import re

    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else ""


def rotate_backups(service, folder_id: str, keep: int = KEEP_ON_DRIVE) -> int:
    """Delete the oldest backup *sets* in the folder, keeping only the last
    ``keep`` sets.  A set is a group of files sharing the same YYYYMMDD_HHMMSS
    date prefix (e.g. graph, manifest, and neo4j dump from the same run).
    Files without a date prefix are left untouched.

    Returns the number of files deleted.
    """
    from collections import defaultdict

    files = list_folder_files(service, folder_id)
    if not files:
        return 0

    # Group by date prefix
    groups: dict[str, list[dict]] = defaultdict(list)
    undated: list[dict] = []
    for f in files:
        prefix = _extract_date_prefix(f["name"])
        if prefix:
            groups[prefix].append(f)
        else:
            undated.append(f)

    # Sort date prefixes newest-first
    sorted_prefixes = sorted(groups.keys(), reverse=True)

    if len(sorted_prefixes) <= keep:
        return 0

    # Delete all files in the oldest sets beyond the keep count
    to_delete_prefixes = sorted_prefixes[keep:]
    deleted = 0
    for prefix in to_delete_prefixes:
        for f in groups[prefix]:
            try:
                service.files().delete(fileId=f["id"]).execute()
                logger.info("  Deleted old backup: %s (id=%s)", f["name"], f["id"])
                deleted += 1
            except HttpError as e:
                logger.warning("  Failed to delete %s: %s", f["name"], e)
    return deleted


def _human_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Upload the most recent ForgeRAG backup to Google Drive."""
    logger.info("=== ForgeRAG Google Drive Backup ===")

    # Find the most recent graph backup JSON
    graph_files = sorted(BACKUP_DIR.glob("graph_*.json"), reverse=True)
    if not graph_files:
        logger.error("No graph backup files found in %s", BACKUP_DIR)
        return 1
    latest_graph = graph_files[0]
    logger.info("Latest graph backup: %s (%s)",
                latest_graph.name, _human_size(latest_graph.stat().st_size))

    # Extract timestamp from the graph filename for consistent naming
    import re
    ts_match = re.search(r"(\d{8}_\d{6})", latest_graph.name)
    backup_ts = ts_match.group(1) if ts_match else ""

    # Find manifest — could be manifest_{timestamp}.json (from GUI backup)
    # or manifest.json inside a timestamped subdirectory (from backup.sh)
    manifest_path: Optional[Path] = None
    # Check for timestamped manifest files at top level (GUI backup pattern)
    manifest_files = sorted(BACKUP_DIR.glob("manifest_*.json"), reverse=True)
    if manifest_files:
        manifest_path = manifest_files[0]
    # Check timestamped subdirectories (backup.sh pattern)
    if manifest_path is None:
        subdirs = sorted(BACKUP_DIR.glob("[0-9]*"), reverse=True)
        for subdir in subdirs:
            candidate = subdir / "manifest.json"
            if candidate.exists():
                manifest_path = candidate
                break
    # Fallback: plain manifest.json at top level
    if manifest_path is None:
        candidate = BACKUP_DIR / "manifest.json"
        if candidate.exists():
            manifest_path = candidate

    if manifest_path:
        logger.info("Manifest found: %s (%s)",
                     manifest_path.name, _human_size(manifest_path.stat().st_size))
    else:
        logger.info("No manifest.json found (will upload graph backup only)")

    # Find the most recent neo4j dump file from timestamped subdirectories
    dump_path: Optional[Path] = None
    dump_subdirs = sorted(BACKUP_DIR.glob("[0-9]*"), reverse=True)
    for subdir in dump_subdirs:
        if not subdir.is_dir():
            continue
        dumps = sorted(subdir.glob("*.dump"), reverse=True)
        if dumps:
            dump_path = dumps[0]
            break
    # Also check top-level backup dir
    if dump_path is None:
        top_level_dumps = sorted(BACKUP_DIR.glob("*.dump"), reverse=True)
        if top_level_dumps:
            dump_path = top_level_dumps[0]

    if dump_path:
        logger.info("Neo4j dump found: %s (%s)",
                     dump_path.name, _human_size(dump_path.stat().st_size))
    else:
        logger.info("No neo4j dump file found (will upload graph backup only)")

    # Authenticate
    try:
        service = get_drive_service()
    except Exception as e:
        logger.error("Google Drive authentication failed: %s", e)
        return 1

    # Find or create the backup folder
    try:
        folder_id = find_or_create_folder(service, DRIVE_FOLDER_NAME)
    except HttpError as e:
        logger.error("Failed to find/create Drive folder: %s", e)
        return 1

    # Upload graph backup
    uploaded = 0
    try:
        result = upload_file(service, latest_graph, folder_id)
        size = int(result.get("size", 0))
        logger.info("Uploaded graph backup: %s (%s) -> %s",
                     result["name"], _human_size(size),
                     result.get("webViewLink", ""))
        uploaded += 1
    except HttpError as e:
        logger.error("Failed to upload graph backup: %s", e)
        return 1

    # Upload manifest if it exists (rename to include timestamp for grouping)
    if manifest_path:
        try:
            manifest_display = (
                f"manifest_{backup_ts}.json" if backup_ts else manifest_path.name
            )
            result = upload_file(
                service, manifest_path, folder_id,
                display_name=manifest_display,
            )
            size = int(result.get("size", 0))
            logger.info("Uploaded manifest: %s (%s) -> %s",
                         result["name"], _human_size(size),
                         result.get("webViewLink", ""))
            uploaded += 1
        except HttpError as e:
            logger.warning("Failed to upload manifest (non-fatal): %s", e)

    # Upload neo4j dump if it exists
    if dump_path:
        try:
            dump_display = (
                f"neo4j_{backup_ts}.dump" if backup_ts else dump_path.name
            )
            logger.info("Starting neo4j dump upload (%s) — this may take a while...",
                         _human_size(dump_path.stat().st_size))
            result = upload_file(
                service, dump_path, folder_id,
                mimetype="application/octet-stream",
                display_name=dump_display,
            )
            size = int(result.get("size", 0))
            logger.info("Uploaded neo4j dump: %s (%s) -> %s",
                         result["name"], _human_size(size),
                         result.get("webViewLink", ""))
            uploaded += 1
        except HttpError as e:
            logger.error("Failed to upload neo4j dump: %s", e)
            # Non-fatal — graph backup is already uploaded

    # Rotate old backups (by date-prefix sets)
    try:
        deleted = rotate_backups(service, folder_id, keep=KEEP_ON_DRIVE)
        if deleted:
            logger.info("Rotated %d old backup file(s) from Drive (keeping last %d sets)",
                         deleted, KEEP_ON_DRIVE)
        else:
            logger.info("No rotation needed (at or under %d backup sets)", KEEP_ON_DRIVE)
    except HttpError as e:
        logger.warning("Backup rotation failed (non-fatal): %s", e)

    logger.info("=== Google Drive backup complete: %d file(s) uploaded ===", uploaded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
