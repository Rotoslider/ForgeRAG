#!/usr/bin/env python3
"""Download ForgeRAG backups from Google Drive.

Authenticates using the same credentials as gdrive_backup.py, lists files
in the "ForgeRAG Backup" folder, and downloads the most recent .dump file
(and optionally manifest.json / graph JSON) to a local directory.

Used by scripts/restore.sh --from-drive, or can be run standalone.

Usage:
    python3 scripts/gdrive_restore.py --download-to <dir>
    python3 scripts/gdrive_restore.py --list
    python3 scripts/gdrive_restore.py --download-to <dir> --all

Requirements (same packages as gdrive_backup.py):
    pip install google-auth-oauthlib google-api-python-client
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — same as gdrive_backup.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHOOM_AUTH_DIR = Path("/home/nuc1/projects/Choom/nextjs-app/services/signal-bridge/google_auth")
LOCAL_AUTH_DIR = Path(__file__).resolve().parent / "google_auth"

DRIVE_FOLDER_NAME = "ForgeRAG Backup"
SCOPES = ["https://www.googleapis.com/auth/drive"]


# ---------------------------------------------------------------------------
# Auth — reuses the same pattern as gdrive_backup.py
# ---------------------------------------------------------------------------

def _find_auth_files() -> tuple[Path, Path]:
    """Locate credentials.json and token.json, preferring Choom's directory."""
    for auth_dir in (CHOOM_AUTH_DIR, LOCAL_AUTH_DIR):
        creds = auth_dir / "credentials.json"
        token = auth_dir / "token.json"
        if creds.exists() and token.exists():
            logger.info("Using auth files from %s", auth_dir)
            return creds, token
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

def find_folder(service, folder_name: str) -> Optional[str]:
    """Find the first Drive folder with the given name.

    Returns the folder's Drive file ID, or None if not found.
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
    return files[0]["id"] if files else None


def list_folder_files(service, folder_id: str) -> list[dict]:
    """List all non-trashed files in a Drive folder, newest first."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=100,
        fields="files(id, name, createdTime, size, mimeType)",
        orderBy="createdTime desc",
    ).execute()
    return results.get("files", [])


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


def download_file(service, file_meta: dict, dest_dir: Path) -> Path:
    """Download a file from Drive to a local directory.

    Shows progress for large files (every 10%).
    Returns the local file path.
    """
    file_id = file_meta["id"]
    file_name = file_meta["name"]
    file_size = int(file_meta.get("size", 0))
    dest_path = dest_dir / file_name

    logger.info("Downloading %s (%s)...", file_name, _human_size(file_size))

    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
        done = False
        last_reported = -1
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                # Report every 10%
                bucket = pct // 10
                if bucket > last_reported:
                    last_reported = bucket
                    logger.info("  %s: %d%%", file_name, pct)
        logger.info("  %s: 100%% (%s)", file_name, _human_size(dest_path.stat().st_size))

    return dest_path


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(service, folder_id: str) -> int:
    """List available backup files on Drive."""
    files = list_folder_files(service, folder_id)

    if not files:
        logger.info("No backup files found in '%s'", DRIVE_FOLDER_NAME)
        return 0

    logger.info("Backup files in '%s' (%d files):", DRIVE_FOLDER_NAME, len(files))
    print()
    print(f"{'Name':<50} {'Size':>12} {'Created':>22}")
    print("-" * 86)
    for f in files:
        size = _human_size(int(f.get("size", 0)))
        created = f.get("createdTime", "")[:19].replace("T", " ")
        print(f"{f['name']:<50} {size:>12} {created:>22}")
    print()
    return 0


def cmd_download(service, folder_id: str, dest_dir: Path, download_all: bool) -> int:
    """Download backup files from Drive."""
    files = list_folder_files(service, folder_id)

    if not files:
        logger.error("No backup files found in '%s'", DRIVE_FOLDER_NAME)
        return 1

    # Categorize files
    dump_files = [f for f in files if f["name"].endswith(".dump")]
    manifest_files = [f for f in files if "manifest" in f["name"].lower() and f["name"].endswith(".json")]
    graph_files = [f for f in files if f["name"].startswith("graph_") and f["name"].endswith(".json")]

    if not dump_files:
        logger.error("No .dump files found on Drive. Available files:")
        for f in files:
            logger.error("  %s (%s)", f["name"], _human_size(int(f.get("size", 0))))
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    # Always download the most recent dump
    latest_dump = dump_files[0]
    path = download_file(service, latest_dump, dest_dir)
    downloaded.append(path)

    if download_all:
        # Also download manifest and graph JSON (most recent of each)
        if manifest_files:
            path = download_file(service, manifest_files[0], dest_dir)
            downloaded.append(path)
        if graph_files:
            path = download_file(service, graph_files[0], dest_dir)
            downloaded.append(path)

    logger.info("Downloaded %d file(s) to %s:", len(downloaded), dest_dir)
    for p in downloaded:
        logger.info("  %s (%s)", p.name, _human_size(p.stat().st_size))

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ForgeRAG backups from Google Drive",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available backup files on Drive",
    )
    parser.add_argument(
        "--download-to", type=str, metavar="DIR",
        help="Download the most recent .dump file to this directory",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="With --download-to, also download manifest and graph JSON",
    )
    args = parser.parse_args()

    if not args.list and not args.download_to:
        parser.print_help()
        return 1

    # Authenticate
    try:
        service = get_drive_service()
    except Exception as e:
        logger.error("Google Drive authentication failed: %s", e)
        return 1

    # Find the backup folder
    folder_id = find_folder(service, DRIVE_FOLDER_NAME)
    if folder_id is None:
        logger.error("Drive folder '%s' not found. Has a backup ever been uploaded?",
                      DRIVE_FOLDER_NAME)
        return 1
    logger.info("Found Drive folder '%s' (id=%s)", DRIVE_FOLDER_NAME, folder_id)

    if args.list:
        return cmd_list(service, folder_id)

    if args.download_to:
        dest_dir = Path(args.download_to)
        return cmd_download(service, folder_id, dest_dir, download_all=args.all)

    return 0


if __name__ == "__main__":
    sys.exit(main())
