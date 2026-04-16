"""Uvicorn entrypoint for ForgeRAG.

Run with: python backend/run.py
Or via systemd: see systemd/forgerag-api.service

Secrets: if /etc/forgerag/env exists and is readable, its KEY=VALUE lines
are loaded into the environment before startup. This matches the systemd
EnvironmentFile= directive so manual and service invocations behave the
same. The file is never shipped — see the install script for how to set
it up.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

from backend.config import get_settings  # noqa: E402


def _load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines from path into os.environ (does not overwrite
    already-set variables). Returns the number of keys loaded. Silent if
    the file doesn't exist or is unreadable."""
    if not path.exists():
        return 0
    try:
        loaded = 0
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
                loaded += 1
        return loaded
    except PermissionError:
        return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run")

    # Load /etc/forgerag/env if present (systemd uses EnvironmentFile= for the
    # same effect; this branch only runs for manual `python backend/run.py`).
    n = _load_env_file(Path("/etc/forgerag/env"))
    if n > 0:
        log.info("Loaded %d env var(s) from /etc/forgerag/env", n)

    settings = get_settings()

    uvicorn.run(
        "backend.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
