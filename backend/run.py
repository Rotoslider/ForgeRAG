"""Uvicorn entrypoint for ForgeRAG.

Run with: python backend/run.py
Or via systemd: see systemd/forgerag-api.service
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

from backend.config import get_settings  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
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
