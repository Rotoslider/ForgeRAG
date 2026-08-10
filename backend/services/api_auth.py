"""N3 bearer-token API auth.

One static token, because this is a one-human instrument on a LAN, not a
multi-tenant service (roadmap N3: users/roles/OAuth explicitly out of
scope; remote access, if ever wanted, is Tailscale).

Rules, in order:
- No token configured -> middleware not installed, loud startup warning.
  This preserves zero-config behavior for dev checkouts.
- OPTIONS passes (CORS preflight carries no Authorization header).
- Localhost clients pass: the Chooms' Next.js app, the bridge, and
  anything else on the box keep working with zero changes.
- /health passes (systemd/monitoring probes), / and /app* pass (static
  SPA shell — the API calls it makes are what auth protects).
- Everything else needs `Authorization: Bearer <token>`, compared
  constant-time.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = ("/health",)
_EXEMPT_PREFIXES = ("/app",)
_LOCALHOST = ("127.0.0.1", "::1")


def install_auth(app: FastAPI, token: str) -> bool:
    """Install the bearer-token middleware. Returns True if enabled."""
    if not token:
        logger.warning(
            "API auth DISABLED — set [server] api_token in forgerag.toml "
            "(or FORGERAG_API_TOKEN) to require a bearer token for "
            "non-localhost requests."
        )
        return False

    @app.middleware("http")
    async def _bearer_auth(request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        client = request.client.host if request.client else ""
        if client in _LOCALHOST:
            return await call_next(request)
        path = request.url.path
        if path == "/" or path in _EXEMPT_PATHS or path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)
        supplied = request.headers.get("authorization", "")
        if supplied.startswith("Bearer ") and secrets.compare_digest(
            supplied[7:], token
        ):
            return await call_next(request)
        return JSONResponse(
            {"success": False, "reason": "unauthorized", "data": None},
            status_code=401,
        )

    logger.info("API auth enabled: bearer token required for non-localhost requests")
    return True
