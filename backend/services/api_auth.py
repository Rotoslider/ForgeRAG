"""N3 bearer-token API auth.

Static tokens, because this is a one-human instrument on a LAN, not a
multi-tenant service (roadmap N3: users/roles/OAuth explicitly out of
scope; remote access, if ever wanted, is Tailscale).

Two tokens, by design:
- admin token: full access (ingest, delete, rebuild, admin, backup...).
- read-only token (optional): lets remote agents *learn from* the library —
  search, read documents/pages, browse the graph — without being able to
  mutate state. Add `api_token_readonly` to [server] in forgerag.toml (or
  set FORGERAG_API_TOKEN_READONLY). A separate token means the human's full
  credentials are never shared with an external harness.

Rules, in order:
- No tokens configured -> middleware not installed, loud startup warning.
  This preserves zero-config behavior for dev checkouts.
- OPTIONS passes (CORS preflight carries no Authorization header).
- Localhost clients pass: the Chooms' Next.js app, the bridge, and
  anything else on the box keep working with zero changes.
- /health passes (systemd/monitoring probes), / and /app* pass (static
  SPA shell — the API calls it makes are what auth protects).
- Admin bearer token -> everything.
- Read-only bearer token -> GET/HEAD/OPTIONS plus a curated set of
  read-only POST endpoints (/search/*, /skills/*, /graph/query,
  /graph/explore, /ingest/check-duplicates). Any other write from the
  read-only token is 403.
- Anything else -> 401. All token comparisons are constant-time.
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

# A read-only token may call any safe method (these never mutate state), plus
# the POST endpoints below, which are read-only by design (they return search
# results / query results / duplicate lookups without writing anything).
_SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
_READONLY_POST_PREFIXES = (
    "/search/",
    "/skills/",
    "/graph/query",
    "/graph/explore",
    "/ingest/check-duplicates",
)


def _is_readonly_request(request: Request) -> bool:
    """True if this request is read-only and safe for a scoped token."""
    if request.method in _SAFE_METHODS:
        return True
    if request.method == "POST":
        path = request.url.path
        return path.startswith(_READONLY_POST_PREFIXES)
    return False


def install_auth(app: FastAPI, token: str, readonly_token: str = "") -> bool:
    """Install the bearer-token middleware. Returns True if enabled."""
    if not token and not readonly_token:
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
        is_bearer = supplied.startswith("Bearer ")
        if is_bearer and token and secrets.compare_digest(supplied[7:], token):
            return await call_next(request)
        if (
            is_bearer
            and readonly_token
            and secrets.compare_digest(supplied[7:], readonly_token)
        ):
            if _is_readonly_request(request):
                return await call_next(request)
            return JSONResponse(
                {
                    "success": False,
                    "reason": "readonly-token: write operation not permitted",
                    "data": None,
                },
                status_code=403,
            )
        return JSONResponse(
            {"success": False, "reason": "unauthorized", "data": None},
            status_code=401,
        )

    if token:
        logger.info(
            "API auth enabled: admin token required for non-localhost requests"
        )
    if readonly_token:
        logger.info(
            "API auth: read-only token enabled for search/read/graph endpoints"
        )
    return True
