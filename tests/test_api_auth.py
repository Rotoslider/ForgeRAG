"""N3 bearer-token auth middleware.

TestClient requests arrive with client host "testclient" — conveniently
NOT localhost, so the require-token path is what gets exercised by
default. Localhost exemption is covered by overriding the transport's
client address.
"""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.api_auth import install_auth

TOKEN = "test-token-123"
READONLY = "readonly-token-456"


def _app(token=TOKEN, readonly_token="") -> FastAPI:
    app = FastAPI()
    enabled = install_auth(app, token, readonly_token)
    assert enabled == bool(token or readonly_token)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/search/thing")
    async def thing():
        return {"ok": True}

    @app.post("/search/keyword")
    async def keyword():
        return {"ok": True}

    @app.post("/graph/query")
    async def graph_query():
        return {"ok": True}

    @app.post("/documents/abc/delete")
    async def delete():
        return {"ok": True}

    @app.post("/ingest")
    async def ingest():
        return {"ok": True}

    @app.delete("/documents/abc")
    async def delete_doc():
        return {"ok": True}

    return app


def test_no_token_configured_means_no_auth():
    app = _app(token="")
    with TestClient(app) as c:
        assert c.get("/search/thing").status_code == 200


def test_remote_request_without_token_is_401():
    with TestClient(_app()) as c:
        r = c.get("/search/thing")
        assert r.status_code == 401
        assert r.json()["reason"] == "unauthorized"


def test_remote_request_with_bearer_passes():
    with TestClient(_app()) as c:
        r = c.get("/search/thing",
                  headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200


def test_wrong_token_is_401():
    with TestClient(_app()) as c:
        r = c.get("/search/thing",
                  headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401


def test_health_exempt():
    with TestClient(_app()) as c:
        assert c.get("/health").status_code == 200


def test_options_preflight_exempt():
    with TestClient(_app()) as c:
        # No Authorization header, as browsers send preflights.
        assert c.options("/search/thing").status_code in (200, 405)


def test_localhost_exempt():
    import asyncio

    app = _app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 51000))

    async def go():
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as c:
            return (await c.get("/search/thing")).status_code

    assert asyncio.run(go()) == 200


# Read-only token scope ---------------------------------------------------

def _readonly_app():
    return _app(token=TOKEN, readonly_token=READONLY)


def _hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_readonly_token_allows_get():
    with TestClient(_readonly_app()) as c:
        assert c.get("/search/thing", headers=_hdr(READONLY)).status_code == 200


def test_readonly_token_allows_readonly_post_search():
    with TestClient(_readonly_app()) as c:
        assert c.post("/search/keyword", headers=_hdr(READONLY)).status_code == 200


def test_readonly_token_allows_readonly_post_graph_query():
    with TestClient(_readonly_app()) as c:
        assert c.post("/graph/query", headers=_hdr(READONLY)).status_code == 200


def test_readonly_token_blocks_write_post():
    with TestClient(_readonly_app()) as c:
        r = c.post("/documents/abc/delete", headers=_hdr(READONLY))
        assert r.status_code == 403
        assert "readonly-token" in r.json()["reason"]


def test_readonly_token_blocks_ingest():
    with TestClient(_readonly_app()) as c:
        assert c.post("/ingest", headers=_hdr(READONLY)).status_code == 403


def test_readonly_token_blocks_delete():
    with TestClient(_readonly_app()) as c:
        assert c.delete("/documents/abc", headers=_hdr(READONLY)).status_code == 403


def test_admin_token_unaffected_by_readonly_scope():
    with TestClient(_readonly_app()) as c:
        assert c.post("/documents/abc/delete", headers=_hdr(TOKEN)).status_code == 200
        assert c.delete("/documents/abc", headers=_hdr(TOKEN)).status_code == 200


def test_readonly_token_with_no_admin_token_also_enforces():
    # Middleware is enabled when ONLY a readonly token is configured.
    app = _app(token="", readonly_token=READONLY)

    with TestClient(app) as c:
        assert c.get("/search/thing", headers=_hdr(READONLY)).status_code == 200
        assert c.delete("/documents/abc", headers=_hdr(READONLY)).status_code == 403
