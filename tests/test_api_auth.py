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


def _app(token=TOKEN) -> FastAPI:
    app = FastAPI()
    enabled = install_auth(app, token)
    assert enabled == bool(token)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/search/thing")
    async def thing():
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
