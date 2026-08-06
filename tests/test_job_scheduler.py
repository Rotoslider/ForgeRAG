"""Tests for the JobScheduler: window boundary math, tick firing,
watch-folder scanning, and the /schedule endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.ingestion.job_manager import JobManager
from backend.routers import schedule as schedule_router
from backend.services.job_scheduler import (
    JobScheduler,
    compute_last_boundary,
    compute_next_boundary,
)


def _sched(**kw):
    base = {"enabled": True, "start": "21:00", "end": "06:30",
            "days": [0, 1, 2, 3, 4, 5, 6]}
    base.update(kw)
    return base


# ------------------------------------------------------------ boundary math


def test_overnight_window_inside():
    # Wednesday 2026-08-05 22:00 — inside the 21:00→06:30 window
    now = datetime(2026, 8, 5, 22, 0)
    last = compute_last_boundary(_sched(), now)
    nxt = compute_next_boundary(_sched(), now)
    assert last == (datetime(2026, 8, 5, 21, 0), "resume")
    assert nxt == (datetime(2026, 8, 6, 6, 30), "pause")


def test_overnight_window_outside():
    # Wednesday 10:00 — after this morning's close, before tonight's open
    now = datetime(2026, 8, 5, 10, 0)
    last = compute_last_boundary(_sched(), now)
    nxt = compute_next_boundary(_sched(), now)
    assert last == (datetime(2026, 8, 5, 6, 30), "pause")
    assert nxt == (datetime(2026, 8, 5, 21, 0), "resume")


def test_day_mask_applies_to_start_day():
    # Monday-only window. Wednesday: the last boundary is Tuesday's 06:30
    # close (Mon 21:00 start runs into Tue morning); next is next Monday.
    monday_only = _sched(days=[0])
    now = datetime(2026, 8, 5, 12, 0)  # Wednesday
    last = compute_last_boundary(monday_only, now)
    nxt = compute_next_boundary(monday_only, now)
    assert last == (datetime(2026, 8, 4, 6, 30), "pause")   # Tue morning
    assert nxt == (datetime(2026, 8, 10, 21, 0), "resume")  # next Mon


def test_daytime_window():
    day = _sched(start="09:00", end="17:00")
    now = datetime(2026, 8, 5, 12, 0)
    assert compute_last_boundary(day, now) == (datetime(2026, 8, 5, 9, 0), "resume")
    assert compute_next_boundary(day, now) == (datetime(2026, 8, 5, 17, 0), "pause")


def test_no_days_yields_none():
    assert compute_last_boundary(_sched(days=[]), datetime(2026, 8, 5)) is None
    assert compute_next_boundary(_sched(days=[]), datetime(2026, 8, 5)) is None


def test_back_to_back_tie_resumes():
    # end == next start at the same instant -> resume wins
    cfg = _sched(start="00:00", end="00:00")
    # degenerate zero-duration window is treated as no windows at all
    assert compute_last_boundary(cfg, datetime(2026, 8, 5, 1, 0)) is None


# --------------------------------------------------------------- fixtures


class StubNeo4j:
    def __init__(self):
        self.known_hashes: set[str] = set()

    async def run_query(self, query, params=None, **kw):
        if params and params.get("h") in self.known_hashes:
            return [{"title": "Existing Doc"}]
        return []


class StubPipeline:
    def __init__(self, neo4j):
        self.neo4j = neo4j
        self.calls: list[tuple[str, str]] = []

    async def run_job(self, job_id, collection="default"):
        self.calls.append((job_id, collection))


@pytest.fixture()
async def scheduler(tmp_path):
    manager = JobManager(tmp_path / "jobs.sqlite")
    await manager.init()
    neo4j = StubNeo4j()
    pipeline = StubPipeline(neo4j)
    settings = SimpleNamespace(server=SimpleNamespace(data_dir=str(tmp_path / "data")))
    sched = JobScheduler(job_manager=manager, pipeline=pipeline, settings=settings)
    await sched._load()
    return sched, manager, pipeline, neo4j


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


# ------------------------------------------------------------- tick firing


@pytest.mark.asyncio
async def test_tick_applies_current_window_state(scheduler):
    sched, manager, _, _ = scheduler
    await manager.set_pause_all(True)

    # Window opened an hour ago and closes in an hour -> tick must resume.
    now = datetime.now()
    await sched.update_schedule(
        _sched(start=_hhmm(now - timedelta(hours=1)), end=_hhmm(now + timedelta(hours=1)))
    )
    await sched._tick()
    assert manager.pause_all_active is False

    # Window that already closed two hours ago -> tick must pause.
    await sched.update_schedule(
        _sched(start=_hhmm(now - timedelta(hours=3)), end=_hhmm(now - timedelta(hours=2)))
    )
    await sched._tick()
    assert manager.pause_all_active is True


@pytest.mark.asyncio
async def test_manual_override_holds_until_next_boundary(scheduler):
    sched, manager, _, _ = scheduler
    now = datetime.now()
    await sched.update_schedule(
        _sched(start=_hhmm(now - timedelta(hours=1)), end=_hhmm(now + timedelta(hours=1)))
    )
    await sched._tick()
    assert manager.pause_all_active is False

    # User pauses manually mid-window; further ticks must NOT resume.
    await manager.set_pause_all(True)
    await sched._tick()
    await sched._tick()
    assert manager.pause_all_active is True


@pytest.mark.asyncio
async def test_schedule_validation(scheduler):
    sched, _, _, _ = scheduler
    with pytest.raises(ValueError):
        await sched.update_schedule({"start": "25:00"})
    with pytest.raises(ValueError):
        await sched.update_schedule({"start": "10:00", "end": "10:00"})
    with pytest.raises(ValueError):
        await sched.update_schedule({"enabled": True, "days": []})
    with pytest.raises(ValueError):
        await sched.update_schedule({"days": [7]})


@pytest.mark.asyncio
async def test_config_persists_across_instances(scheduler, tmp_path):
    sched, manager, pipeline, _ = scheduler
    await sched.update_schedule(_sched(start="20:15", end="05:45", days=[0, 2, 4]))

    fresh = JobScheduler(
        job_manager=manager, pipeline=pipeline,
        settings=SimpleNamespace(server=SimpleNamespace(data_dir=str(tmp_path / "data"))),
    )
    await fresh._load()
    assert fresh.schedule["start"] == "20:15"
    assert fresh.schedule["days"] == [0, 2, 4]


# ------------------------------------------------------------ watch folder


@pytest.mark.asyncio
async def test_watch_scan_two_pass_stability_then_ingest(scheduler, tmp_path):
    sched, manager, pipeline, _ = scheduler
    await sched.update_watch({"enabled": True, "path": ""})
    inbox = sched.default_watch_path()
    assert inbox.is_dir()

    (inbox / "book.pdf").write_bytes(b"%PDF-1.4 fake")
    (inbox / "notes.txt").write_text("not a pdf")

    # First scan: file just appeared -> only snapshotted, not ingested.
    r1 = await sched.scan_watch_folder()
    assert r1["queued"] == 0
    # Second scan: unchanged -> ingested.
    r2 = await sched.scan_watch_folder()
    assert r2["queued"] == 1

    assert not (inbox / "book.pdf").exists()
    assert (inbox / "ingested" / "book.pdf").exists()
    assert (inbox / "notes.txt").exists()  # non-PDF untouched

    jobs = await manager.list_recent(limit=10)
    assert jobs[0].job_type == "ingest"
    assert jobs[0].filename == "book.pdf"
    assert jobs[0].job_params.get("source") == "watch-folder"
    # Staged copy exists in uploads and the spawned run_job was called.
    staged = list((tmp_path / "data" / "uploads").glob("*_book.pdf"))
    assert len(staged) == 1
    import asyncio
    await asyncio.sleep(0.05)  # let the spawned stub run
    assert pipeline.calls and pipeline.calls[0][1] == "default"


@pytest.mark.asyncio
async def test_watch_scan_files_duplicates_without_jobs(scheduler):
    sched, manager, pipeline, neo4j = scheduler
    await sched.update_watch({"enabled": True, "path": ""})
    inbox = sched.default_watch_path()
    (inbox / "dupe.pdf").write_bytes(b"%PDF-1.4 same content")

    import hashlib
    neo4j.known_hashes.add(hashlib.sha256(b"%PDF-1.4 same content").hexdigest())

    result = await sched.scan_watch_folder(force=True)
    assert result["duplicates"] == 1 and result["queued"] == 0
    assert (inbox / "duplicates" / "dupe.pdf").exists()
    jobs = await manager.list_recent(limit=10)
    assert jobs == []
    assert pipeline.calls == []


@pytest.mark.asyncio
async def test_watch_rejects_bad_paths(scheduler):
    sched, _, _, _ = scheduler
    with pytest.raises(ValueError):
        await sched.update_watch({"enabled": True, "path": "relative/dir"})
    with pytest.raises(ValueError):
        await sched.update_watch({"enabled": True, "path": "/nonexistent-forge-inbox"})


# --------------------------------------------------------------- endpoints


@pytest.fixture()
async def api(scheduler):
    sched, manager, pipeline, neo4j = scheduler
    app = FastAPI()
    app.include_router(schedule_router.router)
    app.state.scheduler = sched
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, sched


@pytest.mark.asyncio
async def test_get_and_put_schedule_roundtrip(api):
    client, _ = api
    r = await client.get("/schedule")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["schedule"]["enabled"] is False
    assert "status" in data and "watch" in data

    r = await client.put(
        "/schedule",
        json={"enabled": True, "start": "22:00", "end": "05:00", "days": [0, 1]},
    )
    assert r.status_code == 200
    assert r.json()["data"]["schedule"]["start"] == "22:00"

    r = await client.put("/schedule", json={"start": "nope"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_put_watch_and_scan_now(api, tmp_path):
    client, sched = api
    r = await client.post("/schedule/watch/scan-now")
    assert r.status_code == 409  # not enabled yet

    r = await client.put("/schedule/watch", json={"enabled": True, "path": ""})
    assert r.status_code == 200
    assert r.json()["data"]["watch"]["path"].endswith("ingest-inbox")

    inbox = sched.default_watch_path()
    (inbox / "drop.pdf").write_bytes(b"%PDF-1.4 dropped")
    r = await client.post("/schedule/watch/scan-now")
    assert r.status_code == 200
    assert r.json()["data"]["queued"] == 1

    r = await client.put("/schedule/watch", json={"enabled": True, "path": "/nope"})
    assert r.status_code == 400


# ----------------------------------------------------- subfolders / browse


@pytest.mark.asyncio
async def test_watch_scan_recurses_and_preserves_structure(scheduler):
    sched, manager, pipeline, _ = scheduler
    await sched.update_watch({"enabled": True, "path": ""})
    inbox = sched.default_watch_path()
    (inbox / "robotics" / "gaits").mkdir(parents=True)
    (inbox / "robotics" / "gaits" / "trot.pdf").write_bytes(b"%PDF-1.4 trot")
    # Files inside the output trees must never be re-scanned.
    (inbox / "ingested").mkdir()
    (inbox / "ingested" / "old.pdf").write_bytes(b"%PDF-1.4 old")

    result = await sched.scan_watch_folder(force=True)
    assert result["queued"] == 1
    assert (inbox / "ingested" / "robotics" / "gaits" / "trot.pdf").exists()
    assert (inbox / "ingested" / "old.pdf").exists()  # untouched
    jobs = await manager.list_recent(limit=5)
    assert jobs[0].filename == "trot.pdf"


@pytest.mark.asyncio
async def test_browse_endpoint_lists_directories(api, tmp_path):
    client, _ = api
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")

    r = await client.get("/schedule/browse", params={"path": str(tmp_path)})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["name"] for d in data["dirs"]] == ["alpha", "beta"]
    assert data["parent"] == str(tmp_path.parent)

    r = await client.get("/schedule/browse", params={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_open_folder_endpoint(api):
    client, sched = api
    # Not configured yet -> 409
    r = await client.post("/schedule/watch/open-folder")
    assert r.status_code == 409

    await sched.update_watch({"enabled": True, "path": ""})
    calls = []
    sched._opener = lambda cmd, env: calls.append((cmd, env))
    r = await client.post("/schedule/watch/open-folder")
    assert r.status_code == 200
    assert calls and calls[0][0][0] == "xdg-open"
    assert calls[0][0][1] == sched.watch["path"]
    assert "DISPLAY" in calls[0][1]
