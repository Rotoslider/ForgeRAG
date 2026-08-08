"""LLM failures during entity extraction must fail pages/jobs — never be
silently converted into "extracted, found nothing".

Regression tests for the incident where a fill-missing job ran with the LLM
endpoint down: every call failed, yet the job completed, logged "0/1 pages
failed", and stamped the page entities_extracted_at so the completeness
audit would never retry it.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.config import LLMSettings
from backend.ingestion.entity_extractor import EntityExtractor, PageExtraction
from backend.ingestion.pipeline import IngestionPipeline
from backend.services.llm_service import LLMFatalError, LLMTransientError


# ------------------------------------------------------------------- stubs


class _FailingLLM:
    """LLMService stand-in whose structured call always fails."""

    settings = LLMSettings()

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    async def chat_json_structured(self, *args, **kwargs):
        self.calls += 1
        raise self.exc


class _Neo4jStub:
    """Dispatches on query text; records every write."""

    def __init__(self, doc_rows):
        self.doc_rows = doc_rows
        self.writes: list[tuple[str, dict | None]] = []

    async def run_query(self, query, params=None):
        if "file_hash" in query:
            return [{"h": "hash123"}]
        if "count(p) AS n" in query:
            return [{"n": len(self.doc_rows[0]["pages"])}]
        return self.doc_rows

    async def run_write(self, query, params=None):
        self.writes.append((query, params))


class _JobsStub:
    def __init__(self):
        self.steps: list[tuple[str, str, str | None]] = []
        self.completed: list[str] = []
        self.failures: list[str] = []

    async def set_steps(self, job_id, plan):
        pass

    async def update(self, job_id, **kwargs):
        pass

    async def checkpoint(self, job_id):
        # Pause/stop gate — a no-op here; job-control behaviour is covered
        # by tests/test_job_control.py.
        pass

    async def update_step(self, job_id, step, status, detail=None):
        self.steps.append((step, status, detail))

    async def complete(self, job_id):
        self.completed.append(job_id)

    async def fail(self, job_id, message):
        self.failures.append(message)


def _doc_rows(n_pages: int):
    return [{
        "title": "Remote Area Power Supply",
        "pages": [
            {"page_id": f"p{i}", "page_number": i, "text": f"page {i} text"}
            for i in range(1, n_pages + 1)
        ],
    }]


def _pipeline(neo4j, jobs, extractor, *, max_concurrent: int = 3) -> IngestionPipeline:
    # Bypass __init__ (it builds Docling/PDF machinery); wire only what the
    # extraction paths touch.
    p = object.__new__(IngestionPipeline)
    p.neo4j = neo4j
    p.jobs = jobs
    p.entity_extractor = extractor
    p.llm = extractor.llm
    p.graph_builder = None  # only reached on successful extractions
    p._ingest_semaphore = asyncio.Semaphore(max_concurrent)
    return p


class _StubExtractor:
    """extract_page raises for page numbers in `fail_pages`."""

    def __init__(self, fail_pages: set[int]):
        self.llm = _FailingLLM(LLMFatalError("unused"))
        self.fail_pages = fail_pages

    async def extract_page(self, *, document_title, page_number, page_text):
        if page_number in self.fail_pages:
            raise LLMFatalError(
                "Structured JSON call failed after retries: "
                "LLM service circuit breaker open"
            )
        return PageExtraction()


class _GraphStub:
    def __init__(self):
        self.pages_written: list[str] = []

    async def write_page(self, *, page_id, extraction):
        self.pages_written.append(page_id)
        return {"materials": 0, "processes": 0, "standards": 0,
                "clauses": 0, "equipment": 0,
                "page_rels": 0, "entity_rels": 0}


# --------------------------------------------------- extractor propagation


async def test_extract_page_propagates_fatal_llm_error():
    ex = EntityExtractor(_FailingLLM(LLMFatalError("circuit breaker open")))
    with pytest.raises(LLMFatalError):
        await ex.extract_page(
            document_title="T", page_number=1, page_text="steel plate",
        )


async def test_extract_page_propagates_transient_llm_error():
    ex = EntityExtractor(_FailingLLM(LLMTransientError("Request failed")))
    with pytest.raises(LLMTransientError):
        await ex.extract_page(
            document_title="T", page_number=1, page_text="steel plate",
        )


async def test_extract_page_empty_text_skips_llm_and_returns_empty():
    llm = _FailingLLM(LLMFatalError("should never be called"))
    ex = EntityExtractor(llm)
    result = await ex.extract_page(
        document_title="T", page_number=1, page_text="   ",
    )
    assert isinstance(result, PageExtraction)
    assert llm.calls == 0


# ------------------------------------------- pipeline counting + stamping


async def test_failed_page_is_counted_and_not_stamped():
    neo4j = _Neo4jStub(_doc_rows(2))
    jobs = _JobsStub()
    extractor = _StubExtractor(fail_pages={1})
    p = _pipeline(neo4j, jobs, extractor)
    p.graph_builder = _GraphStub()

    done, failed, last_err = await p._extract_entities("job1", "doc1")

    assert (done, failed) == (2, 1)
    assert last_err is not None
    # Only the successful page may be stamped entities_extracted_at.
    stamped = [params["pid"] for q, params in neo4j.writes
               if "entities_extracted_at" in q]
    assert stamped == ["p2"]


async def test_dense_empty_page_stamped_confirmed_empty():
    # An empty extraction on a DENSE page already survived the extractor's
    # anti-bail retry — the stamp must carry entities_confirmed_empty so
    # the suspicious-empty check/drain never re-pays it.
    rows = _doc_rows(1)
    rows[0]["pages"][0]["text"] = "dense table content " * 200  # > 2000 chars
    rows[0]["pages"][0]["char_count"] = 4000  # density source = stored property
    neo4j = _Neo4jStub(rows)
    p = _pipeline(neo4j, _JobsStub(), _StubExtractor(fail_pages=set()))
    p.graph_builder = _GraphStub()

    done, failed, _ = await p._extract_entities("job1", "doc1")

    assert (done, failed) == (1, 0)
    stamps = [q for q, _ in neo4j.writes if "entities_extracted_at" in q]
    assert len(stamps) == 1
    assert "entities_confirmed_empty = true" in stamps[0]


async def test_entity_bearing_page_never_marked_confirmed_empty():
    # counts["page_rels"] tallies the model's explicit relationship list,
    # which is legitimately zero on entity-rich pages — the confirmed-empty
    # gate must key on the per-type entity counts instead (live bug
    # 2026-08-08: entity-bearing drain pages were wrongly flagged).
    rows = _doc_rows(1)
    rows[0]["pages"][0]["text"] = "dense table content " * 200

    class _EntityGraphStub(_GraphStub):
        async def write_page(self, *, page_id, extraction):
            await super().write_page(page_id=page_id, extraction=extraction)
            return {"materials": 5, "processes": 0, "standards": 0,
                    "clauses": 0, "equipment": 0,
                    "page_rels": 0, "entity_rels": 0}

    neo4j = _Neo4jStub(rows)
    p = _pipeline(neo4j, _JobsStub(), _StubExtractor(fail_pages=set()))
    p.graph_builder = _EntityGraphStub()

    done, failed, _ = await p._extract_entities("job1", "doc1")

    assert (done, failed) == (1, 0)
    stamps = [q for q, _ in neo4j.writes if "entities_extracted_at" in q]
    assert len(stamps) == 1
    # Never marked confirmed-empty — and any STALE flag from a previous
    # empty run must be actively cleared, not left in place.
    assert "entities_confirmed_empty = true" not in stamps[0]
    assert "entities_confirmed_empty = null" in stamps[0]


async def test_sparse_empty_page_stamped_without_confirmed_marker():
    neo4j = _Neo4jStub(_doc_rows(1))  # stub pages have no char_count (sparse)
    p = _pipeline(neo4j, _JobsStub(), _StubExtractor(fail_pages=set()))
    p.graph_builder = _GraphStub()

    done, failed, _ = await p._extract_entities("job1", "doc1")

    assert (done, failed) == (1, 0)
    stamps = [q for q, _ in neo4j.writes if "entities_extracted_at" in q]
    assert len(stamps) == 1
    assert "entities_confirmed_empty = true" not in stamps[0]


async def test_all_pages_failed_reports_every_failure():
    neo4j = _Neo4jStub(_doc_rows(3))
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _StubExtractor(fail_pages={1, 2, 3}))

    done, failed, last_err = await p._extract_entities("job1", "doc1")

    assert (done, failed) == (3, 3)
    assert last_err is not None
    assert neo4j.writes == []  # nothing stamped


# ----------------------------------------------------- job-level outcomes


async def test_fill_missing_fails_job_when_all_extractions_fail():
    neo4j = _Neo4jStub(_doc_rows(1))
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _StubExtractor(fail_pages={1}))

    await p.run_fill_missing(
        "job1", "doc1",
        do_text=False, do_visual=False, do_entities=True,
    )

    assert jobs.completed == []
    assert len(jobs.failures) == 1
    assert "all 1 pages" in jobs.failures[0]
    assert ("extracting_entities", "error") in [
        (s, st) for s, st, _ in jobs.steps
    ]
    assert neo4j.writes == []


async def test_fill_missing_partial_failure_completes_with_warning():
    neo4j = _Neo4jStub(_doc_rows(2))
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _StubExtractor(fail_pages={1}))
    p.graph_builder = _GraphStub()

    await p.run_fill_missing(
        "job1", "doc1",
        do_text=False, do_visual=False, do_entities=True,
    )

    assert jobs.completed == ["job1"]
    assert jobs.failures == []
    warnings = [(s, st, d) for s, st, d in jobs.steps if st == "warning"]
    assert len(warnings) == 1
    assert "1 of 2 pages failed" in warnings[0][2]


async def test_extraction_only_fails_job_when_all_extractions_fail():
    neo4j = _Neo4jStub(_doc_rows(2))
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _StubExtractor(fail_pages={1, 2}))

    await p.run_extraction_only("job1", "doc1")

    assert jobs.completed == []
    assert len(jobs.failures) == 1
    assert "all 2 pages" in jobs.failures[0]


# ------------------------------------------------------ job concurrency cap


async def test_fill_missing_jobs_respect_ingest_semaphore():
    # 6 jobs queued at once with a cap of 2 must never run more than 2
    # concurrently — the unbounded version of this is what overloaded the
    # LLM server (bulk drains queue one job per doc).
    active = 0
    peak = 0

    class _TrackingExtractor:
        def __init__(self):
            self.llm = _FailingLLM(LLMFatalError("unused"))

        async def extract_page(self, *, document_title, page_number, page_text):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return PageExtraction()

    jobs = _JobsStub()
    extractor = _TrackingExtractor()
    p = _pipeline(_Neo4jStub(_doc_rows(1)), jobs, extractor, max_concurrent=2)
    p.graph_builder = _GraphStub()

    await asyncio.gather(*[
        p.run_fill_missing(
            f"job{i}", "doc1",
            do_text=False, do_visual=False, do_entities=True,
        )
        for i in range(6)
    ])

    assert len(jobs.completed) == 6
    assert peak <= 2, f"ran {peak} jobs concurrently despite cap of 2"
