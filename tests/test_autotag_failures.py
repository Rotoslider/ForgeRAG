"""Auto-tagging failures must stay visible and repairable.

Third instance of the swallow pattern: on LLM failure the auto-tagger
returned a default AutoTagResult, so the doc silently landed unorganized
(default collection, no categories, no tags) with nothing recording that
tagging never ran. Now the failure propagates — the ingest step marks it —
and the autotag-missing drain repairs unorganized docs.
"""

from __future__ import annotations

import pytest

from backend.ingestion.auto_tagger import AutoTagger, AutoTagResult
from backend.ingestion.pipeline import IngestionPipeline
from backend.services.llm_service import LLMFatalError, LLMTransientError

from tests.test_entity_extraction_failures import _JobsStub


class _LLMStub:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    async def chat_json_structured(self, *args, **kwargs):
        if self.exc is not None:
            raise self.exc
        return self.result


# --------------------------------------------------------------- suggest()


async def test_suggest_propagates_llm_errors():
    tagger = AutoTagger(_LLMStub(exc=LLMFatalError("circuit breaker open")))
    with pytest.raises(LLMFatalError):
        await tagger.suggest(
            title="T", filename="f.pdf", sample_pages_text=["some text"],
        )
    tagger = AutoTagger(_LLMStub(exc=LLMTransientError("Request failed")))
    with pytest.raises(LLMTransientError):
        await tagger.suggest(
            title="T", filename="f.pdf", sample_pages_text=["some text"],
        )


async def test_suggest_normalizes_successful_result():
    tagger = AutoTagger(_LLMStub(result=AutoTagResult(
        collection="Welding Codes", categories=["Welding"],
        tags=["Arc Welding", " brazing "],
    )))
    r = await tagger.suggest(
        title="T", filename="f.pdf", sample_pages_text=["some text"],
    )
    assert r.collection == "welding_codes"
    assert r.tags == ["arc-welding", "brazing"]


# ------------------------------------------------------ autotag-missing job


class _AutotagNeo4j:
    def __init__(self, doc_ids):
        self.doc_ids = list(doc_ids)
        self.writes: list[tuple[str, dict]] = []

    async def run_query(self, query, params=None, **kwargs):
        return [{"doc_id": d} for d in self.doc_ids]

    async def run_write(self, query, params=None, **kwargs):
        self.writes.append((query, params or {}))


class _TaggerStub:
    """suggest_for_doc raises for fail_ids, returns None for no_text_ids."""

    def __init__(self, fail_ids=(), no_text_ids=()):
        self.fail_ids = set(fail_ids)
        self.no_text_ids = set(no_text_ids)

    async def suggest_for_doc(self, neo4j, doc_id):
        if doc_id in self.fail_ids:
            raise LLMFatalError("Structured JSON call failed after retries")
        if doc_id in self.no_text_ids:
            return None
        return AutoTagResult(
            collection="electronics", categories=["Electronics"],
            tags=["circuits"],
        )


def _pipeline(neo4j, jobs, tagger) -> IngestionPipeline:
    p = object.__new__(IngestionPipeline)
    p.neo4j = neo4j
    p.jobs = jobs
    p.auto_tagger = tagger
    return p


async def test_autotag_missing_tags_unorganized_docs():
    neo4j = _AutotagNeo4j(["d1", "d2"])
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _TaggerStub())

    await p.run_autotag_missing("job1")

    assert jobs.completed == ["job1"]
    # Each doc gets collection + category + tag writes
    assert sum(1 for q, _ in neo4j.writes if "d.collection" in q) == 2
    assert sum(1 for q, _ in neo4j.writes if "IN_CATEGORY" in q) == 2
    assert sum(1 for q, _ in neo4j.writes if "TAGGED_WITH" in q) == 2


async def test_autotag_missing_fails_job_when_all_docs_fail():
    neo4j = _AutotagNeo4j(["d1", "d2"])
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _TaggerStub(fail_ids={"d1", "d2"}))

    await p.run_autotag_missing("job1")

    assert jobs.completed == []
    assert len(jobs.failures) == 1
    assert "all 2 documents" in jobs.failures[0]
    assert neo4j.writes == []


async def test_autotag_missing_partial_failure_warns_and_completes():
    neo4j = _AutotagNeo4j(["d1", "d2"])
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _TaggerStub(fail_ids={"d2"}))

    await p.run_autotag_missing("job1")

    assert jobs.completed == ["job1"]
    warnings = [(s, st, d) for s, st, d in jobs.steps if st == "warning"]
    assert len(warnings) == 1
    assert "1 documents tagged" in warnings[0][2]
    assert "1 failed" in warnings[0][2]


async def test_autotag_missing_counts_textless_docs_as_skipped_not_tagged():
    # A doc with no text can't be tagged — the job must not claim it was.
    neo4j = _AutotagNeo4j(["d1", "d2"])
    jobs = _JobsStub()
    p = _pipeline(neo4j, jobs, _TaggerStub(no_text_ids={"d2"}))

    await p.run_autotag_missing("job1")

    assert jobs.completed == ["job1"]
    warnings = [(s, st, d) for s, st, d in jobs.steps if st == "warning"]
    assert len(warnings) == 1
    assert "1 documents tagged" in warnings[0][2]
    assert "no text to analyze" in warnings[0][2]
