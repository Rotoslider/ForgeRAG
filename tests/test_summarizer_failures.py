"""Chunk-summary failures must stay visible and repairable.

Same disease as entity extraction: when the LLM fails during chunking, the
summarizer falls back to a text preview. Before 2026-08-06 that fallback was
stored indistinguishably from a real summary — 18k+ chunks accumulated fake
summaries. Now the fallback is marked (summary_source='preview') and the
resummarize repair regenerates marked chunks, never converting a failure
into 'done'.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

from backend.ingestion.chunker import StructuralChunk
from backend.ingestion.chunk_summarizer import ChunkSummarizer
from backend.ingestion.pipeline import IngestionPipeline
from backend.services.llm_service import LLMTransientError

from tests.test_entity_extraction_failures import _JobsStub

import asyncio


def _chunk(text: str, chunk_id: str = "c1") -> StructuralChunk:
    return StructuralChunk(
        chunk_id=chunk_id, page_number=1, chunk_index=0,
        chunk_type="text", text=text,
    )


class _LLMStub:
    def __init__(self, response: str | None = None, exc: Exception | None = None,
                 finish_reason: str = "stop"):
        self.response = response
        self.exc = exc
        self.finish_reason = finish_reason
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.response

    async def chat_with_finish_reason(self, *args, **kwargs):
        return await self.chat(*args, **kwargs), self.finish_reason


# ------------------------------------------------------------- summarize()


async def test_short_chunk_is_its_own_summary_no_llm():
    llm = _LLMStub(exc=AssertionError("LLM must not be called"))
    s, source = await ChunkSummarizer(llm).summarize(_chunk("short text"))
    assert (s, source) == ("short text", "short")
    assert llm.calls == 0


async def test_llm_summary_is_marked_llm():
    llm = _LLMStub(response="Tap drill sizes for UNC threads.")
    s, source = await ChunkSummarizer(llm).summarize(_chunk("x" * 500))
    assert source == "llm"
    assert s == "Tap drill sizes for UNC threads."


async def test_llm_failure_falls_back_to_marked_preview():
    llm = _LLMStub(exc=LLMTransientError("Request failed: ReadTimeout"))
    text = "y" * 500
    s, source = await ChunkSummarizer(llm).summarize(_chunk(text))
    assert source == "preview"
    assert s == text[:240]


async def test_garbage_llm_output_falls_back_to_marked_preview():
    llm = _LLMStub(response="z" * 2000)  # far over the 600-char sanity cap
    text = "w" * 500
    s, source = await ChunkSummarizer(llm).summarize(_chunk(text))
    assert source == "preview"
    assert s == text[:240]


# -------------------------------------------------------- run_resummarize


class _Vec:
    def tolist(self):
        return [0.1, 0.2]


class _EmbedStub:
    def embed_documents(self, inputs, batch_size=None):
        return [_Vec() for _ in inputs]


class _GPUStub:
    def load_scope(self, name):
        @asynccontextmanager
        async def cm():
            yield
        return cm()


class _ResummarizeNeo4j:
    """Holds a pending set of fallback chunks; run_write repairs them."""

    def __init__(self, chunk_ids):
        self.pending = {
            cid: {"chunk_id": cid, "page_number": 1, "chunk_type": "text",
                  "text": "t" * 500, "section_path": []}
            for cid in chunk_ids
        }
        self.repaired: list[dict] = []

    async def run_query(self, query, params=None, **kwargs):
        if "count(c) AS n" in query:
            return [{"n": len(self.pending)}]
        skip = set((params or {}).get("skip", []))
        rows = [r for cid, r in self.pending.items() if cid not in skip]
        return rows[: (params or {}).get("batch", 100)]

    async def run_write(self, query, params=None, **kwargs):
        for row in (params or {}).get("rows", []):
            self.pending.pop(row["chunk_id"], None)
            self.repaired.append(row)


class _BatchSummarizer:
    """summarize_batch stub; fails chunks whose id is in fail_ids."""

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)

    async def summarize_batch(self, chunks, concurrency=4):
        return [
            ("preview text", "preview") if c.chunk_id in self.fail_ids
            else (f"summary of {c.chunk_id}", "llm")
            for c in chunks
        ]


def _resum_pipeline(neo4j, jobs, summarizer) -> IngestionPipeline:
    p = object.__new__(IngestionPipeline)
    p.neo4j = neo4j
    p.jobs = jobs
    p.chunk_summarizer = summarizer
    p.text_embedding = _EmbedStub()
    p.gpu = _GPUStub()
    p.settings = SimpleNamespace(
        ingestion=SimpleNamespace(text_embedding_batch_size=32)
    )
    p._ingest_semaphore = asyncio.Semaphore(3)
    return p


async def test_resummarize_repairs_marked_chunks():
    neo4j = _ResummarizeNeo4j(["c1", "c2", "c3"])
    jobs = _JobsStub()
    p = _resum_pipeline(neo4j, jobs, _BatchSummarizer())

    await p.run_resummarize("job1")

    assert jobs.completed == ["job1"]
    assert {r["chunk_id"] for r in neo4j.repaired} == {"c1", "c2", "c3"}
    assert all(r["source"] == "llm" for r in neo4j.repaired)
    assert all(r["embedding"] == [0.1, 0.2] for r in neo4j.repaired)
    assert neo4j.pending == {}


async def test_resummarize_leaves_failed_chunks_marked():
    neo4j = _ResummarizeNeo4j(["c1", "c2"])
    jobs = _JobsStub()
    p = _resum_pipeline(neo4j, jobs, _BatchSummarizer(fail_ids={"c2"}))

    await p.run_resummarize("job1")

    assert jobs.completed == ["job1"]
    assert {r["chunk_id"] for r in neo4j.repaired} == {"c1"}
    assert "c2" in neo4j.pending  # still marked for the next run
    warnings = [(s, st, d) for s, st, d in jobs.steps if st == "warning"]
    assert len(warnings) == 1 and "1 of 2" in warnings[0][2]


async def test_resummarize_fails_job_when_all_chunks_fail():
    neo4j = _ResummarizeNeo4j(["c1", "c2"])
    jobs = _JobsStub()
    p = _resum_pipeline(neo4j, jobs, _BatchSummarizer(fail_ids={"c1", "c2"}))

    await p.run_resummarize("job1")

    assert jobs.completed == []
    assert len(jobs.failures) == 1
    assert "all 2 chunks" in jobs.failures[0]
    assert neo4j.repaired == []
    assert set(neo4j.pending) == {"c1", "c2"}  # nothing lost


async def test_resummarize_noop_when_nothing_marked():
    neo4j = _ResummarizeNeo4j([])
    jobs = _JobsStub()
    p = _resum_pipeline(neo4j, jobs, _BatchSummarizer())

    await p.run_resummarize("job1")

    assert jobs.completed == ["job1"]
    assert neo4j.repaired == []


async def test_truncated_summary_is_marked_preview():
    # A summary cut off at max_tokens (finish_reason="length") is not a
    # summary — storing it as source='llm' would hide it from the
    # resummarize repair forever.
    llm = _LLMStub(response="This summary was cut off mid", finish_reason="length")
    text = "x" * 500
    s, source = await ChunkSummarizer(llm).summarize(_chunk(text))
    assert source == "preview"
    assert s == text[:240]
