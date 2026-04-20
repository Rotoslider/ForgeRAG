"""Per-chunk LLM summarization for richer retrieval.

Each structural chunk gets a 1-3 sentence summary. The summary is embedded
alongside the raw chunk text (in fact they're indexed together in the
fulltext index) so queries phrased in generalized language match even
when the raw chunk uses jargon:

    Query:   "what size hole to drill for a 1/4-20 tap?"
    Raw:     (a table showing tap drill diameters indexed by thread)
    Summary: "Tap drill sizes in inches for Unified National Coarse (UNC)
              and Unified National Fine (UNF) threads from #0 through
              1-1/2 inch."

The summary path catches the match that the raw (mostly numbers) would miss.

Design notes:
- We send the chunk TEXT plus minimal context (section path, chunk type)
  — NOT the surrounding chunks. Summaries should describe what's in *this*
  chunk, not the neighborhood.
- We keep summaries short (~50 words) to keep embeddings focused and
  reduce LLM cost across 100K+ chunks.
- If the chunk text is already short and factual (< 300 chars), we skip
  summarization and use the raw text as both text and summary. No point
  summarizing one sentence into one sentence.
"""

from __future__ import annotations

import logging
from typing import Sequence

from backend.ingestion.chunker import StructuralChunk
from backend.services.llm_service import (
    LLMFatalError,
    LLMService,
    LLMTransientError,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are summarizing individual chunks (paragraphs, tables, figures, and
equations) extracted from engineering reference handbooks so they can be
retrieved by search. Given ONE chunk, produce a single short summary
(1-3 sentences, under 50 words) that describes what the chunk is about
in language a user would naturally search with.

Rules:
- Write ONE summary, no lists, no headers, no commentary.
- For a TABLE: describe the dimensions. "Tap drill sizes for UNC threads
  from #0 through 1-1/2 inch" beats "A table of numbers".
- For a FIGURE or CAPTION: describe what the figure shows.
- For an EQUATION: name the formula and what it computes.
- For PROSE text: capture the main claim or procedure in 1-2 sentences.
- Use engineering terminology from the chunk itself (alloy names,
  standard codes, unit system, process names).
- Do NOT invent content that isn't in the chunk.
- Do NOT copy the chunk verbatim.

Output ONLY the summary text. No JSON, no prose wrapper, no code fences.
/no_think
"""


# Chunks this short don't need a separate summary — their raw text is
# already concise enough to serve as both the searchable text and the
# "summary". Saves one LLM call per short chunk.
_SHORT_CHUNK_CHARS = 300


class ChunkSummarizer:
    """Thin wrapper around LLMService for chunk-level summaries."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def summarize(self, chunk: StructuralChunk) -> str:
        """Return a summary for one chunk. Returns the raw text for short
        chunks; returns an LLM-generated summary for longer ones. On LLM
        failure, falls back to a truncated preview — so the chunk still
        has *some* summary field populated."""
        text = chunk.text.strip()
        if len(text) < _SHORT_CHUNK_CHARS:
            return text

        user_parts = []
        if chunk.section_path:
            user_parts.append("Section: " + " > ".join(chunk.section_path))
        user_parts.append(f"Chunk type: {chunk.chunk_type}")
        user_parts.append(f"Page: {chunk.page_number}")
        user_parts.append("")
        user_parts.append("Chunk content:")
        user_parts.append(text[:4000])  # cap input to avoid runaway prompts

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        try:
            raw = await self.llm.chat(
                messages,
                max_tokens=150,
                temperature=0.1,
            )
        except (LLMTransientError, LLMFatalError) as exc:
            logger.warning(
                "Chunk summarization failed for chunk %s: %s — "
                "falling back to text preview",
                chunk.chunk_id, exc,
            )
            return text[:240]

        summary = (raw or "").strip()
        # Strip common wrappers the model sometimes adds despite instructions.
        for prefix in ("Summary:", "summary:", "- ", "* "):
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
        # Safety: if the LLM returned something far too long or clearly
        # malformed, fall back to the preview.
        if not summary or len(summary) > 600:
            return text[:240]
        return summary

    async def summarize_batch(
        self, chunks: Sequence[StructuralChunk], concurrency: int = 4,
    ) -> list[str]:
        """Summarize many chunks with bounded concurrency. Order of output
        matches order of input."""
        import asyncio

        sem = asyncio.Semaphore(concurrency)

        async def _one(i: int, c: StructuralChunk) -> tuple[int, str]:
            async with sem:
                s = await self.summarize(c)
                return i, s

        tasks = [asyncio.create_task(_one(i, c)) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        out = [""] * len(chunks)
        for i, s in results:
            out[i] = s
        return out
