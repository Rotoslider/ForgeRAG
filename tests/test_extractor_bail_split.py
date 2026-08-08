"""EntityExtractor anti-bail retry and split-on-truncation.

Two live-observed failure modes on dense table pages (2026-08-07):
1. The model bails with a fast, schema-valid EMPTY extraction instead of
   transcribing — the extractor must retry once with an explicit nudge
   before trusting an empty result on a dense page.
2. The genuine extraction overflows even the 32k max_tokens ceiling — the
   extractor must split the page text in half and merge the two halves.
"""

import pytest

from backend.config import LLMSettings
from backend.ingestion.entity_extractor import (
    BAIL_RETRY_MIN_CHARS,
    EntityExtractor,
    MaterialMention,
    PageExtraction,
)
from backend.services.llm_service import LLMTruncationError

pytestmark = pytest.mark.asyncio

DENSE = "AISI 4140 chromoly, hardness 28 HRC. " * 100  # >> BAIL_RETRY_MIN_CHARS
SPARSE = "Intro prose with no entities."


def _extraction(*names: str) -> PageExtraction:
    return PageExtraction(materials=[MaterialMention(name=n) for n in names])


class _ScriptedLLM:
    """chat_json_structured plays back a script of results/exceptions and
    records the messages of every call."""

    settings = LLMSettings()

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.kwargs_log: list[dict] = []

    async def chat_json_structured(self, messages, schema_cls, **kwargs):
        self.calls.append(messages)
        self.kwargs_log.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def test_dense_empty_gets_one_nudged_retry():
    llm = _ScriptedLLM([PageExtraction(), _extraction("AISI 4140")])
    ex = EntityExtractor(llm)

    result = await ex.extract_page(
        document_title="T", page_number=1, page_text=DENSE,
    )

    assert [m.name for m in result.materials] == ["AISI 4140"]
    assert len(llm.calls) == 2
    # Retry carries the anti-bail nudge as an extra user message.
    assert len(llm.calls[1]) == 3
    assert "DOES contain technical content" in llm.calls[1][-1]["content"]


async def test_dense_empty_twice_is_accepted_as_empty():
    llm = _ScriptedLLM([PageExtraction(), PageExtraction()])
    ex = EntityExtractor(llm)

    result = await ex.extract_page(
        document_title="T", page_number=1, page_text=DENSE,
    )

    assert not result.materials
    assert len(llm.calls) == 2


async def test_sparse_empty_is_accepted_without_retry():
    assert len(SPARSE) < BAIL_RETRY_MIN_CHARS
    llm = _ScriptedLLM([PageExtraction()])
    ex = EntityExtractor(llm)

    result = await ex.extract_page(
        document_title="T", page_number=1, page_text=SPARSE,
    )

    assert not result.materials
    assert len(llm.calls) == 1


async def test_ceiling_truncation_splits_page_and_merges():
    llm = _ScriptedLLM([
        LLMTruncationError("response truncated at the max_tokens ceiling"),
        _extraction("Alloy 625"),
        _extraction("ASTM A36"),
    ])
    ex = EntityExtractor(llm)

    result = await ex.extract_page(
        document_title="T", page_number=1, page_text=DENSE,
    )

    assert {m.name for m in result.materials} == {"Alloy 625", "ASTM A36"}
    assert len(llm.calls) == 3
    # The two half-calls together cover the full page text.
    half1 = llm.calls[1][-1]["content"]
    half2 = llm.calls[2][-1]["content"]
    assert len(half1) < len(llm.calls[0][-1]["content"])
    assert len(half2) < len(llm.calls[0][-1]["content"])


async def test_truncating_half_retries_without_strict_grammar():
    llm = _ScriptedLLM([
        LLMTruncationError("ceiling"),      # full page overflows -> split
        LLMTruncationError("ceiling"),      # first half overflows (loop)
        _extraction("Alloy 625"),           # non-strict retry breaks it
        _extraction("ASTM A36"),            # second half fine under grammar
    ])
    ex = EntityExtractor(llm)

    result = await ex.extract_page(
        document_title="T", page_number=1, page_text=DENSE,
    )

    assert {m.name for m in result.materials} == {"Alloy 625", "ASTM A36"}
    # The loop-breaking retry is the only call forcing strict=False.
    stricts = [k.get("strict") for k in llm.kwargs_log]
    assert stricts == [None, None, False, None]


async def test_half_failing_even_non_strict_fails_the_page():
    llm = _ScriptedLLM([
        LLMTruncationError("ceiling"),  # full page -> split
        LLMTruncationError("ceiling"),  # first half, strict
        LLMTruncationError("ceiling"),  # first half, non-strict retry
    ])
    ex = EntityExtractor(llm)

    with pytest.raises(LLMTruncationError):
        await ex.extract_page(
            document_title="T", page_number=1, page_text=DENSE,
        )
