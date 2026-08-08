"""chat_json_structured truncation handling.

A response cut off at max_tokens (finish_reason == "length") is unparseable
JSON for a reason no re-prompt can fix. The retry loop must escalate the
token budget instead of appending the "respond with ONLY JSON" nudge, and
must give up once the escalation ceiling is reached rather than burning
full-generation retries that cannot succeed.
"""

import pytest
from pydantic import BaseModel

from backend.config import LLMSettings
from backend.services.llm_service import (
    LLMFatalError,
    LLMService,
    TRUNCATION_MAX_TOKENS,
)

pytestmark = pytest.mark.asyncio


class _Result(BaseModel):
    items: list[str] = []


TRUNCATED = '{"items": ["a", "b", "c"'  # cut mid-array, unbalanced
COMPLETE = '{"items": ["a", "b", "c"]}'


def _service(monkeypatch, responses):
    """LLMService whose chat_with_finish_reason pops canned responses.

    Each canned entry is (content, finish_reason). Calls are recorded as
    (messages, max_tokens) so tests can assert on escalation and nudges.
    """
    svc = LLMService(LLMSettings())
    calls = []

    async def fake_chat(messages, *, max_tokens=None, temperature=None,
                        response_format=None):
        calls.append((messages, max_tokens))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(svc, "chat_with_finish_reason", fake_chat)
    return svc, calls


async def test_truncation_escalates_max_tokens_and_skips_nudge(monkeypatch):
    svc, calls = _service(monkeypatch, [
        (TRUNCATED, "length"),
        (COMPLETE, "stop"),
    ])

    result = await svc.chat_json_structured(
        [{"role": "user", "content": "extract"}], _Result, max_tokens=8192,
    )

    assert result.items == ["a", "b", "c"]
    assert [mt for _, mt in calls] == [8192, 16384]
    # Truncation must NOT append the "respond with ONLY JSON" nudge — the
    # model did nothing wrong and the nudge just muddies the conversation.
    assert len(calls[1][0]) == len(calls[0][0])


async def test_truncation_at_ceiling_gives_up(monkeypatch):
    svc, calls = _service(monkeypatch, [(TRUNCATED, "length")])

    with pytest.raises(LLMFatalError, match="truncated"):
        await svc.chat_json_structured(
            [{"role": "user", "content": "extract"}], _Result,
            max_tokens=TRUNCATION_MAX_TOKENS,
        )

    # Already at the ceiling: exactly one call, no doomed retries.
    assert len(calls) == 1


async def test_prose_response_still_gets_nudge(monkeypatch):
    svc, calls = _service(monkeypatch, [
        ("I cannot produce JSON right now.", "stop"),
        (COMPLETE, "stop"),
    ])

    result = await svc.chat_json_structured(
        [{"role": "user", "content": "extract"}], _Result, max_tokens=8192,
    )

    assert result.items == ["a", "b", "c"]
    # Non-truncated garbage keeps the corrective-nudge behavior.
    assert len(calls[1][0]) == len(calls[0][0]) + 1
    assert calls[1][0][-1]["role"] == "user"
    assert "valid JSON" in calls[1][0][-1]["content"]
