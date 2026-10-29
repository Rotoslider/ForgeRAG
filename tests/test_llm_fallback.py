"""Interactive answers fall back to [llm] fallback_endpoint.

Live incident 2026-10-29: the NUC's LM Studio had no model loaded and
answered every request with HTTP 400 "No models loaded. Please load a model
in the developer page or use the 'lms load' command." Every Choom
ask_engineering_question failed at the synthesis step, while the Mac's LM
Studio had the same model loaded.

These tests assert:
- an allow_fallback call goes to the fallback server, with fallback_model,
  when the main one has no model loaded or refuses the connection,
- callers without allow_fallback (ingestion) never touch the fallback,
- genuine errors and busy-server timeouts don't fall back,
- a failing fallback reports both errors.
"""

import json

import httpx
import pytest

from backend.config import LLMSettings
from backend.services.llm_service import (
    LLMFatalError,
    LLMModelUnavailableError,
    LLMService,
    LLMTransientError,
    LLMUnreachableError,
)

pytestmark = pytest.mark.asyncio

NO_MODELS_BODY = json.dumps({"error": {
    "message": "No models loaded. Please load a model in the developer "
               "page or use the 'lms load' command.",
    "type": "invalid_request_error", "param": "model", "code": None,
}})
GOOD_BODY = json.dumps(
    {"choices": [{"message": {"content": "from the Mac"},
                  "finish_reason": "stop"}]}
)
MESSAGES = [{"role": "user", "content": "x"}]


class _Resp:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


class _Client:
    """Fake httpx client: returns one canned response or raises."""

    def __init__(self, response):
        self._response = response
        self.payloads = []

    async def post(self, path, json=None):
        self.payloads.append(json)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _service(main, fallback, fallback_model="mac-model"):
    svc = LLMService(LLMSettings(
        model="nuc-model",
        fallback_endpoint="http://mac:1234/v1",
        fallback_model=fallback_model,
    ))
    svc._client = _Client(main)
    svc._fallback_client = _Client(fallback)
    svc.circuit_breaker.failure_threshold = 100
    return svc


async def test_no_models_loaded_falls_back_with_fallback_model():
    svc = _service(_Resp(400, NO_MODELS_BODY), _Resp(200, GOOD_BODY))
    text = await svc.chat(MESSAGES, allow_fallback=True)
    assert text == "from the Mac"
    assert svc._client.payloads[0]["model"] == "nuc-model"
    assert svc._fallback_client.payloads[0]["model"] == "mac-model"
    assert svc._fallback_client.payloads[0]["messages"] == MESSAGES


async def test_empty_fallback_model_keeps_main_model():
    svc = _service(_Resp(400, NO_MODELS_BODY), _Resp(200, GOOD_BODY),
                   fallback_model="")
    await svc.chat(MESSAGES, allow_fallback=True)
    assert svc._fallback_client.payloads[0]["model"] == "nuc-model"


async def test_connection_refused_falls_back():
    svc = _service(httpx.ConnectError("refused"), _Resp(200, GOOD_BODY))
    assert await svc.chat(MESSAGES, allow_fallback=True) == "from the Mac"


async def test_open_breaker_falls_back():
    svc = _service(_Resp(200, GOOD_BODY), _Resp(200, GOOD_BODY))
    svc._client = _Client(httpx.ConnectError("refused"))
    svc.circuit_breaker.failure_threshold = 1
    with pytest.raises(LLMUnreachableError):
        await svc.chat(MESSAGES)
    assert await svc.chat(MESSAGES, allow_fallback=True) == "from the Mac"
    assert len(svc._client.payloads) == 1  # breaker kept the main one shut


async def test_ingestion_never_falls_back():
    svc = _service(_Resp(400, NO_MODELS_BODY), _Resp(200, GOOD_BODY))
    with pytest.raises(LLMModelUnavailableError):
        await svc.chat_with_finish_reason(MESSAGES)
    assert svc._fallback_client.payloads == []


async def test_no_models_loaded_is_still_fatal_without_fallback():
    svc = _service(_Resp(400, NO_MODELS_BODY), _Resp(200, GOOD_BODY))
    svc._fallback_client = None
    with pytest.raises(LLMFatalError):
        await svc.chat(MESSAGES, allow_fallback=True)


async def test_genuine_400_does_not_fall_back():
    svc = _service(_Resp(400, '{"error":"messages field is required"}'),
                   _Resp(200, GOOD_BODY))
    with pytest.raises(LLMFatalError):
        await svc.chat(MESSAGES, allow_fallback=True)
    assert svc._fallback_client.payloads == []


async def test_busy_timeout_does_not_fall_back():
    svc = _service(httpx.ReadTimeout("slow"), _Resp(200, GOOD_BODY))
    with pytest.raises(LLMTransientError) as info:
        await svc.chat(MESSAGES, allow_fallback=True)
    assert not isinstance(info.value, LLMUnreachableError)
    assert svc._fallback_client.payloads == []


async def test_failing_fallback_reports_both():
    svc = _service(_Resp(400, NO_MODELS_BODY),
                   httpx.ConnectError("mac asleep"))
    with pytest.raises(LLMTransientError) as info:
        await svc.chat(MESSAGES, allow_fallback=True)
    assert "mac asleep" in str(info.value)
    assert "No models loaded" in str(info.value)
