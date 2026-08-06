"""LLM client for entity extraction and other LLM-driven steps.

Talks to an OpenAI-compatible HTTP endpoint — llama.cpp server (`llama-server`),
vLLM (`python -m vllm.entrypoints.openai.api_server`), LM Studio, or any
equivalent. The endpoint and model are configured via [llm] in forgerag.toml.

Primary method: chat_json_structured(messages, schema) asks the model for
structured JSON output matching a Pydantic schema. Uses either:
- `response_format={"type": "json_object"}` (widely supported)
- `response_format={"type": "json_schema", "json_schema": {...}}` (stricter,
  supported by llama.cpp and vLLM — we prefer this when available).

Errors are classified so the pipeline can retry (transient) vs. give up (schema
mismatch). Timeouts come from config.llm.timeout_seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.config import LLMSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for LLM service errors."""


class LLMTransientError(LLMError):
    """Network hiccup, server 5xx, timeout — worth retrying."""


class LLMFatalError(LLMError):
    """Schema mismatch, invalid response, auth — don't retry."""


class CircuitBreaker:
    """Circuit breaker for the LLM HTTP endpoint.

    States:
    - CLOSED: requests flow normally, failures are counted.
    - OPEN: after `failure_threshold` consecutive failures, all requests
      are rejected immediately without making the HTTP call.
    - HALF-OPEN: after `recovery_timeout` seconds, one probe request is
      allowed through. If it succeeds the circuit closes; if it fails
      the circuit stays open for another timeout period.
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half-open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._consecutive_failures: int = 0
        self._state: str = self.STATE_CLOSED
        self._opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        """True when the circuit is open (rejecting requests)."""
        return self._state == self.STATE_OPEN

    @property
    def state(self) -> str:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through.

        Returns True if the request may proceed, False if the circuit is
        open and the caller should fail fast.
        """
        if self._state == self.STATE_CLOSED:
            return True

        if self._state == self.STATE_OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                # Transition to half-open: allow one probe request
                self._state = self.STATE_HALF_OPEN
                logger.info(
                    "LLM circuit breaker half-open — allowing probe request "
                    "after %.0fs timeout",
                    elapsed,
                )
                return True
            return False

        # HALF-OPEN: only one request is in flight (the probe). Block
        # additional requests until the probe resolves.
        return False

    def record_success(self) -> None:
        """Record a successful response. Resets the circuit to closed."""
        if self._state != self.STATE_CLOSED:
            logger.info(
                "LLM circuit breaker closed — probe succeeded, "
                "resetting after %d consecutive failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._state = self.STATE_CLOSED

    def record_failure(self) -> None:
        """Record a failed response. Opens the circuit after threshold."""
        self._consecutive_failures += 1
        if self._state == self.STATE_HALF_OPEN:
            # Probe failed — reopen the circuit for another timeout period
            self._state = self.STATE_OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "LLM circuit breaker re-opened — probe failed "
                "(%d consecutive failures)",
                self._consecutive_failures,
            )
        elif (
            self._state == self.STATE_CLOSED
            and self._consecutive_failures >= self.failure_threshold
        ):
            self._state = self.STATE_OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "LLM circuit breaker OPEN — %d consecutive failures",
                self._consecutive_failures,
            )


class LLMService:
    """Async OpenAI-compatible client. One instance per app."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None
        self.circuit_breaker = CircuitBreaker()
        # Local LLM servers (LM Studio, llama-server) serve one request at a
        # time. When many jobs run concurrently (bulk fill-missing queues one
        # per doc), unthrottled requests all sit in the SERVER's queue, where
        # time-in-queue counts against timeout_seconds — so requests time out
        # having never started inference. Queue client-side instead: only
        # max_concurrent_requests are in flight, so each request's timeout
        # covers (almost) only its own inference.
        self._request_slots = asyncio.Semaphore(
            max(1, getattr(settings, "max_concurrent_requests", 2))
        )

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.settings.endpoint.rstrip("/"),
            timeout=httpx.Timeout(self.settings.timeout_seconds),
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_configured(self) -> bool:
        return bool(self.settings.endpoint)

    async def health(self) -> bool:
        """Check whether the LLM endpoint is reachable."""
        if self._client is None:
            return False
        try:
            # Most OpenAI-compatible servers expose /v1/models
            r = await self._client.get("/models", timeout=5.0)
            return r.status_code < 500
        except (httpx.RequestError, asyncio.TimeoutError):
            return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Low-level chat completion. Returns the content string of the first choice."""
        if self._client is None:
            raise LLMFatalError("LLMService not started")

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": (
                temperature if temperature is not None else self.settings.temperature
            ),
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # Qwen3.6 dropped the "/no_think" soft directive — thinking is
        # disabled via chat_template_kwargs.enable_thinking, which LM Studio
        # forwards to llama.cpp's chat-template renderer. Other models that
        # don't recognize the kwarg ignore it silently.
        if getattr(self.settings, "disable_thinking", False):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        async with self._request_slots:
            # Circuit breaker: fail fast if LM Studio is down. Checked after
            # acquiring a slot so callers that waited in line still fail fast
            # when the breaker opened while they were waiting.
            if not self.circuit_breaker.allow_request():
                raise LLMTransientError(
                    f"LLM service circuit breaker open — "
                    f"{self.circuit_breaker.consecutive_failures} consecutive failures"
                )

            try:
                r = await self._client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                self.circuit_breaker.record_failure()
                # str() of httpx timeout exceptions is often empty — always
                # include the exception type so the log says WHAT failed.
                raise LLMTransientError(
                    f"Request failed: {type(exc).__name__}: {exc}"
                    if str(exc) else f"Request failed: {type(exc).__name__} "
                    f"(no detail — likely timeout after "
                    f"{self.settings.timeout_seconds}s)"
                ) from exc

        if r.status_code >= 500:
            self.circuit_breaker.record_failure()
            raise LLMTransientError(f"Server {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise LLMFatalError(f"HTTP {r.status_code}: {r.text[:400]}")

        try:
            body = r.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            # Reasoning models (GLM-4.7-Flash, DeepSeek-R1, etc.) sometimes
            # route their entire output — including structured JSON — into
            # reasoning_content and leave content empty. Fall back to that
            # when content is blank so we don't lose the actual response.
            if not content.strip():
                reasoning = message.get("reasoning_content") or ""
                if reasoning.strip():
                    content = reasoning
            self.circuit_breaker.record_success()
            return content
        except (KeyError, ValueError, TypeError) as exc:
            raise LLMFatalError(f"Malformed response: {r.text[:400]}") from exc

    async def chat_json_structured(
        self,
        messages: list[dict[str, Any]],
        schema_cls: Type[T],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
    ) -> T:
        """Chat completion that returns a validated Pydantic model.

        Asks the model for JSON that matches schema_cls. Validates on our side
        and retries on transient errors or invalid JSON (up to `retries` extra
        attempts beyond the first). Fails fast on auth/schema errors.
        """
        schema = schema_cls.model_json_schema()
        # Some models support strict JSON schema grammar (llama.cpp and
        # vLLM do), some don't. Config toggles whether we try it at all.
        # When enabled we start with json_schema and fall back to plain
        # text mode on a 400 / schema mismatch. When disabled we always
        # use plain text and rely on the prompt (works with Gemma 4 MoE
        # which loops in degenerate ways under strict grammar).
        response_format_primary = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        }

        attempts_left = retries + 1
        last_err: Exception | None = None
        use_strict = self.settings.use_json_schema
        primary_failed = False

        while attempts_left > 0:
            attempts_left -= 1
            try:
                rf = (
                    response_format_primary
                    if (use_strict and not primary_failed)
                    else None
                )
                content = await self.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=rf,
                )
            except LLMTransientError as exc:
                last_err = exc
                logger.warning("LLM transient error (attempts left=%d): %s", attempts_left, exc)
                await asyncio.sleep(1.0)
                continue
            except LLMFatalError as exc:
                # json_schema not supported? fall back to json_object once.
                if not primary_failed and "json_schema" in str(exc).lower():
                    primary_failed = True
                    attempts_left += 1  # retry with fallback doesn't cost an attempt
                    continue
                raise

            # Parse + validate — tolerate models that wrap JSON in prose or
            # markdown code fences. We look for the first {...} span if the
            # raw content doesn't parse directly.
            data = None
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_first_json_object(content)
                if extracted is not None:
                    try:
                        data = json.loads(extracted)
                    except json.JSONDecodeError:
                        data = None

            if data is None:
                last_err = ValueError("non-JSON response")
                logger.warning("LLM returned non-JSON (attempts left=%d): %.200s", attempts_left, content)
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. Respond "
                            "with ONLY a single JSON object, no prose."
                        ),
                    }
                ]
                continue

            try:
                return schema_cls.model_validate(data)
            except ValidationError as exc:
                last_err = exc
                # Log both the validation errors and a preview of what the
                # model returned. Without the preview we can't diagnose
                # cases where the model emits structurally-valid JSON with
                # semantically-wrong content (prompt leaking into fields,
                # etc.) — the only clue is in the raw response.
                logger.warning(
                    "LLM response failed schema validation (attempts left=%d): %s\n"
                    "Response preview: %.400s",
                    attempts_left, exc, content,
                )
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your JSON did not match the required schema. "
                            f"Errors: {exc.errors()[:3]}. Return valid JSON only."
                        ),
                    }
                ]
                continue

        # All retries exhausted. Surface enough detail for a future post-mortem.
        # This MUST propagate to the per-page handler in the pipeline so the
        # page is counted as failed and left unstamped for a later retry —
        # never convert it into an empty result.
        raise LLMFatalError(
            f"Structured JSON call failed after retries: {last_err}"
        )


def _extract_first_json_object(text: str) -> str | None:
    """Find the first balanced {...} JSON object in arbitrary text.

    Handles models that wrap JSON in markdown (```json ... ```) or include
    prose before/after. Uses a depth counter; doesn't handle escapes inside
    strings, but json.loads in the caller will reject malformed spans so
    we just try and move on.
    """
    # Strip common code-fence wrapping
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop opening fence (```json or just ```)
        first_newline = stripped.find("\n")
        if first_newline > 0:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start:i + 1]
    return None


def create_llm_service(settings) -> LLMService:
    """Factory — wire into main.py lifespan."""
    return LLMService(settings.llm)
