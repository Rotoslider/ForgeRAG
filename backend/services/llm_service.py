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


class LLMService:
    """Async OpenAI-compatible client. One instance per app."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

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

        try:
            r = await self._client.post("/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise LLMTransientError(f"Request failed: {exc}") from exc

        if r.status_code >= 500:
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
        # Prefer json_schema response_format (llama.cpp, vLLM); fall back to
        # json_object if the server doesn't grok it.
        response_format_primary = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        response_format_fallback = {"type": "json_object"}

        attempts_left = retries + 1
        last_err: Exception | None = None
        primary_failed = False

        while attempts_left > 0:
            attempts_left -= 1
            try:
                rf = response_format_fallback if primary_failed else response_format_primary
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

            # Parse + validate
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                last_err = exc
                logger.warning("LLM returned non-JSON (attempts left=%d): %.200s", attempts_left, content)
                # Next iteration will retry with a reminder message
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
                logger.warning("LLM response failed schema validation (attempts left=%d): %s", attempts_left, exc)
                # Next iteration: ask the model to fix itself
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

        raise LLMFatalError(f"Structured JSON call failed after retries: {last_err}")


def create_llm_service(settings) -> LLMService:
    """Factory — wire into main.py lifespan."""
    return LLMService(settings.llm)
