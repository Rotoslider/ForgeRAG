# Notes for future-Claude working in ForgeRAG

> Cross-repo handoff from the Choom project, 2026-04-25. Donny is upgrading the local LLM on this machine from Qwen 3.5 35B-A3B → Qwen 3.6 35B-A3B. Qwen 3.6 changes a few things that ForgeRAG's entity-extraction pipeline depends on. Read this before touching `[llm]` config or the entity extractor.

## What ForgeRAG uses today (Qwen 3.5 path)

- `config/forgerag.toml` `[llm] model = "qwen/qwen3.5-35b-a3b"` (line 57)
- Entity extraction (`backend/ingestion/entity_extractor.py`) appends `/no_think` to every user prompt to suppress chain-of-thought. Without it, Qwen 3.5 deliberates for hundreds of tokens before emitting JSON → ~30 s/page instead of ~8 s.
- `backend/services/llm_service.py` already has a `reasoning_content` fallback (lines 115-122) for when content is empty but the model dumped its output into the reasoning channel.

## What changes for Qwen 3.6

**1. Soft-switch directives no longer work.** Per the Qwen 3.6 model card on HuggingFace:

> "Qwen3.6 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`."

So `/no_think` in `entity_extractor.py:695`, `chunk_summarizer.py:62`, `auto_tagger.py:78`, and `routers/search.py:534` will be silently ignored — the model thinks anyway, and per-page latency will regress back to 30 s+.

**2. The replacement is `chat_template_kwargs.enable_thinking`.** Qwen 3.6 honors this flag via OpenAI's `extra_body`:

```python
client.chat.completions.create(
    model=...,
    messages=...,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

`llm_service.py` uses raw httpx (not the OpenAI SDK), so just add it to the JSON payload directly:

```python
payload: dict[str, Any] = {
    "model": self.settings.model,
    "messages": messages,
    ...,
    "chat_template_kwargs": {"enable_thinking": False},  # NEW for Qwen 3.6
}
```

LM Studio passes this kwarg through to llama.cpp's chat-template renderer, which honors it for Qwen 3.6. Older models that don't recognize the kwarg silently ignore it (verified across DeepSeek, GLM, Llama, Gemma — same pattern as `/no_think` was harmless to non-Qwen models).

**3. Sampling defaults for instruct (non-thinking) mode** per the model card: `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, repetition_penalty=1.0`. ForgeRAG uses `temperature=0.1` for entity extraction (deterministic JSON), which is fine; the rest only matter for chat use cases. Don't change `temperature` — JSON extraction needs it low.

**4. Tool-call format is different but ForgeRAG doesn't use it.** Qwen 3.6 emits tool calls as `<function=NAME><parameter=KEY>VAL</parameter></function>` (Anthropic-style). ForgeRAG uses structured JSON output via `response_format`, not function calling, so this doesn't affect us. Mention only because if anyone adds tool-calling later, the parser in Choom's `app/api/chat/route.ts` (`parseXmlToolCalls`, the `<function=...>` branch) is the reference implementation.

**5. Output may route through `reasoning_content` even with thinking disabled.** Choom observed Qwen 3.6 in LM Studio routing the entire completion (including structured JSON) through `delta.reasoning_content` regardless of the `enable_thinking=False` setting. ForgeRAG already handles this via the fallback at `llm_service.py:115-122` — so this case is already covered. No change needed unless we move to streaming (this code path is non-streaming and reads `message.content || message.reasoning_content`).

## Migration checklist

1. Confirm LM Studio has Qwen 3.6 35B-A3B downloaded under the exact identifier `qwen/qwen3.6-35b-a3b` (note the `qwen/` org prefix — LM Studio reports it that way after a HF-org download). Verify via `curl http://localhost:1234/v1/models | jq '.data[].id'`.
2. Update `config/forgerag.toml` line 57: `model = "qwen/qwen3.6-35b-a3b"`.
3. Add `"chat_template_kwargs": {"enable_thinking": False}` to the payload in `backend/services/llm_service.py` `_chat_raw()` (the method around line 85). This replaces the role of `/no_think`.
4. Optionally remove the `/no_think` trailers from user prompts (`entity_extractor.py:695`, `chunk_summarizer.py:62`, `auto_tagger.py:78`, `routers/search.py:534`). They become no-ops on 3.6 but are harmless leftover noise. Decision is cosmetic.
5. Re-run a single PDF ingestion as smoke test. Expected: per-page entity extraction time should match the prior Qwen 3.5 baseline (~8 s/page). If it's still 30+ s/page, `enable_thinking=False` isn't being honored — check LM Studio version (older builds didn't pass kwargs through to llama.cpp; the build that came with the 2026-04 update onward does), or verify the field isn't being stripped by an HTTP middleware.
6. Re-run an entity-extraction sample on a known PDF and diff the output JSON shape against a 3.5 baseline. Schema should be identical; quality may shift slightly but should not regress on standards-heavy pages.

## Reference materials

- HuggingFace model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Choom commits that worked through these issues (read these for prior art on the same model):
  - `4e20768` — salvage `delta.reasoning_content` as content when `enableThinking=false`
  - `cff3051` — hide reasoning prose from user-facing output (only relevant if streaming)
  - `426590c` — Qwen 3.6 model profile with official non-thinking sampling params
  - `ed3f66d` — `<function=NAME><parameter=KEY>VAL</parameter></function>` parser (only relevant if adding tool calls)

## What NOT to change

- The `reasoning_content` fallback in `llm_service.py:115-122` — it's already correct, handles the case where Qwen 3.6 puts JSON in the reasoning channel.
- `temperature=0.1` for entity extraction. JSON extraction is structured, not creative; low temp is correct regardless of model.
- `use_json_schema = true` in forgerag.toml. Qwen 3.6 supports JSON schema grammar via llama.cpp the same way 3.5 did.

## If something breaks after migration

If entity-extraction per-page latency stays high (>20 s) after step 3:
1. Check LM Studio's request log (Settings → Developer → Server Logs) — confirm the request body includes `chat_template_kwargs` and that the response stream isn't sitting in `<think>...</think>` for half the tokens.
2. Try setting `extra_body` instead at the SDK call site if you migrate to the openai SDK.
3. If LM Studio is too old to pass `chat_template_kwargs` through, upgrade it — the build that shipped Qwen 3.6 support also routes the kwarg correctly.

If output JSON schema drifts (extra fields, missing fields, prose leakage):
1. Verify `response_format: {"type": "json_schema", ...}` is being sent — `chat_json_structured()` should still emit it.
2. Bump `max_tokens` from 4096 → 8192 for entity-extraction prompts that hit standards-heavy pages (the existing per-call override in `entity_extractor.py` already does this; just confirm it's still firing).
