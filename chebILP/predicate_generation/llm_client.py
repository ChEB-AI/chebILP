"""LLM access for the auxiliary-generation pipelines.

Each class needs one structured answer (a :class:`~pydantic.BaseModel`).
``structured_completion`` dispatches on the model string to one of two backends:

- **A bare model id** (``claude-opus-5``) runs through the local, already-authenticated
  ``claude`` CLI over the Claude Agent SDK: calls bill to the Claude subscription rather
  than a metered API key. The SDK drives the CLI as a subprocess and returns validated
  JSON matching the schema (``output_format`` / ``structured_output``), re-prompting
  itself on schema mismatch. The CLI must be installed and logged in to a Claude account
  (``claude`` then ``/login``); set ``CHEBILP_CLAUDE_CLI`` to point at the binary if it is
  not on ``PATH``.

- **A ``provider/name`` id** (``openai/gpt-4o``) runs through any OpenAI-compatible HTTP
  endpoint via the ``openai`` SDK, reading ``OPENAI_API_BASE`` and ``OPENAI_API_KEY``.
  This is the path for a self-hosted gateway (e.g. LibreChat). The endpoint need not
  implement strict ``json_schema`` output: the backend asks for a JSON object with the
  schema embedded in the prompt, then validates with Pydantic and reasks on a bad reply.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIConnectionError,
    ProcessError,
    ResultMessage,
    query,
)
from pydantic import BaseModel, ValidationError

# Point the SDK at a specific CLI binary; otherwise it auto-detects ``claude`` on PATH.
_CLI_PATH = os.environ.get("CHEBILP_CLAUDE_CLI")


class ModelRefusal(RuntimeError):
    """The model's safety classifier declined the request (not a malformed reply).

    Surfaced by ``stop_reason == "refusal"`` (CLI) or ``finish_reason == "content_filter"``
    (OpenAI-compatible). Reasking is pointless — the classifier is deterministic per input —
    so this is raised past the retry loop and fails the class.
    """


def _drop_api_key_auth() -> None:
    """Remove API-key auth from the process env so the spawned CLI uses its OAuth login.

    ``generate_auxiliary_*`` calls ``load_dotenv()``, which injects ``ANTHROPIC_API_KEY``
    from ``.env`` into ``os.environ``. The Agent SDK builds the child's environment as
    ``{**os.environ, **options.env}``, so a key present here takes precedence over the
    CLI's logged-in subscription and bills the metered key instead. Passing ``options.env``
    cannot mask it — a merge does not delete an inherited key — so it must be popped here.
    """
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        os.environ.pop(var, None)


async def _run_query(
    model: str, system: str, prompt: str, schema: type[BaseModel], effort: str | None = None
) -> ResultMessage | None:
    """Drive one CLI query to completion and return its final ``ResultMessage``."""
    _drop_api_key_auth()
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,  
        tools=[],
        setting_sources=[],         
        output_format={"type": "json_schema", "schema": schema.model_json_schema()},
        effort=effort,
        # Skips the per-session title request the CLI otherwise sends to a Haiku model.
        env={"CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"},
        **({"cli_path": _CLI_PATH} if _CLI_PATH else {}),
    )
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    return result


# List prices in $/MTok: (input, output, cache read). Cache writes are priced from input.
# The CLI's own ``total_cost_usd`` bills Sonnet 5.5 and Haiku 5.5 at Opus 5.5 rates.
_PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}


def _price_key(model: str) -> str | None:
    """Longest ``_PRICES`` key that prefixes ``model`` (tolerates date suffixes and ``[1m]``)."""
    hits = [k for k in _PRICES if model.startswith(k)]
    return max(hits, key=len) if hits else None


def _usage_summary(model: str, result: ResultMessage) -> tuple[dict | None, float | None]:
    """Tokens of ``model`` in one CLI query, and the list-price cost of every model it used.

    The CLI also makes a small side call on another model (e.g. a Haiku session title); it is
    priced but not counted in the tokens. Cost is ``None`` if any model used is unpriced.
    """
    model_usage = result.model_usage or {}
    split = (result.usage or {}).get("cache_creation") or {}
    w1h, w5m = split.get("ephemeral_1h_input_tokens") or 0, split.get("ephemeral_5m_input_tokens") or 0
    write_mult = (2.0 * w1h + 1.25 * w5m) / (w1h + w5m) if (w1h + w5m) else 1.25
    tokens, cost = None, (0.0 if model_usage else None)
    for name, mu in model_usage.items():
        t_in, t_out = mu.get("inputTokens", 0), mu.get("outputTokens", 0)
        t_w, t_r = mu.get("cacheCreationInputTokens", 0), mu.get("cacheReadInputTokens", 0)
        key = _price_key(mu.get("canonicalModel") or name)
        if key is not None and key == _price_key(model):
            tokens = {"input": t_in, "output": t_out, "thinking": mu.get("thinkingTokens", 0),
                      "cache_write": t_w, "cache_read": t_r}
        if key is None or cost is None:
            cost = None
            continue
        p_in, p_out, p_read = _PRICES[key]
        cost += (t_in * p_in + t_out * p_out + t_w * p_in * write_mult + t_r * p_read) / 1e6
    return tokens, cost


# A wedged ``claude`` CLI subprocess (auth prompt, network stall, ...) otherwise blocks
# ``asyncio.run`` forever with no error and no output, indistinguishable from a slow class.
_CLI_TIMEOUT = float(os.environ.get("CHEBILP_CLAUDE_CLI_TIMEOUT", "300"))


def structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int = 5,
    effort: str | None = None,
):
    """Ask ``model`` for one structured answer. Returns ``(parsed, raw_json_text, attempts)``.

    ``raw`` is the answer re-serialized as JSON, kept for the exchange log. ``attempts`` is
    one record per query made (each ``{"error", "raw", "cost", "tokens"}``), in order — the final entry
    is the successful call (``error`` is ``None``); earlier entries are retried failures. On
    total failure the collected attempts are attached to the raised exception as
    ``_chebilp_attempts`` so the caller can still log them.

    A ``provider/name`` model id (``openai/gpt-4o``) routes to an OpenAI-compatible endpoint;
    a bare id (``claude-opus-5``) routes to the local ``claude`` CLI. ``effort`` applies to
    the CLI only; ``None`` leaves the CLI's per-model default.
    """
    if "/" in model:
        return _openai_structured_completion(model, system, prompt, schema, max_retries=max_retries)
    return _cli_structured_completion(model, system, prompt, schema, max_retries=max_retries, effort=effort)


def _cli_structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int,
    effort: str | None = None,
):
    """Structured answer over the local ``claude`` CLI (Claude Agent SDK).

    The SDK does its own schema-mismatch re-prompting, so a run that finishes without valid
    structured output is terminal and not retried here.
    """
    # The CLI takes a bare model id ("claude-opus-5"); strip any "provider/" prefix.
    cli_model = model.split("/")[-1]
    attempts: list[dict] = []
    last_exc: BaseException | None = None

    for attempt in range(max_retries):
        try:
            started = time.monotonic()
            print(f"  requesting {cli_model} via CLI (attempt {attempt + 1}/{max_retries}, timeout {_CLI_TIMEOUT:.0f}s)...")
            result = asyncio.run(asyncio.wait_for(_run_query(cli_model, system, prompt, schema, effort), timeout=_CLI_TIMEOUT))
            print(f"  response in {time.monotonic() - started:.0f}s")
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"CLI query timed out after {_CLI_TIMEOUT:.0f}s")
            wait = 2 ** attempt
            print(f"  CLI timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s")
            time.sleep(wait)
            continue
        except (CLIConnectionError, ProcessError) as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  CLI error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
            continue

        if result is None:
            last_exc = RuntimeError("Agent SDK query produced no result message")
            wait = 2 ** attempt
            print(f"  No result (attempt {attempt + 1}/{max_retries}), retrying in {wait}s")
            time.sleep(wait)
            continue

        tokens, cost = _usage_summary(cli_model, result)
        if result.stop_reason == "refusal":
            attempts.append({"error": "refusal", "raw": result.result, "cost": cost, "tokens": tokens})
            exc = ModelRefusal(f"{cli_model} declined this request (safety classifier)")
            exc._chebilp_attempts = attempts
            raise exc

        structured = result.structured_output
        if result.subtype == "success" and structured:
            raw = json.dumps(structured, ensure_ascii=False, indent=2)
            parsed = schema.model_validate(structured)
            attempts.append({"error": None, "raw": raw, "cost": cost, "tokens": tokens})
            return parsed, raw, attempts

        # No structured output despite the SDK's own retries — terminal, don't re-ask.
        raw = json.dumps(structured, ensure_ascii=False) if structured else result.result
        detail = f"subtype={result.subtype}, errors={result.errors}"
        attempts.append({"error": detail, "raw": raw, "cost": cost, "tokens": tokens})
        last_exc = RuntimeError(f"no valid structured output ({detail})")
        break

    try:
        last_exc._chebilp_attempts = attempts
    except (AttributeError, TypeError):
        pass  # some exception types forbid attribute assignment
    raise last_exc


def _extract_json(text: str) -> str:
    """Return the outermost ``{...}`` object in ``text``, tolerating prose or code fences."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in reply")
    return text[start : end + 1]


def _openai_structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int,
):
    """Structured answer over an OpenAI-compatible endpoint (``OPENAI_API_BASE``/``OPENAI_API_KEY``).

    The schema is embedded in the prompt and the reply parsed/validated here, so the endpoint
    only needs plain chat completions — strict ``json_schema`` support is not required. A
    malformed or schema-invalid reply is fed back and reasked; transient HTTP errors are
    retried with backoff.
    """
    import openai

    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        raise RuntimeError("OPENAI_API_BASE is not set (needed for the 'provider/name' backend)")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for the 'provider/name' backend)")

    # A hung/slow endpoint must surface as a retryable APITimeoutError, not an invisible
    # block: the SDK default is 600s per request, long enough to look like a wedged run.
    request_timeout = float(os.environ.get("OPENAI_TIMEOUT", "300"))
    client = openai.OpenAI(base_url=api_base, api_key=api_key, timeout=request_timeout)
    api_model = model.split("/", 1)[1]  # drop the "provider/" prefix; keep the rest verbatim

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system_full = (
        f"{system}\n\nRespond with a single JSON object conforming to this JSON Schema:\n"
        f"{schema_json}\nOutput only the JSON object — no prose, no code fences."
    )
    messages = [
        {"role": "system", "content": system_full},
        {"role": "user", "content": prompt},
    ]

    transient = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
    attempts: list[dict] = []
    last_exc: BaseException | None = None

    for attempt in range(max_retries):
        try:
            started = time.monotonic()
            print(f"  requesting {api_model} (attempt {attempt + 1}/{max_retries}, timeout {request_timeout:.0f}s)...")
            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            elapsed = time.monotonic() - started
            finish = response.choices[0].finish_reason
            chars = len(response.choices[0].message.content or "")
            print(f"  response in {elapsed:.0f}s (finish_reason={finish}, {chars} chars)")
        except transient as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
            continue

        choice = response.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "content_filter":
            attempts.append({"error": "refusal", "raw": content, "cost": None})
            exc = ModelRefusal(f"{api_model} declined this request (content filter)")
            exc._chebilp_attempts = attempts
            raise exc

        try:
            parsed = schema.model_validate_json(_extract_json(content))
            raw = json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
            attempts.append({"error": None, "raw": raw, "cost": None})
            return parsed, raw, attempts
        except (ValueError, ValidationError) as e:
            attempts.append({"error": str(e), "raw": content, "cost": None})
            last_exc = RuntimeError(f"invalid structured output: {e}")
            # Reask: show the model its reply and the validation error.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That reply was not valid: {e}. Return only the JSON object required by the schema.",
            })

    try:
        last_exc._chebilp_attempts = attempts
    except (AttributeError, TypeError):
        pass
    raise last_exc
