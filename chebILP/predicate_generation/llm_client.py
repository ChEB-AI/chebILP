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


async def _run_query(model: str, system: str, prompt: str, schema: type[BaseModel]) -> ResultMessage | None:
    """Drive one CLI query to completion and return its final ``ResultMessage``."""
    _drop_api_key_auth()
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,          # our contract, replacing the CLI's default prompt
        allowed_tools=[],              # no tools available, so there is no agentic loop to bound
        setting_sources=[],            # do not load repo/user CLAUDE.md or settings
        output_format={"type": "json_schema", "schema": schema.model_json_schema()},
        **({"cli_path": _CLI_PATH} if _CLI_PATH else {}),
    )
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    return result


def structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int = 5,
):
    """Ask ``model`` for one structured answer. Returns ``(parsed, raw_json_text, attempts)``.

    ``raw`` is the answer re-serialized as JSON, kept for the exchange log. ``attempts`` is
    one record per query made (each ``{"error", "raw", "cost"}``), in order — the final entry
    is the successful call (``error`` is ``None``); earlier entries are retried failures. On
    total failure the collected attempts are attached to the raised exception as
    ``_chebilp_attempts`` so the caller can still log them.

    A ``provider/name`` model id (``openai/gpt-4o``) routes to an OpenAI-compatible endpoint;
    a bare id (``claude-opus-5``) routes to the local ``claude`` CLI.
    """
    if "/" in model:
        return _openai_structured_completion(model, system, prompt, schema, max_retries=max_retries)
    return _cli_structured_completion(model, system, prompt, schema, max_retries=max_retries)


def _cli_structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int,
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
            result = asyncio.run(_run_query(cli_model, system, prompt, schema))
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

        cost = result.total_cost_usd
        if result.stop_reason == "refusal":
            attempts.append({"error": "refusal", "raw": result.result, "cost": cost})
            exc = ModelRefusal(f"{cli_model} declined this request (safety classifier)")
            exc._chebilp_attempts = attempts
            raise exc

        structured = result.structured_output
        if result.subtype == "success" and structured:
            raw = json.dumps(structured, ensure_ascii=False, indent=2)
            parsed = schema.model_validate(structured)
            attempts.append({"error": None, "raw": raw, "cost": cost})
            return parsed, raw, attempts

        # No structured output despite the SDK's own retries — terminal, don't re-ask.
        raw = json.dumps(structured, ensure_ascii=False) if structured else result.result
        detail = f"subtype={result.subtype}, errors={result.errors}"
        attempts.append({"error": detail, "raw": raw, "cost": cost})
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

    client = openai.OpenAI(base_url=api_base, api_key=api_key)
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
            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
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
