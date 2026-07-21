"""Provider-agnostic LLM access for the auxiliary-generation pipelines.

Both generation pipelines ask a model for one structured answer per class and get back a
Pydantic object. LiteLLM lets any provider serve that request behind a single
``provider/name`` model string (``anthropic/claude-haiku-4-5``, ``openai/gpt-4o``,
``gemini/gemini-2.5-pro``, ``ollama/llama3.1``, ``hosted_vllm/<name>`` with an
``api_base``...). Provider API keys are read from the usual environment variables
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ...).
"""

from __future__ import annotations

import time

import litellm
from pydantic import BaseModel, ValidationError

# Validate structured output against the schema even on providers without native strict
# schema support, so weaker models fail loudly (and get reasked) instead of silently.
litellm.enable_json_schema_validation = True

# Provider-agnostic transient errors worth retrying with backoff.
_TRANSIENT = (
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def structured_completion(
    model: str,
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_tokens: int = 16000,
    max_retries: int = 5,
    api_base: str | None = None,
):
    """Ask ``model`` for one structured answer. Returns ``(parsed, raw_json_text)``.

    ``raw`` is the model's JSON string, kept for the exchange log. Retries transient
    connection/timeout/rate-limit errors with exponential backoff, and reasks when a
    (typically weaker) model returns malformed or schema-invalid JSON.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                # Generous: a truncated response is invalid JSON and loses the whole class.
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format=schema,
                api_base=api_base,
            )
            raw = response.choices[0].message.content
            return schema.model_validate_json(raw), raw
        except _TRANSIENT as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  Transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
        except (ValidationError, ValueError) as e:
            last_exc = e
            print(f"  Invalid structured output (attempt {attempt + 1}/{max_retries}), reasking: {e}")
    raise last_exc
