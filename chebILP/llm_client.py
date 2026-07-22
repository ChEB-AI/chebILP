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


def _patch_litellm_native_structured_output() -> None:
    """Route always-on-thinking Anthropic models through native structured output.

    LiteLLM chooses the Anthropic structured-output path from a hardcoded model
    allowlist that predates Fable 5 / Mythos 5, so those fall back to tool-based
    JSON coercion with a forced ``tool_choice``. That is incompatible with their
    always-on thinking (the API can't force a tool while thinking is active), so
    the model returns no tool call and the response parses to an empty ``{}``.
    Reroute them to the native ``output_config.format`` path LiteLLM already uses
    for 4.6/4.7, which is thinking-compatible.
    """
    from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    if getattr(AnthropicConfig, "_chebilp_native_output_patch", False):
        return
    original = AnthropicConfig.map_openai_params

    def patched(self, non_default_params, optional_params, model, drop_params):
        params = original(self, non_default_params, optional_params, model, drop_params)
        response_format = non_default_params.get("response_format")
        if (
            isinstance(response_format, dict)
            and "output_format" not in params
            and any(family in model for family in ("fable", "mythos"))
        ):
            output_format = self.map_response_format_to_anthropic_output_format(response_format)
            if output_format is not None:
                params["output_format"] = output_format
                params["tools"] = [t for t in params.get("tools", []) if t.get("name") != RESPONSE_FORMAT_TOOL_NAME]
                if not params["tools"]:
                    params.pop("tools", None)
                tool_choice = params.get("tool_choice")
                if isinstance(tool_choice, dict) and tool_choice.get("name") == RESPONSE_FORMAT_TOOL_NAME:
                    params.pop("tool_choice", None)
        return params

    AnthropicConfig.map_openai_params = patched
    AnthropicConfig._chebilp_native_output_patch = True


_patch_litellm_native_structured_output()

class ModelRefusal(RuntimeError):
    """The model's safety classifier declined the request (not a malformed reply).

    Reasking is pointless — the classifier is deterministic per input — so this is
    raised past the reask loop and fails the class outright. Seen on Claude Fable /
    Mythos, whose research-biology and cyber classifiers can false-positive on benign
    life-sciences prompts; models without those classifiers (e.g. Opus) are unaffected.
    """


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
    max_tokens: int = 32000,
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
            choice = response.choices[0]
            raw = choice.message.content
            if choice.finish_reason == "content_filter":
                raise ModelRefusal(f"{model} declined this request (safety classifier)")
            if not raw:
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                original = getattr(response, "_hidden_params", {}).get("original_response")
                raise ValueError(
                    f"empty completion content (finish_reason={choice.finish_reason}, "
                    f"reasoning_chars={len(reasoning)}); raw Anthropic blocks: {original!r}"
                )
            return schema.model_validate_json(raw), raw
        except _TRANSIENT as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  Transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
        except (ValidationError, ValueError, litellm.JSONSchemaValidationError) as e:
            last_exc = e
            raw = getattr(e, "raw_response", None)
            detail = f"\n    raw response: {raw!r}" if raw else ""
            print(f"  Invalid structured output (attempt {attempt + 1}/{max_retries}), reasking: {e}{detail}")
    raise last_exc
