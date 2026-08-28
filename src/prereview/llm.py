"""Thin LLM helper over one of two credential paths.

Reasons for a wrapper:
  - Centralize JSON-output parsing so every caller doesn't reinvent it.
  - Easy to mock in tests via monkeypatch.
  - One place to choose the backend, so the four ``acompletion_json`` call sites
    (ingest, verify, synthesize x2) never care which credential is in play.

Backends: ``litellm`` (default) calls the Anthropic API with ``ANTHROPIC_API_KEY``;
``agent-sdk`` drives the Claude Code CLI with a ``CLAUDE_CODE_OAUTH_TOKEN`` so spend
lands on a Claude subscription instead. See ``llm_agent_sdk``.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Optional


# Which credential path this run uses. Set once by the CLI before the pipeline
# starts; module-level because every call site shares the one process-wide choice.
BACKEND_LITELLM = "litellm"
BACKEND_AGENT_SDK = "agent-sdk"
_BACKEND = BACKEND_LITELLM


def set_backend(name: str) -> None:
    global _BACKEND
    if name not in (BACKEND_LITELLM, BACKEND_AGENT_SDK):
        raise ValueError(f"unknown LLM backend: {name!r}")
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


# Some Anthropic models (e.g. Opus 4.7 with adaptive thinking) reject the
# ``temperature`` parameter outright. We cache the per-model decision after the
# first 400 so we don't spend a retry on every call.
_NO_TEMPERATURE_MODELS: set[str] = set()

# Retry knobs for prereview's own direct LLM calls. litellm defaults to NO retries,
# so a transient 429/5xx/timeout fails immediately without this. litellm performs
# library-managed retry-with-backoff and reraises the original exception on
# exhaustion; callers (verify/synthesize) map that to a degraded outcome rather than
# crashing or faking an abstention. Tunable in one place.
_LLM_NUM_RETRIES = 4
_LLM_TIMEOUT_S = 120.0  # litellm's own default is 600s — too long for an interactive CLI
# A malformed/partial JSON response often succeeds on a fresh call; re-attempt once
# before surfacing the parse failure to the caller's boundary.
_JSON_REATTEMPT = True


def _model_takes_temperature(model: str) -> bool:
    if model in _NO_TEMPERATURE_MODELS:
        return False
    # claude-opus-4-7 (and any future model with adaptive thinking) ships
    # without a temperature knob. Hard-code the known case.
    if "claude-opus-4-7" in model:
        return False
    return True


async def acompletion_text(
    *,
    model: str,
    user: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    verbose: bool = False,
) -> str:
    if _BACKEND == BACKEND_AGENT_SDK:
        # Lazy: claude-agent-sdk is an optional extra, so importing it at module
        # scope would break the default install. temperature/max_tokens have no
        # ClaudeAgentOptions equivalent and are dropped on this path.
        from .llm_agent_sdk import agent_sdk_text

        return await agent_sdk_text(
            model=model,
            user=user,
            system=system,
            num_retries=_LLM_NUM_RETRIES,
            timeout_s=_LLM_TIMEOUT_S,
            verbose=verbose,
        )

    from litellm import acompletion  # imported lazily so tests can monkeypatch

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if verbose:
        print(
            f"[llm] {model} call: {len(user)} chars in",
            file=sys.stderr,
            flush=True,
        )

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "num_retries": _LLM_NUM_RETRIES,
        "timeout": _LLM_TIMEOUT_S,
    }
    if _model_takes_temperature(model):
        kwargs["temperature"] = temperature

    try:
        resp = await acompletion(**kwargs)
    except Exception as e:
        # Retry without temperature if the API tells us it's deprecated.
        msg = str(e).lower()
        if "temperature" in kwargs and "temperature" in msg and ("deprecated" in msg or "not supported" in msg):
            _NO_TEMPERATURE_MODELS.add(model)
            kwargs.pop("temperature", None)
            if verbose:
                print(
                    f"[llm] {model} rejects temperature; retrying without it",
                    file=sys.stderr,
                    flush=True,
                )
            resp = await acompletion(**kwargs)
        else:
            raise

    return resp.choices[0].message.content or ""


async def acompletion_json(
    *,
    model: str,
    user: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    verbose: bool = False,
) -> Any:
    """Call the LLM and parse the response as JSON.

    Tolerant: pulls the first balanced JSON object/array out of fenced blocks
    or surrounding prose. Raises ValueError on parse failure.
    """
    sys_prompt = system or "Respond with a single valid JSON value. No prose, no fences."
    attempts = 2 if _JSON_REATTEMPT else 1
    last_exc: Optional[ValueError] = None
    for i in range(attempts):
        text = await acompletion_text(
            model=model,
            user=user,
            system=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        try:
            return parse_json_loose(text)
        except ValueError as e:
            last_exc = e
            if verbose:
                print(
                    f"[llm] JSON parse failed (attempt {i + 1}/{attempts})",
                    file=sys.stderr,
                    flush=True,
                )
    raise last_exc  # type: ignore[misc]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_loose(text: str) -> Any:
    """Best-effort JSON extractor. Tries:
       1. Whole text as JSON.
       2. The contents of the first ```...``` fence.
       3. The first balanced { ... } or [ ... ] in the text.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        body = m.group(1).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"could not parse JSON from LLM response: {text[:200]!r}")
