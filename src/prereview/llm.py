"""Thin LLM helper wrapping litellm.acompletion.

Reasons for a wrapper:
  - Centralize JSON-output parsing so every caller doesn't reinvent it.
  - Easy to mock in tests via monkeypatch.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Optional


# Some Anthropic models (e.g. Opus 4.7 with adaptive thinking) reject the
# ``temperature`` parameter outright. We cache the per-model decision after the
# first 400 so we don't spend a retry on every call.
_NO_TEMPERATURE_MODELS: set[str] = set()


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

    kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
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
    text = await acompletion_text(
        model=model,
        user=user,
        system=sys_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )
    return parse_json_loose(text)


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
