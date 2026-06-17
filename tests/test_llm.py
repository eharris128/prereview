"""Tests for prereview.llm — retry kwargs, error propagation, and JSON re-attempt.

We monkeypatch ``litellm.acompletion`` (the wrapper imports it lazily, so patching
the module attribute is picked up at call time). The kwarg-passthrough test pins
the contract prereview owns — that it opts into retry; litellm's own internal
retry behavior is exercised in integration, not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

from prereview import llm


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_acompletion_text_passes_retry_kwargs(monkeypatch):
    seen: dict = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return _resp("hi")

    monkeypatch.setattr(litellm, "acompletion", fake)
    out = await llm.acompletion_text(model="anthropic/claude-sonnet-4-6", user="q")
    assert out == "hi"
    assert seen["num_retries"] == llm._LLM_NUM_RETRIES
    assert seen["timeout"] == llm._LLM_TIMEOUT_S


async def test_acompletion_text_reraises_exhausted_error(monkeypatch):
    """When litellm raises after its own retries are exhausted, the wrapper must
    surface it so the caller's boundary (verify/synthesize) can mark degraded."""

    class Boom(Exception):
        pass

    async def fake(**kwargs):
        raise Boom("rate limited; retries exhausted")

    monkeypatch.setattr(litellm, "acompletion", fake)
    with pytest.raises(Boom):
        await llm.acompletion_text(model="anthropic/claude-sonnet-4-6", user="q")


async def test_temperature_rejection_retry_keeps_retry_kwargs(monkeypatch):
    """The existing temperature-rejection retry must still fire and must carry the
    new retry kwargs on the second call."""
    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise Exception("temperature is not supported by this model")
        return _resp("ok")

    monkeypatch.setattr(litellm, "acompletion", fake)
    # A throwaway model name (not the hardcoded no-temperature one) so the first call
    # includes temperature and the global no-temp cache mutation stays isolated.
    out = await llm.acompletion_text(model="anthropic/temp-sensitive-test", user="q", temperature=0.0)
    assert out == "ok"
    assert len(calls) == 2
    assert "temperature" not in calls[1]
    assert calls[1]["num_retries"] == llm._LLM_NUM_RETRIES


async def test_acompletion_json_reattempts_on_bad_json(monkeypatch):
    outputs = iter(["not valid json", '{"verdict": "supports"}'])

    async def fake(**kwargs):
        return _resp(next(outputs))

    monkeypatch.setattr(litellm, "acompletion", fake)
    result = await llm.acompletion_json(model="anthropic/claude-sonnet-4-6", user="q")
    assert result == {"verdict": "supports"}


async def test_acompletion_json_raises_on_persistent_bad_json(monkeypatch):
    async def fake(**kwargs):
        return _resp("still not json")

    monkeypatch.setattr(litellm, "acompletion", fake)
    with pytest.raises(ValueError):
        await llm.acompletion_json(model="anthropic/claude-sonnet-4-6", user="q")
