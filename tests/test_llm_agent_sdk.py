"""Tests for the Agent SDK (OAuth) backend.

``agent_sdk_text`` imports ``claude_agent_sdk`` lazily inside the function, so
patching the module attribute is picked up at call time — same trick
``tests/test_llm.py`` uses for ``litellm.acompletion``.

What is worth pinning here is not the happy path (a real call is exercised in the
live smoke run) but the options: this backend only behaves like a completion because
it strips the Claude Code harness back, and a silent default change there would leak
the operator's CLAUDE.md into every citation verdict without any test going red.
"""

from __future__ import annotations

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402

from prereview import llm, llm_agent_sdk  # noqa: E402


def _result(**kw) -> ResultMessage:
    base = dict(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s1",
    )
    base.update(kw)
    return ResultMessage(**base)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-sonnet-4-6")


def _fake_query(messages, seen: dict | None = None):
    """Build a stand-in for claude_agent_sdk.query that yields ``messages``."""

    async def fake(*, prompt, options):
        if seen is not None:
            seen["prompt"] = prompt
            seen["options"] = options
        for m in messages:
            yield m

    return fake


@pytest.fixture(autouse=True)
def _available(monkeypatch):
    """Neutralize the environment gate; these tests never touch a real CLI."""
    monkeypatch.setattr(llm_agent_sdk, "sdk_importable", lambda: True)
    monkeypatch.setattr(llm_agent_sdk, "cli_on_path", lambda: True)


async def _call(messages, seen=None, **kw):
    monkey = pytest.MonkeyPatch()
    monkey.setattr(claude_agent_sdk, "query", _fake_query(messages, seen))
    try:
        return await llm_agent_sdk.agent_sdk_text(
            model=kw.pop("model", "anthropic/claude-sonnet-4-6"),
            user=kw.pop("user", "q"),
            num_retries=kw.pop("num_retries", 4),
            timeout_s=kw.pop("timeout_s", 30.0),
            **kw,
        )
    finally:
        monkey.undo()


# --------------------------------------------------------------------------- text


async def test_accumulates_assistant_text_blocks():
    out = await _call([_assistant("hel"), _assistant("lo"), _result()])
    assert out == "hello"


async def test_falls_back_to_result_text_when_no_blocks_streamed():
    out = await _call([_result(result='{"verdict": "supports"}')])
    assert out == '{"verdict": "supports"}'


async def test_raises_when_result_is_error():
    """Must raise, not return empty — verify/synthesize key their degraded path off
    an exception, and a silent "" would be parsed as an abstention instead."""
    with pytest.raises(RuntimeError, match="agent-sdk call failed"):
        await _call([_assistant("partial"), _result(is_error=True, subtype="error")])


async def test_raises_when_no_text_at_all():
    with pytest.raises(RuntimeError, match="returned no text"):
        await _call([_result(terminal_reason="max_turns")])


# ------------------------------------------------------------------------- model


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("anthropic/claude-opus-4-7", "claude-opus-4-7"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("sonnet", "sonnet"),
    ],
)
def test_resolve_model_strips_anthropic_prefix(given, expected):
    assert llm_agent_sdk.resolve_model(given) == expected


@pytest.mark.parametrize("given", ["openai/gpt-4", "vertex_ai/claude-opus-4-7"])
def test_resolve_model_rejects_other_providers(given):
    with pytest.raises(ValueError, match="only supports Anthropic"):
        llm_agent_sdk.resolve_model(given)


def test_resolve_model_rejects_empty():
    with pytest.raises(ValueError):
        llm_agent_sdk.resolve_model("")


# ----------------------------------------------------------------------- options


async def test_options_strip_the_claude_code_harness():
    seen: dict = {}
    await _call([_assistant("ok"), _result()], seen, system="SYS")
    opts = seen["options"]

    # The load-bearing one: default ["user", "project"] would inject the operator's
    # ~/.claude/CLAUDE.md and project settings into every prompt.
    assert opts.setting_sources == []
    # tools=[] is what emits `--tools ""`; allowed_tools alone disables nothing.
    assert opts.tools == []
    assert opts.allowed_tools == []
    assert opts.mcp_servers == {}
    assert opts.permission_mode == "dontAsk"
    assert opts.max_turns == 1
    # A plain string replaces the system prompt; a preset dict would prepend the
    # whole Claude Code agent prompt.
    assert opts.system_prompt == "SYS"
    assert opts.model == "claude-sonnet-4-6"
    assert seen["prompt"] == "q"


async def test_child_env_blanks_the_api_key_and_carries_retry_knobs():
    """options.env merges over os.environ, so an inherited ANTHROPIC_API_KEY could
    shadow the OAuth token and silently bill the API instead of the subscription."""
    seen: dict = {}
    await _call([_assistant("ok"), _result()], seen, num_retries=4, timeout_s=120.0)
    env = seen["options"].env

    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_MAX_RETRIES"] == "4"
    assert env["API_TIMEOUT_MS"] == "120000"


# ------------------------------------------------------------------- availability


def test_unavailable_reason_names_the_missing_half(monkeypatch):
    monkeypatch.setattr(llm_agent_sdk, "sdk_importable", lambda: False)
    assert "[oauth]" in (llm_agent_sdk.unavailable_reason() or "")

    monkeypatch.setattr(llm_agent_sdk, "sdk_importable", lambda: True)
    monkeypatch.setattr(llm_agent_sdk, "cli_on_path", lambda: False)
    assert "claude` CLI" in (llm_agent_sdk.unavailable_reason() or "")


async def test_agent_sdk_text_refuses_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm_agent_sdk, "cli_on_path", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        await llm_agent_sdk.agent_sdk_text(
            model="anthropic/claude-sonnet-4-6", user="q", num_retries=1, timeout_s=5.0
        )


# --------------------------------------------------------------------- dispatch


@pytest.fixture
def _restore_backend():
    original = llm.get_backend()
    yield
    llm.set_backend(original)


async def test_acompletion_text_routes_to_agent_sdk(monkeypatch, _restore_backend):
    seen: dict = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return "routed"

    monkeypatch.setattr(llm_agent_sdk, "agent_sdk_text", fake)
    llm.set_backend(llm.BACKEND_AGENT_SDK)

    out = await llm.acompletion_text(
        model="anthropic/claude-sonnet-4-6", user="q", system="s", temperature=0.0
    )

    assert out == "routed"
    assert seen["num_retries"] == llm._LLM_NUM_RETRIES
    assert seen["timeout_s"] == llm._LLM_TIMEOUT_S
    # No ClaudeAgentOptions equivalent; must not be forwarded.
    assert "temperature" not in seen
    assert "max_tokens" not in seen


async def test_acompletion_json_parses_agent_sdk_output(monkeypatch, _restore_backend):
    """The JSON tolerance (fences, prose) and one-shot re-attempt must apply on both
    backends, since acompletion_json sits above the branch."""
    outputs = iter(["not json", '```json\n{"format": "numeric"}\n```'])

    async def fake(**kwargs):
        return next(outputs)

    monkeypatch.setattr(llm_agent_sdk, "agent_sdk_text", fake)
    llm.set_backend(llm.BACKEND_AGENT_SDK)

    assert await llm.acompletion_json(
        model="anthropic/claude-sonnet-4-6", user="q"
    ) == {"format": "numeric"}


def test_set_backend_rejects_unknown(_restore_backend):
    with pytest.raises(ValueError):
        llm.set_backend("nope")
