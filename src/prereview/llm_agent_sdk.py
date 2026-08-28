"""Agent SDK backend — run prereview's LLM calls through the Claude Code CLI.

Why a second backend exists: ``ANTHROPIC_API_KEY`` bills metered API credits, while
a Claude Code OAuth token (``CLAUDE_CODE_OAUTH_TOKEN``) draws on a Claude
subscription. The Agent SDK is the supported way to drive the latter
programmatically. It is a *harness*, not a transport — it spawns the local ``claude``
CLI as a subprocess and talks stream-json to it.

The shape we want here is a plain completion, not an agent. Left at its defaults the
SDK loads the user's ``~/.claude/CLAUDE.md``, project settings, and MCP servers into
every prompt — which, for a citation-verification call, means personal instructions
silently steering verdicts. Every option pinned in ``_build_options`` that looks
redundant is there to strip the harness back to "text in, text out"; see the comments
on each.

Deliberately dropped, because ``ClaudeAgentOptions`` has no equivalent: ``temperature``
and ``max_tokens``. Both are accepted and ignored on this path.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from typing import Any, Optional

_CLI_NAME = "claude"

MISSING_SDK_HINT = (
    "the Claude Agent SDK is not installed. Install the optional extra:\n"
    '    uv pip install -e ".[oauth]"'
)
MISSING_CLI_HINT = (
    f"the `{_CLI_NAME}` CLI was not found on PATH. The Agent SDK drives it as a "
    "subprocess, so Claude Code must be installed."
)


def sdk_importable() -> bool:
    """True if ``claude_agent_sdk`` can be imported."""
    try:
        import claude_agent_sdk  # noqa: F401
    except Exception:
        return False
    return True


def cli_on_path() -> bool:
    """True if the ``claude`` CLI the SDK shells out to is available."""
    return shutil.which(_CLI_NAME) is not None


def agent_sdk_available() -> bool:
    """True only if *both* halves of the OAuth path are present.

    Used by ``--auth auto`` so a token present in the environment (``with-keys``
    injects one on every run) can't select a backend that would fail at the first
    model call.
    """
    return sdk_importable() and cli_on_path()


def unavailable_reason() -> Optional[str]:
    """The actionable half of the failure, or None when the path is usable."""
    if not sdk_importable():
        return MISSING_SDK_HINT
    if not cli_on_path():
        return MISSING_CLI_HINT
    return None


def resolve_model(model: str) -> str:
    """Map a litellm-style model name onto what the ``claude`` CLI expects.

    prereview's defaults carry litellm's provider prefix (``anthropic/claude-...``);
    the CLI wants a bare id or alias. Any other provider is a hard error rather than
    something to hand to the CLI and let it fail obscurely.
    """
    name = (model or "").strip()
    if not name:
        raise ValueError("empty model name")
    if "/" not in name:
        return name
    provider, _, rest = name.partition("/")
    if provider != "anthropic":
        raise ValueError(
            f"--auth oauth runs through Claude Code and only supports Anthropic "
            f"models; got {model!r}. Use --auth api-key for other providers."
        )
    if not rest:
        raise ValueError(f"model name has a provider prefix but no model: {model!r}")
    return rest


def _build_options(
    *,
    model: str,
    system: Optional[str],
    num_retries: int,
    timeout_s: float,
) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        model=resolve_model(model),
        # A plain string replaces the system prompt outright (--system-prompt).
        # A {"type": "preset", "preset": "claude_code"} dict would prepend the whole
        # Claude Code agent prompt instead — never use it here.
        system_prompt=system or "",
        # The base tool set. An empty list emits `--tools ""`, which is what actually
        # leaves the model with no tools; allowed_tools below is only a permission
        # allowlist and would not disable anything on its own.
        tools=[],
        allowed_tools=[],
        # No CLAUDE.md, no settings.json, no MCP servers. This is the load-bearing
        # one: the default (["user", "project"]) would inject the operator's personal
        # instructions into every verification prompt.
        setting_sources=[],
        mcp_servers={},
        # If the model somehow emits a tool call anyway, deny it rather than block
        # forever waiting on an interactive prompt that nothing will answer.
        permission_mode="dontAsk",
        max_turns=1,
        env=_child_env(num_retries=num_retries, timeout_s=timeout_s),
    )


def _child_env(*, num_retries: int, timeout_s: float) -> dict[str, str]:
    """Environment overrides for the spawned CLI.

    ``ClaudeAgentOptions.env`` is merged *over* ``os.environ`` by the SDK, so
    everything in the parent environment is inherited — including
    ``ANTHROPIC_API_KEY``. Left alone, that key can shadow the OAuth token in the
    child and quietly bill the API instead of the subscription, which is exactly the
    failure this backend exists to avoid. The merge API offers no way to *remove* a
    variable, so blank it: the CLI is JavaScript and tests these with plain
    truthiness, where "" reads as unset.
    """
    return {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_MAX_RETRIES": str(num_retries),
        "API_TIMEOUT_MS": str(int(timeout_s * 1000)),
    }


async def agent_sdk_text(
    *,
    model: str,
    user: str,
    system: Optional[str] = None,
    num_retries: int,
    timeout_s: float,
    verbose: bool = False,
) -> str:
    """One prompt in, the assistant's text out.

    Raises on an unusable turn so the caller's boundary (verify/synthesize) can mark
    the run degraded, matching how the litellm path surfaces an exhausted retry.
    """
    reason = unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"--auth oauth is unavailable: {reason}")

    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

    options = _build_options(
        model=model, system=system, num_retries=num_retries, timeout_s=timeout_s
    )

    if verbose:
        print(
            f"[llm] agent-sdk {options.model} call: {len(user)} chars in",
            file=sys.stderr,
            flush=True,
        )

    chunks: list[str] = []
    result_text: Optional[str] = None
    terminal_reason: Optional[str] = None
    is_error = False
    api_error_status: Any = None

    # The SDK yields until the subprocess is done; a wedged CLI would otherwise hang
    # the whole pipeline, so bound the turn the way the litellm path bounds a request.
    async with asyncio.timeout(timeout_s):
        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                terminal_reason = message.terminal_reason
                result_text = message.result
                is_error = bool(message.is_error)
                api_error_status = message.api_error_status

    if is_error:
        raise RuntimeError(
            f"agent-sdk call failed (terminal_reason={terminal_reason!r}, "
            f"api_error_status={api_error_status!r})"
        )

    text = "".join(chunks).strip()
    if not text:
        # ResultMessage.result carries the final text when no assistant text block
        # was streamed; fall back to it before giving up.
        text = (result_text or "").strip()
    if not text:
        raise RuntimeError(
            f"agent-sdk call returned no text (terminal_reason={terminal_reason!r})"
        )

    if verbose:
        print(
            f"[llm] agent-sdk returned {len(text)} chars "
            f"(terminal_reason={terminal_reason!r})",
            file=sys.stderr,
            flush=True,
        )
    return text
