"""Capped, jittered HTTP retry for the scholarly-API calls.

httpx has no status-code retry (``AsyncHTTPTransport(retries=)`` is connection-
level only), so we hand-roll one on top of ``tenacity``. This is the *single*
retry layer for prereview's outbound HTTP — do not also enable transport-level
retries on the client, or attempts multiply.

The helper retries transient failures (HTTP 429/5xx, connection/read timeouts)
with full-jitter exponential backoff, honoring a server ``Retry-After`` header
when present (clamped so a hostile value can't hang the CLI). Terminal responses
(404, 400, 401/403, and anything else) are returned to the caller, which decides
whether they mean "not found here" or "infrastructure problem". When every attempt
fails transiently, it raises :class:`TransientExhausted` so the caller can record a
degraded outcome rather than a false "not found".
"""

from __future__ import annotations

import email.utils
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Status codes worth retrying. 429 (rate limit) and 5xx (server/gateway) are
# transient; 409 is deliberately absent — OpenAlex returns it for keyless credit
# exhaustion, which is terminal infrastructure that a retry will not fix.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Never wait longer than this for a single retry, even if the server's Retry-After
# says so — bounds one wait so a garbage/hostile header can't stall the CLI.
RETRY_AFTER_CAP_S = 60.0


@dataclass
class RetryPolicy:
    """Backoff knobs for one logical request. Defaults tuned for an interactive CLI."""

    base: float = 0.75  # exponential base (seconds)
    cap: float = 8.0  # max single backoff wait (seconds), before Retry-After
    max_attempts: int = 3  # total attempts (<=2 retries) per request


DEFAULT_POLICY = RetryPolicy()


class TransientExhausted(Exception):
    """Raised when every attempt for a request failed transiently after retries."""

    def __init__(self, url: str, attempts: int, last_exc: Optional[BaseException] = None):
        self.url = url
        self.attempts = attempts
        self.last_exc = last_exc
        super().__init__(f"transient failure not recovered after {attempts} attempt(s): {url}")


class _TransientStatus(Exception):
    """Internal: a response whose status code is in the retriable set."""

    def __init__(self, response: httpx.Response):
        self.response = response
        super().__init__(f"transient HTTP {response.status_code}")


def classify_status(status_code: int) -> str:
    """``"retry"`` for transient status codes, ``"terminal"`` otherwise."""
    return "retry" if status_code in _RETRY_STATUS else "terminal"


def retry_after_seconds(response: httpx.Response) -> Optional[float]:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds.

    Returns None when the header is absent or unparseable. The result is floored at
    0 (a past date never yields a negative wait) and clamped to
    :data:`RETRY_AFTER_CAP_S` so a hostile value can't produce a huge wait.
    """
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        secs = float(raw)
    else:
        try:
            dt = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = (dt - datetime.now(timezone.utc)).total_seconds()
    if secs < 0:
        secs = 0.0
    return min(secs, RETRY_AFTER_CAP_S)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (_TransientStatus, httpx.TransportError))


def _make_wait(policy: RetryPolicy) -> Callable:
    base_wait = wait_random_exponential(multiplier=policy.base, max=policy.cap)

    def _wait(retry_state) -> float:
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        if isinstance(exc, _TransientStatus):
            ra = retry_after_seconds(exc.response)
            if ra is not None:
                # Honor the server's hint (already clamped) over computed backoff —
                # never retry sooner than it asked.
                return ra
        return base_wait(retry_state)

    return _wait


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: Optional[float] = None,
    policy: RetryPolicy = DEFAULT_POLICY,
    on_retry: Optional[Callable[[], None]] = None,
) -> httpx.Response:
    """GET ``url`` with capped, jittered retry on transient failures.

    Returns the response for any success or terminal status (the caller classifies
    2xx vs 404/4xx). Raises :class:`TransientExhausted` when all attempts failed
    transiently. ``on_retry`` (if given) is invoked once per retry — after a
    transient failure, before the next attempt — so callers can count recoveries.
    """
    before_sleep = (lambda _rs: on_retry()) if on_retry is not None else None
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(policy.max_attempts),
            wait=_make_wait(policy),
            before_sleep=before_sleep,
            reraise=True,
        ):
            with attempt:
                kwargs: dict = {}
                if params is not None:
                    kwargs["params"] = params
                if headers is not None:
                    kwargs["headers"] = headers
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = await client.get(url, **kwargs)
                if classify_status(resp.status_code) == "retry":
                    raise _TransientStatus(resp)
                return resp
    except _TransientStatus as e:
        raise TransientExhausted(url, policy.max_attempts) from e
    except httpx.TransportError as e:
        raise TransientExhausted(url, policy.max_attempts, e) from e
    raise TransientExhausted(url, policy.max_attempts)  # pragma: no cover (loop returns or raises)
