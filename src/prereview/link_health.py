"""Stage 1.5: probe each :class:`LinkCheck` URL extracted by tex_ingest.

The URLs surfaced here are external references the author put in the source —
``\\url{}`` and ``\\href{}`` in the .tex, and ``url = {...}`` fields in the .bib.
We try a HEAD first (cheap), fall back to GET if HEAD is rejected (many servers
return 405 / 403 on HEAD even when GET works), and classify the final response
after redirects.

Failures we surface:
- HTTP 4xx — the URL is gone, moved without redirect, or auth-walled.
- HTTP 5xx — server error; we treat as broken because the user can't verify
  the content right now.
- Connection / DNS / TLS failures — the host doesn't resolve or refused.
- Timeout — the host is too slow to be useful in a review pass.

The check is deliberately quick: a per-URL timeout of a few seconds and a
small concurrency cap. A flaky network shouldn't block the rest of the
pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Iterable

import httpx

from . import __version__
from .models import LinkCheck

log = logging.getLogger("prereview.link_health")


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[link-health] {msg}", file=sys.stderr, flush=True)


_HEAD_FALLBACK_STATUSES = {403, 405, 501}


async def _check_one(
    client: httpx.AsyncClient,
    check: LinkCheck,
    *,
    verbose: bool,
) -> LinkCheck:
    """Probe one URL. Mutates and returns the input :class:`LinkCheck`."""
    try:
        r = await client.head(check.url)
        # Many servers refuse HEAD; retry with GET so we don't false-positive.
        if r.status_code in _HEAD_FALLBACK_STATUSES:
            r = await client.get(check.url)
        check.status = r.status_code
        check.ok = 200 <= r.status_code < 400
        if not check.ok:
            check.error = f"HTTP {r.status_code}"
    except httpx.TimeoutException:
        check.error = "timeout"
    except httpx.ConnectError as e:
        check.error = f"connection error: {e!s}"[:200]
    except httpx.HTTPError as e:
        check.error = f"http error: {e!s}"[:200]
    except Exception as e:
        # Defensive: we never want a single bad URL to break the run.
        check.error = f"unexpected: {e!r}"[:200]

    if verbose:
        if check.ok:
            _log(verbose, f"OK   {check.url} ({check.status})")
        else:
            _log(verbose, f"FAIL {check.url} — {check.error}")
    return check


async def check_links(
    checks: Iterable[LinkCheck],
    *,
    timeout_s: float = 8.0,
    concurrency: int = 8,
    verbose: bool = False,
) -> list[LinkCheck]:
    """Probe every URL in ``checks`` concurrently and return the same list,
    each entry populated with status / ok / error.

    The input list is consumed in-place and returned as a list, so callers
    can pass ``paper.link_checks`` directly without rebinding.
    """
    items = list(checks)
    if not items:
        return items

    headers = {"User-Agent": f"prereview/{__version__} link-health"}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=timeout_s,
        headers=headers,
        follow_redirects=True,
        limits=limits,
    ) as client:

        async def _bounded(item: LinkCheck) -> LinkCheck:
            async with sem:
                return await _check_one(client, item, verbose=verbose)

        return list(await asyncio.gather(*(_bounded(c) for c in items)))
