"""OpenReview decision enrichment for cited papers (U8).

When a cited paper is on OpenReview, surface its accept/reject decision and rating
distribution — catching a paper cited as foundational that was actually rejected.
Advisory only ("OpenReview decision: reject — verify it's the right reference"),
never a defect verdict, and disclosed as degradation (not a finding) when the API
is unreachable.

Two infrastructure realities shape this module:

1. ``openreview-py`` is **synchronous** (``requests``-based) while the pipeline is
   ``asyncio`` — every client call is wrapped in :func:`asyncio.to_thread` so it
   never blocks the event loop.
2. The decision/rating queries require **credentials**. No creds → U8 skips
   entirely as a first-class path (NOT degradation). ``openreview-py`` is an
   optional dependency (``pip install prereview[openreview]``); if it is not
   installed, the feature degrades cleanly.

The library is imported lazily inside :func:`_make_client` so the rest of
``prereview`` neither imports nor requires it.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from .models import OpenReviewInfo, VerificationResult


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[openreview] {msg}", file=sys.stderr, flush=True)


def _make_client(username: str, password: str):
    """Construct an OpenReview client (v2, falling back to v1), or None when the
    library is missing or login fails. Runs in a worker thread."""
    try:
        import openreview  # optional dependency; lazy-imported
    except ImportError:
        return None
    try:
        return openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net", username=username, password=password
        )
    except Exception:
        try:
            return openreview.Client(
                baseurl="https://api.openreview.net", username=username, password=password
            )
        except Exception:
            return None


def _avg_rating(values: list[float]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def _find_decision(client, arxiv_id: Optional[str], doi: Optional[str], title: Optional[str]) -> Optional[OpenReviewInfo]:
    """Best-effort: locate the OpenReview note for a cited paper (by title) and pull
    its decision + rating distribution. Defensive — any API shape mismatch returns
    None rather than raising, so one odd reference never derails the run. Runs in a
    worker thread."""
    if not title:
        return None
    try:
        notes = client.search_notes(term=title)
    except Exception:
        try:
            notes = client.get_notes(content={"title": title})
        except Exception:
            return None
    if not notes:
        return None
    note = notes[0]
    note_id = getattr(note, "id", None) or getattr(note, "forum", None)
    if note_id is None:
        return None

    decision: Optional[str] = None
    ratings: list[float] = []
    try:
        replies = client.get_notes(forum=note_id)
    except Exception:
        replies = []
    for r in replies or []:
        content = getattr(r, "content", {}) or {}
        for key in ("decision", "recommendation"):
            val = _content_value(content, key)
            if val and decision is None:
                decision = str(val)
        rating = _content_value(content, "rating")
        if rating is not None:
            num = _leading_number(rating)
            if num is not None:
                ratings.append(num)

    if decision is None and not ratings:
        return None
    return OpenReviewInfo(
        decision=decision,
        rating_avg=_avg_rating(ratings),
        rating_count=len(ratings),
        url=f"https://openreview.net/forum?id={note_id}",
    )


def _content_value(content: dict, key: str):
    """OpenReview v2 wraps fields as ``{"value": ...}``; v1 stores them flat."""
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _leading_number(val) -> Optional[float]:
    import re

    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(val))
    return float(m.group(1)) if m else None


async def enrich_with_openreview(
    verifications: list[VerificationResult],
    *,
    username: Optional[str],
    password: Optional[str],
    verbose: bool = False,
) -> tuple[int, bool]:
    """Annotate ``verifications`` in place with OpenReview decisions where found.

    Returns ``(n_annotated, degraded)``. No creds → ``(0, False)`` (a clean skip,
    not degradation). A login/API failure → ``degraded=True`` so the coverage
    section can disclose it. Every synchronous client call is ``to_thread``-wrapped.
    """
    if not (username and password):
        return 0, False  # first-class no-creds skip

    client = await asyncio.to_thread(_make_client, username, password)
    if client is None:
        _log(verbose, "openreview-py missing or login failed; skipping enrichment")
        return 0, True

    by_ref: dict[str, list[VerificationResult]] = {}
    for v in verifications:
        by_ref.setdefault(v.ref_id, []).append(v)

    annotated = 0
    degraded = False
    for ref_id, sites in by_ref.items():
        ref = sites[0].reference
        if not ref.title:
            continue  # the lookup matches on title — nothing to search by; silent skip
        try:
            info = await asyncio.to_thread(_find_decision, client, ref.arxiv_id, ref.doi, ref.title)
        except Exception as e:
            _log(verbose, f"openreview lookup failed for {ref_id}: {e!r}")
            degraded = True
            continue
        if info is not None:
            for v in sites:
                v.openreview = info
            annotated += 1
    _log(verbose, f"annotated {annotated} citation(s) with OpenReview decisions")
    return annotated, degraded
