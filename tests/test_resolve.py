"""Tests for prereview.resolve.

We mock the four external APIs with respx and assert priority ordering and
cache behavior. No real network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from prereview.cache import Cache
from prereview.http_retry import RetryPolicy
from prereview.models import Reference, Resolution, ResolutionStatus
from prereview.resolve import (
    Resolver,
    _decode_openalex_abstract,
    _title_matches,
    normalize_arxiv_id,
    normalize_doi,
)


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache")


@pytest.fixture
def fast_resolver(cache: Cache):
    """Return an async-context-managed Resolver with the rate limiters tightened to
    near-zero so tests run fast."""

    class _Fast(Resolver):
        def __init__(self, **kw):
            super().__init__(**kw)
            for g in (self._g_crossref, self._g_s2, self._g_arxiv, self._g_openalex):
                g.min_interval = 0.0
            # Zero the backoff so transient-retry tests don't actually sleep.
            self._policy = RetryPolicy(base=0.0, cap=0.0, max_attempts=3)

    return lambda **kw: _Fast(cache=cache, **kw)


CROSSREF_DOI_HIT = {
    "status": "ok",
    "message": {
        "DOI": "10.1234/example",
        "title": ["A Toy Paper About Toys"],
        "author": [
            {"given": "Alice", "family": "Smith"},
            {"given": "Bob", "family": "Jones"},
        ],
        "issued": {"date-parts": [[2023]]},
        "container-title": ["Journal of Toys"],
        "URL": "https://doi.org/10.1234/example",
        "abstract": "<jats:p>Toys are nice.</jats:p>",
    },
}

S2_DOI_HIT = {
    "title": "A Toy Paper About Toys",
    "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
    "year": 2023,
    "abstract": "Toys are nice.",
    "externalIds": {"DOI": "10.1234/example"},
    "openAccessPdf": {"url": "https://example.org/toy.pdf"},
    "venue": "Journal of Toys",
}

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>A Toy Paper About Toys</title>
    <summary>We discuss toys.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1"/>
  </entry>
</feed>
"""

OPENALEX_DOI_HIT = {
    "id": "https://openalex.org/W123",
    "title": "A Toy Paper About Toys",
    "authorships": [
        {"author": {"display_name": "Alice Smith"}},
        {"author": {"display_name": "Bob Jones"}},
    ],
    "publication_year": 2023,
    "doi": "https://doi.org/10.1234/example",
    "abstract_inverted_index": {"Toys": [0], "are": [1], "nice.": [2]},
    "primary_location": {"source": {"display_name": "Journal of Toys"}},
    "open_access": {"oa_url": "https://example.org/openalex.pdf"},
}


# ---------------------------------------------------------------------------
# helpers


def test_normalize_doi_strips_url_prefix():
    assert normalize_doi("https://doi.org/10.1234/Foo") == "10.1234/foo"
    assert normalize_doi("DOI: 10.1234/foo.") == "10.1234/foo"
    assert normalize_doi(None) is None


def test_normalize_arxiv_id_handles_url_and_v_suffix():
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"
    assert normalize_arxiv_id("arXiv:2401.12345v3") == "2401.12345"
    assert normalize_arxiv_id("hep-ex/0307015") == "hep-ex/0307015"
    assert normalize_arxiv_id(None) is None


def test_title_matches_jaccard():
    assert _title_matches("Attention Is All You Need", ["Attention is all you need"])
    assert _title_matches(
        "On the convergence of Adam and beyond",
        ["On the convergence of Adam and beyond: a study"],
    )
    assert not _title_matches("Apples", ["A History of Banana Republics"])


def test_decode_openalex_abstract():
    inv = {"Hello": [0], "world": [1]}
    assert _decode_openalex_abstract(inv) == "Hello world"
    assert _decode_openalex_abstract(None) is None
    assert _decode_openalex_abstract({}) is None


# ---------------------------------------------------------------------------
# resolution priority


@pytest.mark.asyncio
async def test_resolve_prefers_crossref_when_doi_present(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="Smith et al. 2023.",
        authors=["Alice Smith"],
        title="A Toy Paper About Toys",
        year=2023,
        doi="10.1234/example",
    )
    with respx.mock(base_url="https://api.crossref.org") as cr:
        cr.get("/works/10.1234%2Fexample").respond(json=CROSSREF_DOI_HIT)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "crossref"
    assert rec.title == "A Toy Paper About Toys"
    assert rec.year == 2023
    assert rec.doi == "10.1234/example"
    assert rec.abstract == "Toys are nice."  # JATS tags stripped


@pytest.mark.asyncio
async def test_resolve_falls_through_to_s2_when_crossref_404s(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        doi="10.1234/example",
    )
    with respx.mock() as mock:
        mock.get("https://api.crossref.org/works/10.1234%2Fexample").respond(404)
        # Crossref search also misses (no items)
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": []}}
        )
        mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1234/example"
        ).respond(json=S2_DOI_HIT)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "semanticscholar"
    assert rec.open_access_pdf_url == "https://example.org/toy.pdf"


@pytest.mark.asyncio
async def test_resolve_falls_through_to_arxiv_when_no_doi(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        arxiv_id="2401.12345",
    )
    with respx.mock() as mock:
        # Crossref search returns nothing useful.
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": []}}
        )
        # S2 by ARXIV: 404.
        mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345"
        ).respond(404)
        mock.get(host="api.semanticscholar.org", path="/graph/v1/paper/search").respond(
            json={"data": []}
        )
        mock.get(host="export.arxiv.org", path="/api/query").respond(text=ARXIV_FEED)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "arxiv"
    assert rec.title == "A Toy Paper About Toys"
    assert rec.year == 2024
    assert rec.open_access_pdf_url == "http://arxiv.org/pdf/2401.12345v1"


@pytest.mark.asyncio
async def test_resolve_openalex_last(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        doi="10.1234/example",
    )
    with respx.mock() as mock:
        mock.get(host="api.crossref.org").respond(404)
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(
            "https://api.openalex.org/works/doi:10.1234%2Fexample"
        ).respond(json=OPENALEX_DOI_HIT)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "openalex"
    assert rec.abstract == "Toys are nice."
    assert rec.doi == "10.1234/example"


@pytest.mark.asyncio
async def test_resolve_returns_none_when_all_miss(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="A nonexistent paper",
        title="A Nonexistent Paper",
    )
    with respx.mock() as mock:
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is None


@pytest.mark.asyncio
async def test_resolve_uses_cache_on_second_call(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        doi="10.1234/example",
    )
    with respx.mock() as mock:
        cr_route = mock.get(
            "https://api.crossref.org/works/10.1234%2Fexample"
        ).respond(json=CROSSREF_DOI_HIT)
        async with fast_resolver() as r:
            rec1 = (await r.resolve(ref)).record
            rec2 = (await r.resolve(ref)).record
    assert rec1 is not None and rec2 is not None
    assert rec1.title == rec2.title
    assert cr_route.call_count == 1  # second call hit the cache


@pytest.mark.asyncio
async def test_resolve_picks_up_openalex_retraction_flag(fast_resolver):
    """When OpenAlex itself is the resolver hit, the canonical record's
    is_retracted flag should be populated directly from OpenAlex's payload."""
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        doi="10.1234/example",
    )
    retracted = json.loads(json.dumps(OPENALEX_DOI_HIT))
    retracted["is_retracted"] = True
    with respx.mock() as mock:
        mock.get(host="api.crossref.org").respond(404)
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(
            "https://api.openalex.org/works/doi:10.1234%2Fexample"
        ).respond(json=retracted)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "openalex"
    assert rec.is_retracted is True


@pytest.mark.asyncio
async def test_resolve_cross_checks_openalex_for_retraction_when_crossref_hits(fast_resolver):
    """When a non-OpenAlex source resolves the reference, we still check
    OpenAlex for the retraction flag — it's the only source that mirrors
    Retraction Watch reliably."""
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        authors=["Alice Smith"],
        title="A Toy Paper About Toys",
        doi="10.1234/example",
    )
    retracted = json.loads(json.dumps(OPENALEX_DOI_HIT))
    retracted["is_retracted"] = True
    with respx.mock() as mock:
        mock.get(
            "https://api.crossref.org/works/10.1234%2Fexample"
        ).respond(json=CROSSREF_DOI_HIT)
        oa_route = mock.get(
            "https://api.openalex.org/works/doi:10.1234%2Fexample"
        ).respond(json=retracted)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "crossref"  # Crossref still wins as the primary source
    assert rec.is_retracted is True  # but retraction came from the OpenAlex follow-up
    assert oa_route.call_count == 1


@pytest.mark.asyncio
async def test_resolve_retraction_check_failure_does_not_break_resolve(fast_resolver):
    """If the OpenAlex retraction follow-up errors, we keep the primary record
    (better to miss a retraction than to refuse to resolve a paper)."""
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        authors=["Alice Smith"],
        title="A Toy Paper About Toys",
        doi="10.1234/example",
    )
    with respx.mock() as mock:
        mock.get(
            "https://api.crossref.org/works/10.1234%2Fexample"
        ).respond(json=CROSSREF_DOI_HIT)
        mock.get(
            "https://api.openalex.org/works/doi:10.1234%2Fexample"
        ).respond(500)
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "crossref"
    assert rec.is_retracted is False


@pytest.mark.asyncio
async def test_resolve_skips_retraction_check_when_no_doi(fast_resolver):
    """No DOI means we have no way to look up retraction in OpenAlex; the
    follow-up call should be skipped rather than triggering a search."""
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        arxiv_id="2401.12345",
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="api.semanticscholar.org", path="/graph/v1/paper/search").respond(
            json={"data": []}
        )
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        # Any OpenAlex hit should not happen — the arXiv record has no DOI.
        oa_route = mock.get(host="api.openalex.org")
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is not None
    assert rec.source == "arxiv"
    assert rec.is_retracted is False
    assert oa_route.call_count == 0


@pytest.mark.asyncio
async def test_resolve_rejects_title_mismatch(fast_resolver):
    """Crossref returns a paper whose title doesn't match — should reject and fall through."""
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
    )
    other = json.loads(json.dumps(CROSSREF_DOI_HIT))
    other["message"]["title"] = ["Quantum Theory of the Entirely Different Universe"]
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [other["message"]]}}
        )
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            rec = (await r.resolve(ref)).record
    assert rec is None


# ---------------------------------------------------------------------------
# three-state outcomes (recover-then-disclose): recover / degrade-not-ghost / breaker


@pytest.mark.asyncio
async def test_resolve_recovers_after_transient_then_hit(fast_resolver):
    """A 429 that recovers on retry resolves normally and counts as a recovery —
    it must not leak through as a ghost."""
    ref = Reference(
        ref_id="r1", raw_text="x", title="A Toy Paper About Toys", doi="10.1234/example"
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.crossref.org/works/10.1234%2Fexample").mock(
            side_effect=[httpx.Response(429), httpx.Response(200, json=CROSSREF_DOI_HIT)]
        )
        mock.get(host="api.openalex.org").respond(json={})  # absorbs the retraction follow-up
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            recovered = r.recovered_after_retry
    assert res.status == ResolutionStatus.RESOLVED
    assert res.record is not None and res.record.source == "crossref"
    assert recovered == 1


@pytest.mark.asyncio
async def test_resolve_degraded_not_ghost_when_a_source_transient_fails(fast_resolver):
    """Headline R1/R2 guarantee: a transient failure on one source (with a real 404 on
    another) yields DEGRADED, never UNRESOLVED — a blip must not become a false ghost."""
    ref = Reference(ref_id="r1", raw_text="x", title="A Real Paper")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})  # authoritative miss
        mock.get(host="api.semanticscholar.org").respond(503)  # down — transient
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.DEGRADED
    assert res.record is None


@pytest.mark.asyncio
async def test_resolve_all_terminal_is_unresolved(fast_resolver):
    """When every source gives an authoritative miss, it's a true ghost (UNRESOLVED)."""
    ref = Reference(ref_id="r1", raw_text="x", title="A Nonexistent Paper")
    with respx.mock() as mock:
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.UNRESOLVED
    assert res.record is None


@pytest.mark.asyncio
async def test_resolve_200_unparseable_is_transient_not_ghost(fast_resolver):
    """A 200 with an unparseable body is a transient symptom, so the reference is
    DEGRADED (couldn't verify), never a ghost manufactured from a garbage 200."""
    ref = Reference(ref_id="r1", raw_text="x", title="A Real Paper")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.crossref.org").respond(
            content=b"<html>gateway error</html>", headers={"content-type": "text/html"}
        )
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.DEGRADED


@pytest.mark.asyncio
async def test_resolve_does_not_cache_degraded(fast_resolver):
    """A DEGRADED outcome must not be cached — otherwise a blip freezes into a permanent
    ghost on re-runs. The failing source is retried on the second call."""
    ref = Reference(ref_id="r1", raw_text="x", title="A Real Paper")
    with respx.mock(assert_all_called=False) as mock:
        cr = mock.get(host="api.crossref.org").respond(503)
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res1 = await r.resolve(ref)
            calls_after_first = cr.call_count
            res2 = await r.resolve(ref)
    assert res1.status == ResolutionStatus.DEGRADED
    assert res2.status == ResolutionStatus.DEGRADED
    assert cr.call_count > calls_after_first  # not served from cache


@pytest.mark.asyncio
async def test_resolve_circuit_breaker_trips_after_consecutive_failures(fast_resolver, monkeypatch):
    """After enough consecutive transient failures, a source is skipped for the rest of
    the run instead of being hammered — bounding retry amplification."""
    import prereview.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "_BREAKER_CONSECUTIVE", 2)
    with respx.mock(assert_all_called=False) as mock:
        s2 = mock.get(host="api.semanticscholar.org").respond(503)
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            for i in range(2):
                await r.resolve(Reference(ref_id=f"r{i}", raw_text="x", title=f"Paper {i}"))
            calls_at_trip = s2.call_count
            await r.resolve(Reference(ref_id="r9", raw_text="x", title="Paper 9"))
            tripped = "semanticscholar" in r._tripped
            calls_after = s2.call_count
    assert tripped
    assert calls_after == calls_at_trip  # no further S2 calls after the breaker tripped
