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
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.crossref.org/works/10.1234%2Fexample").respond(404)
        # A Crossref title search would answer — but identifier lookups run first
        # across every source, so the S2 DOI hit wins without a search being made.
        cr_search = mock.get(host="api.crossref.org", path="/works").respond(
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
    assert cr_search.call_count == 0


@pytest.mark.asyncio
async def test_resolve_falls_through_to_arxiv_when_no_doi(fast_resolver):
    ref = Reference(
        ref_id="r1",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        arxiv_id="2401.12345",
    )
    with respx.mock(assert_all_called=False) as mock:
        # Title searches are mocked but must NOT be consulted: the arXiv ID is an
        # identifier, and identifier lookups run before any title search.
        cr_search = mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": []}}
        )
        # S2 by ARXIV: 404.
        mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345"
        ).respond(404)
        s2_search = mock.get(host="api.semanticscholar.org", path="/graph/v1/paper/search").respond(
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
    assert cr_search.call_count == 0
    assert s2_search.call_count == 0


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
        # No arXiv ID on the entry → arXiv is never asked in the identifier phase,
        # and the OpenAlex DOI hit ends resolution before any title search.
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


# ---------------------------------------------------------------------------
# OpenAlex post-2026-02 access model (U4): API key, no mailto, 409 = infra


@pytest.mark.asyncio
async def test_openalex_sends_api_key_not_mailto(fast_resolver):
    """Post-2026-02, OpenAlex needs an API key and ignores the (removed) mailto. The
    request must carry api_key and must NOT carry mailto, even when --mailto is set."""
    ref = Reference(
        ref_id="r1", raw_text="x", title="A Toy Paper About Toys", doi="10.1234/example"
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.crossref.org").respond(404)
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        oa = mock.get("https://api.openalex.org/works/doi:10.1234%2Fexample").respond(
            json=OPENALEX_DOI_HIT
        )
        async with fast_resolver(openalex_api_key="SECRET", polite_mailto="me@x.org") as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.RESOLVED and res.record.source == "openalex"
    assert oa.called
    query = str(oa.calls.last.request.url)
    assert "api_key=SECRET" in query
    assert "mailto" not in query


@pytest.mark.asyncio
async def test_openalex_409_is_degraded_not_ghost(fast_resolver):
    """A 409 from OpenAlex (keyless / daily credits exhausted) is infrastructure-
    degraded, never an authoritative 'not found' — it must not produce a false ghost."""
    ref = Reference(ref_id="r1", raw_text="x", title="A Real Paper", doi="10.1234/real")
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.crossref.org/works/10.1234%2Freal").respond(404)
        mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(409)
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.DEGRADED


# ---------------------------------------------------------------------------
# year gate on title searches: strong / weak / near-miss; identifiers first


def _crossref_item(title: str, year: int, *, authors=(("Alice", "Smith"), ("Bob", "Jones")), doi="10.9999/knockoff", venue="Mirror Journal of Everything"):
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"given": g, "family": f} for g, f in authors],
        "issued": {"date-parts": [[year]]},
        "container-title": [venue],
        "URL": f"https://doi.org/{doi}",
    }


def _s2_item(title: str, year: int, *, authors=("Alice Smith", "Bob Jones"), external_ids=None, venue="Journal of Toys", publication_venue=None):
    item = {
        "title": title,
        "authors": [{"name": a} for a in authors],
        "year": year,
        "abstract": "Toys are nice.",
        "externalIds": external_ids or {},
        "venue": venue,
    }
    if publication_venue is not None:
        item["publicationVenue"] = publication_venue
    return item


@pytest.mark.asyncio
async def test_search_hit_with_wrong_year_is_skipped_for_a_year_compatible_one(fast_resolver):
    """The observed failure: a 2025 reprint of a 2017 paper sits in Crossref with the
    same title and authors. Crossref is asked first, so it used to win. Now the year
    gate rejects it and the year-compatible S2 record resolves instead."""
    ref = Reference(
        ref_id="vaswani",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith", "Bob Jones"],
        year=2017,
        venue="Advances in Neural Information Processing Systems",
    )
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2025)]}}
        )
        mock.get(host="api.semanticscholar.org", path="/graph/v1/paper/search").respond(
            json={"data": [_s2_item("A Toy Paper About Toys", 2017, venue="Neural Information Processing Systems")]}
        )
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.record is not None
    assert res.record.source == "semanticscholar"
    assert res.record.year == 2017
    assert res.record.match_note is None
    assert res.near_misses == []


@pytest.mark.asyncio
async def test_weak_match_accepted_with_note_only_when_nothing_else_matches(fast_resolver):
    """Title+author match with an incompatible year, and no source has anything
    better: accept it rather than manufacture a ghost — but say so, and cache the
    note with the record."""
    ref = Reference(
        ref_id="book",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        year=2016,
    )
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2006, venue="Toy Press")]}}
        )
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            cached = r.cache.get_record(
                __import__("prereview.cache", fromlist=["cache_key"]).cache_key(
                    doi=None, arxiv_id=None, title=ref.title, first_author="Alice Smith"
                )
            )
    assert res.status == ResolutionStatus.RESOLVED
    rec = res.record
    assert rec is not None and rec.source == "crossref" and rec.year == 2006
    assert rec.match_note and "2016" in rec.match_note and "2006" in rec.match_note
    assert cached is not None and cached.match_note == rec.match_note


@pytest.mark.asyncio
async def test_weak_candidates_pick_the_closest_year(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2016)
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [
                _crossref_item("A Toy Paper About Toys", 1999, doi="10.1/far"),
                _crossref_item("A Toy Paper About Toys", 2010, doi="10.1/near"),
            ]}}
        )
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.record is not None and res.record.doi == "10.1/near"


@pytest.mark.asyncio
async def test_year_mismatch_without_author_overlap_is_a_ghost_with_near_misses(fast_resolver):
    """Same title, wrong year, different people: not even a weak match. The ghost
    verdict should still be able to say what was found."""
    ref = Reference(
        ref_id="r",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        year=2016,
    )
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [
                _crossref_item("A Toy Paper About Toys", 1999, authors=(("Zed", "Zulu"),), venue="Other Journal")
            ]}}
        )
        mock.get(host="api.semanticscholar.org").respond(json={"data": []})
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.UNRESOLVED
    assert res.record is None
    assert len(res.near_misses) == 1
    assert "1999" in res.near_misses[0] and "Other Journal" in res.near_misses[0]


@pytest.mark.asyncio
async def test_weak_match_is_not_accepted_over_a_transient_failure(fast_resolver):
    """If a source that might hold the real record failed transiently, the honest
    state is DEGRADED — a weak hit must not paper over it."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2016)
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2006)]}}
        )
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text="<feed/>")
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.DEGRADED
    assert res.record is None


@pytest.mark.asyncio
async def test_unknown_year_on_either_side_does_not_gate(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"])  # no year
    with respx.mock() as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2025)]}}
        )
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.RESOLVED and res.record.match_note is None


@pytest.mark.asyncio
async def test_identifier_lookup_beats_a_same_year_knockoff_in_title_search(fast_resolver):
    """A knockoff with a matching year would pass the year gate — but an arXiv ID is
    an identifier, and identifier lookups run before any title search."""
    ref = Reference(
        ref_id="r",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        year=2024,
        venue="arXiv preprint arXiv:2401.12345",
        arxiv_id="2401.12345",
    )
    with respx.mock(assert_all_called=False) as mock:
        cr_search = mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2024)]}}
        )
        mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345").respond(
            json=_s2_item("A Toy Paper About Toys", 2024, external_ids={"ArXiv": "2401.12345"}, venue="arXiv.org")
        )
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.record is not None and res.record.source == "semanticscholar"
    assert cr_search.call_count == 0


@pytest.mark.asyncio
async def test_arxiv_error_entry_is_a_terminal_miss_not_a_hit(fast_resolver):
    """arXiv answers an unknown ID with HTTP 200 and an <entry> titled 'Error'."""
    ref = Reference(ref_id="r", raw_text="x", title="Nothing Here", arxiv_id="2401.99999")
    error_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_2401.99999</id>
    <title>Error</title>
    <summary>incorrect id format for 2401.99999</summary>
  </entry>
</feed>
"""
    with respx.mock() as mock:
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text=error_feed)
        mock.get(host="api.crossref.org").respond(json={"message": {"items": []}})
        mock.get(host="api.openalex.org").respond(json={"results": []})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.status == ResolutionStatus.UNRESOLVED


def test_guess_arxiv_id_from_datacite_doi_url_and_context_only():
    from prereview.resolve import _guess_arxiv_id, arxiv_id_of, cites_arxiv_version

    assert _guess_arxiv_id(Reference(ref_id="a", raw_text="x", doi="10.48550/arXiv.2401.12345")) == "2401.12345"
    assert _guess_arxiv_id(Reference(ref_id="b", raw_text="x", url="https://arxiv.org/abs/2401.12345v2")) == "2401.12345"
    assert _guess_arxiv_id(Reference(ref_id="c", raw_text="journal = {arXiv preprint arXiv:2409.06927}")) == "2409.06927"
    # A bare NNNN.NNNNN inside a DOI is not an arXiv ID.
    assert _guess_arxiv_id(Reference(ref_id="d", raw_text="doi = {10.1000/1234.5678}")) is None
    assert arxiv_id_of(Reference(ref_id="e", raw_text="x", arxiv_id="arXiv:2401.12345v1")) == "2401.12345"

    assert cites_arxiv_version(Reference(ref_id="f", raw_text="x", arxiv_id="2401.12345"))
    assert cites_arxiv_version(Reference(ref_id="g", raw_text="x", venue="arXiv preprint arXiv:2401.12345"))
    assert cites_arxiv_version(Reference(ref_id="h", raw_text="x", venue="CoRR"))
    assert cites_arxiv_version(Reference(ref_id="i", raw_text="x", doi="10.48550/arXiv.2401.12345"))
    # A real publisher DOI means the published version is cited, whatever else the entry says.
    assert not cites_arxiv_version(Reference(ref_id="j", raw_text="x", doi="10.18653/v1/2021.acl-long.1", venue="arXiv"))
    assert not cites_arxiv_version(Reference(ref_id="k", raw_text="x", venue="Proceedings of NeurIPS"))


# ---------------------------------------------------------------------------
# outdated arXiv citations: published-version detection and follow-up


def test_s2_published_version_signals():
    from prereview.resolve import _s2_published_version

    # DBLP conf/ key + typed conference venue, no DOI (the NeurIPS shape).
    pv = _s2_published_version(_s2_item(
        "T", 2017, external_ids={"ArXiv": "1706.03762", "DBLP": "conf/nips/VaswaniSPUJGKP17"},
        venue="Neural Information Processing Systems",
        publication_venue={"name": "Neural Information Processing Systems", "type": "conference"},
    ))
    assert pv is not None and pv.venue == "Neural Information Processing Systems" and pv.year == 2017 and pv.doi is None
    # Real publisher DOI.
    pv = _s2_published_version(_s2_item("T", 2021, external_ids={"DOI": "10.18653/v1/2021.acl-long.1"}, venue="ACL"))
    assert pv is not None and pv.doi == "10.18653/v1/2021.acl-long.1" and pv.url == "https://doi.org/10.18653/v1/2021.acl-long.1"
    # arXiv-only: DataCite DOI, journals/corr DBLP key, arXiv venue → not published.
    assert _s2_published_version(_s2_item(
        "T", 2024, external_ids={"ArXiv": "2401.12345", "DOI": "10.48550/arXiv.2401.12345", "DBLP": "journals/corr/abs-2401-12345"},
        venue="arXiv.org", publication_venue={"name": "arXiv.org", "type": "journal"},
    )) is None
    assert _s2_published_version({}) is None
    assert _s2_published_version(None) is None


@pytest.mark.asyncio
async def test_s2_identifier_hit_carries_published_version_for_arxiv_cited_entry(fast_resolver):
    ref = Reference(
        ref_id="r",
        raw_text="x",
        title="A Toy Paper About Toys",
        authors=["Alice Smith"],
        year=2023,
        venue="arXiv preprint arXiv:2301.00001",
        arxiv_id="2301.00001",
    )
    with respx.mock() as mock:
        mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2301.00001").respond(
            json=_s2_item(
                "A Toy Paper About Toys", 2024,
                external_ids={"ArXiv": "2301.00001", "DBLP": "conf/iclr/Smith24"},
                venue="International Conference on Learning Representations",
                publication_venue={"name": "International Conference on Learning Representations", "type": "conference"},
            )
        )
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    rec = res.record
    assert rec is not None and rec.published_version_checked
    assert rec.published_version is not None
    assert rec.published_version.venue == "International Conference on Learning Representations"
    assert rec.published_version.year == 2024


@pytest.mark.asyncio
async def test_arxiv_record_with_registered_journal_doi_carries_published_version(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    feed = ARXIV_FEED.replace(
        '<link title="pdf" href="http://arxiv.org/pdf/2401.12345v1"/>',
        '<link title="pdf" href="http://arxiv.org/pdf/2401.12345v1"/>'
        "<arxiv:doi>10.1000/toys.2024.1</arxiv:doi>"
        "<arxiv:journal_ref>Journal of Toys 12(3), 2024</arxiv:journal_ref>",
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(404)
        mock.get(host="export.arxiv.org").respond(text=feed)
        mock.get(host="api.openalex.org").respond(json={"is_retracted": False})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    rec = res.record
    assert rec is not None and rec.source == "arxiv"
    assert rec.published_version is not None
    assert rec.published_version.doi == "10.1000/toys.2024.1"
    assert rec.published_version.venue == "Journal of Toys 12(3), 2024"
    assert rec.published_version_checked


@pytest.mark.asyncio
async def test_arxiv_hit_without_doi_triggers_s2_followup(fast_resolver):
    """S2 down for the ID lookup, arXiv resolves, S2 back for the follow-up: the
    follow-up asks S2 by arXiv ID and records the answer."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        s2 = mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345")
        s2.side_effect = [
            httpx.Response(404),  # identifier phase: S2 doesn't have it (yet)
            httpx.Response(200, json=_s2_item(
                "A Toy Paper About Toys", 2024,
                external_ids={"ArXiv": "2401.12345", "DOI": "10.1000/toys.2024.1"},
                venue="Journal of Toys",
            )),
        ]
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json={"result": {"hits": {"@total": "0"}}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    rec = res.record
    assert rec is not None and rec.source == "arxiv"
    assert s2.call_count == 2
    assert rec.published_version_checked
    assert rec.published_version is not None and rec.published_version.doi == "10.1000/toys.2024.1"


@pytest.mark.asyncio
async def test_followup_transient_leaves_record_unchecked_and_is_counted(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json={"result": {"hits": {"@total": "0"}}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            key = __import__("prereview.cache", fromlist=["cache_key"]).cache_key(
                doi=None, arxiv_id="2401.12345", title=ref.title, first_author="Alice Smith"
            )
            cached = r.cache.get_record(key)
            unchecked = r.published_version_unchecked
    assert res.status == ResolutionStatus.RESOLVED  # the S2 outage didn't stop arXiv from resolving it
    assert res.record.published_version_checked is False
    assert res.record.published_version is None
    assert unchecked == 1
    assert cached is not None and cached.published_version_checked is False  # retried next run


@pytest.mark.asyncio
async def test_followup_runs_on_cache_hit_for_records_cached_before_the_check_existed(fast_resolver, tmp_path):
    """A pre-existing cache file without the new fields must be upgraded on the next
    run, not silently treated as 'no published version'."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    from prereview.cache import cache_key

    key = cache_key(doi=None, arxiv_id="2401.12345", title=ref.title, first_author="Alice Smith")
    with respx.mock() as mock:
        s2 = mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345").respond(
            json=_s2_item("A Toy Paper About Toys", 2024, external_ids={"DBLP": "conf/iclr/Smith24"},
                          publication_venue={"name": "ICLR", "type": "conference"})
        )
        async with fast_resolver() as r:
            # Old-schema record on disk: no published_version* keys at all.
            (r.cache.refs_dir / f"{key}.json").write_text(json.dumps({
                "source": "arxiv", "title": "A Toy Paper About Toys", "authors": ["Alice Smith"],
                "year": 2024, "venue": "arXiv", "url": "http://arxiv.org/abs/2401.12345v1",
            }))
            res = await r.resolve(ref)
            cached = r.cache.get_record(key)
    assert res.status == ResolutionStatus.RESOLVED and res.record.source == "arxiv"
    assert s2.call_count == 1
    assert res.record.published_version is not None and res.record.published_version.venue == "ICLR"
    assert cached.published_version_checked and cached.published_version.venue == "ICLR"


@pytest.mark.asyncio
async def test_no_followup_for_entries_that_cite_a_published_doi(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2023, doi="10.1234/example")
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.crossref.org/works/10.1234%2Fexample").respond(json=CROSSREF_DOI_HIT)
        mock.get(host="api.openalex.org").respond(json={"is_retracted": False})
        s2 = mock.get(host="api.semanticscholar.org").respond(json={})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.record is not None
    assert s2.call_count == 0
    assert res.record.published_version is None
    assert res.record.published_version_checked is False  # never asked; not arXiv-cited


@pytest.mark.asyncio
async def test_strong_crossref_venue_hit_is_itself_the_published_version(fast_resolver):
    """An arXiv-cited entry that title-resolves to a real venue record (same year,
    publisher DOI, non-arXiv container) needs no follow-up call."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2023, venue="arXiv preprint")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.crossref.org", path="/works").respond(
            json={"message": {"items": [_crossref_item("A Toy Paper About Toys", 2023, doi="10.1234/example", venue="Journal of Toys")]}}
        )
        mock.get(host="api.openalex.org").respond(json={"is_retracted": False})
        s2 = mock.get(host="api.semanticscholar.org").respond(json={})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    rec = res.record
    assert rec is not None and rec.source == "crossref"
    assert s2.call_count == 0
    assert rec.published_version_checked
    assert rec.published_version is not None and rec.published_version.venue == "Journal of Toys"
    assert rec.published_version.doi == "10.1234/example"


@pytest.mark.asyncio
async def test_crossref_fallback_confirms_published_version_when_s2_is_down(fast_resolver):
    ref = Reference(
        ref_id="bert", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"],
        year=2018, venue="arXiv preprint arXiv:2401.12345", arxiv_id="2401.12345",
    )
    item = _crossref_item("A Toy Paper About Toys", 2019, doi="10.18653/v1/n19-1423", venue="Proceedings of NAACL-HLT")
    item["type"] = "proceedings-article"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json={"result": {"hits": {"@total": "0"}}})
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": [item]}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            unchecked = r.published_version_unchecked
    rec = res.record
    assert rec is not None and rec.source == "arxiv"
    assert cr.call_count == 1
    assert rec.published_version_checked
    assert rec.published_version is not None
    assert rec.published_version.source == "crossref"
    assert rec.published_version.doi == "10.18653/v1/n19-1423"
    assert rec.published_version.venue == "Proceedings of NAACL-HLT"
    assert unchecked == 0


@pytest.mark.asyncio
async def test_crossref_fallback_ignores_preprint_records_and_wrong_years(fast_resolver):
    ref = Reference(
        ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"],
        year=2018, venue="arXiv preprint arXiv:2401.12345", arxiv_id="2401.12345",
    )
    preprint = _crossref_item("A Toy Paper About Toys", 2018, doi="10.48550/arxiv.2401.12345", venue="arXiv")
    preprint["type"] = "posted-content"
    reprint = _crossref_item("A Toy Paper About Toys", 2025, doi="10.9999/knockoff", venue="Mirror Journal")
    reprint["type"] = "journal-article"
    strangers = _crossref_item("A Toy Paper About Toys", 2018, authors=(("Zed", "Zulu"),), doi="10.5555/other", venue="Other Proc.")
    strangers["type"] = "proceedings-article"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json={"result": {"hits": {"@total": "0"}}})
        mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": [preprint, reprint, strangers]}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            unchecked = r.published_version_unchecked
    rec = res.record
    assert rec is not None and rec.published_version is None
    assert rec.published_version_checked is False  # S2 was transient and Crossref could not confirm
    assert unchecked == 1


@pytest.mark.asyncio
async def test_crossref_fallback_not_consulted_when_s2_answers(fast_resolver):
    """S2 says 'no venue version'. DBLP (curated) is still asked in case S2 lags;
    Crossref (uncurated) is not allowed to overrule S2."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        s2 = mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345")
        s2.side_effect = [
            httpx.Response(404),
            httpx.Response(200, json=_s2_item("A Toy Paper About Toys", 2024, external_ids={"ArXiv": "2401.12345"}, venue="arXiv.org")),
        ]
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json={"result": {"hits": {"@total": "0"}}})
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    assert res.record is not None and res.record.published_version_checked and res.record.published_version is None
    assert cr.call_count == 0


# ---------------------------------------------------------------------------
# DBLP as the published-version confirmer


def _dblp_payload(*hits):
    return {"result": {"hits": {"@total": str(len(hits)), "hit": [{"info": h} for h in hits]}}}


DBLP_NAACL = {
    "authors": {"author": [{"@pid": "1", "text": "Alice Smith"}, {"@pid": "2", "text": "Bob Jones"}]},
    "title": "A Toy Paper About Toys.",
    "venue": "NAACL-HLT",
    "year": "2019",
    "type": "Conference and Workshop Papers",
    "key": "conf/naacl/SmithJ19",
    "doi": "10.18653/V1/N19-1423",
    "ee": "https://doi.org/10.18653/v1/n19-1423",
    "url": "https://dblp.org/rec/conf/naacl/SmithJ19",
}
DBLP_CORR = {
    "authors": {"author": [{"@pid": "1", "text": "Alice Smith"}]},
    "title": "A Toy Paper About Toys.",
    "venue": "CoRR",
    "year": "2018",
    "type": "Informal and Other Publications",
    "key": "journals/corr/abs-2401-12345",
    "ee": "http://arxiv.org/abs/2401.12345",
}


@pytest.mark.asyncio
async def test_dblp_confirms_published_version_when_s2_is_down(fast_resolver):
    ref = Reference(
        ref_id="bert", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith", "Bob Jones"],
        year=2018, venue="arXiv preprint arXiv:2401.12345", arxiv_id="2401.12345",
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        dblp = mock.get(host="dblp.org", path="/search/publ/api").respond(json=_dblp_payload(DBLP_CORR, DBLP_NAACL))
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            unchecked = r.published_version_unchecked
    rec = res.record
    assert rec is not None and rec.published_version_checked
    pv = rec.published_version
    assert pv is not None and pv.source == "dblp"
    assert pv.venue == "NAACL-HLT" and pv.year == 2019
    assert pv.doi == "10.18653/v1/n19-1423"  # lower-cased
    assert pv.url == "https://doi.org/10.18653/v1/n19-1423"
    assert cr.call_count == 0  # DBLP confirmed; Crossref never needed
    assert unchecked == 0
    # The query carries the first author's surname so DBLP ANDs it with the title.
    assert "Smith" in str(dblp.calls.last.request.url)


@pytest.mark.asyncio
async def test_dblp_corr_only_record_is_not_a_confirmation(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2018, arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json=_dblp_payload(DBLP_CORR))
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            unchecked = r.published_version_unchecked
    assert res.record.published_version is None
    assert res.record.published_version_checked is False
    assert cr.call_count == 1  # fell through to the Crossref last resort
    assert unchecked == 1  # S2 was transient and nothing confirmed


@pytest.mark.asyncio
async def test_dblp_can_lift_a_venue_version_s2_does_not_know_yet(fast_resolver):
    """S2 lags for fresh proceedings; DBLP is consulted even after S2 says 'none'."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], year=2018, arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345").respond(
            json=_s2_item("A Toy Paper About Toys", 2018, external_ids={"ArXiv": "2401.12345"}, venue="arXiv.org")
        )
        mock.get(host="dblp.org").respond(json=_dblp_payload(DBLP_NAACL))
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    rec = res.record
    assert rec.source == "semanticscholar"
    assert rec.published_version is not None and rec.published_version.source == "dblp"
    assert cr.call_count == 0


@pytest.mark.asyncio
async def test_dblp_gates_year_authors_and_single_author_dict_shape(fast_resolver):
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Zhe Li"], year=2018, arxiv_id="2401.12345")
    wrong_year = dict(DBLP_NAACL, year="2025", authors={"author": {"text": "Zhe Li 0030"}})
    strangers = dict(DBLP_NAACL, authors={"author": [{"text": "Zed Zulu"}]})
    right = dict(DBLP_NAACL, year="2019", authors={"author": {"@pid": "9", "text": "Zhe Li 0030"}}, venue=["ICASSP"], doi=None, ee="https://ieeexplore.example/1")
    with respx.mock(assert_all_called=False) as mock:
        mock.get(host="api.semanticscholar.org").respond(500)
        mock.get(host="export.arxiv.org").respond(text=ARXIV_FEED)
        mock.get(host="dblp.org").respond(json=_dblp_payload(wrong_year, strangers, right))
        async with fast_resolver() as r:
            res = await r.resolve(ref)
    pv = res.record.published_version
    assert pv is not None and pv.venue == "ICASSP" and pv.year == 2019
    assert pv.doi is None and pv.url == "https://ieeexplore.example/1"


@pytest.mark.asyncio
async def test_dblp_transient_does_not_disturb_s2_none(fast_resolver):
    """S2 answered 'none'; a DBLP hiccup must not turn that into 'unchecked'."""
    ref = Reference(ref_id="r", raw_text="x", title="A Toy Paper About Toys", authors=["Alice Smith"], arxiv_id="2401.12345")
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.semanticscholar.org/graph/v1/paper/ARXIV:2401.12345").respond(
            json=_s2_item("A Toy Paper About Toys", 2024, external_ids={"ArXiv": "2401.12345"}, venue="arXiv.org")
        )
        mock.get(host="dblp.org").respond(503)
        cr = mock.get(host="api.crossref.org", path="/works").respond(json={"message": {"items": []}})
        async with fast_resolver() as r:
            res = await r.resolve(ref)
            unchecked = r.published_version_unchecked
    assert res.record.published_version_checked and res.record.published_version is None
    assert unchecked == 0 and cr.call_count == 0
