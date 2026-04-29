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
from prereview.models import Reference
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
            rec = await r.resolve(ref)
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
            rec = await r.resolve(ref)
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
            rec = await r.resolve(ref)
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
            rec = await r.resolve(ref)
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
            rec = await r.resolve(ref)
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
            rec1 = await r.resolve(ref)
            rec2 = await r.resolve(ref)
    assert rec1 is not None and rec2 is not None
    assert rec1.title == rec2.title
    assert cr_route.call_count == 1  # second call hit the cache


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
            rec = await r.resolve(ref)
    assert rec is None
