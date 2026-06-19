"""Tests for prereview.openreview_enrich.

The OpenReview client is mocked (the library is an optional dependency). The
first-class contracts: no creds → a clean skip (not degradation), a mapped
citation gains an advisory annotation, an unmapped one is silently skipped, and an
API failure degrades without crashing the run.
"""

from __future__ import annotations

import pytest

from prereview import openreview_enrich as ore
from prereview.models import Citation, OpenReviewInfo, Reference, VerificationResult, Verdict


def _vr(ref_id: str, *, arxiv_id=None, doi=None, title="A Paper") -> VerificationResult:
    ref = Reference(ref_id=ref_id, raw_text="x", title=title, arxiv_id=arxiv_id, doi=doi)
    cit = Citation(ref_id=ref_id, sentence="We cite this.")
    return VerificationResult(
        ref_id=ref_id, citation=cit, reference=ref, verdict=Verdict.SUPPORTS, rationale="r"
    )


@pytest.mark.asyncio
async def test_no_creds_is_clean_skip_not_degradation():
    vrs = [_vr("1", arxiv_id="2401.00001")]
    annotated, degraded = await ore.enrich_with_openreview(vrs, username=None, password=None)
    assert (annotated, degraded) == (0, False)
    assert vrs[0].openreview is None


@pytest.mark.asyncio
async def test_mapped_citation_gets_advisory_annotation(monkeypatch):
    monkeypatch.setattr(ore, "_make_client", lambda u, p: object())  # non-None fake client
    monkeypatch.setattr(
        ore, "_find_decision",
        lambda client, arxiv, doi, title: OpenReviewInfo(
            decision="Reject", rating_avg=3.5, rating_count=4, url="https://openreview.net/forum?id=x"
        ),
    )
    vrs = [_vr("1", arxiv_id="2401.00001")]
    annotated, degraded = await ore.enrich_with_openreview(vrs, username="u", password="p")
    assert annotated == 1 and degraded is False
    assert vrs[0].openreview is not None
    assert vrs[0].openreview.decision == "Reject"


@pytest.mark.asyncio
async def test_unmapped_reference_is_silently_skipped(monkeypatch):
    called: list = []
    monkeypatch.setattr(ore, "_make_client", lambda u, p: object())
    monkeypatch.setattr(ore, "_find_decision", lambda *a: called.append(1) or None)
    vrs = [_vr("1")]  # no arxiv_id, no doi → nothing to map
    annotated, degraded = await ore.enrich_with_openreview(vrs, username="u", password="p")
    assert annotated == 0 and degraded is False
    assert not called  # the lookup was never attempted


@pytest.mark.asyncio
async def test_api_failure_degrades_without_crashing(monkeypatch):
    def boom(*a):
        raise RuntimeError("openreview down")

    monkeypatch.setattr(ore, "_make_client", lambda u, p: object())
    monkeypatch.setattr(ore, "_find_decision", boom)
    vrs = [_vr("1", arxiv_id="2401.00001")]
    annotated, degraded = await ore.enrich_with_openreview(vrs, username="u", password="p")
    assert annotated == 0 and degraded is True
    assert vrs[0].openreview is None  # no annotation, but the run completed


@pytest.mark.asyncio
async def test_missing_library_or_login_degrades(monkeypatch):
    monkeypatch.setattr(ore, "_make_client", lambda u, p: None)  # import/login failure
    vrs = [_vr("1", arxiv_id="2401.00001")]
    annotated, degraded = await ore.enrich_with_openreview(vrs, username="u", password="p")
    assert annotated == 0 and degraded is True


def test_content_value_unwraps_v2_and_v1():
    assert ore._content_value({"decision": {"value": "Accept"}}, "decision") == "Accept"  # v2
    assert ore._content_value({"decision": "Reject"}, "decision") == "Reject"  # v1


def test_leading_number_parses_rating_strings():
    assert ore._leading_number("7: Good paper") == 7.0
    assert ore._leading_number(8) == 8.0
    assert ore._leading_number("not a number") is None
