"""Tests for prereview.synthesize.

The deterministic sections (Citation issues, Methodology) and the stitching
logic are tested directly. The LLM prose pass is mocked.
"""

from __future__ import annotations

import pytest

from prereview import synthesize as syn
from prereview.models import (
    CanonicalRecord,
    Citation,
    IngestedPaper,
    Reference,
    ReviewBundle,
    VerificationResult,
    Verdict,
)


def _make_bundle(verifications: list[VerificationResult]) -> ReviewBundle:
    paper = IngestedPaper(
        title="A Toy Paper",
        abstract="We propose toys.",
        sections=[("body", "Toys are great. We claim X [1].")],
        references={v.reference.ref_id: v.reference for v in verifications},
        citations=[v.citation for v in verifications],
    )
    return ReviewBundle(
        paper=paper,
        verifications=verifications,
        model="anthropic/claude-sonnet-4-6",
        synthesis_model="anthropic/claude-opus-4-7",
        fetched_full_text_count=sum(1 for v in verifications if not v.abstract_only and v.canonical),
        abstract_only_count=sum(1 for v in verifications if v.abstract_only and v.canonical),
        unresolved_count=sum(1 for v in verifications if v.canonical is None),
    )


def _v(verdict: Verdict, *, ref_id="1", canonical=True, abstract_only=False, rationale="rationale.") -> VerificationResult:
    ref = Reference(
        ref_id=ref_id,
        raw_text=f"[{ref_id}] Smith. A Toy Paper. 2023.",
        authors=["Alice Smith"],
        title="A Toy Paper",
        year=2023,
    )
    cit = Citation(ref_id=ref_id, sentence="We claim X [1].")
    can = (
        CanonicalRecord(
            source="crossref",
            title="A Toy Paper",
            authors=["Alice Smith"],
            year=2023,
            doi="10.1234/toy",
            url="https://doi.org/10.1234/toy",
            abstract="abstract",
        )
        if canonical
        else None
    )
    return VerificationResult(
        ref_id=ref_id,
        citation=cit,
        reference=ref,
        canonical=can,
        verdict=verdict,
        rationale=rationale,
        abstract_only=abstract_only,
    )


def test_render_citation_issues_lists_unresolved():
    bundle = _make_bundle([_v(Verdict.TARGET_UNAVAILABLE, ref_id="ghost", canonical=False, rationale="No record found.")])
    md = syn.render_citation_issues(bundle)
    assert "Citation issues" in md
    assert "ghost" in md
    assert "ghost reference" in md.lower()
    assert "No record found." in md


def test_render_citation_issues_lists_does_not_support():
    bundle = _make_bundle([_v(Verdict.DOES_NOT_SUPPORT, ref_id="r2", rationale="Cited paper actually says the opposite.")])
    md = syn.render_citation_issues(bundle)
    assert "r2" in md
    assert "Does not support" in md
    assert "opposite" in md


def test_render_citation_issues_flags_abstract_only_supports():
    """Abstract-only `supports` should still be flagged so the user can verify."""
    bundle = _make_bundle([_v(Verdict.SUPPORTS, abstract_only=True, rationale="Abstract states the claim.")])
    md = syn.render_citation_issues(bundle)
    assert "abstract-only" in md.lower()


def test_render_citation_issues_clean_when_no_problems():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, abstract_only=False, rationale="Body text confirms.")])
    md = syn.render_citation_issues(bundle)
    assert "No citation issues detected" in md


def test_render_methodology_mentions_models_and_counts():
    bundle = _make_bundle([
        _v(Verdict.SUPPORTS, ref_id="1", abstract_only=False),
        _v(Verdict.SUPPORTS, ref_id="2", abstract_only=True),
        _v(Verdict.TARGET_UNAVAILABLE, ref_id="3", canonical=False),
    ])
    md = syn.render_methodology(bundle)
    assert "anthropic/claude-sonnet-4-6" in md
    assert "anthropic/claude-opus-4-7" in md
    assert "3 in-text citations" in md
    assert "1 bibliography entr" in md  # one ghost
    assert "Missing prior work" in md


@pytest.mark.asyncio
async def test_synthesize_review_full_pass(monkeypatch):
    bundle = _make_bundle([
        _v(Verdict.DOES_NOT_SUPPORT, ref_id="badcite", rationale="Cited paper does not back this claim."),
        _v(Verdict.SUPPORTS, ref_id="goodcite", abstract_only=False, rationale="Body confirms."),
        _v(Verdict.TARGET_UNAVAILABLE, ref_id="ghost", canonical=False, rationale="Resolved nowhere."),
    ])

    async def fake_json(**kwargs):
        return {
            "summary": "This paper claims X.\n\nThey also do Y.",
            "strengths": ["Clear motivation.", "Well-written abstract."],
            "weaknesses": ["Several citations do not support the claims they back."],
            "questions": ["How does X scale?", "What if Y is removed?"],
            "rating_low": 4,
            "rating_high": 6,
            "rating_justification": "Mid-range; this is an LLM judgement.",
        }

    monkeypatch.setattr(syn, "acompletion_json", fake_json)
    md = await syn.synthesize_review(bundle)

    # Section headers all present.
    for h in (
        "# Pre-submission review",
        "## Summary",
        "## Strengths",
        "## Weaknesses",
        "## Citation issues",
        "## Questions for the author",
        "## Suggested rating",
        "## Methodology and limits of this review",
    ):
        assert h in md, f"missing header: {h}"

    # Every problematic citation is named in the citation-issues section.
    issues_start = md.index("## Citation issues")
    issues_end = md.index("## Questions for the author")
    issues_block = md[issues_start:issues_end]
    assert "badcite" in issues_block
    assert "ghost" in issues_block
    # The well-supported full-text cite should not appear in issues.
    assert "goodcite" not in issues_block

    # Rating range present.
    assert "4–6/10" in md or "4-6/10" in md

    # Methodology lists what the tool does NOT do.
    assert "Missing prior work" in md
