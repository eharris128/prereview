"""Tests for prereview.synthesize.

The deterministic sections (Citation issues, Methodology) and the stitching
logic are tested directly. The LLM prose pass is mocked.
"""

from __future__ import annotations

import pytest

from prereview import synthesize as syn
from prereview.models import (
    BrokenRef,
    CanonicalRecord,
    ChecklistFinding,
    ChecklistFindingKind,
    Citation,
    CitationRole,
    IngestedPaper,
    LinkCheck,
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


def _v(verdict: Verdict, *, ref_id="1", canonical=True, abstract_only=False, rationale="rationale.", sentence="We claim X [1].", role=None) -> VerificationResult:
    ref = Reference(
        ref_id=ref_id,
        raw_text=f"[{ref_id}] Smith. A Toy Paper. 2023.",
        authors=["Alice Smith"],
        title="A Toy Paper",
        year=2023,
    )
    cit = Citation(ref_id=ref_id, sentence=sentence)
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
        role=role,
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


def test_render_citation_issues_groups_bib_level_sites():
    """A broken .bib entry cited at N sites should produce one section, not N."""
    bundle = _make_bundle([
        _v(
            Verdict.METADATA_MISMATCH,
            ref_id="broken",
            sentence="First use of broken cite [broken].",
            rationale="No author-surname overlap with resolved record.",
        ),
        _v(
            Verdict.METADATA_MISMATCH,
            ref_id="broken",
            sentence="Second use of broken cite [broken].",
            rationale="No author-surname overlap with resolved record.",
        ),
        _v(
            Verdict.METADATA_MISMATCH,
            ref_id="broken",
            sentence="Third use of broken cite [broken].",
            rationale="No author-surname overlap with resolved record.",
        ),
    ])
    md = syn.render_citation_issues(bundle)

    # Exactly one numbered section for the broken bibkey, not three.
    assert md.count("### 1.") == 1
    assert "### 2." not in md
    # All three cite sites are still surfaced, under a single entry.
    assert "First use of broken cite" in md
    assert "Second use of broken cite" in md
    assert "Third use of broken cite" in md
    assert "Cited at 3 sites" in md
    # Header summary mentions 1 unique reference / 3 cite sites.
    assert "1 unique reference" in md
    assert "3 cite sites" in md


def test_render_citation_issues_site_level_multi_site():
    """A bib entry that resolves correctly but is mis-attached to multiple
    different claims should list each cite site's own verdict and rationale."""
    bundle = _make_bundle([
        _v(
            Verdict.PARTIALLY_SUPPORTS,
            ref_id="paper",
            sentence="Mild claim about X [paper].",
            rationale="Cited paper covers X tangentially.",
        ),
        _v(
            Verdict.DOES_NOT_SUPPORT,
            ref_id="paper",
            sentence="Strong unrelated claim about Y [paper].",
            rationale="Cited paper says nothing about Y.",
        ),
    ])
    md = syn.render_citation_issues(bundle)

    # One grouped section, both sites surfaced with their own verdicts.
    assert md.count("### 1.") == 1
    assert "2 cite sites flagged" in md
    assert "Does not support" in md
    assert "Partially supports" in md
    assert "Mild claim about X" in md
    assert "Strong unrelated claim about Y" in md
    # Most-severe site listed first (does_not_support before partially_supports).
    assert md.index("Strong unrelated claim about Y") < md.index("Mild claim about X")


def test_method_attribution_abstract_only_supports_not_flagged():
    """A method/tool attribution cite is structurally fine when title+authors
    match — abstract-only `supports` should not be surfaced as 'verify'."""
    bundle = _make_bundle([
        _v(
            Verdict.SUPPORTS,
            ref_id="hst",
            abstract_only=True,
            role=CitationRole.METHOD_ATTRIBUTION,
            sentence="We use HST [hst].",
            rationale="The cited paper introduces the HST anomaly detector.",
        ),
    ])
    md = syn.render_citation_issues(bundle)
    assert "No citation issues detected" in md


def test_claim_support_abstract_only_supports_still_flagged():
    """A claim-support cite that only resolves at the abstract still gets
    flagged for verification — the body might tell a different story."""
    bundle = _make_bundle([
        _v(
            Verdict.SUPPORTS,
            ref_id="claim",
            abstract_only=True,
            role=CitationRole.CLAIM_SUPPORT,
            sentence="X occurs in 30% of cases [claim].",
            rationale="Abstract reports the 30% figure.",
        ),
    ])
    md = syn.render_citation_issues(bundle)
    assert "claim" in md
    assert "abstract-only" in md.lower()


def test_method_attribution_abstract_too_thin_not_flagged():
    """If the model returns abstract_too_thin on a method attribution, that
    means it couldn't tell from abstract+title alone. We treat that as not
    actionable: pushing the user to fetch full text won't change the verdict
    on whether this is the right method's paper."""
    bundle = _make_bundle([
        _v(
            Verdict.ABSTRACT_TOO_THIN,
            ref_id="lof",
            abstract_only=True,
            role=CitationRole.METHOD_ATTRIBUTION,
            sentence="LOF [lof] is one of the baselines.",
            rationale="Abstract describes LOF without naming the citing paper's claim.",
        ),
    ])
    md = syn.render_citation_issues(bundle)
    assert "No citation issues detected" in md


def test_does_not_support_flagged_regardless_of_role():
    """does_not_support is always actionable — wrong cite is wrong cite."""
    bundle = _make_bundle([
        _v(
            Verdict.DOES_NOT_SUPPORT,
            ref_id="bad_method",
            role=CitationRole.METHOD_ATTRIBUTION,
            rationale="Cited paper is about a different algorithm entirely.",
        ),
    ])
    md = syn.render_citation_issues(bundle)
    assert "bad_method" in md
    assert "Does not support" in md


def test_render_hygiene_section_shows_broken_refs_and_unused_bibkeys():
    refs = [_v(Verdict.SUPPORTS, ref_id="1", abstract_only=False)]
    paper = IngestedPaper(
        title="A Toy Paper",
        abstract="x",
        sections=[("body", "x")],
        references={v.reference.ref_id: v.reference for v in refs},
        citations=[v.citation for v in refs],
        unused_bibkeys=["dead1", "dead2"],
        broken_refs=[
            BrokenRef(command="ref", target="app:missing", surrounding="See App.~\\ref{app:missing}."),
            BrokenRef(command="cref", target="tab:nope", surrounding="\\cref{tab:nope} shows nothing."),
        ],
    )
    bundle = ReviewBundle(
        paper=paper,
        verifications=refs,
        model="m",
        synthesis_model="s",
    )
    md = syn.render_hygiene_section(bundle)
    assert md is not None
    assert "Broken cross-references (2)" in md
    assert "app:missing" in md
    assert "tab:nope" in md
    assert "Unused bibliography entries (2)" in md
    assert "dead1" in md
    assert "dead2" in md


def test_render_hygiene_section_returns_none_when_clean():
    paper = IngestedPaper(title="t", references={}, citations=[])
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    assert syn.render_hygiene_section(bundle) is None


def _v_retracted(ref_id="retracted_paper", sentence=None) -> VerificationResult:
    """A verification whose canonical record is flagged as retracted."""
    ref = Reference(
        ref_id=ref_id,
        raw_text=f"[{ref_id}] Wakefield. Wrong Paper. 1998.",
        authors=["A. Wakefield"],
        title="Wrong Paper",
        year=1998,
    )
    sentence = sentence or f"We rely on prior findings [{ref_id}]."
    return VerificationResult(
        ref_id=ref_id,
        citation=Citation(ref_id=ref_id, sentence=sentence),
        reference=ref,
        canonical=CanonicalRecord(
            source="openalex",
            title="Wrong Paper",
            authors=["A. Wakefield"],
            year=1998,
            doi="10.1234/retracted",
            url="https://doi.org/10.1234/retracted",
            abstract="abstract",
            is_retracted=True,
        ),
        verdict=Verdict.SUPPORTS,
        rationale="abstract supports the claim",
        abstract_only=False,
        role=CitationRole.CLAIM_SUPPORT,
    )


def test_render_hygiene_section_surfaces_retractions_even_when_otherwise_clean():
    """A retracted-but-otherwise-supports cite should appear in Hygiene even
    when there are no broken refs or unused bibkeys — it's exactly the case
    Citation Issues would silently miss."""
    v = _v_retracted()
    paper = IngestedPaper(
        title="A Paper",
        references={v.reference.ref_id: v.reference},
        citations=[v.citation],
    )
    bundle = ReviewBundle(paper=paper, verifications=[v], model="m", synthesis_model="s")
    md = syn.render_hygiene_section(bundle)
    assert md is not None
    assert "Retracted citations (1)" in md
    assert "retracted_paper" in md
    assert "1 cite site" in md
    assert "10.1234/retracted" in md


def test_render_hygiene_section_groups_retraction_by_bibkey():
    """A retracted paper cited at multiple sites groups into one entry with
    the cite-site count."""
    v1 = _v_retracted(sentence="We follow prior work [retracted_paper].")
    v2 = _v_retracted(sentence="As shown in earlier studies [retracted_paper].")
    paper = IngestedPaper(
        title="A Paper",
        references={v1.reference.ref_id: v1.reference},
        citations=[v1.citation, v2.citation],
    )
    bundle = ReviewBundle(paper=paper, verifications=[v1, v2], model="m", synthesis_model="s")
    md = syn.render_hygiene_section(bundle)
    assert md is not None
    assert "Retracted citations (1)" in md
    assert "2 cite sites" in md


def test_render_resolved_record_marks_retraction_in_citation_issues():
    """When a retracted paper is also flagged for non-support, the resolved-
    record block carries the ⚠ RETRACTED line so the reader doesn't have to
    cross-check the Hygiene section."""
    ref = Reference(
        ref_id="bad",
        raw_text="[bad] Wakefield. Retracted. 1998.",
        authors=["A. Wakefield"],
        title="Retracted",
    )
    canonical = CanonicalRecord(
        source="openalex",
        title="Retracted",
        authors=["A. Wakefield"],
        year=1998,
        doi="10.1234/retracted",
        is_retracted=True,
    )
    v = VerificationResult(
        ref_id="bad",
        citation=Citation(ref_id="bad", sentence="A claim [bad]."),
        reference=ref,
        canonical=canonical,
        verdict=Verdict.DOES_NOT_SUPPORT,
        rationale="Cited paper does not back the claim.",
        abstract_only=False,
    )
    bundle = _make_bundle([v])
    md = syn.render_citation_issues(bundle)
    assert "⚠ **RETRACTED**" in md
    assert "Does not support" in md


def test_render_hygiene_section_lists_unreachable_urls():
    """An unreachable URL surfaced from the .tex/.bib should land in Hygiene."""
    paper = IngestedPaper(
        title="A Paper",
        link_checks=[
            LinkCheck(url="https://ok.example.org/a", source="tex_url", status=200, ok=True),
            LinkCheck(
                url="https://gone.example.org/x",
                source="bib_url",
                bibkey="r1",
                status=404,
                ok=False,
                error="HTTP 404",
            ),
            LinkCheck(
                url="https://broken.example.org/z",
                source="tex_href",
                ok=False,
                error="connection error: ...",
            ),
        ],
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    md = syn.render_hygiene_section(bundle)
    assert md is not None
    assert "Unreachable URLs (2 of 3)" in md
    # The healthy URL should NOT appear in the failures list.
    assert "ok.example.org" not in md
    # Both broken URLs surface, with their origin and reason.
    assert "https://gone.example.org/x" in md
    assert "404" in md
    assert "r1" in md  # bibkey carried through
    assert "https://broken.example.org/z" in md
    assert "connection error" in md


def test_render_hygiene_section_omits_link_block_when_all_pass():
    paper = IngestedPaper(
        title="t",
        link_checks=[
            LinkCheck(url="https://ok.example.org/a", source="tex_url", status=200, ok=True),
        ],
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    md = syn.render_hygiene_section(bundle)
    assert md is None  # all healthy + no other hygiene issues -> no section


def test_render_methodology_includes_link_check_summary():
    paper = IngestedPaper(
        title="t",
        link_checks=[
            LinkCheck(url="https://ok.example.org/a", source="tex_url", status=200, ok=True),
            LinkCheck(url="https://gone.example.org/x", source="tex_url", status=404, ok=False, error="HTTP 404"),
        ],
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    md = syn.render_methodology(bundle)
    assert "1 of 2 URLs unreachable" in md


def test_render_methodology_omits_link_check_summary_when_no_urls():
    paper = IngestedPaper(title="t")
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    md = syn.render_methodology(bundle)
    assert "URL" not in md  # no URL clause when there's nothing to check


def test_render_methodology_includes_retraction_count():
    v = _v_retracted()
    paper = IngestedPaper(
        title="A Paper",
        references={v.reference.ref_id: v.reference},
        citations=[v.citation],
    )
    bundle = ReviewBundle(paper=paper, verifications=[v], model="m", synthesis_model="s")
    md = syn.render_methodology(bundle)
    assert "1 retracted citation" in md
    assert "Retraction Watch" in md


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


# ---------------------------------------------------------------------------
# checklist section


def _checklist_bundle(*, found: bool, findings: list[ChecklistFinding]) -> ReviewBundle:
    paper = IngestedPaper(
        title="A Paper",
        references={},
        citations=[],
        checklist_found=found,
        checklist_findings=findings,
    )
    return ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")


def _one_of_each_kind() -> list[ChecklistFinding]:
    return [
        ChecklistFinding(
            kind=ChecklistFindingKind.UNANSWERED,
            section="General Paper Structure",
            question="Includes a conceptual outline of AI methods introduced",
        ),
        ChecklistFinding(
            kind=ChecklistFindingKind.INVALID_RESPONSE,
            section="General Paper Structure",
            question="Clearly delineates opinions from objective facts",
            response="maybe",
            detail="allowed: yes, no",
        ),
        ChecklistFinding(
            kind=ChecklistFindingKind.GATE_INCONSISTENCY,
            section="Theoretical Contributions",
            question="Does this paper make theoretical contributions?",
            response="no",
            detail='answered "no", but 1 of its sub-items carry substantive answers',
        ),
        ChecklistFinding(
            kind=ChecklistFindingKind.CLAIM_UNSUPPORTED,
            section="Computational Experiments",
            question="All source code will be made publicly available",
            response="yes",
            detail="no a public repository URL (e.g. github.com / zenodo.org) found in the paper — verify",
        ),
    ]


def test_render_checklist_section_groups_all_kinds():
    bundle = _checklist_bundle(found=True, findings=_one_of_each_kind())
    md = syn.render_checklist_section(bundle)
    assert md is not None
    assert "## Reproducibility checklist" in md
    for heading in (
        "Unanswered items (1)",
        "Invalid responses (1)",
        "Gate inconsistencies (1)",
        "Answers not supported by the paper (1)",
    ):
        assert heading in md
    # Each finding's question is quoted.
    assert "Includes a conceptual outline of AI methods introduced" in md
    assert "Clearly delineates opinions from objective facts" in md
    assert "Does this paper make theoretical contributions?" in md
    assert "All source code will be made publicly available" in md


def test_render_checklist_section_tier2_is_advisory_not_accusatory():
    findings = [f for f in _one_of_each_kind() if f.kind == ChecklistFindingKind.CLAIM_UNSUPPORTED]
    bundle = _checklist_bundle(found=True, findings=findings)
    md = syn.render_checklist_section(bundle)
    assert md is not None
    assert "verify" in md.lower()
    # No accusatory language.
    for bad in ("lied", "false", "dishonest", "untruthful"):
        assert bad not in md.lower()


def test_render_checklist_section_clean_when_found_but_no_findings():
    bundle = _checklist_bundle(found=True, findings=[])
    md = syn.render_checklist_section(bundle)
    assert md is not None
    assert "## Reproducibility checklist" in md
    assert "no unanswered" in md.lower() or "no issues" in md.lower()


def test_render_checklist_section_none_when_not_found():
    bundle = _checklist_bundle(found=False, findings=[])
    assert syn.render_checklist_section(bundle) is None


def test_stitch_omits_checklist_section_when_not_found():
    bundle = _checklist_bundle(found=False, findings=[])
    md = syn.stitch_review({"summary": "s"}, bundle)
    assert "## Reproducibility checklist" not in md


def test_stitch_includes_checklist_section_when_found():
    bundle = _checklist_bundle(found=True, findings=_one_of_each_kind())
    md = syn.stitch_review({"summary": "s"}, bundle)
    assert "## Reproducibility checklist" in md
    # Rendered after Hygiene (when present) and before Questions.
    assert md.index("## Reproducibility checklist") < md.index("## Questions for the author")


def test_render_methodology_checklist_line_found_and_flagged():
    bundle = _checklist_bundle(found=True, findings=_one_of_each_kind())
    md = syn.render_methodology(bundle)
    assert "reproducibility checklist was located and linted" in md.lower()
    assert "4 issue(s) flagged" in md


def test_render_methodology_checklist_line_found_and_clean():
    bundle = _checklist_bundle(found=True, findings=[])
    md = syn.render_methodology(bundle)
    assert "no issues flagged" in md.lower()


def test_render_methodology_omits_checklist_line_when_not_found():
    bundle = _checklist_bundle(found=False, findings=[])
    md = syn.render_methodology(bundle)
    assert "checklist" not in md.lower()


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
