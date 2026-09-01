"""Tests for prereview.synthesize.

The deterministic sections (Citation issues, Methodology) and the stitching
logic are tested directly. The LLM prose pass is mocked.
"""

from __future__ import annotations

import pytest

from prereview import synthesize as syn
from prereview.models import (
    AnonymizationFinding,
    AnonymizationFindingKind,
    BrokenRef,
    CanonicalRecord,
    ChecklistFinding,
    ChecklistFindingKind,
    Citation,
    CitationRole,
    CoverageReport,
    IngestedPaper,
    LinkCheck,
    NumericFinding,
    NumericFindingKind,
    OpenReviewInfo,
    Reference,
    ReviewBundle,
    SubmissionFinding,
    SubmissionFindingKind,
    SubmissionSeverity,
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


# ---------------------------------------------------------------------------
# U2: anonymization audit rendering


def _anon_bundle(*, checked: bool, findings=None, page_count=None) -> ReviewBundle:
    paper = IngestedPaper(
        title="A Paper",
        references={},
        citations=[],
        anonymization_checked=checked,
        anonymization_findings=findings or [],
        page_count=page_count,
    )
    return ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")


def _anon_findings():
    K = AnonymizationFindingKind
    return [
        AnonymizationFinding(kind=K.RESIDUAL_IDENTITY, evidence="Jane Smith", detail="verify the build is anonymized"),
        AnonymizationFinding(kind=K.SELF_REVEALING_PHRASE, evidence="In our previous work we showed", detail="verify it does not point to your identity"),
        AnonymizationFinding(kind=K.IDENTITY_URL, evidence="https://github.com/jsmith/repo", detail="verify it is anonymized"),
    ]


def test_render_anonymization_none_when_not_checked_on_tex():
    # --no-anonymize on a .tex (no page_count) → section absent.
    assert syn.render_anonymization_section(_anon_bundle(checked=False)) is None


def test_render_anonymization_note_on_pdf_input():
    # PDF input (page_count set) but audit not run → a one-line TeX-only note.
    md = syn.render_anonymization_section(_anon_bundle(checked=False, page_count=9))
    assert md is not None
    assert "## Anonymization audit" in md
    assert ".tex" in md and "PDF" in md


def test_render_anonymization_clean_when_checked_no_findings():
    md = syn.render_anonymization_section(_anon_bundle(checked=True, findings=[]))
    assert md is not None
    assert "## Anonymization audit" in md
    assert "verify" in md.lower()


def test_render_anonymization_groups_findings_advisory():
    md = syn.render_anonymization_section(_anon_bundle(checked=True, findings=_anon_findings()))
    assert md is not None
    assert "Residual identity blocks (1)" in md
    assert "Self-revealing phrasing (1)" in md
    assert "Identity-revealing URLs (1)" in md
    assert "Jane Smith" in md
    assert "github.com/jsmith" in md
    assert "verify" in md.lower()
    for bad in ("lied", "false", "dishonest", "untruthful"):
        assert bad not in md.lower()


def test_stitch_includes_anonymization_section_before_summary():
    bundle = _anon_bundle(checked=True, findings=_anon_findings())
    md = syn.stitch_review({"summary": "s"}, bundle)
    assert "## Anonymization audit" in md
    # Desk-reject guard leads the narrative.
    assert md.index("## Anonymization audit") < md.index("## Summary")


def test_stitch_omits_anonymization_section_when_not_checked():
    bundle = _anon_bundle(checked=False)
    md = syn.stitch_review({"summary": "s"}, bundle)
    assert "## Anonymization audit" not in md


# ---------------------------------------------------------------------------
# U3: submission-readiness rendering


def _submission_bundle(*, checked: bool, findings=None) -> ReviewBundle:
    paper = IngestedPaper(
        title="A Paper",
        references={},
        citations=[],
        submission_checked=checked,
        submission_findings=findings or [],
    )
    return ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")


def _submission_findings():
    K = SubmissionFindingKind
    S = SubmissionSeverity
    return [
        SubmissionFinding(kind=K.PLACEHOLDER_TITLE, severity=S.BLOCKER, detail="the title is empty — verify"),
        SubmissionFinding(kind=K.OVER_LENGTH, severity=S.WARNING, detail="approximately 9 pages — verify against the compiled PDF"),
    ]


def test_render_submission_none_when_not_checked():
    assert syn.render_submission_section(_submission_bundle(checked=False)) is None


def test_render_submission_clean_confirmation_when_checked_empty():
    md = syn.render_submission_section(_submission_bundle(checked=True, findings=[]))
    assert md is not None
    assert "## Submission readiness" in md
    assert "verify" in md.lower()


def test_render_submission_groups_blockers_and_warnings():
    md = syn.render_submission_section(_submission_bundle(checked=True, findings=_submission_findings()))
    assert md is not None
    assert "Blockers (1)" in md
    assert "Warnings (1)" in md
    assert "the title is empty" in md
    assert "approximately 9 pages" in md


def _changed_abstract_finding():
    return SubmissionFinding(
        kind=SubmissionFindingKind.CHANGED_ABSTRACT, severity=SubmissionSeverity.BLOCKER,
        detail="the abstract has changed substantially from the recorded baseline — verify",
    )


def test_render_submission_renders_venue_independent_findings_when_not_checked():
    # No --venue, but --abstract-baseline fired: the section must still surface it.
    md = syn.render_submission_section(
        _submission_bundle(checked=False, findings=[_changed_abstract_finding()])
    )
    assert md is not None
    assert "No `--venue` was selected" in md
    assert "Blockers (1)" in md
    assert "changed substantially" in md


def test_render_methodology_discloses_no_venue_selected():
    md = syn.render_methodology(_submission_bundle(checked=False))
    assert "No venue was selected" in md
    assert "Submission-readiness checks ran" not in md


def test_render_methodology_no_venue_counts_abstract_diff():
    md = syn.render_methodology(
        _submission_bundle(checked=False, findings=[_changed_abstract_finding()])
    )
    assert "abstract-baseline diff flagged 1 issue(s)" in md


def test_render_methodology_with_venue_keeps_ran_line():
    md = syn.render_methodology(_submission_bundle(checked=True))
    assert "Submission-readiness checks ran against the venue rules: no desk-reject issues flagged" in md


def test_stitch_orders_desk_reject_guards_then_summary():
    paper = IngestedPaper(
        title="A Paper", references={}, citations=[],
        anonymization_checked=True, anonymization_findings=[],
        submission_checked=True, submission_findings=_submission_findings(),
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    md = syn.stitch_review({"summary": "s"}, bundle)
    # Order: Anonymization → Submission readiness → Summary.
    assert md.index("## Anonymization audit") < md.index("## Submission readiness")
    assert md.index("## Submission readiness") < md.index("## Summary")


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
        "## Overall assessment",
        "## Summary",
        "## Strengths",
        "## Weaknesses",
        "## Citation issues",
        "## Questions for the author",
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

    # U5: the 1–10 rating is dropped from the default output; the review leads with
    # the severity-bucketed Overall assessment instead.
    assert "LLM-estimated rating" not in md
    assert md.index("## Overall assessment") < md.index("## Summary")

    # Methodology lists what the tool does NOT do.
    assert "Missing prior work" in md


# ---------------------------------------------------------------------------
# coverage & reliability section + graceful degradation (U7)


def test_coverage_section_none_when_all_resolved():
    """A clean run with nothing to disclose omits the coverage section entirely."""
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1"), _v(Verdict.SUPPORTS, ref_id="2")])
    assert syn.render_coverage_section(bundle) is None


def test_coverage_section_reports_degraded_and_keeps_it_out_of_citation_issues():
    """A VERIFICATION_UNAVAILABLE citation surfaces in the coverage section (named, framed
    as 'could not verify') and must NOT appear in Citation issues — it's a tool problem,
    not a paper defect."""
    verifs = [
        _v(Verdict.SUPPORTS, ref_id="1"),
        _v(Verdict.VERIFICATION_UNAVAILABLE, ref_id="2", canonical=False, rationale="API failed."),
    ]
    bundle = _make_bundle(verifs)
    cov = syn.render_coverage_section(bundle)
    assert cov is not None
    assert "could not be" in cov and "verified" in cov
    assert "`2`" in cov  # the degraded ref_id is named so the user can re-check it
    issues = syn.render_citation_issues(bundle)
    assert "`2`" not in issues  # excluded from paper-level findings


def test_coverage_section_reports_recovered_and_breaker():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport(
        recovered_after_retry=3, circuit_broken_sources=["semanticscholar"]
    )
    cov = syn.render_coverage_section(bundle)
    assert cov is not None
    assert "recovered on" in cov
    assert "semanticscholar" in cov


def test_methodology_counts_true_ghosts_not_degraded():
    """The unresolved count in the methodology section must reflect true ghosts only —
    an infrastructure-degraded citation is not a ghost."""
    verifs = [
        _v(Verdict.TARGET_UNAVAILABLE, ref_id="1", canonical=False),
        _v(Verdict.VERIFICATION_UNAVAILABLE, ref_id="2", canonical=False),
    ]
    bundle = _make_bundle(verifs)
    meth = syn.render_methodology(bundle)
    assert "1 bibliography entry did not" in meth  # 1 true ghost, not 2


@pytest.mark.asyncio
async def test_synthesize_review_survives_prose_failure(monkeypatch):
    """If the prose (LLM) pass fails, the deterministic sections still render and the
    coverage section discloses the gap — the review is never lost to an LLM blip."""
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport()

    async def boom(bundle, *, verbose=False):
        raise RuntimeError("opus down")

    monkeypatch.setattr(syn, "_generate_prose", boom)
    # run_reviewer2=False so this stays a pure prose-failure test (no real LLM call).
    md = await syn.synthesize_review(bundle, run_reviewer2=False)
    assert "# Pre-submission review" in md
    assert "Methodology and limits" in md  # deterministic section survived
    assert bundle.coverage.synthesis_degraded is True
    assert "could not be generated" in md  # coverage section discloses the prose failure


# ---------------------------------------------------------------------------
# U4: adversarial Reviewer-2 pass


def _reviewer2_payload():
    return {
        "findings": [
            {"dimension": "novelty", "severity": "critical", "issue": "The contribution collapses into prior work.", "evidence": "Section 3 restates an earlier objective.", "suggested_fix": "Differentiate explicitly."},
            {"dimension": "baselines", "severity": "major", "issue": "No SOTA baseline.", "evidence": "Table 2 omits the standard transformer baseline.", "suggested_fix": "Add the SOTA baseline."},
            {"dimension": "claims", "severity": "minor", "issue": "Slightly over-strong wording.", "evidence": "'always outperforms' in the abstract.", "suggested_fix": "Qualify the claim."},
            {"dimension": "reproducibility", "severity": "none", "issue": "no issue found", "evidence": "", "suggested_fix": ""},
            # clarity intentionally omitted → renderer still shows it as "No issue found".
        ]
    }


def test_render_reviewer2_groups_by_dimension_with_severity():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn.render_reviewer2_section(_reviewer2_payload(), bundle)
    assert md is not None
    assert "## Reviewer 2 (adversarial)" in md
    for heading in (
        "Novelty & positioning",
        "Baselines & experiments",
        "Claims vs. evidence",
        "Reproducibility",
        "Clarity",
    ):
        assert heading in md
    assert "**critical**" in md
    assert "**major**" in md
    # Dimensions with severity "none" or omitted entirely both render "No issue found".
    assert md.count("_No issue found._") >= 2  # reproducibility (none) + clarity (omitted)
    assert "Table 2 omits the standard transformer baseline." in md  # evidence quoted


def test_render_reviewer2_drops_verdict_and_score():
    """The model tries to sneak in an accept verdict + score; neither may leak — this
    is the anti-generosity guarantee (only known fields are rendered)."""
    payload = {
        "verdict": "accept",
        "overall_score": 8,
        "findings": [{"dimension": "novelty", "severity": "none", "issue": "no issue found"}],
    }
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn.render_reviewer2_section(payload, bundle)
    assert md is not None
    low = md.lower()
    assert "accept" not in low
    assert "reject" not in low


def test_render_reviewer2_none_when_not_run():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport()  # reviewer2_degraded is False
    assert syn.render_reviewer2_section(None, bundle) is None


def test_render_reviewer2_discloses_degradation():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport(reviewer2_degraded=True)
    md = syn.render_reviewer2_section(None, bundle)
    assert md is not None
    assert "Reviewer 2" in md
    assert "could not be generated" in md


def test_reviewer2_prompt_carries_rubric_and_grounding():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="goodref")])
    system = syn._REVIEWER2_SYSTEM
    prompt = syn._reviewer2_prompt(bundle)
    # Rubric anchors live in the system prompt (the anti-generosity mechanism).
    assert "critical:" in system.lower()
    assert "do not default" in system.lower()
    assert "sota baseline" in system.lower()
    assert "accept/reject" in system.lower()  # the no-verdict rule
    # Novelty grounding: the paper's own cited title reaches the prompt.
    assert "A Toy Paper" in prompt
    for dim in ("novelty", "baselines", "claims", "reproducibility", "clarity"):
        assert dim in prompt


def test_reviewer2_flag_summary_reaches_prompt_for_calibration():
    """A deterministic signal (submission findings) must reach the prompt so the model
    can calibrate severity against the tool's own flags."""
    paper = IngestedPaper(
        title="A Paper", abstract="x", sections=[("body", "b")],
        references={}, citations=[],
        submission_checked=True,
        submission_findings=[
            SubmissionFinding(kind=SubmissionFindingKind.OVER_LENGTH, severity=SubmissionSeverity.WARNING, detail="over length"),
        ],
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    prompt = syn._reviewer2_prompt(bundle)
    assert "submission-readiness issue" in prompt.lower()


def test_stitch_places_reviewer2_after_weaknesses_before_citations():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn.stitch_review({"summary": "s"}, bundle, reviewer2=_reviewer2_payload())
    assert "## Reviewer 2 (adversarial)" in md
    assert md.index("## Weaknesses") < md.index("## Reviewer 2 (adversarial)")
    assert md.index("## Reviewer 2 (adversarial)") < md.index("## Citation issues")


@pytest.mark.asyncio
async def test_reviewer2_failure_does_not_mislabel_prose(monkeypatch):
    """A Reviewer-2-only failure sets reviewer2_degraded, NOT synthesis_degraded, and
    must not print the prose/narrative-degradation message."""
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport()

    async def good_prose(bundle, *, verbose=False):
        return {
            "summary": "ok", "strengths": ["s"], "weaknesses": ["w"], "questions": ["q"],
            "rating_low": 5, "rating_high": 6, "rating_justification": "LLM judgement.",
        }

    async def boom_reviewer2(bundle, *, verbose=False):
        raise RuntimeError("opus down")

    monkeypatch.setattr(syn, "_generate_prose", good_prose)
    monkeypatch.setattr(syn, "_generate_reviewer2", boom_reviewer2)
    md = await syn.synthesize_review(bundle)

    assert bundle.coverage.reviewer2_degraded is True
    assert bundle.coverage.synthesis_degraded is False
    assert "Reviewer 2" in md and "could not be generated" in md
    assert "narrative sections" not in md  # the prose sections are NOT mislabeled


# ---------------------------------------------------------------------------
# U5: rating reconciliation → severity-bucketed overall assessment


def test_overall_assessment_headline_counts():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    reviewer2 = {"findings": [
        {"dimension": "novelty", "severity": "critical", "issue": "a", "evidence": "e1"},
        {"dimension": "baselines", "severity": "major", "issue": "b", "evidence": "e2"},
        {"dimension": "claims", "severity": "major", "issue": "c", "evidence": "e3"},
    ]}
    md = syn._render_overall_assessment({}, bundle, reviewer2, show_rating=False)
    assert "critical: 1 · major: 2 · minor: 0" in md
    assert "LLM-estimated rating" not in md  # no rating number by default


def test_overall_assessment_no_findings_headline():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn._render_overall_assessment({}, bundle, {"findings": []}, show_rating=False)
    assert "no critical or major concerns" in md.lower()


def test_overall_assessment_dedups_same_issue_across_sources():
    """One underlying issue surfacing as both a submission blocker and a Reviewer-2
    critical (same quoted element) is counted once, not twice."""
    paper = IngestedPaper(
        title="P", references={}, citations=[],
        submission_checked=True,
        submission_findings=[
            SubmissionFinding(
                kind=SubmissionFindingKind.COLOR_TABLE, severity=SubmissionSeverity.BLOCKER,
                detail="color table", evidence="cellcolor in Table 2",
            )
        ],
    )
    bundle = ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")
    reviewer2 = {"findings": [
        {"dimension": "clarity", "severity": "critical", "issue": "colored table", "evidence": "cellcolor in Table 2"},
    ]}
    counts = syn._overall_assessment(reviewer2, bundle)
    assert counts["critical"] == 1  # deduped — NOT the sum (2)


def test_stitch_default_omits_rating_number():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn.stitch_review(
        {"rating_low": 8, "rating_high": 9, "rating_justification": "j"}, bundle
    )
    assert "## Overall assessment" in md
    assert "LLM-estimated rating" not in md


def test_stitch_show_rating_includes_caveated_number():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    md = syn.stitch_review(
        {"rating_low": 8, "rating_high": 9, "rating_justification": "j"}, bundle, show_rating=True
    )
    assert "LLM-estimated rating" in md
    assert "8–9/10" in md
    assert "keuper" in md.lower() or "skew generous" in md.lower()


def test_overall_assessment_leads_with_severity_not_number():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    reviewer2 = {"findings": [{"dimension": "novelty", "severity": "critical", "issue": "x", "evidence": "e"}]}
    md = syn._render_overall_assessment(
        {"rating_low": 9, "rating_high": 10, "rating_justification": "j"}, bundle, reviewer2, show_rating=True
    )
    # Even with the number shown, the severity headline comes first and the number
    # is labeled secondary with the caveat.
    assert md.index("critical:") < md.index("/10")
    assert "secondary" in md.lower()


# ---------------------------------------------------------------------------
# U6: numerical-sanity rendering + methodology coherence


def _numeric_bundle(findings=None, *, checked=True) -> ReviewBundle:
    paper = IngestedPaper(
        title="P", references={}, citations=[],
        numeric_checked=checked, numeric_findings=findings or [],
    )
    return ReviewBundle(paper=paper, verifications=[], model="m", synthesis_model="s")


def test_render_numeric_none_when_no_findings():
    assert syn.render_numeric_section(_numeric_bundle([])) is None


def test_render_numeric_groups_and_advisory():
    K = NumericFindingKind
    findings = [
        NumericFinding(kind=K.BOUNDED_METRIC, detail="an accuracy of 102% exceeds the 100% ceiling", evidence="accuracy of 102%"),
        NumericFinding(kind=K.SPLIT_MISMATCH, detail="train+val+test does not equal the stated total"),
    ]
    md = syn.render_numeric_section(_numeric_bundle(findings))
    assert md is not None
    assert "## Numerical sanity" in md
    assert "Out-of-range metrics (1)" in md
    assert "Dataset split arithmetic (1)" in md
    assert "verify" in md.lower()
    assert "accuracy of 102%" in md  # evidence quoted


def test_stitch_includes_numeric_section():
    bundle = _numeric_bundle([NumericFinding(kind=NumericFindingKind.BOUNDED_METRIC, detail="d", evidence="e")])
    md = syn.stitch_review({"summary": "s"}, bundle)
    assert "## Numerical sanity" in md


def test_methodology_qualifies_stats_bullet_once_numeric_ran():
    """Shipping U6 makes the blanket 'does not check reported statistics' promise
    false — the methodology must no longer claim it."""
    meth = syn.render_methodology(_numeric_bundle([], checked=True))
    assert "Reported statistics (no statcheck / GRIM-style audit)." not in meth
    assert "Numerical sanity section above checks" in meth


def test_methodology_keeps_stats_bullet_when_numeric_off():
    meth = syn.render_methodology(_numeric_bundle([], checked=False))
    assert "Reported statistics (no statcheck / GRIM-style audit)." in meth


# ---------------------------------------------------------------------------
# U8: OpenReview decision rendering


def test_render_openreview_section_annotates_decision():
    v = _v(Verdict.SUPPORTS, ref_id="found")
    v.openreview = OpenReviewInfo(
        decision="Reject", rating_avg=3.0, rating_count=4, url="https://openreview.net/forum?id=abc"
    )
    md = syn.render_openreview_section(_make_bundle([v]))
    assert md is not None
    assert "## OpenReview decisions" in md
    assert "Reject" in md
    assert "verify" in md.lower()  # advisory framing
    assert "openreview.net/forum?id=abc" in md


def test_render_openreview_none_when_no_enrichment():
    assert syn.render_openreview_section(_make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])) is None


def test_coverage_section_discloses_openreview_degradation():
    bundle = _make_bundle([_v(Verdict.SUPPORTS, ref_id="1")])
    bundle.coverage = CoverageReport(openreview_degraded=True)
    cov = syn.render_coverage_section(bundle)
    assert cov is not None
    assert "OpenReview enrichment could not be completed" in cov


# ---------------------------------------------------------------------------
# resolver guard + outdated-arXiv rendering


def _v_for_hygiene(canonical, *, ref_id="r", reference=None) -> VerificationResult:
    from prereview.models import Citation as _Cit, Reference as _Ref

    ref = reference or _Ref(ref_id=ref_id, raw_text="x", authors=["Alice Smith"], title="A Toy Paper", year=2017)
    return VerificationResult(
        ref_id=ref_id,
        citation=_Cit(ref_id=ref_id, sentence="We use toys [r]."),
        reference=ref,
        canonical=canonical,
        verdict=Verdict.SUPPORTS,
        rationale="ok",
        abstract_only=False,
        role=CitationRole.METHOD_ATTRIBUTION,
    )


def _weak_record():
    return CanonicalRecord(
        source="crossref", title="A Toy Paper", authors=["Alice Smith"], year=2025,
        venue="Mirror Journal", doi="10.9999/knockoff",
        match_note="Matched on title and author surnames only: the bibliography gives 2017 but the resolved record is dated 2025 (Mirror Journal).",
    )


def test_render_hygiene_section_lists_weak_matches():
    md = syn.render_hygiene_section(_make_bundle([_v_for_hygiene(_weak_record())]))
    assert md is not None
    assert "### Resolved on title and authors only — year differs (1)" in md
    assert "`r` — bibliography: 2017 · resolved: 2025, Mirror Journal (DOI [10.9999/knockoff]" in md


def test_render_resolved_record_shows_weak_match_note():
    from prereview.synthesize import _render_resolved_record

    lines = _render_resolved_record(_weak_record())
    assert any("Weak match" in ln and "2025" in ln for ln in lines)


def test_render_hygiene_section_lists_outdated_arxiv_only_for_arxiv_cited_entries():
    from prereview.models import PublishedVersion, Reference as _Ref

    pv = PublishedVersion(venue="Neural Information Processing Systems", year=2017, source="semanticscholar")
    arxiv_cited = _Ref(
        ref_id="vaswani", raw_text="x", authors=["Ashish Vaswani"], title="Attention Is All You Need",
        year=2017, venue="arXiv preprint arXiv:1706.03762", arxiv_id="1706.03762",
    )
    doi_cited = _Ref(
        ref_id="proper", raw_text="x", authors=["Ashish Vaswani"], title="Attention Is All You Need",
        year=2017, venue="NeurIPS", doi="10.5555/3295222.3295349",
    )
    rec = lambda: CanonicalRecord(  # noqa: E731
        source="semanticscholar", title="Attention Is All You Need", authors=["Ashish Vaswani"],
        year=2017, venue="Neural Information Processing Systems",
        published_version=pv, published_version_checked=True,
    )
    md = syn.render_hygiene_section(_make_bundle([
        _v_for_hygiene(rec(), ref_id="vaswani", reference=arxiv_cited),
        _v_for_hygiene(rec(), ref_id="proper", reference=doi_cited),
    ]))
    assert md is not None
    assert "### arXiv preprints with a published version (1)" in md
    assert "`vaswani` — cited as arXiv:1706.03762 (2017) → published: **Neural Information Processing Systems** (2017)" in md
    assert "`proper`" not in md


def test_render_hygiene_section_none_when_published_version_but_not_arxiv_cited():
    from prereview.models import PublishedVersion, Reference as _Ref

    ref = _Ref(ref_id="p", raw_text="x", authors=["A B"], title="T", year=2020, doi="10.1/x")
    rec = CanonicalRecord(source="crossref", title="T", authors=["A B"], year=2020, venue="J",
                          published_version=PublishedVersion(venue="J", year=2020), published_version_checked=True)
    assert syn.render_hygiene_section(_make_bundle([_v_for_hygiene(rec, ref_id="p", reference=ref)])) is None


def test_methodology_counts_weak_matches_and_outdated_arxiv():
    from prereview.models import PublishedVersion, Reference as _Ref

    arxiv_cited = _Ref(ref_id="a", raw_text="x", authors=["Alice Smith"], title="A Toy Paper", year=2017, arxiv_id="1706.03762")
    outdated = CanonicalRecord(source="arxiv", title="A Toy Paper", authors=["Alice Smith"], year=2017, venue="arXiv",
                               published_version=PublishedVersion(venue="NeurIPS", year=2017), published_version_checked=True)
    md = syn.render_methodology(_make_bundle([
        _v_for_hygiene(_weak_record(), ref_id="w"),
        _v_for_hygiene(outdated, ref_id="a", reference=arxiv_cited),
    ]))
    assert "1 resolved on title/author only (year differs)" in md
    assert "1 arXiv-cited entry with a published version" in md


def test_coverage_section_discloses_unchecked_published_versions():
    from prereview.models import CoverageReport

    bundle = _make_bundle([_v(Verdict.SUPPORTS)])
    bundle.coverage = CoverageReport(published_version_unchecked=2)
    assert bundle.coverage.has_coverage_gap is False  # an enrichment gap never drives exit 3
    md = syn.render_coverage_section(bundle)
    assert md is not None
    assert "2 arXiv-cited entries could not be checked for a published version (Semantic Scholar / DBLP / Crossref" in md
    assert "NOT a finding about the paper" in md
