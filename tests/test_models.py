"""Tests for prereview.models — checklist model construction and defaults.

These are schema-shape guards: the checklist linter (TeX-only) adds fields to
IngestedPaper that the PDF path never sets, so the defaults must stay back-compat.
"""

from __future__ import annotations

import pytest

from prereview.models import (
    ChecklistFinding,
    ChecklistFindingKind,
    ChecklistItem,
    CoverageReport,
    IngestedPaper,
    Resolution,
    ResolutionStatus,
    ReviewBundle,
    Verdict,
)


@pytest.mark.parametrize("kind", list(ChecklistFindingKind))
def test_checklist_finding_round_trips_each_kind(kind: ChecklistFindingKind):
    f = ChecklistFinding(
        kind=kind,
        section="Computational Experiments",
        question="This paper states the number of algorithm runs",
        response="yes",
        detail="evidence sought: number of runs / random seeds",
    )
    dumped = f.model_dump()
    restored = ChecklistFinding(**dumped)
    assert restored == f
    assert restored.kind == kind


def test_checklist_finding_defaults():
    f = ChecklistFinding(kind=ChecklistFindingKind.UNANSWERED, question="Q?")
    assert f.section is None
    assert f.response == ""
    assert f.detail == ""


def test_checklist_item_defaults():
    item = ChecklistItem(question="Does this paper include computational experiments?")
    assert item.section is None
    assert item.options == []
    assert item.response == ""
    assert item.is_gate is False
    assert item.gate_question is None


def test_ingested_paper_defaults_checklist_fields():
    """The PDF path never touches the checklist fields, so a bare IngestedPaper
    must default to 'no checklist seen' — not crash, not look clean-but-found."""
    paper = IngestedPaper()
    assert paper.checklist_found is False
    assert paper.checklist_findings == []


def test_ingested_paper_defaults_structure_fields():
    """U1 fields must be back-compat: a bare IngestedPaper (the PDF path, or any
    existing construction site) leaves them all unset, never crashing."""
    paper = IngestedPaper()
    assert paper.author_block is None
    assert paper.acknowledgments is None
    assert paper.section_titles == []
    assert paper.page_count is None
    assert paper.references_start_page is None


def test_ingested_paper_structure_fields_round_trip():
    paper = IngestedPaper(
        author_block="Jane Smith",
        acknowledgments="We thank the lab.",
        section_titles=["Introduction", "Method"],
        page_count=9,
        references_start_page=7,
    )
    restored = IngestedPaper(**paper.model_dump())
    assert restored.author_block == "Jane Smith"
    assert restored.acknowledgments == "We thank the lab."
    assert restored.section_titles == ["Introduction", "Method"]
    assert restored.page_count == 9
    assert restored.references_start_page == 7


# ---------------------------------------------------------------------------
# Robustness models (recover-then-disclose): Resolution / Verdict / CoverageReport


@pytest.mark.parametrize("status", list(ResolutionStatus))
def test_resolution_round_trips_each_status(status: ResolutionStatus):
    res = Resolution(status=status)
    restored = Resolution(**res.model_dump())
    assert restored.status == status
    assert restored.record is None  # record optional, defaults None regardless of status


def test_verification_unavailable_is_distinct_verdict():
    """The infrastructure-failure verdict must be its own value, not collapsed into an
    honest abstention or a genuine ghost — the whole de-conflation depends on it."""
    assert Verdict.VERIFICATION_UNAVAILABLE.value == "verification_unavailable"
    assert Verdict.VERIFICATION_UNAVAILABLE is not Verdict.ABSTRACT_TOO_THIN
    assert Verdict.VERIFICATION_UNAVAILABLE is not Verdict.TARGET_UNAVAILABLE


def test_coverage_report_defaults_have_no_gap():
    report = CoverageReport()
    assert report.has_coverage_gap is False
    assert report.ghost_unresolved == 0
    assert report.circuit_broken_sources == []
    assert report.synthesis_degraded is False


@pytest.mark.parametrize("field", ["resolution_degraded", "verification_degraded"])
def test_coverage_report_gap_flags_on_degraded(field: str):
    report = CoverageReport(**{field: 1})
    assert report.has_coverage_gap is True


def test_coverage_report_resolved_only_has_no_gap():
    report = CoverageReport(references_parsed=10, citations_checked=10, resolved=10)
    assert report.has_coverage_gap is False


def test_review_bundle_defaults_coverage_none():
    """Back-compat: existing construction paths that don't set coverage must not break."""
    bundle = ReviewBundle(
        paper=IngestedPaper(),
        verifications=[],
        model="anthropic/claude-sonnet-4-6",
        synthesis_model="anthropic/claude-opus-4-7",
    )
    assert bundle.coverage is None
