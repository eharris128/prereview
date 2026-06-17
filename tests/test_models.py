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
    IngestedPaper,
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
