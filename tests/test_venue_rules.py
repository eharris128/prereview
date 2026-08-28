"""Tests for prereview.venue_rules — submission-readiness detectors.

Deterministic, so concrete outputs are asserted. The precision contract: an
uncertain length measurement degrades to a warning (never a false hard block),
and a color macro in body prose is not mistaken for a color-coded table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prereview.models import (
    AnonymizationFinding,
    AnonymizationFindingKind,
    ChecklistFinding,
    ChecklistFindingKind,
    IngestedPaper,
    SubmissionFindingKind,
    SubmissionSeverity,
)
from prereview.venue_rules import (
    DEFAULT_VENUE,
    VENUE_RULES,
    VenueRules,
    audit_submission_pdf,
    audit_submission_tex,
    check_changed_abstract,
    check_checklist_completeness,
    check_color_tables,
    check_length,
    check_placeholder_abstract,
    check_placeholder_title,
    collect_gate_blockers,
    get_rules,
)

# A synthetic venue carrying the values the (since-removed) AAAI-27 entry had, so the
# detector contracts below stay pinned to concrete numbers.
RULES = VenueRules(
    venue_id="test-venue",
    page_limit_technical=7,
    references_excluded=True,
    requires_abstract=True,
    min_abstract_words=20,
    checklist_required=True,
    placeholder_markers=(
        "type your", "todo", "tbd", "lorem ipsum", "xxx", "goes here", "placeholder",
        "your title here", "your abstract here", "fixme", "title here", "abstract here",
    ),
    color_table_macros=(r"\cellcolor", r"\rowcolor", r"\columncolor", r"\colorbox", r"\textcolor"),
)
_K = SubmissionFindingKind
_S = SubmissionSeverity


def _one(findings, kind):
    matches = [f for f in findings if f.kind == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, got {findings}"
    return matches[0]


# ---------------------------------------------------------------------------
# length (PDF only, approximate)


def test_length_over_limit_with_isolated_boundary_is_blocker():
    # refs start page 8 → technical pages = 8 > 7 → over-length blocker.
    findings = check_length(RULES, page_count=9, references_start_page=8)
    f = _one(findings, _K.OVER_LENGTH)
    assert f.severity == _S.BLOCKER
    assert "8" in f.detail


def test_length_at_limit_is_clean():
    # refs start page 7 → 7 technical pages == limit → no finding.
    assert check_length(RULES, page_count=9, references_start_page=7) == []


def test_length_uncertain_boundary_degrades_to_warning():
    # No isolable references boundary → warning using the whole PDF, never a blocker.
    findings = check_length(RULES, page_count=9, references_start_page=None)
    f = _one(findings, _K.OVER_LENGTH)
    assert f.severity == _S.WARNING


def test_length_unmeasurable_on_tex_yields_nothing():
    assert check_length(RULES, page_count=None, references_start_page=None) == []


def test_length_short_pdf_with_unknown_boundary_is_clean():
    assert check_length(RULES, page_count=6, references_start_page=None) == []


# ---------------------------------------------------------------------------
# placeholder / empty title & abstract


def test_empty_title_is_blocker_in_strict_mode():
    f = _one(check_placeholder_title(RULES, "", strict=True), _K.PLACEHOLDER_TITLE)
    assert f.severity == _S.BLOCKER


def test_placeholder_marker_title_is_blocker():
    f = _one(check_placeholder_title(RULES, "Your Title Here", strict=True), _K.PLACEHOLDER_TITLE)
    assert f.severity == _S.BLOCKER


def test_real_title_is_clean():
    assert check_placeholder_title(RULES, "A Robust Method for Graph Learning", strict=True) == []


def test_empty_title_not_blocked_in_nonstrict_pdf_mode():
    assert check_placeholder_title(RULES, "", strict=False) == []


def test_placeholder_marker_abstract_is_blocker():
    f = _one(check_placeholder_abstract(RULES, "Abstract goes here.", strict=True), _K.PLACEHOLDER_ABSTRACT)
    assert f.severity == _S.BLOCKER


def test_too_short_abstract_is_blocker_in_strict_mode():
    f = _one(check_placeholder_abstract(RULES, "We do things.", strict=True), _K.PLACEHOLDER_ABSTRACT)
    assert f.severity == _S.BLOCKER


def test_real_abstract_is_clean():
    abstract = " ".join(["word"] * 60)
    assert check_placeholder_abstract(RULES, abstract, strict=True) == []


def test_short_abstract_not_blocked_in_nonstrict_pdf_mode():
    # PDF abstract extraction is unreliable, so the too-short heuristic is off there.
    assert check_placeholder_abstract(RULES, "We do things.", strict=False) == []


# ---------------------------------------------------------------------------
# changed-abstract snapshot/diff


def test_changed_abstract_first_run_snapshots_silently(tmp_path: Path):
    baseline = tmp_path / "abstract.txt"
    assert check_changed_abstract("Our method reaches 91.2 accuracy.", baseline) == []
    assert baseline.exists()  # snapshot written


def test_changed_abstract_flags_number_edit(tmp_path: Path):
    baseline = tmp_path / "abstract.txt"
    check_changed_abstract("Our method reaches 91.2 accuracy on the benchmark.", baseline)
    findings = check_changed_abstract("Our method reaches 95.8 accuracy on the benchmark.", baseline)
    f = _one(findings, _K.CHANGED_ABSTRACT)
    assert f.severity == _S.BLOCKER


def test_changed_abstract_unchanged_is_clean(tmp_path: Path):
    baseline = tmp_path / "abstract.txt"
    text = "Our method reaches 91.2 accuracy on the benchmark."
    check_changed_abstract(text, baseline)
    assert check_changed_abstract(text, baseline) == []


def test_changed_abstract_none_path_is_noop():
    assert check_changed_abstract("anything", None) == []


# ---------------------------------------------------------------------------
# color result tables


def test_color_macro_inside_tabular_is_blocker():
    tex = r"""
\begin{tabular}{ll}
\cellcolor{green!25} 91.2 & baseline \\
\end{tabular}
"""
    f = _one(check_color_tables(RULES, tex), _K.COLOR_TABLE)
    assert f.severity == _S.BLOCKER
    assert "cellcolor" in f.detail


def test_textcolor_in_body_prose_is_not_a_color_table():
    tex = r"In the text we say \textcolor{red}{this is important} but no table is colored."
    assert check_color_tables(RULES, tex) == []


# ---------------------------------------------------------------------------
# checklist completeness


def test_unanswered_checklist_item_is_blocker():
    findings = check_checklist_completeness(
        RULES,
        checklist_found=True,
        checklist_findings=[
            ChecklistFinding(kind=ChecklistFindingKind.UNANSWERED, question="Q?"),
        ],
    )
    f = _one(findings, _K.CHECKLIST_INCOMPLETE)
    assert f.severity == _S.BLOCKER


def test_no_checklist_means_no_completeness_finding():
    assert check_checklist_completeness(RULES, checklist_found=False, checklist_findings=[]) == []


# ---------------------------------------------------------------------------
# orchestrator + gate aggregation


def test_audit_submission_tex_combines_source_checks():
    tex = r"""\title{}
\begin{tabular}{l}\rowcolor{gray} x \\\end{tabular}
"""
    findings = audit_submission_tex(
        RULES, title="", abstract="We do things.", tex_text=tex,
        checklist_found=False, checklist_findings=[], abstract_baseline=None,
    )
    kinds = {f.kind for f in findings}
    assert _K.PLACEHOLDER_TITLE in kinds
    assert _K.PLACEHOLDER_ABSTRACT in kinds
    assert _K.COLOR_TABLE in kinds


def test_collect_gate_blockers_spans_anonymization_and_submission():
    paper = IngestedPaper(
        anonymization_findings=[
            AnonymizationFinding(
                kind=AnonymizationFindingKind.RESIDUAL_IDENTITY,
                evidence="Jane Smith",
                detail="verify the build is anonymized",
            ),
            AnonymizationFinding(
                kind=AnonymizationFindingKind.SELF_REVEALING_PHRASE,  # not a gate blocker
                evidence="in our previous work",
                detail="verify",
            ),
        ],
        submission_findings=audit_submission_tex(
            RULES, title="", abstract=" ".join(["w"] * 60), tex_text="",
            checklist_found=False, checklist_findings=[], abstract_baseline=None,
        ),
    )
    blockers = collect_gate_blockers(paper)
    # Residual identity (1) + empty-title blocker (1); the self-revealing phrase
    # and any warnings do NOT gate.
    assert len(blockers) == 2
    assert any("residual author identity" in b for b in blockers)


def test_collect_gate_blockers_empty_when_clean():
    assert collect_gate_blockers(IngestedPaper()) == []


# ---------------------------------------------------------------------------
# venue table / no-venue path


def test_default_venue_is_none_and_aaai27_is_gone():
    # AAAI-27 was removed on 2026-08-28 after its deadline passed; the guard is now
    # opt-in per venue and nothing is configured until the next target is added.
    assert DEFAULT_VENUE is None
    assert "aaai-27" not in VENUE_RULES


def test_get_rules_none_returns_none():
    assert get_rules(None) is None


def test_get_rules_unknown_venue_raises_naming_it():
    with pytest.raises(ValueError, match="unknown venue 'nope-99'"):
        get_rules("nope-99")


def test_get_rules_configured_venue(monkeypatch):
    monkeypatch.setitem(VENUE_RULES, "test-venue", RULES)
    assert get_rules("test-venue") is RULES


def test_audit_tex_without_venue_runs_only_abstract_diff(tmp_path: Path):
    baseline = tmp_path / "abstract.txt"
    baseline.write_text("Our method reaches 91.2 accuracy on the benchmark.")
    tex = r"""\title{}
\begin{tabular}{l}\rowcolor{gray} x \\\end{tabular}
"""
    findings = audit_submission_tex(
        None, title="", abstract="Our method reaches 95.8 accuracy on the benchmark.",
        tex_text=tex, checklist_found=True,
        checklist_findings=[ChecklistFinding(kind=ChecklistFindingKind.UNANSWERED, question="Q?")],
        abstract_baseline=baseline,
    )
    # Empty title, color table, and the unanswered checklist item all need a venue;
    # only the venue-independent changed-abstract diff fires.
    assert [f.kind for f in findings] == [_K.CHANGED_ABSTRACT]


def test_audit_tex_without_venue_and_no_baseline_is_empty():
    assert audit_submission_tex(
        None, title="", abstract="", tex_text=r"\rowcolor{gray}",
        checklist_found=False, checklist_findings=[], abstract_baseline=None,
    ) == []


def test_audit_pdf_without_venue_is_empty():
    assert audit_submission_pdf(None, title="", page_count=40, references_start_page=39) == []
