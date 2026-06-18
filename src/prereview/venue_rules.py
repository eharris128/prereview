"""Submission-readiness / venue-rules guard (AAAI-27 first).

Why this exists: AAAI summarily desk-rejects papers for mechanically-detectable
problems — over-length technical content, a placeholder or changed title/abstract,
a color-coded result table, or an incomplete reproducibility checklist. Catching
these before submission is pure expected value: a tripped rule means acceptance
probability zero regardless of quality.

Design (KTD-2): the *detectors are generic* and the *per-venue facts are data*
(:data:`VENUE_RULES`), the same "generic parser, venue-specific data" decision the
checklist linter made — adding NeurIPS/ACL later is data, not code. Everything is
deterministic and advisory; only ``BLOCKER``-severity findings change the exit code,
and only under ``--gate`` (KTD-8).

Length is PDF-only and approximate (KTD-4): two-column references routinely start
mid-page, so the technical-page count is ±1. When the references boundary is not
isolable the finding degrades to a warning (never a false hard block), and on TeX
input length is unmeasurable, so no length finding is produced at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import (
    AnonymizationFindingKind,
    ChecklistFinding,
    ChecklistFindingKind,
    IngestedPaper,
    SubmissionFinding,
    SubmissionFindingKind,
    SubmissionSeverity,
)

_Kind = SubmissionFindingKind
_Sev = SubmissionSeverity


# ---------------------------------------------------------------------------
# venue data table


@dataclass(frozen=True)
class VenueRules:
    venue_id: str
    page_limit_technical: int  # max pages of technical content (references excluded)
    references_excluded: bool
    requires_abstract: bool
    min_abstract_words: int  # below this, a non-empty abstract reads as a placeholder
    checklist_required: bool
    placeholder_markers: tuple[str, ...]  # lowercased substrings that betray a stub
    color_table_macros: tuple[str, ...]  # color macros AAAI press rules reject in tables


VENUE_RULES: dict[str, VenueRules] = {
    "aaai-27": VenueRules(
        venue_id="aaai-27",
        page_limit_technical=7,
        references_excluded=True,
        requires_abstract=True,
        min_abstract_words=20,
        checklist_required=True,
        placeholder_markers=(
            "type your",
            "todo",
            "tbd",
            "lorem ipsum",
            "xxx",
            "goes here",
            "placeholder",
            "your title here",
            "your abstract here",
            "fixme",
            "title here",
            "abstract here",
        ),
        color_table_macros=(
            r"\cellcolor",
            r"\rowcolor",
            r"\columncolor",
            r"\colorbox",
            r"\textcolor",
        ),
    ),
}

DEFAULT_VENUE = "aaai-27"


def get_rules(venue: str) -> VenueRules:
    return VENUE_RULES[venue]


# ---------------------------------------------------------------------------
# small helpers


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _has_placeholder(text: str, markers: tuple[str, ...]) -> Optional[str]:
    low = text.lower()
    for marker in markers:
        if marker in low:
            return marker
    return None


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _numbers(s: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


# ---------------------------------------------------------------------------
# detectors (each pure and individually testable)


def check_length(
    rules: VenueRules, page_count: Optional[int], references_start_page: Optional[int]
) -> list[SubmissionFinding]:
    """Over-length check (PDF only). technical pages = references_start_page (when
    isolated) else the whole PDF; an un-isolable boundary degrades to a warning."""
    if page_count is None:
        return []  # TeX / unknown — length is unmeasurable, no finding (renderer notes it)
    isolated = references_start_page is not None
    technical = references_start_page if isolated else page_count
    if technical <= rules.page_limit_technical:
        return []
    if isolated:
        return [
            SubmissionFinding(
                kind=_Kind.OVER_LENGTH,
                severity=_Sev.BLOCKER,
                detail=(
                    f"approximately {technical} technical pages (references heading detected "
                    f"on page {references_start_page}) exceeds the {rules.page_limit_technical}-page "
                    "limit — verify against the compiled PDF"
                ),
            )
        ]
    return [
        SubmissionFinding(
            kind=_Kind.OVER_LENGTH,
            severity=_Sev.WARNING,
            detail=(
                f"the PDF is {page_count} pages and the references boundary could not be "
                f"isolated, so this may exceed the {rules.page_limit_technical}-page technical "
                "limit — verify against the compiled PDF"
            ),
        )
    ]


def check_placeholder_title(
    rules: VenueRules, title: Optional[str], *, strict: bool
) -> list[SubmissionFinding]:
    """Empty title (``strict`` only — i.e. TeX, where extraction is reliable) or a
    placeholder marker in the title → blocker."""
    if not (title or "").strip():
        if strict:
            return [
                SubmissionFinding(
                    kind=_Kind.PLACEHOLDER_TITLE,
                    severity=_Sev.BLOCKER,
                    detail="the title is empty — verify the manuscript has its real title",
                )
            ]
        return []
    marker = _has_placeholder(title, rules.placeholder_markers)
    if marker is not None:
        return [
            SubmissionFinding(
                kind=_Kind.PLACEHOLDER_TITLE,
                severity=_Sev.BLOCKER,
                detail=f"the title contains a placeholder (\"{marker}\") — replace it before submitting",
                evidence=title.strip(),
            )
        ]
    return []


def check_placeholder_abstract(
    rules: VenueRules, abstract: Optional[str], *, strict: bool
) -> list[SubmissionFinding]:
    """Empty/too-short (``strict`` only) or placeholder-marked abstract → blocker.

    The too-short heuristic runs only in ``strict`` (TeX) mode: PDF abstract
    extraction is unreliable, so a short extracted string there would be a false
    blocker.
    """
    if not (abstract or "").strip():
        if strict and rules.requires_abstract:
            return [
                SubmissionFinding(
                    kind=_Kind.PLACEHOLDER_ABSTRACT,
                    severity=_Sev.BLOCKER,
                    detail="the abstract is empty or missing — verify it is present before submitting",
                )
            ]
        return []
    marker = _has_placeholder(abstract, rules.placeholder_markers)
    if marker is not None:
        return [
            SubmissionFinding(
                kind=_Kind.PLACEHOLDER_ABSTRACT,
                severity=_Sev.BLOCKER,
                detail=f"the abstract contains a placeholder (\"{marker}\") — replace it before submitting",
                evidence=abstract.strip()[:200],
            )
        ]
    if strict and len(abstract.split()) < rules.min_abstract_words:
        return [
            SubmissionFinding(
                kind=_Kind.PLACEHOLDER_ABSTRACT,
                severity=_Sev.BLOCKER,
                detail=(
                    f"the abstract is only {len(abstract.split())} words — likely a stub or "
                    "incomplete; verify it is the final abstract"
                ),
                evidence=abstract.strip()[:200],
            )
        ]
    return []


def _abstract_changed_substantially(old: str, new: str) -> bool:
    """True when the abstract diverged beyond a token threshold OR any number
    changed (KTD-9 / Open-Q2 recommended policy: any numeric/claim edit OR >20%
    token change)."""
    if _numbers(old) != _numbers(new):
        return True
    a, b = set(_tokens(old)), set(_tokens(new))
    if not a and not b:
        return False
    union = a | b
    changed_fraction = 1 - (len(a & b) / len(union)) if union else 0.0
    return changed_fraction > 0.20


def check_changed_abstract(
    abstract: Optional[str], baseline_path: Optional[Path]
) -> list[SubmissionFinding]:
    """Snapshot/diff the abstract against a local baseline (KTD-9).

    First run (baseline absent): write the current abstract and return nothing.
    Later runs: a substantial divergence → blocker. Purely local and deterministic.
    """
    if baseline_path is None or abstract is None:
        return []
    if not baseline_path.exists():
        try:
            baseline_path.write_text(abstract, encoding="utf-8")
        except OSError:
            pass  # can't snapshot (e.g. missing parent dir) — silently skip, never crash ingest
        return []
    try:
        baseline = baseline_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if _abstract_changed_substantially(baseline, abstract):
        return [
            SubmissionFinding(
                kind=_Kind.CHANGED_ABSTRACT,
                severity=_Sev.BLOCKER,
                detail=(
                    "the abstract has changed substantially from the recorded baseline — AAAI "
                    "can reject a paper whose final abstract diverges from the registered one; "
                    "verify the change is intended"
                ),
                evidence=abstract.strip()[:200],
            )
        ]
    return []


_TABLE_ENV_RE = re.compile(
    r"\\begin\s*\{(table\*?|tabular\*?|tabularx|longtable)\}(.*?)\\end\s*\{\1\}",
    re.DOTALL,
)


def check_color_tables(rules: VenueRules, tex_text: str) -> list[SubmissionFinding]:
    """Flag color macros scoped *inside* a table/tabular environment (AAAI press
    rules reject color-coded result tables). ``\\textcolor`` in body prose is not
    flagged — only inside a table environment."""
    text = _strip_comments(tex_text)
    found: list[str] = []
    for m in _TABLE_ENV_RE.finditer(text):
        body = m.group(2)
        for macro in rules.color_table_macros:
            if macro in body and macro not in found:
                found.append(macro)
    if not found:
        return []
    return [
        SubmissionFinding(
            kind=_Kind.COLOR_TABLE,
            severity=_Sev.BLOCKER,
            detail=(
                "color macros (" + ", ".join(f"`{m}`" for m in found) + ") appear inside a "
                "table — AAAI press rules reject color-coded result tables; verify before submitting"
            ),
            evidence=found[0],
        )
    ]


def check_checklist_completeness(
    rules: VenueRules, checklist_found: bool, checklist_findings: list[ChecklistFinding]
) -> list[SubmissionFinding]:
    """An unanswered mandatory checklist item → blocker (AAAI desk-rejects
    incomplete checklists). Reuses the shipped checklist findings."""
    if not rules.checklist_required or not checklist_found:
        return []
    unanswered = [f for f in checklist_findings if f.kind == ChecklistFindingKind.UNANSWERED]
    if not unanswered:
        return []
    return [
        SubmissionFinding(
            kind=_Kind.CHECKLIST_INCOMPLETE,
            severity=_Sev.BLOCKER,
            detail=(
                f"{len(unanswered)} reproducibility-checklist item(s) are unanswered — AAAI "
                "desk-rejects incomplete checklists; complete them before submitting"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# orchestrators (one per input kind)


def audit_submission_tex(
    rules: VenueRules,
    *,
    title: Optional[str],
    abstract: Optional[str],
    tex_text: str,
    checklist_found: bool,
    checklist_findings: list[ChecklistFinding],
    abstract_baseline: Optional[Path] = None,
) -> list[SubmissionFinding]:
    """Source-level checks for TeX input (length is unmeasurable here, by design)."""
    findings: list[SubmissionFinding] = []
    findings += check_placeholder_title(rules, title, strict=True)
    findings += check_placeholder_abstract(rules, abstract, strict=True)
    findings += check_changed_abstract(abstract, abstract_baseline)
    findings += check_color_tables(rules, tex_text)
    findings += check_checklist_completeness(rules, checklist_found, checklist_findings)
    return findings


def audit_submission_pdf(
    rules: VenueRules,
    *,
    title: Optional[str],
    page_count: Optional[int],
    references_start_page: Optional[int],
) -> list[SubmissionFinding]:
    """The length check plus a safe (marker-only) placeholder-title check for PDF
    input. The source-level checks need .tex and do not run here."""
    findings: list[SubmissionFinding] = []
    findings += check_length(rules, page_count, references_start_page)
    findings += check_placeholder_title(rules, title, strict=False)
    return findings


# ---------------------------------------------------------------------------
# gate aggregation (spans U2 anonymization + U3 submission)


def collect_gate_blockers(paper: IngestedPaper) -> list[str]:
    """Aggregate the hard desk-reject blockers for ``--gate`` (KTD-8): residual
    identity blocks (U2) plus blocker-severity submission findings (U3). Returns
    short human-readable descriptions; an empty list means nothing hard-blocks."""
    blockers: list[str] = []
    for f in paper.anonymization_findings:
        if f.kind == AnonymizationFindingKind.RESIDUAL_IDENTITY:
            blockers.append(f"residual author identity ({f.evidence})")
    for f in paper.submission_findings:
        if f.severity == _Sev.BLOCKER:
            blockers.append(f.detail)
    return blockers
