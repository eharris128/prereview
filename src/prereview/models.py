"""Shared data models for the prereview pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORTS = "supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    DOES_NOT_SUPPORT = "does_not_support"
    ABSTRACT_TOO_THIN = "abstract_too_thin"
    TARGET_UNAVAILABLE = "target_unavailable"
    METADATA_MISMATCH = "metadata_mismatch"
    # Verification could not be completed for infrastructure reasons — the reference
    # failed to resolve transiently (every source errored after retries), or the
    # verify model call failed after retries. Distinct from a genuine
    # TARGET_UNAVAILABLE (authoritatively resolved nowhere) and from an honest
    # ABSTRACT_TOO_THIN. Never rendered as a paper-level citation problem.
    VERIFICATION_UNAVAILABLE = "verification_unavailable"


class CitationRole(str, Enum):
    """How a citation functions in its surrounding sentence.

    The verifier asks a different question for each role: a method-attribution
    cite is judged on whether the cited paper is the canonical paper for the
    named method/tool/concept; a claim-support cite is judged on whether the
    cited evidence backs the specific claim; a background cite is judged on
    whether the cited paper is on-topic for the example or context being
    summarized.
    """

    METHOD_ATTRIBUTION = "method_attribution"
    CLAIM_SUPPORT = "claim_support"
    BACKGROUND = "background"


class Reference(BaseModel):
    """A bibliography entry as parsed from the draft paper."""

    ref_id: str  # Stable identifier within the paper, e.g. "ref-12" or "Smith2023"
    raw_text: str
    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None


class CanonicalRecord(BaseModel):
    """A reference resolved against an external scholarly API."""

    source: str  # "crossref" | "semanticscholar" | "arxiv" | "openalex"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    open_access_pdf_url: Optional[str] = None
    is_retracted: bool = False


class ResolutionStatus(str, Enum):
    """Outcome of resolving one reference against the scholarly APIs.

    Separates a genuine "not found anywhere" (every source gave an authoritative
    terminal answer) from an infrastructure failure (at least one source failed
    transiently after retries). The verifier maps these to different verdicts so a
    transient outage is never laundered into a false ghost citation.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"  # every source authoritatively "not here" — a true ghost
    DEGRADED = "degraded"  # >=1 source failed transiently after retries; not confirmed absent


class Resolution(BaseModel):
    """Result of resolving one Reference: a status plus the record when found."""

    status: ResolutionStatus
    record: Optional[CanonicalRecord] = None


class Citation(BaseModel):
    """An in-text citation: a sentence (or two) plus the reference it points to."""

    ref_id: str
    sentence: str
    section: Optional[str] = None  # e.g. "Introduction", "Methods"


class BrokenRef(BaseModel):
    """A ``\\ref``-family command whose target has no matching ``\\label``.

    These are detected directly from the .tex source (PDF ingest cannot see
    them, since by the time the PDF is built the broken refs render as
    ``??`` or as silent placeholders).
    """

    command: str  # e.g. "ref", "eqref", "cref", "Cref", "autoref", "pageref", "nameref"
    target: str  # the label name the command is pointing at
    surrounding: str  # short context around the broken ref so the user can locate it


class LinkCheck(BaseModel):
    """Reachability check on a URL surfaced by the source paper.

    URLs come from three places: ``\\url{...}`` in the .tex body, ``\\href{...}{...}``
    in the .tex body, and ``url = {...}`` fields in the .bib. ``ok`` is True iff
    the final response (after redirects) was a 2xx or 3xx; 4xx/5xx, timeouts,
    and connection errors are all flagged.
    """

    url: str
    source: str  # "tex_url" | "tex_href" | "bib_url"
    bibkey: Optional[str] = None  # set when source == "bib_url"
    status: Optional[int] = None  # final HTTP status, or None if no response
    ok: bool = False
    error: Optional[str] = None  # short failure description, or None on success


class ChecklistFindingKind(str, Enum):
    """What kind of reproducibility-checklist issue a finding represents.

    The first three are tier-1 (self-consistency, need only the checklist file);
    ``CLAIM_UNSUPPORTED`` is tier-2 (the checklist answer claims something the
    paper body does not appear to support).
    """

    UNANSWERED = "unanswered"  # left as "Type your response here" or blank
    INVALID_RESPONSE = "invalid_response"  # response not in the item's option set
    GATE_INCONSISTENCY = "gate_inconsistency"  # gate answer contradicts its sub-items
    CLAIM_UNSUPPORTED = "claim_unsupported"  # "yes" with no supporting evidence in the paper


class ChecklistItem(BaseModel):
    """One parsed ``\\question`` from a venue reproducibility checklist.

    Parsed structurally (not by hard-coded question strings) so the same parser
    works across AAAI years and, later, other venues. ``is_gate`` marks the
    "Does this paper ...?" yes/no questions that gate a nested sub-block; the
    sub-items of that block carry the gate's text in ``gate_question`` so tier-1
    can check gate-vs-subitem consistency.
    """

    section: Optional[str] = None  # the \checksubsection this item lives under
    question: str
    options: list[str] = Field(default_factory=list)  # lowercased, e.g. ["yes", "partial", "no", "na"]
    response: str = ""  # the author's answer, e.g. "yes", "", or "Type your response here"
    is_gate: bool = False  # a "Does this paper ...?" (yes/no) question gating a sub-block
    gate_question: Optional[str] = None  # parent gate's text for a sub-item, else None


class ChecklistFinding(BaseModel):
    """A single reproducibility-checklist issue surfaced to the author.

    Mirrors :class:`BrokenRef`/:class:`LinkCheck`: a deterministic, source-level
    finding rendered in its own review section. Every finding quotes the offending
    ``question`` so the author can locate it; ``detail`` carries the
    human-readable specifics (the allowed options for an invalid response, or the
    evidence that was sought for a tier-2 ``claim_unsupported``).
    """

    kind: ChecklistFindingKind
    section: Optional[str] = None
    question: str
    response: str = ""
    detail: str = ""  # allowed options (invalid_response) or evidence sought (claim_unsupported)


class AnonymizationFindingKind(str, Enum):
    """What kind of double-blind anonymization risk a finding represents.

    All are deterministic, source-level, and **advisory** — each is framed
    "verify this does not deanonymize you", never an accusation that the author
    leaked their identity. ``DUAL_SUBMISSION_TELL`` is a cheap in-paper phrasing
    signal, not real cross-venue detection.
    """

    RESIDUAL_IDENTITY = "residual_identity"  # \author/\thanks/\affiliation/\email with real content
    SELF_REVEALING_PHRASE = "self_revealing_phrase"  # "in our previous work [4]"
    IDENTITY_URL = "identity_url"  # github/gitlab/personal-homepage URL
    ACKNOWLEDGMENTS_PRESENT = "acknowledgments_present"  # acks section in a submission build
    AUTHOR_NAME_IN_BODY = "author_name_in_body"  # name-aware (--authors) surname in running prose
    DUAL_SUBMISSION_TELL = "dual_submission_tell"  # "under review at X" phrasing


class AnonymizationFinding(BaseModel):
    """A single double-blind anonymization risk surfaced to the author.

    Mirrors :class:`ChecklistFinding`: a deterministic, source-level finding
    rendered in its own review section. ``evidence`` quotes the exact offending
    fragment so the author can locate it; ``detail`` says why it is a risk and
    keeps the advisory "verify" framing.
    """

    kind: AnonymizationFindingKind
    evidence: str  # the exact offending fragment, quoted verbatim
    detail: str = ""  # why it is a deanonymization risk, framed as "verify"


class SubmissionFindingKind(str, Enum):
    """A mechanical, desk-reject-eligible submission problem (U3).

    These map to the AAAI rules a paper is summarily rejected for tripping:
    over-length, a placeholder/empty/changed title or abstract, a color-coded
    result table, or an incomplete reproducibility checklist.
    """

    OVER_LENGTH = "over_length"
    PLACEHOLDER_TITLE = "placeholder_title"
    PLACEHOLDER_ABSTRACT = "placeholder_abstract"
    CHANGED_ABSTRACT = "changed_abstract"
    COLOR_TABLE = "color_table"
    CHECKLIST_INCOMPLETE = "checklist_incomplete"


class SubmissionSeverity(str, Enum):
    """Whether a submission finding hard-blocks (gateable) or only warns.

    ``BLOCKER`` is a confident desk-reject trigger and contributes to ``--gate``'s
    exit code 4. ``WARNING`` is approximate (e.g. an over-length count from an
    un-isolable references boundary) and never hard-blocks — the precision-first
    convention: an uncertain measurement degrades to a warning, never a false
    desk-reject.
    """

    BLOCKER = "blocker"
    WARNING = "warning"


class SubmissionFinding(BaseModel):
    """One submission-readiness issue surfaced to the author.

    Mirrors :class:`ChecklistFinding` / :class:`AnonymizationFinding`: a
    deterministic, source-level finding rendered in its own section. ``detail``
    carries the human-readable specifics; ``evidence`` quotes the offending
    fragment when there is one. Advisory by default; only ``BLOCKER`` findings
    change the exit code, and only under ``--gate``.
    """

    kind: SubmissionFindingKind
    severity: SubmissionSeverity
    detail: str
    evidence: str = ""


class IngestedPaper(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    sections: list[tuple[str, str]] = Field(default_factory=list)  # (heading, body)
    references: dict[str, Reference] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    unused_bibkeys: list[str] = Field(default_factory=list)
    broken_refs: list[BrokenRef] = Field(default_factory=list)
    link_checks: list[LinkCheck] = Field(default_factory=list)
    # Reproducibility-checklist linter output (TeX-only; the PDF path never sets
    # these, so they default to "no checklist seen"). ``checklist_found``
    # distinguishes "no checklist" from "checklist clean" for the renderer.
    checklist_found: bool = False
    checklist_findings: list[ChecklistFinding] = Field(default_factory=list)
    # Manuscript-structure signals the desk-reject guards need (U1). The TeX path
    # populates the source-level fields (``author_block``/``acknowledgments``/
    # ``section_titles``); the PDF path populates the layout fields
    # (``page_count``/``references_start_page``). Each path leaves the other's
    # fields at their back-compat defaults, so neither disturbs the existing
    # citation/checklist flow. ``references_start_page`` is 1-based and stays
    # ``None`` when the boundary cannot be isolated (never a guess).
    author_block: Optional[str] = None
    acknowledgments: Optional[str] = None
    section_titles: list[str] = Field(default_factory=list)
    page_count: Optional[int] = None
    references_start_page: Optional[int] = None
    # Anonymization audit output (TeX-only; U2). ``anonymization_checked``
    # distinguishes "audit ran (and was clean if no findings)" from "audit did
    # not run" (the PDF path, or ``--no-anonymize``), which the renderer needs to
    # tell a clean confirmation apart from a skipped section.
    anonymization_checked: bool = False
    anonymization_findings: list[AnonymizationFinding] = Field(default_factory=list)
    # Submission-readiness / venue-rules guard output (U3). ``submission_checked``
    # distinguishes "guard ran" from "guard not run" for the renderer and the
    # methodology note. The PDF path runs the length check; the TeX path runs the
    # source-level checks (placeholder/abstract-diff/color-tables/checklist).
    submission_checked: bool = False
    submission_findings: list[SubmissionFinding] = Field(default_factory=list)


class VerificationResult(BaseModel):
    ref_id: str
    citation: Citation
    reference: Reference
    canonical: Optional[CanonicalRecord] = None
    verdict: Verdict
    rationale: str
    abstract_only: bool = False  # True if the verdict was based on abstract alone
    role: Optional[CitationRole] = None  # None for non-LLM verdicts (ghost, mismatch).


class CoverageReport(BaseModel):
    """Run-level integrity signal: did the review actually cover everything?

    Separates recovered-after-retry, honestly-uncertain, and non-recoverable
    outcomes so a review that *looks* complete can be verified complete. Rendered as
    a dedicated review section and used to choose the CLI exit code.
    """

    references_parsed: int = 0
    citations_checked: int = 0
    resolved: int = 0
    ghost_unresolved: int = 0  # true ghosts: every source authoritatively not-found
    resolution_degraded: int = 0  # could not resolve (infrastructure failure after retries)
    verification_degraded: int = 0  # resolved, but the verify model call failed after retries
    recovered_after_retry: int = 0  # transient calls that succeeded on a retry
    circuit_broken_sources: list[str] = Field(default_factory=list)  # sources stopped mid-run
    synthesis_degraded: bool = False  # prose pass failed; deterministic sections still written
    # Hard desk-reject blockers for ``--gate`` (U3): residual-identity (U2) plus
    # blocker-severity submission findings. Populated every run; the CLI only acts
    # on it (exit code 4, which outranks the coverage-gap exit 3) when ``--gate``
    # is passed. Distinct from infrastructure coverage — it never affects
    # ``has_coverage_gap``.
    gate_blockers: list[str] = Field(default_factory=list)

    @property
    def has_coverage_gap(self) -> bool:
        """True when some citation could not be checked for infrastructure reasons."""
        return (self.resolution_degraded + self.verification_degraded) > 0


class ReviewBundle(BaseModel):
    """Everything stage 4 (synthesize) needs to write the review."""

    paper: IngestedPaper
    verifications: list[VerificationResult]
    model: str
    synthesis_model: str
    fetched_full_text_count: int = 0
    abstract_only_count: int = 0
    unresolved_count: int = 0
    coverage: Optional[CoverageReport] = None
