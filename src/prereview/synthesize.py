"""Stage 4: synthesize the final Markdown review from ingest + verifications.

The "Citation issues" and "Methodology and limits of this review" sections are
rendered deterministically from the bundle. This guarantees that every flagged
citation surfaces, that no metadata gets fabricated, and that the user can
always trust those sections regardless of model behavior.

The remaining sections — Summary, Strengths, Weaknesses, Questions for the
author, and Suggested rating — are written by Opus from a JSON-structured
prompt with low temperature.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Iterable, Optional

from .llm import acompletion_json
from .models import (
    AnonymizationFindingKind,
    ArtifactStatus,
    ChecklistFinding,
    ChecklistFindingKind,
    CitationRole,
    NumericFindingKind,
    ReviewBundle,
    SubmissionSeverity,
    VerificationResult,
    Verdict,
)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[synthesize] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# deterministic sections


_PROBLEM_VERDICTS = (
    Verdict.DOES_NOT_SUPPORT,
    Verdict.PARTIALLY_SUPPORTS,
    Verdict.TARGET_UNAVAILABLE,
    Verdict.METADATA_MISMATCH,
)


def _is_problematic(v: VerificationResult, *, body_load_bearing: bool = True) -> bool:
    """A citation is problematic if it didn't resolve, didn't support, or only
    abstract-supports a load-bearing claim.

    Method-attribution cites are judged more leniently: a method paper from
    year N can't be expected to contain the citing paper's evaluation, so
    abstract-only "supports" or "abstract too thin" outcomes there are not
    flagged. The rationale is the same one we put in the verifier prompt —
    these are structural noise, not real review issues.
    """
    if v.verdict in _PROBLEM_VERDICTS:
        return True
    is_method = v.role == CitationRole.METHOD_ATTRIBUTION
    if v.verdict == Verdict.ABSTRACT_TOO_THIN:
        return not is_method
    if body_load_bearing and v.abstract_only and v.verdict == Verdict.SUPPORTS:
        # Abstract-only "supports" — flag for verification, except for method
        # attributions where title+authors+abstract are sufficient by design.
        return not is_method
    return False


def _verdict_label(v: Verdict) -> str:
    return {
        Verdict.SUPPORTS: "Supports (abstract-only — verify)",
        Verdict.PARTIALLY_SUPPORTS: "Partially supports / qualified",
        Verdict.DOES_NOT_SUPPORT: "Does not support / contradicts",
        Verdict.ABSTRACT_TOO_THIN: "Abstract too thin to tell",
        Verdict.TARGET_UNAVAILABLE: "Unresolved (potential ghost reference)",
        Verdict.METADATA_MISMATCH: "Metadata mismatch (likely wrong DOI or misattributed entry)",
        Verdict.VERIFICATION_UNAVAILABLE: "Could not verify (infrastructure — re-run)",
    }[v]


# Severity ordering for sorting groups and picking a headline verdict within a
# group. Lower number = more severe / higher in the list.
_SEVERITY: dict[Verdict, int] = {
    Verdict.TARGET_UNAVAILABLE: 0,
    Verdict.METADATA_MISMATCH: 1,
    Verdict.DOES_NOT_SUPPORT: 2,
    Verdict.PARTIALLY_SUPPORTS: 3,
    Verdict.ABSTRACT_TOO_THIN: 4,
    Verdict.VERIFICATION_UNAVAILABLE: 4,  # not a paper problem; never enters a flagged group
    Verdict.SUPPORTS: 5,
}

_BIB_LEVEL_VERDICTS = (Verdict.METADATA_MISMATCH, Verdict.TARGET_UNAVAILABLE)


def _group_by_bibkey(
    verifications: list[VerificationResult],
) -> dict[str, list[VerificationResult]]:
    """Group problematic verifications by bibkey, preserving site order."""
    groups: dict[str, list[VerificationResult]] = {}
    for v in verifications:
        if not _is_problematic(v):
            continue
        groups.setdefault(v.reference.ref_id, []).append(v)
    return groups


def _headline_verdict(sites: list[VerificationResult]) -> Verdict:
    return min((s.verdict for s in sites), key=lambda v: _SEVERITY[v])


def _is_bib_level(sites: list[VerificationResult]) -> bool:
    """True if the cause is the .bib entry itself (broken DOI / ghost reference)
    rather than a per-site mis-attachment."""
    return any(s.verdict in _BIB_LEVEL_VERDICTS for s in sites)


def _render_resolved_record(canonical) -> list[str]:
    if canonical is None:
        return [
            "**Resolved record:** _none — this entry could not be matched in "
            "Crossref, Semantic Scholar, arXiv, or OpenAlex._"
        ]
    lines = ["**Resolved record:**", ""]
    if canonical.is_retracted:
        lines.append("- ⚠ **RETRACTED** — this paper has been retracted.")
    lines.append(f"- Source: `{canonical.source}`")
    lines.append(f"- Title: {canonical.title or '_(unknown)_'}")
    if canonical.authors:
        lines.append("- Authors: " + ", ".join(canonical.authors[:8]))
    if canonical.year:
        lines.append(f"- Year: {canonical.year}")
    if canonical.venue:
        lines.append(f"- Venue: {canonical.venue}")
    if canonical.doi:
        lines.append(f"- DOI: [{canonical.doi}](https://doi.org/{canonical.doi})")
    if canonical.url and not canonical.doi:
        lines.append(f"- URL: {canonical.url}")
    return lines


def _render_group(index: int, ref_id: str, sites: list[VerificationResult]) -> list[str]:
    """Render a single bibkey group as Markdown lines."""
    bib_level = _is_bib_level(sites)
    headline = _headline_verdict(sites)
    ref = sites[0].reference
    canonical = sites[0].canonical
    n = len(sites)

    out: list[str] = []
    if bib_level:
        out.append(f"### {index}. `{ref_id}` — {_verdict_label(headline)}")
    elif n == 1:
        out.append(f"### {index}. `{ref_id}` — {_verdict_label(headline)}")
    else:
        out.append(
            f"### {index}. `{ref_id}` — {n} cite sites flagged "
            f"(worst: {_verdict_label(headline)})"
        )
    out.append("")
    out.append(f"**Bibliography entry (verbatim):** {_quote(ref.raw_text)}")
    out.append("")
    out.extend(_render_resolved_record(canonical))
    out.append("")

    if bib_level:
        # All sites share a single root cause: the .bib entry. Show the rationale
        # once, then list each cite site compactly.
        primary = next((s for s in sites if s.verdict in _BIB_LEVEL_VERDICTS), sites[0])
        evidence_tag = "abstract-only" if primary.abstract_only else "full-text"
        out.append(
            f"**Verdict:** {_verdict_label(primary.verdict)} _(evidence: {evidence_tag})_"
        )
        out.append("")
        out.append(f"**Rationale:** {primary.rationale}")
        out.append("")
        out.append(f"**Cited at {n} site{'s' if n != 1 else ''}:**")
        out.append("")
        for j, s in enumerate(sites, start=1):
            out.append(f"{j}. {_quote(s.citation.sentence)}")
        out.append("")
    elif n == 1:
        s = sites[0]
        evidence_tag = "abstract-only" if s.abstract_only else "full-text"
        out.append(f"**In-text sentence:** {_quote(s.citation.sentence)}")
        out.append("")
        out.append(f"**Verdict:** {_verdict_label(s.verdict)} _(evidence: {evidence_tag})_")
        out.append("")
        out.append(f"**Rationale:** {s.rationale}")
        out.append("")
    else:
        out.append(
            "The reference itself resolves correctly, but its cite sites have "
            "varying support:"
        )
        out.append("")
        sorted_sites = sorted(sites, key=lambda s: _SEVERITY[s.verdict])
        for j, s in enumerate(sorted_sites, start=1):
            evidence_tag = "abstract-only" if s.abstract_only else "full-text"
            out.append(
                f"**Site {j} — {_verdict_label(s.verdict)}** _(evidence: {evidence_tag})_"
            )
            out.append("")
            out.append(f"- Sentence: {_quote(s.citation.sentence)}")
            out.append(f"- Rationale: {s.rationale}")
            out.append("")
    return out


def render_citation_issues(bundle: ReviewBundle) -> str:
    """Markdown for the Citation issues section, grouped by bibkey.

    Each unique bibkey gets one section. If the bib entry itself is the cause
    (wrong DOI / ghost reference), every cite site is listed under one rationale.
    Otherwise, each cite site's verdict and rationale are shown individually.
    """
    out: list[str] = ["## Citation issues", ""]

    groups = _group_by_bibkey(bundle.verifications)
    if not groups:
        out.append(
            "_No citation issues detected._ Every cited reference resolved against "
            "Crossref / Semantic Scholar / arXiv / OpenAlex, and every claim was "
            "supported by the cited paper's full text."
        )
        out.append("")
        return "\n".join(out)

    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (_SEVERITY[_headline_verdict(groups[k])], k),
    )
    total_sites = sum(len(groups[k]) for k in sorted_keys)
    bib_level = sum(1 for k in sorted_keys if _is_bib_level(groups[k]))

    out.append(
        f"**{len(sorted_keys)} unique reference{'s' if len(sorted_keys) != 1 else ''} "
        f"flagged across {total_sites} cite site{'s' if total_sites != 1 else ''}.** "
        f"{bib_level} of these {'are bibliography-entry issues (likely wrong DOI / ghost reference)' if bib_level != 1 else 'is a bibliography-entry issue (likely wrong DOI / ghost reference)'}; "
        f"the rest are cite sites whose claim isn't fully supported by the cited paper."
    )
    out.append("")
    out.append(
        "Each entry shows the bibliography entry as the author wrote it, the "
        "canonical record we resolved (if any), and the in-text sentence(s) "
        "where it is cited."
    )
    out.append("")

    for i, key in enumerate(sorted_keys, start=1):
        out.extend(_render_group(i, key, groups[key]))

    return "\n".join(out)


def _quote(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    if not s:
        return "_(empty)_"
    return f"> {s}"


def _retracted_groups(
    verifications: list[VerificationResult],
) -> dict[str, list[VerificationResult]]:
    """Group verifications whose resolved canonical record is retracted, by
    bibkey. Every cite site of a retracted paper is preserved so the user can
    see which sentences depend on retracted work."""
    groups: dict[str, list[VerificationResult]] = {}
    for v in verifications:
        if v.canonical is not None and v.canonical.is_retracted:
            groups.setdefault(v.reference.ref_id, []).append(v)
    return groups


def render_hygiene_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Hygiene checks section, or None if there's nothing
    to surface.

    Reports source- and metadata-level issues that the per-citation Citation
    Issues section can't see on its own: retractions (which can apply even to
    a paper that supports its claim), broken cross-references (\\ref to a
    non-existent \\label), and unused bibliography entries (in the .bib but
    never \\cite-d).
    """
    paper = bundle.paper
    broken = paper.broken_refs
    unused = paper.unused_bibkeys
    retracted = _retracted_groups(bundle.verifications)
    bad_links = [c for c in paper.link_checks if not c.ok]
    if not broken and not unused and not retracted and not bad_links:
        return None

    out: list[str] = ["## Hygiene checks", ""]
    out.append(
        "Source-level issues detected by parsing the .tex and .bib directly, "
        "plus metadata-level red flags from the resolver (retractions). "
        "These are mechanical findings — the kind a copyeditor or pre-submission "
        "checklist would catch."
    )
    out.append("")

    if retracted:
        out.append(f"### ⚠ Retracted citations ({len(retracted)})")
        out.append("")
        out.append(
            "The cited paper has been retracted (per OpenAlex / Retraction Watch). "
            "Even if the cited claim is technically supported, citing a retracted "
            "paper is almost always a problem — verify in person and either drop "
            "the citation or explicitly justify it."
        )
        out.append("")
        for ref_id, sites in retracted.items():
            n = len(sites)
            canonical = sites[0].canonical
            doi_str = ""
            if canonical and canonical.doi:
                doi_str = f" — DOI [{canonical.doi}](https://doi.org/{canonical.doi})"
            site_str = f"{n} cite site{'s' if n != 1 else ''}"
            out.append(f"- `{ref_id}` ({site_str}){doi_str}")
        out.append("")

    if broken:
        out.append(f"### Broken cross-references ({len(broken)})")
        out.append("")
        out.append(
            "Each entry is a `\\ref`/`\\cref`/etc. whose target has no matching "
            "`\\label` anywhere in the source. These render as empty placeholders "
            "or `??` in the compiled PDF."
        )
        out.append("")
        for br in broken:
            out.append(f"- `\\{br.command}{{{br.target}}}` — context: > {br.surrounding}")
        out.append("")

    if unused:
        out.append(f"### Unused bibliography entries ({len(unused)})")
        out.append("")
        out.append(
            "These bibkeys are in the .bib but appear in no `\\cite{...}` in the body. "
            "They are dead weight in the bibliography — either remove them, or cite "
            "them where intended."
        )
        out.append("")
        for key in unused:
            out.append(f"- `{key}`")
        out.append("")

    if bad_links:
        n_total = len(paper.link_checks)
        out.append(f"### Unreachable URLs ({len(bad_links)} of {n_total})")
        out.append("")
        out.append(
            "URLs surfaced from `\\url{}` / `\\href{}` in the .tex and `url = {...}` "
            "fields in the .bib that did not resolve to a 2xx/3xx response. "
            "Reviewers click these — broken links are an easy fix that signal care."
        )
        out.append("")
        for c in bad_links:
            origin = {
                "tex_url": "\\url{}",
                "tex_href": "\\href{}",
                "bib_url": f".bib `{c.bibkey}`" if c.bibkey else ".bib `url=`",
            }.get(c.source, c.source)
            reason = c.error or (f"HTTP {c.status}" if c.status else "no response")
            out.append(f"- {c.url} — {reason} _(from {origin})_")
        out.append("")

    return "\n".join(out)


# Fixed render order; also the only place finding kinds map to user-facing labels.
_CHECKLIST_SUBHEADINGS: dict[ChecklistFindingKind, str] = {
    ChecklistFindingKind.UNANSWERED: "Unanswered items",
    ChecklistFindingKind.INVALID_RESPONSE: "Invalid responses",
    ChecklistFindingKind.GATE_INCONSISTENCY: "Gate inconsistencies",
    ChecklistFindingKind.CLAIM_UNSUPPORTED: "Answers not supported by the paper",
}


def _checklist_counts(findings: list[ChecklistFinding]) -> dict[ChecklistFindingKind, int]:
    counts = {k: 0 for k in ChecklistFindingKind}
    for f in findings:
        counts[f.kind] += 1
    return counts


def _checklist_bullet(f: ChecklistFinding) -> str:
    loc = f"_{f.section}_: " if f.section else ""
    question = (f.question or "").strip()
    line = f'- {loc}"{question}"'
    if f.kind == ChecklistFindingKind.INVALID_RESPONSE:
        line += f' — responded "{f.response}" ({f.detail})'
    elif f.kind == ChecklistFindingKind.GATE_INCONSISTENCY:
        line += f" — {f.detail}"
    elif f.kind == ChecklistFindingKind.CLAIM_UNSUPPORTED:
        line += f' — responded "{f.response}"; {f.detail}' if f.response else f" — {f.detail}"
    return line


def render_checklist_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Reproducibility checklist section, or None when no
    checklist was found.

    Mirrors :func:`render_hygiene_section`'s contract: returns None when there's
    nothing to anchor a section on (no checklist located), a short confirmation
    when a checklist was found but is clean, and a grouped breakdown otherwise.
    The tier-2 "answers not supported by the paper" group is framed as "verify",
    never as an accusation — these are presence checks, not truth judgements.
    """
    paper = bundle.paper
    if not paper.checklist_found:
        return None
    findings = paper.checklist_findings

    out: list[str] = ["## Reproducibility checklist", ""]
    if not findings:
        out.append(
            "The reproducibility checklist was located and parsed; no unanswered "
            "items, invalid responses, gate inconsistencies, or unsupported "
            "answers were detected."
        )
        out.append("")
        return "\n".join(out)

    out.append(
        "Deterministic checks against the venue reproducibility checklist, parsed "
        "from the .tex source. Self-consistency findings need only the checklist; "
        'the "answers not supported by the paper" findings are presence checks '
        "against the paper body — the checklist answer may still be correct, so "
        'treat them as "verify", not as accusations.'
    )
    out.append("")

    for kind, heading in _CHECKLIST_SUBHEADINGS.items():
        group = [f for f in findings if f.kind == kind]
        if not group:
            continue
        out.append(f"### {heading} ({len(group)})")
        out.append("")
        out.extend(_checklist_bullet(f) for f in group)
        out.append("")

    return "\n".join(out)


def _checklist_methodology_sentence(bundle: ReviewBundle) -> str:
    cf = bundle.paper.checklist_findings
    if not cf:
        return "A reproducibility checklist was located and linted: no issues flagged."
    counts = _checklist_counts(cf)
    return (
        f"A reproducibility checklist was located and linted: {len(cf)} issue(s) flagged "
        f"({counts[ChecklistFindingKind.UNANSWERED]} unanswered, "
        f"{counts[ChecklistFindingKind.INVALID_RESPONSE]} invalid, "
        f"{counts[ChecklistFindingKind.GATE_INCONSISTENCY]} gate inconsistency(ies), "
        f"{counts[ChecklistFindingKind.CLAIM_UNSUPPORTED]} not evidenced in the paper). "
        "The claim-vs-paper cross-checks are presence-based and advisory — they flag "
        "missing supporting evidence, not incorrect answers."
    )


# Fixed render order; also where anonymization kinds map to user-facing headings.
_ANON_SUBHEADINGS: dict[AnonymizationFindingKind, str] = {
    AnonymizationFindingKind.RESIDUAL_IDENTITY: "Residual identity blocks",
    AnonymizationFindingKind.SELF_REVEALING_PHRASE: "Self-revealing phrasing",
    AnonymizationFindingKind.IDENTITY_URL: "Identity-revealing URLs",
    AnonymizationFindingKind.ACKNOWLEDGMENTS_PRESENT: "Acknowledgments present",
    AnonymizationFindingKind.AUTHOR_NAME_IN_BODY: "Author names in the body",
    AnonymizationFindingKind.DUAL_SUBMISSION_TELL: "Dual-submission phrasing",
}


def render_anonymization_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Anonymization audit section, or None when the audit did
    not run.

    Three states, mirroring :func:`render_checklist_section`'s contract: a one-line
    note when the input was a PDF (the audit is .tex-source-only), ``None`` when it
    was explicitly disabled (``--no-anonymize``), a clean confirmation when it ran
    with nothing flagged, and a grouped breakdown otherwise. Every finding is
    framed "verify" — the audit surfaces candidates, it does not confirm a leak.
    """
    paper = bundle.paper
    if not paper.anonymization_checked:
        # Distinguish "PDF input — can't run" from "--no-anonymize — truly off".
        # Only the PDF path sets page_count, so it is the reliable input-kind tell.
        if paper.page_count is not None:
            return (
                "## Anonymization audit\n\n_The anonymization audit runs on .tex "
                "source only; it was skipped for this PDF input._\n"
            )
        return None

    findings = paper.anonymization_findings
    out: list[str] = ["## Anonymization audit", ""]
    if not findings:
        out.append(
            "No residual author identity, self-revealing phrasing, identity-revealing "
            "URLs, or acknowledgments section were detected in the .tex source. These "
            "checks cannot confirm anonymization — still verify manually before submitting."
        )
        out.append("")
        return "\n".join(out)

    out.append(
        "Deterministic double-blind checks against the .tex source. AAAI desk-rejects "
        "deanonymized submissions, so each item below is worth a look — but all are "
        "**advisory**: they quote a candidate fragment for you to **verify**, and "
        "cannot themselves confirm a leak."
    )
    out.append("")
    for kind, heading in _ANON_SUBHEADINGS.items():
        group = [f for f in findings if f.kind == kind]
        if not group:
            continue
        out.append(f"### {heading} ({len(group)})")
        out.append("")
        for f in group:
            out.append(f"- {f.detail}")
            out.append(f"  > {f.evidence}")
        out.append("")
    return "\n".join(out)


def render_submission_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Submission readiness (desk-reject guard) section, or None
    when the guard did not run.

    Like the anonymization audit, this is a desk-reject guard that renders even on
    a clean run (a positive "you won't be desk-rejected for these" confirmation is
    the headline value). Blockers and warnings are shown separately; everything is
    advisory unless ``--gate`` is passed.
    """
    paper = bundle.paper
    if not paper.submission_checked:
        return None

    findings = paper.submission_findings
    out: list[str] = ["## Submission readiness (desk-reject guard)", ""]
    if not findings:
        out.append(
            "No over-length, placeholder/empty title or abstract, color-coded result "
            "table, or unanswered mandatory checklist items were detected for this venue "
            "(length is measured on PDF input only). Still verify against the official "
            "author kit before submitting."
        )
        out.append("")
        return "\n".join(out)

    out.append(
        "Mechanical checks against the venue's submission rules. **Blockers** are "
        "desk-reject-eligible; **warnings** are approximate and worth a manual look. "
        "All are advisory unless you pass `--gate`."
    )
    out.append("")
    blockers = [f for f in findings if f.severity == SubmissionSeverity.BLOCKER]
    warnings = [f for f in findings if f.severity == SubmissionSeverity.WARNING]
    for label, group in (("Blockers", blockers), ("Warnings", warnings)):
        if not group:
            continue
        out.append(f"### {label} ({len(group)})")
        out.append("")
        for f in group:
            out.append(f"- {f.detail}")
            if f.evidence:
                out.append(f"  > {f.evidence}")
        out.append("")
    return "\n".join(out)


# Fixed render order; also where numeric kinds map to user-facing headings.
_NUMERIC_SUBHEADINGS: dict[NumericFindingKind, str] = {
    NumericFindingKind.BOUNDED_METRIC: "Out-of-range metrics",
    NumericFindingKind.SPLIT_MISMATCH: "Dataset split arithmetic",
    NumericFindingKind.MEAN_STD_RANGE: "Implausible mean ± std",
    NumericFindingKind.HYPERPARAM_DRIFT: "Prose / table hyperparameter drift",
    NumericFindingKind.ABSTRACT_TABLE_DELTA: "Abstract vs. table deltas",
}


def render_numeric_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Numerical sanity section, or None when nothing was flagged.

    Like :func:`render_hygiene_section`, it renders only when there is something to
    surface (the checks are high-precision, so a finding is worth attention). Every
    item is advisory — a number to verify, not a finding of error."""
    findings = bundle.paper.numeric_findings
    if not findings:
        return None
    out: list[str] = ["## Numerical sanity", ""]
    out.append(
        "High-precision deterministic checks for the numerical errors reviewers catch — "
        "out-of-range metrics, split arithmetic, mean ± std ranges, prose/table "
        "hyperparameter drift, and abstract-vs-table deltas. Each is **advisory**: a number "
        "to **verify**, not a finding of error."
    )
    out.append("")
    for kind, heading in _NUMERIC_SUBHEADINGS.items():
        group = [f for f in findings if f.kind == kind]
        if not group:
            continue
        out.append(f"### {heading} ({len(group)})")
        out.append("")
        for f in group:
            out.append(f"- {f.detail}")
            if f.evidence:
                out.append(f"  > {f.evidence}")
        out.append("")
    return "\n".join(out)


def render_artifacts_section(bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Artifact availability section, or None when nothing
    unresolved. Resolved artifacts are silent; transient failures are disclosed in
    the coverage section instead (they are infrastructure, not author defects)."""
    misses = [c for c in bundle.paper.artifact_checks if c.status == ArtifactStatus.TERMINAL_MISS]
    if not misses:
        return None
    out: list[str] = ["## Artifact availability", ""]
    out.append(
        "Existence checks for the models, datasets, and repos this paper claims. Resolved "
        "artifacts are not listed; the items below did **not** resolve — **verify** each link "
        "before submitting. (Hugging Face's unauthenticated API cannot tell a missing artifact "
        "from a private one; set `HF_TOKEN` to disambiguate.)"
    )
    out.append("")
    for c in misses:
        out.append(f"- {c.detail}")
        out.append(f"  > {c.url}")
    out.append("")
    return "\n".join(out)


def render_coverage_section(bundle: ReviewBundle) -> Optional[str]:
    """Coverage & reliability — the trust signal. Renders only when there is an
    infrastructure outcome to disclose (degradation, recovery, a tripped source, or a
    failed prose pass); a clean run leaves it out (the methodology section carries the
    normal stats). Deterministic, like ``render_hygiene_section``.
    """
    verifs = bundle.verifications
    total = len(verifs)
    ghost = sum(1 for v in verifs if v.verdict == Verdict.TARGET_UNAVAILABLE)
    degraded = [v for v in verifs if v.verdict == Verdict.VERIFICATION_UNAVAILABLE]
    cov = bundle.coverage
    recovered = cov.recovered_after_retry if cov else 0
    broken = list(cov.circuit_broken_sources) if cov else []
    synth_degraded = bool(cov.synthesis_degraded) if cov else False
    artifact_degraded = [
        c for c in bundle.paper.artifact_checks if c.status == ArtifactStatus.TRANSIENT_FAIL
    ]

    if not degraded and recovered == 0 and not broken and not synth_degraded and not artifact_degraded:
        return None  # nothing infrastructure-related to disclose

    resolved = total - ghost - len(degraded)
    lines = ["## Review coverage & reliability", ""]
    if degraded or broken or synth_degraded or artifact_degraded:
        lines.append(
            "_Some parts of this review could not be completed due to infrastructure "
            "issues. The items below are **not** findings about your paper — re-running "
            "may clear them._"
        )
    else:
        lines.append(
            "_This run hit transient API failures and recovered them; coverage is complete._"
        )
    lines.append("")
    lines.append(
        f"- {resolved} of {total} citation{'' if total == 1 else 's'} resolved and verified; "
        f"{ghost} genuinely unresolved (shown under Citation issues)."
    )
    if recovered:
        lines.append(
            f"- {recovered} transient API failure{'' if recovered == 1 else 's'} recovered on "
            "retry (would otherwise have been reported as unresolved)."
        )
    if degraded:
        ids = ", ".join(f"`{v.ref_id}`" for v in degraded)
        lines.append(
            f"- **{len(degraded)} citation{'' if len(degraded) == 1 else 's'} could not be "
            "verified** — a scholarly API or the model failed after retries. These are NOT "
            f"ghost citations; re-run to confirm: {ids}."
        )
    if broken:
        srcs = ", ".join(f"`{s}`" for s in broken)
        lines.append(
            f"- Stopped querying after repeated failures this run: {srcs}. Set the relevant "
            "API key (`S2_API_KEY` / `OPENALEX_API_KEY`) and re-run for fuller coverage."
        )
    if synth_degraded:
        lines.append(
            "- The narrative sections (summary, strengths, weaknesses, questions, rating) "
            "could not be generated this run; the deterministic sections are complete."
        )
    if artifact_degraded:
        n = len(artifact_degraded)
        lines.append(
            f"- {n} artifact existence check{'' if n == 1 else 's'} could not be completed "
            "(Hugging Face / GitHub unreachable after retries) — these are NOT missing "
            "artifacts; re-run to confirm."
        )
    return "\n".join(lines) + "\n"


def render_methodology(bundle: ReviewBundle) -> str:
    total = len(bundle.verifications)
    ghost = sum(1 for v in bundle.verifications if v.verdict == Verdict.TARGET_UNAVAILABLE)
    flagged = sum(1 for v in bundle.verifications if _is_problematic(v))
    n_broken = len(bundle.paper.broken_refs)
    n_unused = len(bundle.paper.unused_bibkeys)
    n_retracted = len(_retracted_groups(bundle.verifications))
    n_links = len(bundle.paper.link_checks)
    n_bad_links = sum(1 for c in bundle.paper.link_checks if not c.ok)
    link_clause = (
        f", {n_bad_links} of {n_links} URL{'s' if n_links != 1 else ''} unreachable"
        if n_links
        else ""
    )
    hygiene_line = (
        f"Source-level hygiene checks ran on the .tex source: "
        f"{n_broken} broken cross-reference{'' if n_broken == 1 else 's'} "
        f"(\\ref/\\cref to a non-existent \\label), "
        f"{n_unused} unused bibliography entr{'y' if n_unused == 1 else 'ies'} "
        f"(in the .bib but never \\cite-d), "
        f"{n_retracted} retracted citation{'' if n_retracted == 1 else 's'} "
        f"(per OpenAlex / Retraction Watch){link_clause}."
    )
    if bundle.paper.checklist_found:
        # 4-space indent + blank line so it dedents uniformly with the template below.
        hygiene_line += "\n\n    " + _checklist_methodology_sentence(bundle)
    if bundle.paper.submission_checked:
        n_sub = len(bundle.paper.submission_findings)
        hygiene_line += "\n\n    " + (
            f"Submission-readiness checks ran against the venue rules: {n_sub} issue(s) flagged "
            "(length, placeholder/abstract-diff, color tables, checklist completeness)."
            if n_sub
            else "Submission-readiness checks ran against the venue rules: no desk-reject issues flagged."
        )
    # Once the numerical-sanity pack (U6) has run, the blanket "does not check
    # reported statistics" promise is no longer true — qualify it to what the pack
    # actually covers vs. the p-value auditing it still does not.
    if bundle.paper.numeric_checked:
        stats_bullet = (
            "Statistical-test reporting (no statcheck / GRIM-style p-value audit) — though the "
            "Numerical sanity section above checks metric bounds, split arithmetic, mean ± std "
            "ranges, and prose/table hyperparameter consistency."
        )
    else:
        stats_bullet = "Reported statistics (no statcheck / GRIM-style audit)."
    return textwrap.dedent(f"""
    ## Methodology and limits of this review

    This review was produced by `prereview`, an AI-assisted pre-submission tool. The retrieval
    and extraction passes used `{bundle.model}`; the synthesis pass used `{bundle.synthesis_model}`.

    Of {total} in-text citations, {ghost} bibliography entr{"y" if ghost == 1 else "ies"} did not
    resolve to any record in Crossref, Semantic Scholar, arXiv, or OpenAlex. Of the citations
    that did resolve, {bundle.fetched_full_text_count} were verified against the cited paper's full text and
    {bundle.abstract_only_count} were verified against the abstract only. {flagged} citation{"" if flagged == 1 else "s"} {"was" if flagged == 1 else "were"} flagged
    in the section above.

    {hygiene_line}

    What this tool does **not** check:

    - Figures, tables, captions.
    - Mathematical proofs or formal correctness.
    - {stats_bullet}
    - Plagiarism or text-overlap with prior work.
    - **Missing prior work** — `prereview` only verifies citations the author already made; it does
      not search for relevant work the author should have cited.

    The review leads with a severity-bucketed **Overall assessment**, not a recommendation. The
    1–10 rating an LLM would assign is shown only under `--show-rating` and is known to skew
    generous (Keuper 2025) — treat it as secondary.
    """).strip() + "\n"


# ---------------------------------------------------------------------------
# LLM prose sections


_PROSE_SYSTEM = (
    "You are a senior peer reviewer for a top venue (NeurIPS / ICLR-style). "
    "You write specific, grounded, fair-minded reviews. You do not hedge generically; "
    "every strength and weakness names a concrete element of the paper. You output "
    "JSON exactly matching the requested schema, no prose, no fences."
)


def _prose_prompt(bundle: ReviewBundle) -> str:
    paper = bundle.paper
    body_text = "\n\n".join(b for _, b in paper.sections) if paper.sections else ""
    if len(body_text) > 200_000:
        # Take the head and tail; the middle is rarely as informative for a review.
        body_text = body_text[:120_000] + "\n\n[...truncated middle...]\n\n" + body_text[-60_000:]

    groups = _group_by_bibkey(bundle.verifications)
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (_SEVERITY[_headline_verdict(groups[k])], k),
    )

    def _summary_line(key: str) -> str:
        sites = groups[key]
        headline = _headline_verdict(sites)
        n = len(sites)
        site_phrase = f"{n} cite site{'s' if n != 1 else ''}"
        primary = next(
            (s for s in sites if s.verdict in _BIB_LEVEL_VERDICTS),
            min(sites, key=lambda s: _SEVERITY[s.verdict]),
        )
        return (
            f"- `{key}` ({_verdict_label(headline)}, {site_phrase}): {primary.rationale}"
        )

    flag_summary = "\n".join(_summary_line(k) for k in sorted_keys[:30]) or (
        "_None — every cited reference checked out._"
    )
    if len(sorted_keys) > 30:
        flag_summary += f"\n- ... and {len(sorted_keys) - 30} more"

    return f"""You are reviewing this draft paper for a top venue. Write a structured review.

The "Citation issues" and "Methodology" sections will be rendered deterministically and appended
to your output — do NOT write them yourself. You may, and should, mention citation problems in
the Weaknesses section if they affect the paper's claims.

PAPER TITLE: {paper.title or "(unknown)"}

PAPER ABSTRACT:
\"\"\"{paper.abstract or "(not extracted)"}\"\"\"

PAPER BODY:
\"\"\"{body_text}\"\"\"

CITATION VERIFICATION SUMMARY:
- Total in-text citations checked: {len(bundle.verifications)}
- Unresolved bibliography entries (potential ghost references): {sum(1 for v in bundle.verifications if v.verdict == Verdict.TARGET_UNAVAILABLE)}
- Verified against full text: {bundle.fetched_full_text_count}
- Verified against abstract only: {bundle.abstract_only_count}
- Flagged citations:
{flag_summary}

Return a single JSON object exactly matching this schema:

{{
  "summary": "2-4 short paragraphs summarizing what the paper claims and how. Concrete and specific.",
  "strengths": ["bullet 1", "bullet 2", ...],
  "weaknesses": ["bullet 1", "bullet 2", ...],
  "questions": ["question 1", "question 2", ...],
  "rating_low": 1-10,
  "rating_high": 1-10,
  "rating_justification": "one sentence"
}}

Rules:
- "summary" should be 2-4 paragraphs, separated by blank lines (use \\n\\n).
- "strengths" and "weaknesses": 3-7 bullets each, each bullet specific to this paper. No filler.
- "questions": 2-6 questions a reviewer would actually ask the authors in a discussion period.
- "rating_low" and "rating_high" are integers on the standard 1-10 scale (1=trivial reject, 10=strong accept).
  Provide a range that honestly captures your uncertainty (typical range width: 2-3).
- "rating_justification" is one sentence; it must explicitly acknowledge that the rating is a model judgement.
- If the paper has serious citation problems (>= 3 unresolved or contradicting), mention this in weaknesses.
"""


async def _generate_prose(bundle: ReviewBundle, *, verbose: bool) -> dict:
    return await acompletion_json(
        model=bundle.synthesis_model,
        system=_PROSE_SYSTEM,
        user=_prose_prompt(bundle),
        temperature=0.0,
        max_tokens=8192,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# adversarial Reviewer-2 pass (U4)
#
# One rubric-anchored LLM pass that surfaces the objections a tough reviewer
# raises, bucketed by per-dimension severity. Designed to AVOID the documented
# LLM-reviewer generosity collapse (>95% accept rates): the persona is explicitly
# adversarial, every dimension must report a finding or an explicit "none", each
# severity is anchored to a rubric baked into the system prompt (without anchors
# the model defaults everything to "minor"), and it emits NO accept/reject verdict
# or score. It is grounded only in the paper plus the tool's own deterministic
# flags; the novelty dimension cross-examines the paper's OWN citation list.

# (key, user-facing heading) in fixed render order.
_REVIEWER2_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("novelty", "Novelty & positioning"),
    ("baselines", "Baselines & experiments"),
    ("claims", "Claims vs. evidence"),
    ("reproducibility", "Reproducibility"),
    ("clarity", "Clarity"),
)
_REVIEWER2_DIM_KEYS = tuple(k for k, _ in _REVIEWER2_DIMENSIONS)

# Severity coercion (mirrors verify.py's _VALID_* maps). "none"/unknown collapses
# to None → rendered as "no issue found".
_VALID_REVIEWER2_SEVERITY = {"critical": "critical", "major": "major", "minor": "minor"}
_REVIEWER2_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}

_REVIEWER2_SYSTEM = (
    "You are Reviewer 2: a sharp, adversarial, but fair-minded reviewer for a top venue "
    "(AAAI / NeurIPS / ICLR). Your job is to surface the objections a tough reviewer WILL "
    "raise so the authors can fix them BEFORE submission. You are deliberately skeptical — "
    "you hunt for what is weak, unsupported, or missing — and every objection you raise "
    "must quote or name a concrete element of the paper (a specific claim, table, baseline, "
    "or a named omission). You never praise to be nice.\n\n"
    "HARD RULES:\n"
    "- You do NOT output an accept/reject decision or any numeric score, anywhere. Surfacing "
    "issues, not rating, is your entire job.\n"
    "- Return an entry for EVERY dimension. If a dimension genuinely has no problem, return "
    'severity "none" for it — never silently skip a dimension.\n'
    "- Assign severity strictly against the rubric below. Do NOT default everything to "
    '"minor"; if a real critical/major issue exists, say so.\n\n'
    "PER-DIMENSION SEVERITY RUBRIC:\n"
    "- novelty/positioning — critical: the core contribution is already published (name the "
    "cited prior work it collapses into); major: the paper overclaims novelty relative to "
    "cited work; minor: related-work gaps that don't threaten the contribution.\n"
    "- baselines/experiments — critical: omits the standard SOTA baseline for the task; "
    "major: missing an obvious ablation or a standard dataset; minor: a baseline is "
    "under-tuned or a detail is unreported.\n"
    "- claims-vs-evidence — critical: a headline claim is unsupported by the reported "
    "results; major: a general claim is backed only by one narrow setting; minor: wording "
    "stronger than the evidence strictly supports.\n"
    "- reproducibility — critical: neither code nor enough detail to reproduce the main "
    "result; major: key hyperparameters / data splits / seeds are missing; minor: incomplete "
    "environment or compute detail.\n"
    "- clarity — critical: the method as written cannot be understood or implemented; major: "
    "a key definition, figure, or symbol is missing or ambiguous; minor: local wording or "
    "notation issues.\n\n"
    "Output a single JSON object only — no prose, no fences."
)


def _reviewer2_citation_list(bundle: ReviewBundle, limit: int = 60) -> str:
    """The paper's own bibliography, as grounding for the novelty dimension's
    claimed-novelty-vs-cited-prior-art cross-examination."""
    refs = bundle.paper.references
    lines: list[str] = []
    for ref_id, ref in list(refs.items())[:limit]:
        title = (ref.title or "(no title)").strip()
        year = f", {ref.year}" if ref.year else ""
        lines.append(f"- [{ref_id}] {title}{year}")
    extra = len(refs) - limit
    if extra > 0:
        lines.append(f"- ... and {extra} more")
    return "\n".join(lines) or "(no references parsed)"


def _reviewer2_flag_summary(bundle: ReviewBundle) -> str:
    """A compact summary of the tool's OWN deterministic flags, so Reviewer-2 is
    grounded in them (e.g. a missing-baseline signal feeds the baselines rubric)."""
    paper = bundle.paper
    parts: list[str] = []
    groups = _group_by_bibkey(bundle.verifications)
    if groups:
        parts.append(f"- {len(groups)} cited reference(s) flagged by the citation verifier (unsupported / ghost / mismatch).")
    if paper.checklist_findings:
        parts.append(f"- {len(paper.checklist_findings)} reproducibility-checklist issue(s).")
    if paper.submission_findings:
        parts.append(f"- {len(paper.submission_findings)} submission-readiness issue(s).")
    numeric = getattr(paper, "numeric_findings", None)
    if numeric:
        parts.append(f"- {len(numeric)} numerical-sanity flag(s) (bounded metrics, split arithmetic, etc.).")
    if paper.anonymization_findings:
        parts.append(f"- {len(paper.anonymization_findings)} anonymization risk(s).")
    return "\n".join(parts) or "(no deterministic flags raised)"


def _reviewer2_prompt(bundle: ReviewBundle) -> str:
    paper = bundle.paper
    body_text = "\n\n".join(b for _, b in paper.sections) if paper.sections else ""
    if len(body_text) > 200_000:
        body_text = body_text[:120_000] + "\n\n[...truncated middle...]\n\n" + body_text[-60_000:]

    return f"""Adversarially review this draft paper. Surface the objections a tough reviewer will raise.

PAPER TITLE: {paper.title or "(unknown)"}

PAPER ABSTRACT:
\"\"\"{paper.abstract or "(not extracted)"}\"\"\"

PAPER BODY:
\"\"\"{body_text}\"\"\"

THE PAPER'S OWN CITED REFERENCES (use these for the novelty/positioning dimension — does the
paper's claimed novelty hold up against what it already cites?):
{_reviewer2_citation_list(bundle)}

THIS TOOL'S OWN DETERMINISTIC FLAGS (corroborating signal — weave in where relevant):
{_reviewer2_flag_summary(bundle)}

Return a single JSON object exactly matching this schema:

{{
  "findings": [
    {{
      "dimension": "novelty" | "baselines" | "claims" | "reproducibility" | "clarity",
      "severity": "critical" | "major" | "minor" | "none",
      "issue": "the objection, one or two sentences",
      "evidence": "the concrete paper element it is grounded in (a quoted claim, a table, a baseline name, or a named omission)",
      "suggested_fix": "what the authors should do about it"
    }}
  ]
}}

Rules:
- Include an entry for EVERY dimension: novelty, baselines, claims, reproducibility, clarity.
  Use severity "none" with issue "no issue found" for a dimension that genuinely has no problem.
- Apply the per-dimension severity rubric from the system prompt; do NOT default to "minor".
- Every non-"none" finding's "evidence" must name a concrete element of THIS paper.
- Do NOT include any accept/reject decision or numeric score anywhere in the output.
"""


async def _generate_reviewer2(bundle: ReviewBundle, *, verbose: bool) -> dict:
    return await acompletion_json(
        model=bundle.synthesis_model,
        system=_REVIEWER2_SYSTEM,
        user=_reviewer2_prompt(bundle),
        temperature=0.0,
        max_tokens=8192,
        verbose=verbose,
    )


def _coerce_reviewer2_finding(raw: dict) -> Optional[dict]:
    """Keep only the known fields (so a stray ``verdict``/``score`` the model adds
    can never leak into the output), and coerce the severity. Returns None for a
    finding with no usable issue text."""
    if not isinstance(raw, dict):
        return None
    dim = str(raw.get("dimension") or "").strip().lower()
    sev = _VALID_REVIEWER2_SEVERITY.get(str(raw.get("severity") or "").strip().lower())  # None for "none"/unknown
    issue = str(raw.get("issue") or "").strip()
    evidence = str(raw.get("evidence") or "").strip()
    fix = str(raw.get("suggested_fix") or "").strip()
    if sev is None:
        return None  # an explicit "none" / unrated dimension carries no finding
    if not issue:
        return None
    return {"dimension": dim, "severity": sev, "issue": issue, "evidence": evidence, "suggested_fix": fix}


def render_reviewer2_section(reviewer2: Optional[dict], bundle: ReviewBundle) -> Optional[str]:
    """Markdown for the Reviewer 2 (adversarial) section.

    None when the pass did not run; a disclosed degradation note when it ran but
    failed; otherwise a per-dimension breakdown. Only known fields are rendered,
    so no accept/reject verdict or score can leak even if the model emits one.
    """
    cov = bundle.coverage
    if reviewer2 is None:
        if cov is not None and cov.reviewer2_degraded:
            return (
                "## Reviewer 2 (adversarial)\n\n_The adversarial Reviewer-2 pass could not be "
                "generated this run (the model call failed after retries). The rest of the "
                "review is unaffected — re-run to include it._\n"
            )
        return None

    raw_findings = reviewer2.get("findings") if isinstance(reviewer2, dict) else None
    coerced = [c for c in (_coerce_reviewer2_finding(f) for f in (raw_findings or [])) if c]
    by_dim: dict[str, list[dict]] = {}
    for f in coerced:
        by_dim.setdefault(f["dimension"], []).append(f)

    out: list[str] = ["## Reviewer 2 (adversarial)", ""]
    out.append(
        "A rubric-anchored adversarial pass that surfaces the objections a tough reviewer is "
        "likely to raise, grouped by dimension and tagged by severity. It deliberately leads "
        "with what is weak; it is **not** a verdict and carries no score. Treat each item as a "
        "prompt to strengthen the paper before submission."
    )
    out.append("")
    for key, heading in _REVIEWER2_DIMENSIONS:
        group = sorted(by_dim.get(key, []), key=lambda f: _REVIEWER2_SEVERITY_ORDER[f["severity"]])
        out.append(f"### {heading}")
        out.append("")
        if not group:
            out.append("- _No issue found._")
            out.append("")
            continue
        for f in group:
            out.append(f"- **{f['severity']}** — {f['issue']}")
            if f["evidence"]:
                out.append(f"  > {f['evidence']}")
            if f["suggested_fix"]:
                out.append(f"  _Fix:_ {f['suggested_fix']}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# stitch


def _bullets(items: Iterable[str]) -> str:
    out = []
    for it in items:
        s = (it or "").strip()
        if not s:
            continue
        out.append(f"- {s}")
    return "\n".join(out) if out else "- _(none provided)_"


def _format_rating(low, high, justification: str) -> str:
    try:
        lo = int(low)
        hi = int(high)
    except (TypeError, ValueError):
        lo, hi = 5, 7
    lo = max(1, min(10, lo))
    hi = max(1, min(10, hi))
    if hi < lo:
        lo, hi = hi, lo
    if lo == hi:
        scale = f"**{lo}/10**"
    else:
        scale = f"**{lo}–{hi}/10**"
    return f"{scale}\n\n{justification.strip()}"


# ---------------------------------------------------------------------------
# rating reconciliation → severity-bucketed overall assessment (U5)
#
# Resolves the standing suggested-rating generosity bias by leading the review
# with a de-duplicated severity summary aggregated from the findings already
# produced (Reviewer-2 + submission + citation issues) and dropping the 1–10
# number from the default output (KTD-6 — the safer default against the exact
# false-confidence failure mode the plan exists to prevent). The number is still
# computed by the prose pass but rendered only under --show-rating, with the
# generosity caveat.

_BUCKET_RANK = {"critical": 0, "major": 1, "minor": 2}
# Citation verdicts that read as a "major" issue; everything else problematic is "minor".
_CITATION_MAJOR = (Verdict.TARGET_UNAVAILABLE, Verdict.METADATA_MISMATCH, Verdict.DOES_NOT_SUPPORT)


def _overall_assessment(reviewer2: Optional[dict], bundle: ReviewBundle) -> dict[str, int]:
    """De-duplicated severity counts across Reviewer-2, submission, and citation
    issues. Keyed by the quoted element so one underlying issue surfacing in two
    sources is counted once (highest severity wins)."""
    best_by_key: dict[str, str] = {}
    uid = 0

    def add(bucket: str, quoted: str) -> None:
        nonlocal uid
        key = " ".join((quoted or "").strip().lower().split())
        if not key:
            key = f"__anon_{uid}__"
            uid += 1
        prev = best_by_key.get(key)
        if prev is None or _BUCKET_RANK[bucket] < _BUCKET_RANK[prev]:
            best_by_key[key] = bucket

    for raw in (reviewer2 or {}).get("findings", []) or []:
        c = _coerce_reviewer2_finding(raw)
        if c:
            add(c["severity"], c["evidence"] or c["issue"])

    for f in bundle.paper.submission_findings:
        bucket = "critical" if f.severity == SubmissionSeverity.BLOCKER else "major"
        add(bucket, f.evidence or f.detail)

    for ref_id, sites in _group_by_bibkey(bundle.verifications).items():
        bucket = "major" if _headline_verdict(sites) in _CITATION_MAJOR else "minor"
        add(bucket, ref_id)

    counts = {"critical": 0, "major": 0, "minor": 0}
    for bucket in best_by_key.values():
        counts[bucket] += 1
    return counts


def _render_overall_assessment(
    prose: dict, bundle: ReviewBundle, reviewer2: Optional[dict], *, show_rating: bool
) -> str:
    counts = _overall_assessment(reviewer2, bundle)
    out = ["## Overall assessment", ""]
    if counts["critical"] == 0 and counts["major"] == 0:
        minor_clause = f" ({counts['minor']} minor)" if counts["minor"] else ""
        out.append(
            f"**No critical or major concerns surfaced**{minor_clause} across submission "
            "readiness, citation verification, and the adversarial Reviewer-2 pass. This is "
            "**not** an accept signal — verify against the venue's bar."
        )
    else:
        out.append(
            f"**critical: {counts['critical']} · major: {counts['major']} · "
            f"minor: {counts['minor']}** — across submission readiness, citation "
            "verification, and Reviewer-2. Address the critical and major items first."
        )
    out.append("")
    out.append(
        "_A severity summary of the issues surfaced elsewhere in this review, de-duplicated "
        "across sources. It is **not** an accept/reject recommendation — prereview does not "
        "issue a verdict._"
    )
    if show_rating:
        rating = _format_rating(
            prose.get("rating_low"),
            prose.get("rating_high"),
            prose.get("rating_justification") or "Rating reflects an LLM judgement; treat as suggestive.",
        )
        out.append("")
        out.append("### LLM-estimated rating (secondary)")
        out.append("")
        out.append(rating)
        out.append("")
        out.append(
            "_The 1–10 rating is an LLM estimate and is known to skew generous "
            "(Keuper 2025); treat it as secondary to the severity summary above._"
        )
    return "\n".join(out)


def stitch_review(
    prose: dict, bundle: ReviewBundle, *, reviewer2: Optional[dict] = None, show_rating: bool = False
) -> str:
    paper = bundle.paper
    summary = (prose.get("summary") or "_(no summary returned)_").strip()
    strengths = _bullets(prose.get("strengths") or [])
    weaknesses = _bullets(prose.get("weaknesses") or [])
    questions = _bullets(prose.get("questions") or [])

    head = f"# Pre-submission review\n\n**Paper:** {paper.title or '(title not extracted)'}\n"

    sections = [head]
    coverage = render_coverage_section(bundle)
    if coverage is not None:
        sections.append(coverage.strip())
    # Desk-reject guards lead the review: a deanonymized or non-compliant
    # submission is summarily rejected regardless of content quality.
    anonymization = render_anonymization_section(bundle)
    if anonymization is not None:
        sections.append(anonymization.strip())
    submission = render_submission_section(bundle)
    if submission is not None:
        sections.append(submission.strip())
    # Honest overall signal: lead with the de-duplicated severity summary, not an
    # inflated 1–10 (U5). The number is opt-in via --show-rating.
    sections.append(_render_overall_assessment(prose, bundle, reviewer2, show_rating=show_rating))
    sections += [
        "## Summary",
        summary,
        "## Strengths",
        strengths,
        "## Weaknesses",
        weaknesses,
    ]
    reviewer2_md = render_reviewer2_section(reviewer2, bundle)
    if reviewer2_md is not None:
        sections.append(reviewer2_md.strip())
    sections.append(render_citation_issues(bundle).strip())
    hygiene = render_hygiene_section(bundle)
    if hygiene is not None:
        sections.append(hygiene.strip())
    checklist = render_checklist_section(bundle)
    if checklist is not None:
        sections.append(checklist.strip())
    numeric = render_numeric_section(bundle)
    if numeric is not None:
        sections.append(numeric.strip())
    artifacts = render_artifacts_section(bundle)
    if artifacts is not None:
        sections.append(artifacts.strip())
    sections.extend(
        [
            "## Questions for the author",
            questions,
            render_methodology(bundle).strip(),
            "",
        ]
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# entry point


async def synthesize_review(
    bundle: ReviewBundle, *, verbose: bool = False, run_reviewer2: bool = True, show_rating: bool = False
) -> str:
    _log(verbose, f"prose pass via {bundle.synthesis_model}")
    try:
        prose = await _generate_prose(bundle, verbose=verbose)
    except Exception as e:
        # The prose pass is one of two LLM steps here. If it fails (after retries), keep the
        # deterministic sections — citation issues, hygiene, checklist, coverage,
        # methodology — rather than losing the whole review, and disclose the gap.
        _log(verbose, f"prose pass failed: {e!r}; writing deterministic sections only")
        if bundle.coverage is not None:
            bundle.coverage.synthesis_degraded = True
        prose = {}

    # The adversarial Reviewer-2 pass is a SEPARATE LLM call with its own failure
    # boundary: a Reviewer-2 failure must not mislabel the prose/narrative sections
    # as degraded (KTD-5 / U4), so it sets its own flag and never touches prose.
    reviewer2: Optional[dict] = None
    if run_reviewer2:
        _log(verbose, f"reviewer-2 pass via {bundle.synthesis_model}")
        try:
            reviewer2 = await _generate_reviewer2(bundle, verbose=verbose)
        except Exception as e:
            _log(verbose, f"reviewer-2 pass failed: {e!r}; continuing without it")
            if bundle.coverage is not None:
                bundle.coverage.reviewer2_degraded = True
            reviewer2 = None

    return stitch_review(prose, bundle, reviewer2=reviewer2, show_rating=show_rating)
