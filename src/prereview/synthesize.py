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
    ChecklistFinding,
    ChecklistFindingKind,
    CitationRole,
    ReviewBundle,
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
    }[v]


# Severity ordering for sorting groups and picking a headline verdict within a
# group. Lower number = more severe / higher in the list.
_SEVERITY: dict[Verdict, int] = {
    Verdict.TARGET_UNAVAILABLE: 0,
    Verdict.METADATA_MISMATCH: 1,
    Verdict.DOES_NOT_SUPPORT: 2,
    Verdict.PARTIALLY_SUPPORTS: 3,
    Verdict.ABSTRACT_TOO_THIN: 4,
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


def render_methodology(bundle: ReviewBundle) -> str:
    total = len(bundle.verifications)
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
    return textwrap.dedent(f"""
    ## Methodology and limits of this review

    This review was produced by `prereview`, an AI-assisted pre-submission tool. The retrieval
    and extraction passes used `{bundle.model}`; the synthesis pass used `{bundle.synthesis_model}`.

    Of {total} in-text citations, {bundle.unresolved_count} bibliography entr{"y" if bundle.unresolved_count == 1 else "ies"} did not
    resolve to any record in Crossref, Semantic Scholar, arXiv, or OpenAlex. Of the citations
    that did resolve, {bundle.fetched_full_text_count} were verified against the cited paper's full text and
    {bundle.abstract_only_count} were verified against the abstract only. {flagged} citation{"" if flagged == 1 else "s"} {"was" if flagged == 1 else "were"} flagged
    in the section above.

    {hygiene_line}

    What this tool does **not** check:

    - Figures, tables, captions.
    - Mathematical proofs or formal correctness.
    - Reported statistics (no statcheck / GRIM-style audit).
    - Plagiarism or text-overlap with prior work.
    - **Missing prior work** — `prereview` only verifies citations the author already made; it does
      not search for relevant work the author should have cited.

    The suggested rating is a model judgement and not authoritative.
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
- Unresolved bibliography entries (potential ghost references): {bundle.unresolved_count}
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


def stitch_review(prose: dict, bundle: ReviewBundle) -> str:
    paper = bundle.paper
    summary = (prose.get("summary") or "_(no summary returned)_").strip()
    strengths = _bullets(prose.get("strengths") or [])
    weaknesses = _bullets(prose.get("weaknesses") or [])
    questions = _bullets(prose.get("questions") or [])
    rating = _format_rating(
        prose.get("rating_low"),
        prose.get("rating_high"),
        prose.get("rating_justification") or "Rating reflects an LLM judgement; treat as suggestive.",
    )

    head = f"# Pre-submission review\n\n**Paper:** {paper.title or '(title not extracted)'}\n"

    sections = [
        head,
        "## Summary",
        summary,
        "## Strengths",
        strengths,
        "## Weaknesses",
        weaknesses,
        render_citation_issues(bundle).strip(),
    ]
    hygiene = render_hygiene_section(bundle)
    if hygiene is not None:
        sections.append(hygiene.strip())
    checklist = render_checklist_section(bundle)
    if checklist is not None:
        sections.append(checklist.strip())
    sections.extend(
        [
            "## Questions for the author",
            questions,
            "## Suggested rating",
            rating,
            render_methodology(bundle).strip(),
            "",
        ]
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# entry point


async def synthesize_review(bundle: ReviewBundle, *, verbose: bool = False) -> str:
    _log(verbose, f"prose pass via {bundle.synthesis_model}")
    prose = await _generate_prose(bundle, verbose=verbose)
    return stitch_review(prose, bundle)
