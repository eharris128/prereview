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
from typing import Iterable

from .llm import acompletion_json
from .models import ReviewBundle, VerificationResult, Verdict


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
    abstract-supports a load-bearing claim."""
    if v.verdict in _PROBLEM_VERDICTS:
        return True
    if v.verdict == Verdict.ABSTRACT_TOO_THIN:
        return True
    if body_load_bearing and v.abstract_only and v.verdict == Verdict.SUPPORTS:
        # Abstract-only "supports" — flag it so the user can verify.
        return True
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


def render_citation_issues(bundle: ReviewBundle) -> str:
    """Markdown for the Citation issues section. Lists every problematic citation."""
    out: list[str] = ["## Citation issues", ""]

    flagged = [v for v in bundle.verifications if _is_problematic(v)]
    if not flagged:
        out.append(
            "_No citation issues detected._ Every cited reference resolved against "
            "Crossref / Semantic Scholar / arXiv / OpenAlex, and every claim was "
            "supported by the cited paper's full text."
        )
        out.append("")
        return "\n".join(out)

    counts: dict[str, int] = {}
    for v in flagged:
        counts[v.verdict.value] = counts.get(v.verdict.value, 0) + 1

    out.append(
        "The following citations were flagged. Every entry shows the in-text "
        "sentence, the bibliography entry as the author wrote it, the canonical "
        "record we resolved (if any), the verdict, and a one-sentence rationale."
    )
    out.append("")
    out.append("**Summary of flags:** " + ", ".join(
        f"{counts[k]}× {k.replace('_', ' ')}" for k in sorted(counts)
    ))
    out.append("")

    for i, v in enumerate(flagged, start=1):
        out.append(f"### {i}. `{v.reference.ref_id}` — {_verdict_label(v.verdict)}")
        out.append("")
        out.append(f"**In-text sentence:** {_quote(v.citation.sentence)}")
        out.append("")
        out.append(f"**Bibliography entry (verbatim):** {_quote(v.reference.raw_text)}")
        out.append("")
        if v.canonical is not None:
            out.append("**Resolved record:**")
            out.append("")
            out.append(f"- Source: `{v.canonical.source}`")
            out.append(f"- Title: {v.canonical.title or '_(unknown)_'}")
            if v.canonical.authors:
                out.append("- Authors: " + ", ".join(v.canonical.authors[:8]))
            if v.canonical.year:
                out.append(f"- Year: {v.canonical.year}")
            if v.canonical.venue:
                out.append(f"- Venue: {v.canonical.venue}")
            if v.canonical.doi:
                out.append(f"- DOI: [{v.canonical.doi}](https://doi.org/{v.canonical.doi})")
            if v.canonical.url and not v.canonical.doi:
                out.append(f"- URL: {v.canonical.url}")
        else:
            out.append(
                "**Resolved record:** _none — this entry could not be matched in "
                "Crossref, Semantic Scholar, arXiv, or OpenAlex._"
            )
        out.append("")
        evidence_tag = "abstract-only" if v.abstract_only else "full-text"
        out.append(f"**Verdict:** {_verdict_label(v.verdict)} _(evidence: {evidence_tag})_")
        out.append("")
        out.append(f"**Rationale:** {v.rationale}")
        out.append("")

    return "\n".join(out)


def _quote(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    if not s:
        return "_(empty)_"
    return f"> {s}"


def render_methodology(bundle: ReviewBundle) -> str:
    total = len(bundle.verifications)
    flagged = sum(1 for v in bundle.verifications if _is_problematic(v))
    return textwrap.dedent(f"""
    ## Methodology and limits of this review

    This review was produced by `prereview`, an AI-assisted pre-submission tool. The retrieval
    and extraction passes used `{bundle.model}`; the synthesis pass used `{bundle.synthesis_model}`.

    Of {total} in-text citations, {bundle.unresolved_count} bibliography entr{"y" if bundle.unresolved_count == 1 else "ies"} did not
    resolve to any record in Crossref, Semantic Scholar, arXiv, or OpenAlex. Of the citations
    that did resolve, {bundle.fetched_full_text_count} were verified against the cited paper's full text and
    {bundle.abstract_only_count} were verified against the abstract only. {flagged} citation{"" if flagged == 1 else "s"} {"was" if flagged == 1 else "were"} flagged
    in the section above.

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

    flagged = [v for v in bundle.verifications if _is_problematic(v)]
    flag_summary = "\n".join(
        f"- `{v.reference.ref_id}` ({_verdict_label(v.verdict)}): {v.rationale}"
        for v in flagged[:30]
    ) or "_None — every cited reference checked out._"
    if len(flagged) > 30:
        flag_summary += f"\n- ... and {len(flagged) - 30} more"

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

    return "\n\n".join(
        [
            head,
            "## Summary",
            summary,
            "## Strengths",
            strengths,
            "## Weaknesses",
            weaknesses,
            render_citation_issues(bundle).strip(),
            "## Questions for the author",
            questions,
            "## Suggested rating",
            rating,
            render_methodology(bundle).strip(),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# entry point


async def synthesize_review(bundle: ReviewBundle, *, verbose: bool = False) -> str:
    _log(verbose, f"prose pass via {bundle.synthesis_model}")
    prose = await _generate_prose(bundle, verbose=verbose)
    return stitch_review(prose, bundle)
