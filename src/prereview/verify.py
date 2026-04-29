"""Stage 3: classify whether a resolved reference supports the surrounding claim.

For each in-text citation that resolved in stage 2:

- If the cited paper has an open-access PDF and ``fetch_cited`` is True, download
  it (caching the bytes) and pass the extracted text to the LLM as evidence.
- Otherwise, fall back to the abstract and mark ``abstract_only=True``.
- If neither full text nor abstract are available, return ``abstract_too_thin``.

The LLM never invents canonical metadata; it only judges retrieved text.
The verdict ``abstract_too_thin`` is a first-class output, not papered over.
"""

from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from typing import Optional

import httpx


def _looks_like_oa_host(url: str) -> bool:
    """Hosts that are reliably OA: arXiv, biorxiv, Springer-OA, ACL, etc."""
    u = url.lower()
    return any(
        h in u
        for h in (
            "arxiv.org/pdf",
            "biorxiv.org",
            "openreview.net/pdf",
            "aclanthology.org",
            "patricegodefroid.github.io",
            ".github.io/",
        )
    )

from . import __version__
from .cache import Cache, cache_key, verification_key
from .llm import acompletion_json
from .models import (
    CanonicalRecord,
    Citation,
    CitationRole,
    Reference,
    VerificationResult,
    Verdict,
)


# Bump when the verification prompt changes meaningfully — included in the
# cache key so that cached verdicts from older prompts don't shadow new ones.
_PROMPT_VERSION = "v2-roles"


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[verify] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# prompt


_VERIFY_SYSTEM = (
    "You evaluate whether a paper's cited reference actually supports the surrounding "
    "claim in the citing paper. You are precise, grounded in the evidence text you are "
    "shown, and you abstain when the evidence is insufficient. You never invent facts "
    "about the cited paper that aren't in the evidence."
)


_VALID_VERDICTS = {
    "supports": Verdict.SUPPORTS,
    "partially_supports": Verdict.PARTIALLY_SUPPORTS,
    "does_not_support": Verdict.DOES_NOT_SUPPORT,
    "abstract_too_thin": Verdict.ABSTRACT_TOO_THIN,
}

_VALID_ROLES = {
    "method_attribution": CitationRole.METHOD_ATTRIBUTION,
    "claim_support": CitationRole.CLAIM_SUPPORT,
    "background": CitationRole.BACKGROUND,
}


_TITLE_TOK_RE = re.compile(r"[a-z0-9]+")


def _norm_title_tokens(s: str) -> set[str]:
    s = (s or "").lower()
    return set(_TITLE_TOK_RE.findall(s))


def _surname(author: str) -> str:
    s = author.strip()
    if "," in s:
        return s.split(",", 1)[0].strip().lower()
    return s.split()[-1].lower() if s else ""


def detect_metadata_mismatch(
    reference: Reference,
    canonical: CanonicalRecord,
) -> Optional[str]:
    """Return a one-sentence reason if the bibliography entry disagrees with
    the resolved canonical record on authorship, else None.

    Author-surname mismatch is the high-confidence signal: a resolver that
    fetches by DOI never validates against bibliography authors, so a wrong
    DOI in the .bib silently substitutes the wrong paper. We surface that
    separately because the LLM can't catch it without an abstract.

    Title divergence on its own is intentionally not flagged — Crossref
    sometimes truncates titles ("Optuna" instead of "Optuna: A Next-generation
    ...") on records whose author list matches perfectly, and we don't want
    those false positives to drown out the real cases.
    """
    if not (reference.authors and canonical.authors):
        return None

    bib_surnames = {s for s in (_surname(a) for a in reference.authors[:5]) if s}
    can_surnames = {s for s in (_surname(a) for a in canonical.authors[:5]) if s}
    if not (bib_surnames and can_surnames):
        return None
    if bib_surnames & can_surnames:
        return None

    # Build a clear rationale that names the diverging title too if relevant
    # — context for the user, but not used as a trigger.
    title_clause = ""
    if reference.title and canonical.title:
        title_clause = (
            f' The resolved record\'s title is "{canonical.title}" '
            f'(bibliography says "{reference.title}").'
        )
    return (
        f"No author-surname overlap between bibliography "
        f"({', '.join(sorted(bib_surnames))}) and resolved record "
        f"({', '.join(sorted(can_surnames))}) — likely a misattributed entry "
        f"or a wrong DOI in the .bib.{title_clause}"
    )


def _verify_prompt(
    citation: Citation,
    reference: Reference,
    canonical: CanonicalRecord,
    evidence: str,
    abstract_only: bool,
) -> str:
    cited_authors = ", ".join(canonical.authors[:5]) or "(unknown)"
    raw = reference.raw_text or "(not extracted)"
    evidence_label = "abstract" if abstract_only else "excerpt(s) from full text"
    return f"""Verify a citation in a paper. First classify the citation's role in its sentence, then judge it by the criteria appropriate to that role.

CITING SENTENCE (where the citation appears in the paper under review):
\"\"\"{citation.sentence}\"\"\"

BIBLIOGRAPHY ENTRY (as the author wrote it):
\"\"\"{raw}\"\"\"

CITED PAPER (resolved canonically against {canonical.source}):
- Title: {canonical.title}
- Authors: {cited_authors}
- Year: {canonical.year}
- Venue: {canonical.venue or "(unknown)"}

EVIDENCE ({evidence_label}, from the cited paper):
\"\"\"{evidence}\"\"\"

STEP 1 — Classify the citation's ROLE:

- "method_attribution" — the citation attributes a named method, tool, library, dataset, format, framework, or algorithm that the citing sentence uses or references by name (e.g., "we use HST [cite]", "tuned with Optuna [cite]", "Croissant metadata [cite]", "our reimplementation of Anomalicious [cite]"). The relevant question is: is the cited paper the canonical/defining paper for the named thing?
- "claim_support" — the citing sentence makes a specific factual or quantitative claim (a number, a comparison, a mechanism, a finding) and the citation is offered as evidence for it (e.g., "X has been shown to occur in Y% of cases [cite]", "payloads use RFC 2606 hosts [cite]"). The relevant question is: does the cited paper actually back that specific claim?
- "background" — the citation gestures at related work, an example, an incident, or a body of prior literature that the citing sentence summarizes or alludes to (e.g., "incidents like SolarWinds [cite]", "prior surveys of supply-chain attacks [cite]", "recent in-depth case studies [cite]"). The relevant question is: is the cited paper on-topic for the background or example being invoked?

When the citation could plausibly fit two roles, prefer the more specific one: method_attribution > claim_support > background.

STEP 2 — Apply role-appropriate criteria and choose ONE verdict.

For "method_attribution":
- "supports" — the cited paper is the canonical/original paper for the named method/tool/concept (judge by title, authors, abstract). A method paper from year N CANNOT be expected to contain results from a citing paper written in year N+M; absence of the citing paper's own numbers is NOT a reason to downgrade.
- "partially_supports" — the cited paper is a follow-up, variant, or survey covering the named thing rather than the canonical defining paper.
- "does_not_support" — the cited paper is clearly the wrong paper for the attribution (e.g., wrong topic entirely).
- "abstract_too_thin" — only when title, authors, and abstract together are insufficient to decide. This should be rare for method attributions.

For "claim_support":
- "supports" — the cited evidence directly substantiates the specific claim in the citing sentence.
- "partially_supports" — the cited evidence backs a related, narrower, or qualified version of the claim, but not the claim as written.
- "does_not_support" — the cited evidence does not substantiate the specific claim, or contradicts it.
- "abstract_too_thin" — the claim is specific (a number, a comparison, a precise mechanism) and only an abstract is available, so the body would be needed to decide.

For "background":
- "supports" — the cited paper is on-topic for the background, example, or body of work the citing sentence is gesturing at.
- "partially_supports" — the cited paper is tangentially relevant (related field, but not really about the example invoked).
- "does_not_support" — the cited paper is off-topic for the background being invoked.
- "abstract_too_thin" — only when even the title is too vague to judge on-topic-ness.

Return JSON of the form:
{{"role": "method_attribution|claim_support|background", "verdict": "supports|partially_supports|does_not_support|abstract_too_thin", "rationale": "one sentence grounded in the cited evidence"}}

Hard rules:
- The rationale must paraphrase or quote concrete content from the EVIDENCE above. Do not speculate beyond it.
- Do NOT downgrade a method_attribution to "partially_supports" merely because the cited paper does not contain numbers or results from the citing paper. That is structural and expected.
- Do NOT downgrade a background to "partially_supports" merely because the citing sentence's wording does not appear verbatim in the cited paper.
- Reserve "does_not_support" for cases where the cited paper is genuinely the wrong paper for the citation's role — wrong method, contradicted claim, off-topic background.
"""


# ---------------------------------------------------------------------------
# verifier


class Verifier:
    def __init__(
        self,
        *,
        cache: Cache,
        polite_mailto: Optional[str] = None,
        timeout_s: float = 60.0,
        max_evidence_chars: int = 80_000,
    ):
        self.cache = cache
        self.max_evidence_chars = max_evidence_chars
        ua = f"prereview/{__version__}"
        if polite_mailto:
            ua += f" (mailto:{polite_mailto})"
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": ua},
            follow_redirects=True,
        )

    async def __aenter__(self) -> "Verifier":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def verify(
        self,
        citation: Citation,
        reference: Reference,
        canonical: Optional[CanonicalRecord],
        *,
        model: str,
        fetch_cited: bool = True,
        verbose: bool = False,
    ) -> VerificationResult:
        if canonical is None:
            return VerificationResult(
                ref_id=reference.ref_id,
                citation=citation,
                reference=reference,
                canonical=None,
                verdict=Verdict.TARGET_UNAVAILABLE,
                rationale="Reference did not resolve to a record in Crossref, Semantic Scholar, arXiv, or OpenAlex.",
                abstract_only=True,
            )

        mismatch = detect_metadata_mismatch(reference, canonical)
        if mismatch:
            return VerificationResult(
                ref_id=reference.ref_id,
                citation=citation,
                reference=reference,
                canonical=canonical,
                verdict=Verdict.METADATA_MISMATCH,
                rationale=mismatch,
                abstract_only=True,
            )

        evidence: Optional[str] = None
        abstract_only = True

        if fetch_cited:
            full_text = await self._fetch_full_text(canonical, reference, verbose=verbose)
            if full_text:
                evidence = full_text[: self.max_evidence_chars]
                abstract_only = False

        if evidence is None and canonical.abstract:
            evidence = canonical.abstract

        if evidence is None or not evidence.strip():
            # Crossref/etc returned a canonical record but no abstract, and no
            # OA PDF was findable. Don't short-circuit — let the role-aware
            # prompt judge from title and authors alone. For method_attribution
            # cites, title+authors are usually enough; for claim_support, the
            # model will correctly return abstract_too_thin.
            evidence = "(no abstract or full text was retrievable for this paper)"
            abstract_only = True

        # Cache lookup. Key includes the evidence marker so that re-running
        # with full-text fetched (or with --no-fetch-cited) doesn't reuse the
        # other mode's verdict. The prompt version invalidates cached verdicts
        # whenever the verification prompt's semantics change.
        evidence_marker = (
            f"prompt={_PROMPT_VERSION}|abstract_only={int(abstract_only)}|evlen={len(evidence)}"
        )
        vkey = verification_key(
            model=model,
            ref_id=reference.ref_id,
            sentence=citation.sentence,
            evidence_marker=evidence_marker,
        )
        cached = self.cache.get_verification_json(vkey)
        if cached is not None:
            try:
                _log(verbose, f"verify cache hit for {reference.ref_id}")
                return VerificationResult.model_validate_json(cached)
            except Exception:
                pass  # fall through and re-compute

        try:
            data = await acompletion_json(
                model=model,
                system=_VERIFY_SYSTEM,
                user=_verify_prompt(citation, reference, canonical, evidence, abstract_only),
                temperature=0.0,
                verbose=verbose,
            )
        except Exception as e:
            _log(verbose, f"LLM call failed for {reference.ref_id}: {e!r}")
            return VerificationResult(
                ref_id=reference.ref_id,
                citation=citation,
                reference=reference,
                canonical=canonical,
                verdict=Verdict.ABSTRACT_TOO_THIN,
                rationale=f"Verification model call failed; treating as undecided.",
                abstract_only=abstract_only,
            )

        verdict_key = (data.get("verdict") or "").strip().lower()
        verdict = _VALID_VERDICTS.get(verdict_key, Verdict.ABSTRACT_TOO_THIN)
        role_key = (data.get("role") or "").strip().lower()
        role = _VALID_ROLES.get(role_key)
        rationale = (data.get("rationale") or "").strip() or "(no rationale returned)"
        result = VerificationResult(
            ref_id=reference.ref_id,
            citation=citation,
            reference=reference,
            canonical=canonical,
            verdict=verdict,
            rationale=rationale,
            abstract_only=abstract_only,
            role=role,
        )
        try:
            self.cache.put_verification_json(vkey, result.model_dump_json())
        except Exception as e:
            _log(verbose, f"failed to cache verification for {reference.ref_id}: {e!r}")
        return result

    # ----- full-text fetch ------------------------------------------------

    async def _fetch_full_text(
        self,
        canonical: CanonicalRecord,
        reference: Optional[Reference] = None,
        *,
        verbose: bool,
    ) -> Optional[str]:
        """Download an OA PDF for ``canonical`` and return its extracted text.

        Tries several discovery routes in order: the canonical record's own OA
        URL, the bibliography entry's URL field (if it points to a PDF or
        a known OA host), Unpaywall by DOI, and OpenAlex by DOI. The first
        route that yields a real PDF wins; the others are silently skipped.
        """
        candidates: list[str] = []
        if canonical.open_access_pdf_url:
            candidates.append(canonical.open_access_pdf_url)

        # Reference URL: the .bib often has a direct PDF link.
        if reference and reference.url:
            u = reference.url.strip()
            if u and u not in candidates:
                if u.lower().endswith(".pdf") or _looks_like_oa_host(u):
                    candidates.append(u)

        if canonical.doi:
            for fn in (self._unpaywall_oa, self._openalex_oa):
                try:
                    extra = await fn(canonical.doi, verbose=verbose)
                except Exception as e:
                    _log(verbose, f"  OA discovery {fn.__name__} failed: {e!r}")
                    extra = None
                if extra and extra not in candidates:
                    candidates.append(extra)

        key = cache_key(
            doi=canonical.doi,
            arxiv_id=None,
            title=canonical.title,
            first_author=canonical.authors[0] if canonical.authors else None,
        )
        path = self.cache.pdf_path(key)

        if self.cache.has_pdf(key):
            try:
                return _extract_pdf_text(path.read_bytes())
            except Exception:
                pass  # fall through to re-fetch

        for url in candidates:
            try:
                _log(verbose, f"fetching {url}")
                r = await self.client.get(url)
                if r.status_code != 200 or not r.content:
                    _log(verbose, f"  HTTP {r.status_code}")
                    continue
                ct = r.headers.get("content-type", "").lower()
                if "pdf" not in ct and not r.content.startswith(b"%PDF"):
                    _log(verbose, f"  not a PDF ({ct})")
                    continue
                path.write_bytes(r.content)
                return _extract_pdf_text(r.content)
            except Exception as e:
                _log(verbose, f"  fetch failed: {e!r}")
                continue
        return None

    async def _unpaywall_oa(self, doi: str, *, verbose: bool) -> Optional[str]:
        """Look up an OA PDF URL via Unpaywall. Free; needs an email for politeness."""
        # Unpaywall requires an email parameter. We use whatever the user
        # provided via --mailto, or fall back to a generic "anonymous" form
        # which Unpaywall still accepts but rate-limits more aggressively.
        from urllib.parse import quote

        email = os.environ.get("PREREVIEW_MAILTO") or "anonymous@example.org"
        url = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email={quote(email)}"
        r = await self.client.get(url, timeout=15.0)
        if r.status_code != 200:
            return None
        data = r.json()
        # Prefer best_oa_location, fall back to first oa_locations entry.
        for slot in (data.get("best_oa_location"),) + tuple(data.get("oa_locations") or ()):
            if isinstance(slot, dict):
                pdf = slot.get("url_for_pdf") or slot.get("url")
                if pdf:
                    return pdf
        return None

    async def _openalex_oa(self, doi: str, *, verbose: bool) -> Optional[str]:
        from urllib.parse import quote

        url = f"https://api.openalex.org/works/doi:{quote(doi, safe='')}"
        params = {}
        mailto = os.environ.get("PREREVIEW_MAILTO")
        if mailto:
            params["mailto"] = mailto
        r = await self.client.get(url, params=params, timeout=15.0)
        if r.status_code != 200:
            return None
        data = r.json()
        oa = data.get("open_access") or {}
        if isinstance(oa, dict) and oa.get("oa_url"):
            return oa["oa_url"]
        # primary_location.pdf_url
        loc = data.get("primary_location") or {}
        if isinstance(loc, dict):
            pdf = loc.get("pdf_url")
            if pdf:
                return pdf
        return None


# ---------------------------------------------------------------------------
# helpers


def _extract_pdf_text(data: bytes) -> str:
    import logging as _logging

    from pypdf import PdfReader

    # pypdf chatters about "Ignoring wrong pointing object" on many real PDFs.
    # These are recoverable parse warnings, not errors — silence them so they
    # don't drown out our own --verbose output.
    pypdf_logger = _logging.getLogger("pypdf")
    prev_level = pypdf_logger.level
    pypdf_logger.setLevel(_logging.ERROR)
    try:
        reader = PdfReader(BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
    finally:
        pypdf_logger.setLevel(prev_level)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# convenience wrapper used by the pipeline


async def verify_citation(
    citation: Citation,
    reference: Reference,
    canonical: Optional[CanonicalRecord],
    *,
    model: str,
    fetch_cited: bool = True,
    verbose: bool = False,
    cache: Optional[Cache] = None,
) -> VerificationResult:
    if cache is None:
        cache = Cache()
    async with Verifier(cache=cache) as v:
        return await v.verify(
            citation,
            reference,
            canonical,
            model=model,
            fetch_cited=fetch_cited,
            verbose=verbose,
        )
