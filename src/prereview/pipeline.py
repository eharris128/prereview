"""End-to-end orchestration: ingest → resolve → verify → synthesize → write."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from .cache import Cache
from .ingest import ingest_pdf
from .link_health import check_links
from .models import (
    CoverageReport,
    Resolution,
    ResolutionStatus,
    ReviewBundle,
    VerificationResult,
    Verdict,
)
from .resolve import Resolver
from .synthesize import synthesize_review
from .tex_ingest import ingest_tex
from .venue_rules import DEFAULT_VENUE, collect_gate_blockers
from .verify import Verifier


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[prereview] {msg}", file=sys.stderr, flush=True)


def _backup_if_exists(out: Path) -> Optional[Path]:
    if not out.exists():
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = out.with_suffix(out.suffix + f".bak.{ts}")
    out.rename(bak)
    return bak


async def run_pipeline(
    pdf_path: Path,
    *,
    out: Path,
    model: str,
    synthesis_model: str,
    fetch_cited: bool = True,
    cache_dir: Optional[Path] = None,
    polite_mailto: Optional[str] = None,
    bib_path: Optional[Path] = None,
    checklist_path: Optional[Path] = None,
    run_checklist: bool = True,
    run_anonymize: bool = True,
    authors: Optional[str] = None,
    venue: str = DEFAULT_VENUE,
    abstract_baseline: Optional[Path] = None,
    run_reviewer2: bool = True,
    show_rating: bool = False,
    run_numeric: bool = True,
    run_artifacts: bool = False,
    run_openreview: bool = False,
    verbose: bool = False,
) -> tuple[Path, CoverageReport]:
    cache = Cache(cache_dir) if cache_dir else Cache()
    s2_key = os.environ.get("S2_API_KEY")
    openalex_key = os.environ.get("OPENALEX_API_KEY")

    suffix = pdf_path.suffix.lower()
    if suffix == ".tex":
        _log(verbose, f"Stage 1: ingesting (tex mode) {pdf_path}")
        paper = await ingest_tex(
            pdf_path,
            model=model,
            verbose=verbose,
            bib_path=bib_path,
            checklist_path=checklist_path,
            run_checklist=run_checklist,
            run_anonymize=run_anonymize,
            authors=authors,
            venue=venue,
            abstract_baseline=abstract_baseline,
            run_numeric=run_numeric,
        )
    else:
        _log(verbose, f"Stage 1: ingesting (pdf mode) {pdf_path}")
        paper = await ingest_pdf(pdf_path, model=model, verbose=verbose, venue=venue)
    _log(verbose, f"  parsed {len(paper.references)} references, {len(paper.citations)} in-text citations")
    if not paper.references:
        print(
            "prereview: warning — no bibliography entries were parsed from the input; "
            "the review will contain no citation checks.",
            file=sys.stderr,
            flush=True,
        )
    elif not paper.citations:
        print(
            "prereview: warning — bibliography entries parsed but no in-text citations "
            "were found; there is nothing to verify.",
            file=sys.stderr,
            flush=True,
        )

    if paper.link_checks:
        _log(verbose, f"Stage 1.5: probing {len(paper.link_checks)} URLs for reachability")
        paper.link_checks = await check_links(paper.link_checks, verbose=verbose)
        n_bad = sum(1 for c in paper.link_checks if not c.ok)
        _log(verbose, f"  {n_bad} URL{'s' if n_bad != 1 else ''} unreachable")

    if run_artifacts:
        _log(verbose, "Stage 1.6: checking claimed artifacts (HF / GitHub)")
        from .artifacts import check_artifacts

        body_text = "\n\n".join(b for _, b in paper.sections) if paper.sections else ""
        paper.artifact_checks = await check_artifacts(
            paper.link_checks, body_text, hf_token=os.environ.get("HF_TOKEN"), verbose=verbose
        )

    _log(verbose, "Stage 2: resolving references")
    canonical_by_ref: dict[str, Resolution] = {}
    async with Resolver(
        cache=cache,
        s2_api_key=s2_key,
        openalex_api_key=openalex_key,
        polite_mailto=polite_mailto,
        verbose=verbose,
    ) as resolver:
        for ref_id, ref in paper.references.items():
            resolution = await resolver.resolve(ref)
            canonical_by_ref[ref_id] = resolution
            if verbose:
                tag = resolution.record.source if resolution.record else resolution.status.value
                _log(verbose, f"  {ref_id}: {tag}")

    _log(verbose, "Stage 3: verifying claim support")
    verifications: list[VerificationResult] = []
    async with Verifier(cache=cache, polite_mailto=polite_mailto) as verifier:
        for cit in paper.citations:
            ref = paper.references.get(cit.ref_id)
            if ref is None:
                continue
            resolution = canonical_by_ref.get(cit.ref_id)
            if resolution is None:
                continue  # defensive: resolve populated an entry for every reference
            result = await verifier.verify(
                cit,
                ref,
                resolution,
                model=model,
                fetch_cited=fetch_cited,
                verbose=verbose,
            )
            verifications.append(result)

    openreview_degraded = False
    if run_openreview:
        _log(verbose, "Stage 3.5: enriching cited papers with OpenReview decisions")
        from .openreview_enrich import enrich_with_openreview

        _, openreview_degraded = await enrich_with_openreview(
            verifications,
            username=os.environ.get("OPENREVIEW_USERNAME"),
            password=os.environ.get("OPENREVIEW_PASSWORD"),
            verbose=verbose,
        )

    abstract_only_count = sum(1 for v in verifications if v.abstract_only)
    fetched_full_text_count = sum(
        1 for v in verifications if not v.abstract_only and v.verdict != Verdict.TARGET_UNAVAILABLE
    )

    # Coverage integrity: partition each citation's outcome so the review can disclose
    # what it could and couldn't check, and the CLI can pick an honest exit code.
    resolution_degraded = verification_degraded = ghost = resolved = 0
    for v in verifications:
        res = canonical_by_ref.get(v.ref_id)
        if res is not None and res.status == ResolutionStatus.DEGRADED:
            resolution_degraded += 1
        elif v.verdict == Verdict.VERIFICATION_UNAVAILABLE:
            verification_degraded += 1  # resolved record, but the verify call failed after retries
        elif v.verdict == Verdict.TARGET_UNAVAILABLE:
            ghost += 1
        else:
            resolved += 1

    coverage = CoverageReport(
        references_parsed=len(paper.references),
        citations_checked=len(verifications),
        resolved=resolved,
        ghost_unresolved=ghost,
        resolution_degraded=resolution_degraded,
        verification_degraded=verification_degraded,
        recovered_after_retry=resolver.recovered_after_retry,
        circuit_broken_sources=resolver.circuit_broken_sources,
        gate_blockers=collect_gate_blockers(paper),
        openreview_degraded=openreview_degraded,
    )

    bundle = ReviewBundle(
        paper=paper,
        verifications=verifications,
        model=model,
        synthesis_model=synthesis_model,
        fetched_full_text_count=fetched_full_text_count,
        abstract_only_count=abstract_only_count,
        unresolved_count=ghost,  # true ghosts only; coverage is the authoritative report
        coverage=coverage,
    )

    _log(verbose, "Stage 4: synthesizing review")
    markdown = await synthesize_review(
        bundle, verbose=verbose, run_reviewer2=run_reviewer2, show_rating=show_rating
    )

    _log(verbose, f"Stage 5: writing {out}")
    bak = _backup_if_exists(out)
    if bak is not None:
        _log(verbose, f"  backed up previous review to {bak}")
    out.write_text(markdown)
    return out, bundle.coverage
