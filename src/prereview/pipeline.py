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
from .models import Resolution, ReviewBundle, VerificationResult, Verdict
from .resolve import Resolver
from .synthesize import synthesize_review
from .tex_ingest import ingest_tex
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
    verbose: bool = False,
) -> Path:
    cache = Cache(cache_dir) if cache_dir else Cache()
    s2_key = os.environ.get("S2_API_KEY")

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
        )
    else:
        _log(verbose, f"Stage 1: ingesting (pdf mode) {pdf_path}")
        paper = await ingest_pdf(pdf_path, model=model, verbose=verbose)
    _log(verbose, f"  parsed {len(paper.references)} references, {len(paper.citations)} in-text citations")

    if paper.link_checks:
        _log(verbose, f"Stage 1.5: probing {len(paper.link_checks)} URLs for reachability")
        paper.link_checks = await check_links(paper.link_checks, verbose=verbose)
        n_bad = sum(1 for c in paper.link_checks if not c.ok)
        _log(verbose, f"  {n_bad} URL{'s' if n_bad != 1 else ''} unreachable")

    _log(verbose, "Stage 2: resolving references")
    canonical_by_ref: dict[str, Resolution] = {}
    async with Resolver(
        cache=cache,
        s2_api_key=s2_key,
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
            canonical = resolution.record if resolution else None
            result = await verifier.verify(
                cit,
                ref,
                canonical,
                model=model,
                fetch_cited=fetch_cited,
                verbose=verbose,
            )
            verifications.append(result)

    abstract_only_count = sum(1 for v in verifications if v.abstract_only)
    fetched_full_text_count = sum(
        1 for v in verifications if not v.abstract_only and v.verdict != Verdict.TARGET_UNAVAILABLE
    )
    unresolved_count = sum(1 for res in canonical_by_ref.values() if res.record is None)

    bundle = ReviewBundle(
        paper=paper,
        verifications=verifications,
        model=model,
        synthesis_model=synthesis_model,
        fetched_full_text_count=fetched_full_text_count,
        abstract_only_count=abstract_only_count,
        unresolved_count=unresolved_count,
    )

    _log(verbose, "Stage 4: synthesizing review")
    markdown = await synthesize_review(bundle, verbose=verbose)

    _log(verbose, f"Stage 5: writing {out}")
    bak = _backup_if_exists(out)
    if bak is not None:
        _log(verbose, f"  backed up previous review to {bak}")
    out.write_text(markdown)
    return out
