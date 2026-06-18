"""End-to-end orchestration tests for prereview.pipeline.

Each stage is mocked at the module level. Asserts:
- Output file is created at the expected path.
- Pre-existing review file is backed up to <name>.review.md.bak.<ts>.
- The synthesized Markdown contains every flagged citation in the issues section.
- ReviewBundle counts reflect what the resolver and verifier returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prereview import pipeline
from prereview.models import (
    CanonicalRecord,
    Citation,
    IngestedPaper,
    Reference,
    Resolution,
    ResolutionStatus,
    ReviewBundle,
    VerificationResult,
    Verdict,
)


def _ref(ref_id: str, surname: str = "Smith", year: int = 2023) -> Reference:
    return Reference(
        ref_id=ref_id,
        raw_text=f"[{ref_id}] {surname}. A toy. {year}.",
        authors=[f"{surname}, A."],
        title=f"A toy {ref_id}",
        year=year,
    )


def _canonical(ref_id: str, source: str = "crossref") -> CanonicalRecord:
    return CanonicalRecord(
        source=source,
        title=f"A toy {ref_id}",
        authors=["Alice Smith"],
        year=2023,
        doi=f"10.1234/{ref_id}",
        url=f"https://doi.org/10.1234/{ref_id}",
        abstract="abstract",
    )


@pytest.fixture
def fake_paper() -> IngestedPaper:
    return IngestedPaper(
        title="A Toy Paper",
        abstract="abstract",
        sections=[("body", "We claim X [1] and Y [2] and Z [3].")],
        references={
            "1": _ref("1"),
            "2": _ref("2"),
            "3": _ref("3"),  # this one will be unresolved
        },
        citations=[
            Citation(ref_id="1", sentence="We claim X [1]."),
            Citation(ref_id="2", sentence="We claim Y [2]."),
            Citation(ref_id="3", sentence="We claim Z [3]."),
        ],
    )


@pytest.mark.asyncio
async def test_pipeline_end_to_end(tmp_path: Path, monkeypatch, fake_paper):
    pdf = tmp_path / "draft.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really a pdf")

    async def fake_ingest(pdf_path, *, model, verbose=False, **_kw):
        return fake_paper

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)

    class FakeResolver:
        recovered_after_retry = 0
        circuit_broken_sources: list = []

        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def resolve(self, ref: Reference):
            if ref.ref_id == "3":
                return Resolution(status=ResolutionStatus.UNRESOLVED)
            return Resolution(
                status=ResolutionStatus.RESOLVED,
                record=_canonical(ref.ref_id, source="crossref"),
            )

    class FakeVerifier:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def verify(self, cit, ref, resolution, *, model, fetch_cited, verbose):
            canonical = resolution.record
            if canonical is None:
                return VerificationResult(
                    ref_id=ref.ref_id,
                    citation=cit,
                    reference=ref,
                    canonical=None,
                    verdict=Verdict.TARGET_UNAVAILABLE,
                    rationale="No record found.",
                    abstract_only=True,
                )
            if ref.ref_id == "1":
                return VerificationResult(
                    ref_id=ref.ref_id,
                    citation=cit,
                    reference=ref,
                    canonical=canonical,
                    verdict=Verdict.SUPPORTS,
                    rationale="Body confirms.",
                    abstract_only=False,
                )
            return VerificationResult(
                ref_id=ref.ref_id,
                citation=cit,
                reference=ref,
                canonical=canonical,
                verdict=Verdict.DOES_NOT_SUPPORT,
                rationale="Cited paper says the opposite.",
                abstract_only=False,
            )

    monkeypatch.setattr(pipeline, "Resolver", FakeResolver)
    monkeypatch.setattr(pipeline, "Verifier", FakeVerifier)

    captured_bundle: dict = {}

    async def fake_synth(bundle: ReviewBundle, *, verbose=False):
        captured_bundle["bundle"] = bundle
        # Mirror the real shape so structural assertions are meaningful.
        from prereview.synthesize import render_citation_issues, render_methodology

        return (
            "# Pre-submission review\n\n"
            "## Summary\nstub\n\n"
            "## Strengths\n- s\n\n"
            "## Weaknesses\n- w\n\n"
            + render_citation_issues(bundle).strip()
            + "\n\n## Questions for the author\n- q\n\n"
            + "## Suggested rating\n**5–7/10**\n\nLLM judgement.\n\n"
            + render_methodology(bundle).strip()
            + "\n"
        )

    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    out = tmp_path / "draft.review.md"
    out_path, report = await pipeline.run_pipeline(
        pdf,
        out=out,
        model="anthropic/claude-sonnet-4-6",
        synthesis_model="anthropic/claude-opus-4-7",
        fetch_cited=False,
        cache_dir=tmp_path / "cache",
        verbose=False,
    )

    assert out_path == out
    assert out.exists()

    # run_pipeline now returns a CoverageReport: ref "3" is a genuine ghost (UNRESOLVED),
    # not an infrastructure gap, so coverage is "clean" (no gap → exit 0).
    assert report.references_parsed == 3
    assert report.ghost_unresolved == 1
    assert report.resolution_degraded == 0 and report.verification_degraded == 0
    assert report.has_coverage_gap is False

    bundle = captured_bundle["bundle"]
    assert bundle.unresolved_count == 1
    assert len(bundle.verifications) == 3

    body = out.read_text()
    # Every problematic citation surfaces in the issues block.
    issues_block = body[body.index("## Citation issues"):body.index("## Questions")]
    assert "2" in issues_block  # does_not_support
    assert "3" in issues_block  # ghost
    # The supported one with full text should not be flagged.
    assert "ref_id` — Supports" not in issues_block.replace("`1`", "REDACTED")


@pytest.mark.asyncio
async def test_pipeline_backs_up_existing_review(tmp_path: Path, monkeypatch, fake_paper):
    pdf = tmp_path / "draft.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    out = tmp_path / "draft.review.md"
    out.write_text("OLD REVIEW")

    async def fake_ingest(pdf_path, *, model, verbose=False, **_kw):
        return IngestedPaper(title="t", references={}, citations=[])

    class _NoOpCtx:
        recovered_after_retry = 0
        circuit_broken_sources: list = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def resolve(self, ref):
            return Resolution(status=ResolutionStatus.UNRESOLVED)

        async def verify(self, *a, **kw):
            raise AssertionError("verify should not be called for empty paper")

    async def fake_synth(bundle, *, verbose=False):
        return "# new review\n"

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(pipeline, "Resolver", lambda **kw: _NoOpCtx())
    monkeypatch.setattr(pipeline, "Verifier", lambda **kw: _NoOpCtx())
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    await pipeline.run_pipeline(
        pdf,
        out=out,
        model="x",
        synthesis_model="y",
        fetch_cited=False,
        cache_dir=tmp_path / "cache",
    )

    backups = list(tmp_path.glob("draft.review.md.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "OLD REVIEW"
    assert out.read_text() == "# new review\n"


# ---------------------------------------------------------------------------
# checklist linter thread-through (.tex mode)


class _NoOpCtx:
    recovered_after_retry = 0
    circuit_broken_sources: list = []

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def resolve(self, ref):
        return None

    async def verify(self, *a, **kw):
        raise AssertionError("verify should not be called when there are no citations")


_CHECKLIST_TEX = r"""
\checksubsection{Computational Experiments}
\begin{itemize}
\question{Does this paper include computational experiments?}{(yes/no)}
yes
\ifyespoints{If yes, please address the following points:}
\begin{itemize}
\question{All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper}{(yes/partial/no)}
yes
\end{itemize}
\end{itemize}
"""


def _tex_project(tmp_path: Path) -> Path:
    (tmp_path / "ReproducibilityChecklist.tex").write_text(_CHECKLIST_TEX)
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\title{Toy}
\begin{document}
\section{Intro}
We run experiments. No repository URL appears anywhere in this paper.
\end{document}
"""
    )
    return tex


@pytest.mark.asyncio
async def test_pipeline_tex_mode_runs_checklist(tmp_path: Path, monkeypatch):
    from prereview.models import ChecklistFindingKind

    tex = _tex_project(tmp_path)

    captured: dict = {}

    async def fake_synth(bundle, *, verbose=False):
        captured["bundle"] = bundle
        return "# review\n"

    monkeypatch.setattr(pipeline, "Resolver", _NoOpCtx)
    monkeypatch.setattr(pipeline, "Verifier", _NoOpCtx)
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    await pipeline.run_pipeline(
        tex,
        out=tmp_path / "paper.review.md",
        model="m",
        synthesis_model="s",
        fetch_cited=False,
        cache_dir=tmp_path / "cache",
    )

    bundle = captured["bundle"]
    assert bundle.paper.checklist_found is True
    assert any(
        f.kind == ChecklistFindingKind.CLAIM_UNSUPPORTED for f in bundle.paper.checklist_findings
    )


@pytest.mark.asyncio
async def test_pipeline_no_checklist_flag_threads_through(tmp_path: Path, monkeypatch):
    tex = _tex_project(tmp_path)

    captured: dict = {}

    async def fake_synth(bundle, *, verbose=False):
        captured["bundle"] = bundle
        return "# review\n"

    monkeypatch.setattr(pipeline, "Resolver", _NoOpCtx)
    monkeypatch.setattr(pipeline, "Verifier", _NoOpCtx)
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    await pipeline.run_pipeline(
        tex,
        out=tmp_path / "paper.review.md",
        model="m",
        synthesis_model="s",
        fetch_cited=False,
        cache_dir=tmp_path / "cache",
        run_checklist=False,
    )

    bundle = captured["bundle"]
    assert bundle.paper.checklist_found is False
    assert bundle.paper.checklist_findings == []


@pytest.mark.asyncio
async def test_pipeline_pdf_mode_leaves_checklist_defaults(tmp_path: Path, monkeypatch):
    """The checklist linter is TeX-only; a PDF input never touches its fields."""
    pdf = tmp_path / "draft.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    async def fake_ingest(pdf_path, *, model, verbose=False, **_kw):
        return IngestedPaper(title="t", references={}, citations=[])

    captured: dict = {}

    async def fake_synth(bundle, *, verbose=False):
        captured["bundle"] = bundle
        return "# review\n"

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(pipeline, "Resolver", _NoOpCtx)
    monkeypatch.setattr(pipeline, "Verifier", _NoOpCtx)
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    await pipeline.run_pipeline(
        pdf,
        out=tmp_path / "draft.review.md",
        model="m",
        synthesis_model="s",
        fetch_cited=False,
        cache_dir=tmp_path / "cache",
    )

    assert captured["bundle"].paper.checklist_found is False


# ---------------------------------------------------------------------------
# coverage aggregation + zero-citation warning (U8)


@pytest.mark.asyncio
async def test_pipeline_reports_degraded_coverage(tmp_path: Path, monkeypatch):
    """A reference that resolves DEGRADED (transient infra failure) is a coverage gap,
    not a ghost: the CoverageReport shows resolution_degraded == 1, ghost == 0, and a
    gap — plus the resolver's recovered/circuit-broken stats."""
    paper = IngestedPaper(
        title="t",
        abstract="a",
        sections=[("body", "x [1].")],
        references={"1": _ref("1")},
        citations=[Citation(ref_id="1", sentence="x [1].")],
    )

    async def fake_ingest(pdf_path, *, model, verbose=False, **_kw):
        return paper

    class FakeResolver:
        recovered_after_retry = 2
        circuit_broken_sources = ["semanticscholar"]

        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def resolve(self, ref):
            return Resolution(status=ResolutionStatus.DEGRADED)

    class FakeVerifier:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def verify(self, cit, ref, resolution, *, model, fetch_cited, verbose):
            verdict = (
                Verdict.VERIFICATION_UNAVAILABLE
                if resolution.status == ResolutionStatus.DEGRADED
                else Verdict.SUPPORTS
            )
            return VerificationResult(
                ref_id=ref.ref_id, citation=cit, reference=ref, canonical=None,
                verdict=verdict, rationale="r", abstract_only=True,
            )

    async def fake_synth(bundle, *, verbose=False):
        return "# review\n"

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(pipeline, "Resolver", FakeResolver)
    monkeypatch.setattr(pipeline, "Verifier", FakeVerifier)
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    pdf = tmp_path / "p.pdf"
    pdf.write_text("x")
    _, report = await pipeline.run_pipeline(
        pdf, out=tmp_path / "p.review.md", model="m", synthesis_model="s",
        fetch_cited=False, cache_dir=tmp_path / "cache",
    )
    assert report.resolution_degraded == 1
    assert report.ghost_unresolved == 0
    assert report.has_coverage_gap is True
    assert report.recovered_after_retry == 2
    assert report.circuit_broken_sources == ["semanticscholar"]


@pytest.mark.asyncio
async def test_pipeline_computes_gate_blockers_from_submission(tmp_path: Path, monkeypatch):
    """run_pipeline populates coverage.gate_blockers from the U2/U3 findings so the
    CLI can apply --gate precedence (4 > 3 > 0)."""
    (tmp_path / "references.bib").write_text("@article{x, author={A}, title={t}, year={2023}}\n")
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\title{}
\begin{document}
\section{Intro}
Body text here with plenty of words so the section is not empty at all.
\end{document}
"""
    )

    async def fake_synth(bundle, *, verbose=False):
        return "# review\n"

    monkeypatch.setattr(pipeline, "Resolver", _NoOpCtx)
    monkeypatch.setattr(pipeline, "Verifier", _NoOpCtx)
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    _, report = await pipeline.run_pipeline(
        tex, out=tmp_path / "paper.review.md", model="m", synthesis_model="s",
        fetch_cited=False, cache_dir=tmp_path / "cache",
    )
    # Empty \title{} is a hard desk-reject blocker that reaches gate_blockers.
    assert any("title is empty" in b for b in report.gate_blockers)


@pytest.mark.asyncio
async def test_pipeline_warns_loudly_on_zero_references(tmp_path: Path, monkeypatch, capsys):
    """Zero parsed references emits a loud stderr warning even without --verbose, so a
    silent-empty extraction never looks like a clean run."""

    async def fake_ingest(pdf_path, *, model, verbose=False, **_kw):
        return IngestedPaper(title="t", references={}, citations=[])

    async def fake_synth(bundle, *, verbose=False):
        return "# review\n"

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(pipeline, "Resolver", lambda **kw: _NoOpCtx())
    monkeypatch.setattr(pipeline, "Verifier", lambda **kw: _NoOpCtx())
    monkeypatch.setattr(pipeline, "synthesize_review", fake_synth)

    pdf = tmp_path / "p.pdf"
    pdf.write_text("x")
    await pipeline.run_pipeline(
        pdf, out=tmp_path / "p.review.md", model="m", synthesis_model="s",
        fetch_cited=False, cache_dir=tmp_path / "cache", verbose=False,
    )
    assert "no bibliography entries" in capsys.readouterr().err
