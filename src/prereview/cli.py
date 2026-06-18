"""prereview command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import DEFAULT_MODEL, DEFAULT_SYNTHESIS_MODEL, __version__
from .cache import DEFAULT_CACHE_DIR
from .pipeline import run_pipeline
from .venue_rules import DEFAULT_VENUE, VENUE_RULES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prereview",
        description=(
            "AI-assisted pre-submission review of an academic preprint. "
            "Verifies every cited reference against Crossref / Semantic Scholar / "
            "arXiv / OpenAlex, classifies whether each citation supports its "
            "surrounding claim, and writes one structured Markdown review. "
            "Accepts either a PDF or a TeX source (with sibling .bib)."
        ),
    )
    p.add_argument(
        "pdf",
        type=Path,
        metavar="INPUT",
        help="Path to the draft. Either a .pdf or a .tex source.",
    )
    p.add_argument(
        "--bib",
        type=Path,
        default=None,
        help="(.tex mode only) Explicit path to the .bib file. Default: auto-detect from \\bibliography{} or sibling references.bib.",
    )
    p.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="(.tex mode only) Explicit path to a reproducibility checklist .tex (AAAI-27). Default: auto-detect from \\input{...} or sibling ReproducibilityChecklist.tex.",
    )
    p.add_argument(
        "--no-checklist",
        dest="run_checklist",
        action="store_false",
        help="(.tex mode only) Skip the reproducibility-checklist linter even if a checklist is found.",
    )
    p.set_defaults(run_checklist=True)
    p.add_argument(
        "--no-anonymize",
        dest="run_anonymize",
        action="store_false",
        help="(.tex mode only) Skip the double-blind anonymization audit (default: on for .tex).",
    )
    p.set_defaults(run_anonymize=True)
    p.add_argument(
        "--authors",
        default=None,
        metavar="\"Surname,Surname\"",
        help=(
            "(.tex mode only) Comma-separated author surnames to also grep for in the body "
            "as an extra anonymization check. Hits inside \\cite{} and hyphenated method "
            "names (e.g. Smith-Waterman) are suppressed."
        ),
    )
    p.add_argument(
        "--venue",
        default=DEFAULT_VENUE,
        choices=sorted(VENUE_RULES),
        help=f"Venue whose submission rules to check against. Default: {DEFAULT_VENUE}.",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Exit non-zero (code 4) if a hard desk-reject blocker is found (residual "
            "identity, placeholder/empty/changed title or abstract, color result table, "
            "unanswered mandatory checklist item, or — PDF only — over-length). Default: "
            "advisory only, exit code unchanged."
        ),
    )
    p.add_argument(
        "--abstract-baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "(.tex mode only) Snapshot the abstract to PATH on first run, then flag on "
            "later runs if it has changed substantially (AAAI two-deadline guard)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path. Default: <pdf-stem>.review.md next to the PDF.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for retrieval/extraction passes. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--synthesis-model",
        default=DEFAULT_SYNTHESIS_MODEL,
        help=f"Model for the final review-writing pass. Default: {DEFAULT_SYNTHESIS_MODEL}",
    )
    p.add_argument(
        "--no-fetch-cited",
        dest="fetch_cited",
        action="store_false",
        help="Skip downloading cited works' full text. Verifications fall back to abstract-only.",
    )
    p.set_defaults(fetch_cited=True)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for resolved references and fetched PDFs. Default: {DEFAULT_CACHE_DIR}",
    )
    p.add_argument(
        "--mailto",
        default=os.environ.get("PREREVIEW_MAILTO"),
        help=(
            "Email address to include in the polite-pool User-Agent for Crossref / OpenAlex. "
            "Defaults to $PREREVIEW_MAILTO if set."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Log every retrieval and verification step to stderr.")
    p.add_argument("--version", action="version", version=f"prereview {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pdf_path: Path = args.pdf.expanduser().resolve()
    if not pdf_path.exists():
        parser.error(f"input not found: {pdf_path}")
    suffix = pdf_path.suffix.lower()
    if suffix not in (".pdf", ".tex"):
        print(
            f"warning: unrecognized input extension {suffix!r} (expected .pdf or .tex)",
            file=sys.stderr,
        )

    checklist_path: Path | None = None
    if args.checklist is not None:
        checklist_path = args.checklist.expanduser().resolve()
        if not checklist_path.exists():
            parser.error(f"checklist not found: {checklist_path}")

    out: Path = (args.out or pdf_path.with_suffix(".review.md")).expanduser().resolve()

    _autoload_env(pdf_path)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error(
            "ANTHROPIC_API_KEY is not set. Export it, or place it in a .env file "
            "next to the input or in the project root."
        )

    try:
        out_path, report = asyncio.run(
            run_pipeline(
                pdf_path,
                out=out,
                model=args.model,
                synthesis_model=args.synthesis_model,
                fetch_cited=args.fetch_cited,
                cache_dir=args.cache_dir,
                polite_mailto=args.mailto,
                bib_path=args.bib,
                checklist_path=checklist_path,
                run_checklist=args.run_checklist,
                run_anonymize=args.run_anonymize,
                authors=args.authors,
                venue=args.venue,
                abstract_baseline=(
                    args.abstract_baseline.expanduser().resolve()
                    if args.abstract_baseline is not None
                    else None
                ),
                verbose=args.verbose,
            )
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except NotImplementedError as e:
        print(f"prereview: not yet implemented — {e}", file=sys.stderr)
        return 2
    except Exception as e:
        # Clean, actionable message instead of a raw traceback — the review either was
        # not written or is incomplete. Run with --verbose for the per-stage log.
        print(f"prereview: failed — {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(str(out_path))

    # Exit-code precedence (KTD-8): 4 (hard desk-reject blocker under --gate) outranks
    # 3 (coverage/degradation gap) outranks 0 (clean). A blocker is a confident,
    # mechanically-detected desk-reject trigger; gating on it is opt-in so the default
    # run keeps its existing advisory, exit-code-neutral semantics.
    if args.gate and report is not None and report.gate_blockers:
        n = len(report.gate_blockers)
        print(
            f"prereview: --gate failed — {n} hard desk-reject blocker{'' if n == 1 else 's'}: "
            + "; ".join(report.gate_blockers)
            + ". See 'Submission readiness' / 'Anonymization audit' in the output.",
            file=sys.stderr,
        )
        return 4

    # The exit code communicates coverage integrity: 0 = ran clean and complete; 3 = ran
    # but at least one citation could not be checked (infrastructure) or the prose pass
    # degraded — an honest, scriptable signal that the review discloses gaps. Honest
    # verdicts (does-not-support, abstract-too-thin, genuine ghosts) still exit 0.
    if report is not None and (report.has_coverage_gap or report.synthesis_degraded):
        n = report.resolution_degraded + report.verification_degraded
        parts: list[str] = []
        if n:
            parts.append(f"{n} citation{'' if n == 1 else 's'} could not be verified (infrastructure)")
        if report.circuit_broken_sources:
            parts.append("sources stopped mid-run: " + ", ".join(report.circuit_broken_sources))
        if report.synthesis_degraded:
            parts.append("narrative sections degraded")
        print(
            "prereview: completed with coverage gaps — "
            + "; ".join(parts)
            + ". See 'Review coverage & reliability' in the output.",
            file=sys.stderr,
        )
        return 3
    return 0


_DOTENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "S2_API_KEY",
    "OPENALEX_API_KEY",
    "PREREVIEW_MAILTO",
)


def _autoload_env(input_path: Path) -> None:
    """Load a small set of known keys from ``.env`` files, if present.

    Search order: cwd, the input file's directory, and the package install root.
    Existing environment values always win.
    """
    candidates: list[Path] = []
    cwd_env = Path.cwd() / ".env"
    if cwd_env not in candidates:
        candidates.append(cwd_env)
    inp_env = input_path.parent / ".env"
    if inp_env not in candidates:
        candidates.append(inp_env)
    pkg_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
    if pkg_root_env not in candidates:
        candidates.append(pkg_root_env)

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if k in _DOTENV_KEYS and not os.environ.get(k):
                os.environ[k] = v


if __name__ == "__main__":
    raise SystemExit(main())
