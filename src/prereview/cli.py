"""prereview command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import DEFAULT_MODEL, DEFAULT_SYNTHESIS_MODEL, __version__
from .cache import DEFAULT_CACHE_DIR
from .llm import BACKEND_AGENT_SDK, BACKEND_LITELLM, set_backend
from .pipeline import run_pipeline
from .venue_rules import DEFAULT_VENUE, VENUE_RULES


def _venue_arg(value: str) -> str:
    """argparse ``type`` for ``--venue``: validate against the (possibly empty) venue
    table at parse time so an unknown id is a usage error (exit 2), not a traceback."""
    if value not in VENUE_RULES:
        known = ", ".join(sorted(VENUE_RULES)) or "none"
        raise argparse.ArgumentTypeError(f"unknown venue {value!r} (configured venues: {known})")
    return value


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
        help="(.tex mode only) Explicit path to a reproducibility checklist .tex (AAAI-27 kit format). Default: auto-detect from \\input{...} or sibling ReproducibilityChecklist.tex.",
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
        "--no-reviewer2",
        dest="run_reviewer2",
        action="store_false",
        help=(
            "Skip the adversarial Reviewer-2 pass (one extra synthesis-model call). "
            "Default: on."
        ),
    )
    p.set_defaults(run_reviewer2=True)
    p.add_argument(
        "--show-rating",
        action="store_true",
        help=(
            "Include the LLM-estimated 1–10 rating (with a generosity caveat) in the Overall "
            "assessment. Default: off — the review leads with severity buckets only."
        ),
    )
    p.add_argument(
        "--no-numeric",
        dest="run_numeric",
        action="store_false",
        help="(.tex mode only) Skip the ML numerical-sanity pack (default: on for .tex).",
    )
    p.set_defaults(run_numeric=True)
    p.add_argument(
        "--artifacts",
        dest="run_artifacts",
        action="store_true",
        help=(
            "Verify that claimed Hugging Face models/datasets and GitHub repos exist "
            "(network; opt-in, default off). Set HF_TOKEN to probe gated/private HF artifacts."
        ),
    )
    p.add_argument("--no-artifacts", dest="run_artifacts", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(run_artifacts=False)
    p.add_argument(
        "--openreview",
        dest="run_openreview",
        action="store_true",
        help=(
            "Annotate cited papers with their OpenReview accept/reject decision and ratings "
            "(network; opt-in, default off). Needs OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD "
            "and the optional openreview-py extra."
        ),
    )
    p.add_argument("--no-openreview", dest="run_openreview", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(run_openreview=False)
    p.add_argument(
        "--venue",
        type=_venue_arg,
        default=DEFAULT_VENUE,
        metavar="ID",
        help=(
            "Venue whose submission rules to check against (desk-reject guard). "
            f"Configured venues: {', '.join(sorted(VENUE_RULES)) or 'none'}. "
            "Default: none — venue-specific checks are skipped."
        ),
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Exit non-zero (code 4) if a hard desk-reject blocker is found: residual "
            "author identity (anonymization audit), a substantially changed abstract "
            "(--abstract-baseline), or a blocker from the selected --venue rules "
            "(placeholder/empty title or abstract, color result table, unanswered "
            "mandatory checklist item, or — PDF only — over-length). Default: advisory "
            "only, exit code unchanged."
        ),
    )
    p.add_argument(
        "--abstract-baseline",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "(.tex mode only) Snapshot the abstract to PATH on first run, then flag on "
            "later runs if it has changed substantially — guards venues that reject a "
            "final abstract diverging from the registered one. Runs with or without --venue."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path. Default: <input-stem>.review.md next to the input.",
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
            "Contact email for the scholarly APIs: joins Crossref's polite pool and is "
            "required by Unpaywall, whose open-access PDF lookups are skipped when unset. "
            "Defaults to $PREREVIEW_MAILTO if set."
        ),
    )
    p.add_argument(
        "--auth",
        choices=("auto", "api-key", "oauth"),
        default="auto",
        help=(
            "Credential path. 'oauth' drives the Claude Code CLI, which authenticates "
            "from $CLAUDE_CODE_OAUTH_TOKEN or an interactive `claude login` (spend "
            "counts against a Claude subscription; needs the [oauth] extra and the "
            "`claude` CLI). 'api-key' calls the Anthropic API with $ANTHROPIC_API_KEY. "
            "'auto' (default) picks oauth only when $CLAUDE_CODE_OAUTH_TOKEN is set "
            "and usable, else the API key."
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

    set_backend(_resolve_auth(args.auth, parser))

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
                run_reviewer2=args.run_reviewer2,
                show_rating=args.show_rating,
                run_numeric=args.run_numeric,
                run_artifacts=args.run_artifacts,
                run_openreview=args.run_openreview,
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


_ENV_HINT = (
    "Export it, or place it in a .env file next to the input or in the project root."
)


def _resolve_auth(choice: str, parser: argparse.ArgumentParser) -> str:
    """Pick the credential path, or exit with an actionable message.

    ``auto`` prefers OAuth but only when the path is actually usable: ``with-keys``
    injects both variables on every run, so a token in the environment is not on its
    own evidence that the Agent SDK and the ``claude`` CLI are installed. Explicit
    ``--auth oauth`` skips that fallback and fails loudly instead.
    """
    from . import llm_agent_sdk

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_token = bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))

    if choice == "api-key":
        if not has_key:
            parser.error(f"--auth api-key needs ANTHROPIC_API_KEY. {_ENV_HINT}")
        return BACKEND_LITELLM

    if choice == "oauth":
        # Deliberately does not require CLAUDE_CODE_OAUTH_TOKEN: the CLI resolves
        # credentials itself, and an interactive `claude login` is a working OAuth
        # credential with no env var in sight. Let the CLI be the authority on
        # whether it can authenticate; we only check that it is there to ask.
        reason = llm_agent_sdk.unavailable_reason()
        if reason is not None:
            parser.error(f"--auth oauth is unavailable: {reason}")
        return BACKEND_AGENT_SDK

    if has_token and llm_agent_sdk.agent_sdk_available():
        return BACKEND_AGENT_SDK
    if has_key:
        return BACKEND_LITELLM
    if has_token:
        parser.error(
            "CLAUDE_CODE_OAUTH_TOKEN is set but the OAuth path is unavailable: "
            f"{llm_agent_sdk.unavailable_reason()}\n"
            f"Set ANTHROPIC_API_KEY to use the API instead. {_ENV_HINT}"
        )
    parser.error(
        "no Anthropic credentials found. Set CLAUDE_CODE_OAUTH_TOKEN (Claude "
        f"subscription, needs the [oauth] extra) or ANTHROPIC_API_KEY. {_ENV_HINT}"
    )


_DOTENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "S2_API_KEY",
    "OPENALEX_API_KEY",
    "PREREVIEW_MAILTO",
    "HF_TOKEN",
    "OPENREVIEW_USERNAME",
    "OPENREVIEW_PASSWORD",
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
