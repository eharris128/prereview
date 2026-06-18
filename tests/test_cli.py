"""Tests for prereview.cli argument plumbing.

Only the deterministic, no-network parts: flag parsing and the early validation
errors that fire before the pipeline (and before the API-key check) runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prereview.cli import _autoload_env, _DOTENV_KEYS, _build_parser, main


def test_parser_defaults_run_checklist_true():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_checklist is True
    assert args.checklist is None


def test_parser_no_checklist_flag():
    args = _build_parser().parse_args(["paper.tex", "--no-checklist"])
    assert args.run_checklist is False


def test_parser_explicit_checklist_path():
    args = _build_parser().parse_args(["paper.tex", "--checklist", "c.tex"])
    assert args.run_checklist is True
    assert args.checklist == Path("c.tex")


def test_parser_defaults_run_anonymize_true():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_anonymize is True
    assert args.authors is None


def test_parser_no_anonymize_flag():
    args = _build_parser().parse_args(["paper.tex", "--no-anonymize"])
    assert args.run_anonymize is False


def test_parser_authors_flag_parses():
    args = _build_parser().parse_args(["paper.tex", "--authors", "Smith,Jones"])
    assert args.authors == "Smith,Jones"


def test_main_threads_anonymize_args_to_pipeline(tmp_path: Path, monkeypatch):
    """--no-anonymize and --authors must reach run_pipeline."""
    from prereview import cli
    from prereview.models import CoverageReport

    captured = {}
    out = tmp_path / "p.review.md"

    async def fake_run(*a, **kw):
        captured.update(kw)
        out.write_text("# review\n")
        return out, CoverageReport()

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    tex = tmp_path / "p.tex"
    tex.write_text(r"\documentclass{article}")

    main([str(tex), "--no-anonymize", "--authors", "Smith"])
    assert captured["run_anonymize"] is False
    assert captured["authors"] == "Smith"


def test_main_errors_on_missing_checklist(tmp_path: Path):
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\documentclass{article}")
    with pytest.raises(SystemExit) as exc:
        main([str(tex), "--checklist", str(tmp_path / "does_not_exist.tex")])
    assert exc.value.code == 2  # argparse error exit code


def test_openalex_api_key_autoloaded_from_env(tmp_path: Path, monkeypatch):
    """OpenAlex now requires a key; OPENALEX_API_KEY placed in a .env next to the input
    must be loaded so the resolver can authenticate."""
    assert "OPENALEX_API_KEY" in _DOTENV_KEYS
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # so cwd/.env is the temp one, not the repo's real .env
    (tmp_path / ".env").write_text("OPENALEX_API_KEY=oa-secret\n")
    inp = tmp_path / "paper.tex"
    inp.write_text("x")
    _autoload_env(inp)
    assert os.environ.get("OPENALEX_API_KEY") == "oa-secret"
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# exit-code semantics + clean error surface (U8)


def _stub_run_pipeline(monkeypatch, tmp_path, *, report=None, raises=None):
    """Replace cli.run_pipeline so main() can be exercised without network/LLM."""
    from prereview import cli

    out = tmp_path / "p.review.md"

    async def fake_run(*a, **kw):
        if raises is not None:
            raise raises
        out.write_text("# review\n")
        return out, report

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    tex = tmp_path / "p.tex"
    tex.write_text(r"\documentclass{article}")
    return tex


def test_cli_exit_0_on_clean_coverage(tmp_path: Path, monkeypatch):
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(
        monkeypatch, tmp_path, report=CoverageReport(references_parsed=2, citations_checked=2, resolved=2)
    )
    assert main([str(tex)]) == 0


def test_cli_exit_3_on_coverage_gap(tmp_path: Path, monkeypatch, capsys):
    """A non-recoverable coverage gap exits 3 (scriptable) and explains itself; honest
    verdicts would still exit 0."""
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(
        monkeypatch, tmp_path, report=CoverageReport(citations_checked=1, verification_degraded=1)
    )
    assert main([str(tex)]) == 3
    assert "coverage gaps" in capsys.readouterr().err


def test_cli_exit_1_clean_message_no_traceback(tmp_path: Path, monkeypatch, capsys):
    """An unexpected pipeline error surfaces as a clean one-line message and exit 1 — no
    raw Python traceback."""
    tex = _stub_run_pipeline(monkeypatch, tmp_path, raises=RuntimeError("kaboom"))
    assert main([str(tex)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "prereview: failed" in err and "kaboom" in err
