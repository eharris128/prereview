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


def test_parser_openreview_default_off():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_openreview is False


def test_parser_openreview_flag_opts_in():
    args = _build_parser().parse_args(["paper.tex", "--openreview"])
    assert args.run_openreview is True


def test_openreview_creds_autoloaded_from_env(tmp_path: Path, monkeypatch):
    for k in ("OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"):
        assert k in _DOTENV_KEYS
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENREVIEW_USERNAME=u@x.com\nOPENREVIEW_PASSWORD=secret\n")
    inp = tmp_path / "paper.tex"
    inp.write_text("x")
    _autoload_env(inp)
    assert os.environ.get("OPENREVIEW_USERNAME") == "u@x.com"
    assert os.environ.get("OPENREVIEW_PASSWORD") == "secret"
    for k in ("OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"):
        monkeypatch.delenv(k, raising=False)


def test_parser_artifacts_default_off():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_artifacts is False


def test_parser_artifacts_flag_opts_in():
    args = _build_parser().parse_args(["paper.tex", "--artifacts"])
    assert args.run_artifacts is True


def test_hf_token_autoloaded_from_env(tmp_path: Path, monkeypatch):
    assert "HF_TOKEN" in _DOTENV_KEYS
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf-secret\n")
    inp = tmp_path / "paper.tex"
    inp.write_text("x")
    _autoload_env(inp)
    assert os.environ.get("HF_TOKEN") == "hf-secret"
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_parser_numeric_default_on():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_numeric is True


def test_parser_no_numeric_flag():
    args = _build_parser().parse_args(["paper.tex", "--no-numeric"])
    assert args.run_numeric is False


def test_parser_show_rating_default_off():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.show_rating is False


def test_parser_show_rating_flag():
    args = _build_parser().parse_args(["paper.tex", "--show-rating"])
    assert args.show_rating is True


def test_parser_reviewer2_default_on():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.run_reviewer2 is True


def test_parser_no_reviewer2_flag():
    args = _build_parser().parse_args(["paper.tex", "--no-reviewer2"])
    assert args.run_reviewer2 is False


def test_main_threads_reviewer2_flag_to_pipeline(tmp_path: Path, monkeypatch):
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

    main([str(tex), "--no-reviewer2"])
    assert captured["run_reviewer2"] is False


def test_parser_venue_gate_baseline_defaults():
    args = _build_parser().parse_args(["paper.tex"])
    assert args.venue == "aaai-27"
    assert args.gate is False
    assert args.abstract_baseline is None


def test_parser_unknown_venue_errors():
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["paper.tex", "--venue", "nope-99"])
    assert exc.value.code == 2  # argparse invalid-choice exit code


def test_parser_gate_and_baseline_parse():
    args = _build_parser().parse_args(["paper.tex", "--gate", "--abstract-baseline", "ab.txt"])
    assert args.gate is True
    assert args.abstract_baseline == Path("ab.txt")


def test_cli_gate_exit_4_on_hard_blocker(tmp_path: Path, monkeypatch, capsys):
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(
        monkeypatch, tmp_path,
        report=CoverageReport(gate_blockers=["residual author identity (Jane Smith)"]),
    )
    assert main([str(tex), "--gate"]) == 4
    assert "--gate failed" in capsys.readouterr().err


def test_cli_gate_clean_exits_0(tmp_path: Path, monkeypatch):
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(monkeypatch, tmp_path, report=CoverageReport(gate_blockers=[]))
    assert main([str(tex), "--gate"]) == 0


def test_cli_blockers_without_gate_do_not_change_exit(tmp_path: Path, monkeypatch):
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(
        monkeypatch, tmp_path, report=CoverageReport(gate_blockers=["something"])
    )
    # No --gate → advisory only → exit 0 despite blockers present.
    assert main([str(tex)]) == 0


def test_cli_gate_precedence_over_coverage_gap(tmp_path: Path, monkeypatch):
    """A hard blocker (4) outranks a coverage gap (3)."""
    from prereview.models import CoverageReport

    tex = _stub_run_pipeline(
        monkeypatch, tmp_path,
        report=CoverageReport(verification_degraded=1, gate_blockers=["blk"]),
    )
    assert main([str(tex), "--gate"]) == 4


def test_main_threads_venue_and_baseline_to_pipeline(tmp_path: Path, monkeypatch):
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

    main([str(tex), "--venue", "aaai-27", "--abstract-baseline", str(tmp_path / "ab.txt"), "--show-rating"])
    assert captured["venue"] == "aaai-27"
    assert captured["abstract_baseline"] == (tmp_path / "ab.txt").resolve()
    assert captured["show_rating"] is True


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
