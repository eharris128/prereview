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
