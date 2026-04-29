"""Unit tests for prereview.ingest.

The LLM bibliography parse is mocked. Real PDF extraction is exercised in the
end-to-end tests with fixture PDFs in tests/fixtures/.
"""

from __future__ import annotations

import pytest

from prereview import ingest
from prereview.ingest import (
    _expand_numeric_range,
    find_abstract,
    find_citations,
    guess_title,
    parse_references,
    split_at_references,
)
from prereview.models import Reference


def test_split_at_references_picks_last_header():
    text = (
        "Introduction\nWe cite References A.\n"
        "Methods\n...\n"
        "References\n[1] First entry.\n[2] Second.\n"
    )
    body, refs = split_at_references(text)
    assert "[1] First entry." in refs
    assert "Methods" in body
    assert "[1] First entry." not in body


def test_split_at_references_no_header():
    text = "Just a body.\nNo refs."
    body, refs = split_at_references(text)
    assert body == text
    assert refs == ""


def test_guess_title_picks_first_real_line():
    text = "\n\n  An Interesting Paper Title  \nAlice Smith, Bob Jones\nAbstract"
    assert guess_title(text) == "An Interesting Paper Title"


def test_find_abstract():
    text = (
        "Title\n"
        "Authors\n"
        "Abstract\n"
        "We propose a new method for doing things. It works well.\n"
        "1. Introduction\n"
        "Let us begin."
    )
    a = find_abstract(text)
    assert a is not None
    assert "new method" in a


def test_expand_numeric_range():
    assert _expand_numeric_range("1") == ["1"]
    assert _expand_numeric_range("1, 2, 3") == ["1", "2", "3"]
    assert _expand_numeric_range("1, 5–7") == ["1", "5", "6", "7"]
    assert _expand_numeric_range("1-3, 9") == ["1", "2", "3", "9"]


def _ref(ref_id: str, surname: str, year: int) -> Reference:
    return Reference(
        ref_id=ref_id,
        raw_text=f"{surname} ({year})",
        authors=[f"{surname}, A."],
        title="Some title",
        year=year,
    )


def test_find_citations_numeric():
    refs = {"1": _ref("1", "Smith", 2023), "2": _ref("2", "Jones", 2022)}
    body = "We base this on prior work [1] and follow-ups [1, 2]. Other things happen."
    cites = find_citations(body, refs, fmt="numeric")
    assert {c.ref_id for c in cites} == {"1", "2"}
    assert all("[1]" in c.sentence or "[1, 2]" in c.sentence for c in cites)


def test_find_citations_author_year_paren():
    refs = {"Smith2023": _ref("Smith2023", "Smith", 2023), "Jones2022": _ref("Jones2022", "Jones", 2022)}
    body = "We follow prior work (Smith, 2023) and (Jones, 2022). Some unrelated text."
    cites = find_citations(body, refs, fmt="author-year")
    assert {c.ref_id for c in cites} == {"Smith2023", "Jones2022"}


def test_find_citations_author_year_inline():
    refs = {"Smith2023": _ref("Smith2023", "Smith", 2023)}
    body = "Smith (2023) showed that things happen."
    cites = find_citations(body, refs, fmt="author-year")
    assert any(c.ref_id == "Smith2023" for c in cites)


def test_find_citations_author_year_etal():
    refs = {"Smith2023": _ref("Smith2023", "Smith", 2023)}
    body = "Recent work (Smith et al., 2023) makes the case."
    cites = find_citations(body, refs, fmt="author-year")
    assert any(c.ref_id == "Smith2023" for c in cites)


@pytest.mark.asyncio
async def test_parse_references_uses_llm(monkeypatch):
    refs_section = (
        "[1] Alice Smith and Bob Jones. A Toy Paper. Toy Journal, 2023.\n"
        "[2] Alice Smith. Another Paper. arXiv:2401.12345, 2024.\n"
    )

    async def fake_json(*, model, user, system=None, temperature=0.0, max_tokens=8192, verbose=False):
        return {
            "format": "numeric",
            "refs": [
                {
                    "ref_id": "1",
                    "raw_text": "[1] Alice Smith and Bob Jones. A Toy Paper. Toy Journal, 2023.",
                    "authors": ["Alice Smith", "Bob Jones"],
                    "title": "A Toy Paper",
                    "year": 2023,
                    "venue": "Toy Journal",
                    "doi": None,
                    "arxiv_id": None,
                    "url": None,
                },
                {
                    "ref_id": "2",
                    "raw_text": "[2] Alice Smith. Another Paper. arXiv:2401.12345, 2024.",
                    "authors": ["Alice Smith"],
                    "title": "Another Paper",
                    "year": 2024,
                    "venue": None,
                    "doi": None,
                    "arxiv_id": "2401.12345",
                    "url": None,
                },
            ],
        }

    monkeypatch.setattr(ingest, "acompletion_json", fake_json)

    fmt, refs = await parse_references(refs_section, model="anthropic/claude-sonnet-4-6")
    assert fmt == "numeric"
    assert "1" in refs and "2" in refs
    assert refs["1"].title == "A Toy Paper"
    assert refs["2"].arxiv_id == "2401.12345"


@pytest.mark.asyncio
async def test_parse_references_drops_hallucinated_entries(monkeypatch):
    refs_section = "[1] Real entry.\n"

    async def fake_json(**_):
        return {
            "format": "numeric",
            "refs": [
                {
                    "ref_id": "1",
                    "raw_text": "[1] Real entry.",
                    "authors": [],
                    "title": "Real",
                    "year": 2023,
                },
                {
                    "ref_id": "2",
                    "raw_text": "[2] Fabricated entry that's not in the source.",
                    "authors": [],
                    "title": "Fake",
                    "year": 2023,
                },
            ],
        }

    monkeypatch.setattr(ingest, "acompletion_json", fake_json)
    fmt, refs = await parse_references(refs_section, model="anthropic/claude-sonnet-4-6")
    assert "1" in refs
    assert "2" not in refs  # hallucinated entry was dropped


@pytest.mark.asyncio
async def test_parse_references_empty_input(monkeypatch):
    fmt, refs = await parse_references("", model="anthropic/claude-sonnet-4-6")
    assert refs == {}
