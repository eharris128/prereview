"""Tests for prereview.tex_ingest.

The .tex/.bib path is fully deterministic — no LLM. So these tests assert
concrete parse outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prereview.tex_ingest import (
    bib_to_reference,
    extract_abstract,
    extract_title,
    extract_urls,
    find_bib_file,
    find_broken_refs,
    find_citations_tex,
    find_unused_bibkeys,
    ingest_tex,
    parse_bib,
    parse_bib_authors,
    strip_tex_to_text,
)


# ---------------------------------------------------------------------------
# .bib parsing


def test_parse_bib_basic():
    text = """
    @article{smith2023, author={Smith, Alice and Jones, Bob}, title={A Toy Paper}, year={2023}, journal={Toys}}
    @inproceedings{doe2022, author={Doe, J.}, title={Other Paper}, year={2022}, booktitle={ICTOY}}
    """
    e = parse_bib(text)
    assert "smith2023" in e and "doe2022" in e
    assert e["smith2023"]["title"] == "A Toy Paper"
    assert e["smith2023"]["author"] == "Smith, Alice and Jones, Bob"
    assert e["smith2023"]["year"] == "2023"
    assert e["smith2023"]["_type"] == "article"
    assert e["doe2022"]["_type"] == "inproceedings"


def test_parse_bib_handles_nested_braces_and_accents():
    text = r"""
    @article{x, author={Sch{\"o}lkopf, Bernhard and Smola, Alex J.}, title={A {S}upport-{V}ector Method}, year={2001}}
    """
    e = parse_bib(text)
    assert "x" in e
    assert "Scholkopf" in e["x"]["author"]
    assert "Support-Vector Method" in e["x"]["title"]


def test_parse_bib_quoted_values():
    text = '@article{q, author = "Smith, A.", title = "Quoted title", year = 2023}'
    e = parse_bib(text)
    assert e["q"]["author"] == "Smith, A."
    assert e["q"]["title"] == "Quoted title"
    assert e["q"]["year"] == "2023"


def test_parse_bib_authors_lastname_first():
    assert parse_bib_authors("Smith, Alice and Jones, Bob") == [
        "Alice Smith",
        "Bob Jones",
    ]
    assert parse_bib_authors("Alice Smith and Bob Jones") == [
        "Alice Smith",
        "Bob Jones",
    ]
    assert parse_bib_authors("") == []


def test_bib_to_reference_picks_doi_and_arxiv():
    fields = {
        "_type": "article",
        "author": "Smith, A.",
        "title": "Toy",
        "year": "2024",
        "journal": "arXiv preprint arXiv:2401.12345",
    }
    ref = bib_to_reference("toy24", fields)
    assert ref.year == 2024
    assert ref.arxiv_id == "2401.12345"
    assert ref.title == "Toy"
    assert ref.authors == ["A. Smith"]
    fields2 = dict(fields)
    fields2["doi"] = "10.1234/example"
    ref2 = bib_to_reference("toy24", fields2)
    assert ref2.doi == "10.1234/example"


# ---------------------------------------------------------------------------
# .tex stripping and discovery


def test_strip_tex_keeps_cite_markers_and_drops_math():
    src = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}
    Some text \cite{foo,bar} and more.
    Equation: $E = mc^2$ should drop.
    \begin{equation} x = y \end{equation}
    Another sentence \citep{baz}.
    \end{document}
    """
    out = strip_tex_to_text(src)
    assert "[CITE:foo,bar]" in out
    assert "[CITE:baz]" in out
    assert "mc^2" not in out
    assert "x = y" not in out
    assert "Some text" in out


def test_strip_tex_drops_figure_environments():
    src = r"""
    \begin{document}
    Before figure. \begin{figure}[t]\includegraphics[width=\linewidth]{x.pdf}\caption{fig}\end{figure} After figure.
    \end{document}
    """
    out = strip_tex_to_text(src)
    assert "Before figure." in out
    assert "After figure." in out
    # The caption text and the includegraphics path are dropped along with the env.
    assert "x.pdf" not in out
    assert "caption" not in out
    assert "linewidth" not in out


def test_strip_tex_handles_text_formatting():
    src = r"\begin{document}\textbf{bold} and \emph{italic} and \textsc{small caps}\end{document}"
    out = strip_tex_to_text(src)
    assert "bold" in out and "italic" in out and "small caps" in out
    assert "\\textbf" not in out


def test_extract_title_strips_commands():
    src = r"\title{\textsc{Gitcha}: A Generator for Things}"
    assert extract_title(src) == "Gitcha: A Generator for Things"


def test_extract_abstract_drops_cite_markers():
    src = r"""
    \begin{abstract}
    We propose a thing \citep{foo}. It is good.
    \end{abstract}
    """
    a = extract_abstract(src)
    assert a is not None
    assert "We propose a thing" in a
    assert "[CITE:" not in a
    assert "It is good." in a


def test_find_bib_file_explicit(tmp_path: Path):
    bib = tmp_path / "myrefs.bib"
    bib.write_text("@article{x, title={t}}")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\bibliography{myrefs}")
    assert find_bib_file(tex, tex.read_text()) == bib


def test_find_bib_file_addbibresource(tmp_path: Path):
    bib = tmp_path / "biblio.bib"
    bib.write_text("@article{x, title={t}}")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\addbibresource{biblio.bib}")
    assert find_bib_file(tex, tex.read_text()) == bib


def test_find_bib_file_default_sibling(tmp_path: Path):
    bib = tmp_path / "references.bib"
    bib.write_text("@article{x, title={t}}")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\documentclass{article}\begin{document}\end{document}")
    assert find_bib_file(tex, tex.read_text()) == bib


# ---------------------------------------------------------------------------
# citation finding


def test_find_citations_tex_links_known_keys():
    refs = {
        "smith2023": bib_to_reference("smith2023", {"author": "Smith, A.", "title": "T", "year": "2023"}),
    }
    body = "We claim X [CITE:smith2023]. Another sentence."
    cites, refs2 = find_citations_tex(body, refs)
    assert len(cites) == 1
    assert cites[0].ref_id == "smith2023"
    assert "claim X" in cites[0].sentence
    assert "smith2023" in refs2 and len(refs2) == 1


def test_find_citations_tex_surfaces_unknown_keys_as_ghosts():
    refs = {}
    body = "Prior work [CITE:nonexistent_key] suggests this."
    cites, refs2 = find_citations_tex(body, refs)
    assert len(cites) == 1
    assert cites[0].ref_id == "nonexistent_key"
    assert "nonexistent_key" in refs2
    assert "not present in bibliography" in refs2["nonexistent_key"].raw_text


# ---------------------------------------------------------------------------
# end-to-end


@pytest.mark.asyncio
async def test_ingest_tex_end_to_end(tmp_path: Path):
    bib = tmp_path / "references.bib"
    bib.write_text(
        "@article{good2023, author={Smith, A.}, title={A real paper}, year={2023}, doi={10.1/x}}\n"
    )
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\title{Toy Paper}
\begin{document}
\begin{abstract}
We do things \citep{good2023}.
\end{abstract}
\section{Introduction}
Prior work \citep{good2023} matters. We also reference \citep{ghost2099}.
Equation: $a + b = c$.
\begin{figure}\caption{fig}\end{figure}
\end{document}
"""
    )
    paper = await ingest_tex(tex, model="ignored", verbose=False)
    assert paper.title == "Toy Paper"
    assert paper.abstract is not None and "We do things" in paper.abstract
    assert "good2023" in paper.references
    # ghost citation key surfaces as a synthetic Reference
    assert "ghost2099" in paper.references
    assert "not present in bibliography" in paper.references["ghost2099"].raw_text
    # body has the cite markers
    assert any(c.ref_id == "good2023" for c in paper.citations)
    assert any(c.ref_id == "ghost2099" for c in paper.citations)
    # math and figure were stripped
    body = paper.sections[0][1]
    assert "a + b" not in body
    assert "fig" not in body


# ---------------------------------------------------------------------------
# hygiene checks


def test_find_broken_refs_flags_undefined_targets():
    tex = r"""
\section{Intro}\label{sec:intro}
See Section~\ref{sec:intro} and Appendix~\ref{app:missing}.
Equation~\eqref{eq:nope} is great. Also \cref{tab:nope}.
"""
    out = find_broken_refs(tex)
    targets = {(b.command, b.target) for b in out}
    assert ("ref", "app:missing") in targets
    assert ("eqref", "eq:nope") in targets
    assert ("cref", "tab:nope") in targets
    # Defined target should not appear.
    assert ("ref", "sec:intro") not in targets


def test_find_broken_refs_ignores_commented_out_refs():
    tex = r"""
\section{Intro}\label{sec:x}
Real ref: \ref{sec:x}.
% \ref{will:not:exist}  -- commented out, should be ignored.
"""
    out = find_broken_refs(tex)
    assert out == []


def test_find_broken_refs_dedupes_repeated_targets():
    """A broken \\ref reused at three sites should produce one entry, not three."""
    tex = r"""
First mention: \ref{missing}. Second: \ref{missing}. Third: \ref{missing}.
"""
    out = find_broken_refs(tex)
    assert len(out) == 1
    assert out[0].target == "missing"


def test_find_unused_bibkeys_lists_uncited_entries():
    bib_text = """
    @article{used, author={A}, title={t}, year={2023}}
    @article{also_used, author={B}, title={t}, year={2023}}
    @article{unused1, author={C}, title={t}, year={2023}}
    @article{unused2, author={D}, title={t}, year={2023}}
    """
    refs = {k: bib_to_reference(k, v) for k, v in parse_bib(bib_text).items()}
    body = "Cite [CITE:used] and [CITE:also_used]."
    citations, refs_with_ghosts = find_citations_tex(body, refs)

    out = find_unused_bibkeys(refs_with_ghosts, citations)
    assert out == ["unused1", "unused2"]


def test_extract_urls_pulls_url_and_href_from_tex():
    tex = r"""
    See \url{https://github.com/foo/bar} for code.
    Also \href{https://example.org/page}{the project page}.
    Commented out: % \url{https://commented.org}
    """
    out = extract_urls(tex, references={})
    urls = {(c.source, c.url) for c in out}
    assert ("tex_url", "https://github.com/foo/bar") in urls
    assert ("tex_href", "https://example.org/page") in urls
    # Commented URL should not appear.
    assert all("commented.org" not in c.url for c in out)


def test_extract_urls_pulls_bib_url_and_tags_bibkey():
    bib_text = """
    @misc{repo, author={X}, title={t}, year={2023}, url={https://github.com/x/y}}
    @article{noiurl, author={Y}, title={t}, year={2023}}
    """
    refs = {k: bib_to_reference(k, v) for k, v in parse_bib(bib_text).items()}
    out = extract_urls("", refs)
    assert len(out) == 1
    assert out[0].source == "bib_url"
    assert out[0].url == "https://github.com/x/y"
    assert out[0].bibkey == "repo"


def test_extract_urls_normalizes_missing_scheme():
    """A bare github.com/foo/bar in the .bib should get https:// prepended."""
    bib_text = """
    @misc{r, author={X}, title={t}, year={2023}, url={github.com/foo/bar}}
    """
    refs = {k: bib_to_reference(k, v) for k, v in parse_bib(bib_text).items()}
    out = extract_urls("", refs)
    assert len(out) == 1
    assert out[0].url == "https://github.com/foo/bar"


def test_extract_urls_skips_mailto_and_javascript():
    tex = r"\url{mailto:author@example.org} \url{javascript:alert(1)} \url{https://ok.example.org}"
    out = extract_urls(tex, references={})
    urls = {c.url for c in out}
    assert urls == {"https://ok.example.org"}


def test_extract_urls_dedupes_within_same_source():
    """Two \\url{...} of the same URL should yield one entry; same URL appearing
    in both .tex and .bib should yield two (different sources, both worth flagging)."""
    bib_text = """
    @misc{r, author={X}, title={t}, year={2023}, url={https://github.com/x/y}}
    """
    refs = {k: bib_to_reference(k, v) for k, v in parse_bib(bib_text).items()}
    tex = r"\url{https://github.com/x/y} and again \url{https://github.com/x/y}."
    out = extract_urls(tex, refs)
    sources = sorted(c.source for c in out)
    assert sources == ["bib_url", "tex_url"]


def test_ingest_tex_populates_hygiene_fields(tmp_path: Path):
    """End-to-end: hygiene fields should land on IngestedPaper."""
    import asyncio

    bib = tmp_path / "references.bib"
    bib.write_text(
        "@article{cited, author={A}, title={A}, year={2023}}\n"
        "@article{never_cited, author={B}, title={B}, year={2023}}\n"
    )
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\title{Hygiene Toy}
\begin{document}
\section{Intro}\label{sec:intro}
See \ref{sec:intro} (good). Also \ref{nope:gone} (broken).
We cite \citep{cited}.
\end{document}
"""
    )
    paper = asyncio.run(ingest_tex(tex, model="ignored", verbose=False))
    assert paper.unused_bibkeys == ["never_cited"]
    targets = {(b.command, b.target) for b in paper.broken_refs}
    assert ("ref", "nope:gone") in targets
    assert ("ref", "sec:intro") not in targets
