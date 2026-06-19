"""Native ``.tex``/``.bib`` ingest mode.

Why this exists: when a draft is in TeX source, every ``\\cite{key}`` and every
bibliography entry is *explicit*. Parsing the source is far more reliable than
PDF text extraction + LLM bibliography parsing, and it is deterministic, so the
verifier judgements become the dominant source of variance instead of the
ingest.

This module produces the same :class:`IngestedPaper` shape as
:mod:`prereview.ingest`, so the rest of the pipeline doesn't care which mode
ran.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from .anonymize import audit_anonymization
from .checklist import (
    check_claims_vs_paper,
    check_self_consistency,
    find_checklist_file,
    parse_checklist,
)
from .ingest import _surrounding_sentence
from .models import (
    AnonymizationFinding,
    BrokenRef,
    ChecklistFinding,
    Citation,
    IngestedPaper,
    LinkCheck,
    NumericFinding,
    Reference,
    SubmissionFinding,
)
from .numeric_sanity import audit_numeric
from .texutils import flatten_tex as _flatten_tex
from .texutils import read_balanced as _read_balanced
from .texutils import strip_comments as _strip_tex_comments
from .venue_rules import DEFAULT_VENUE, audit_submission_tex, get_rules

_CHECKSUB_RE = re.compile(r"\\checksubsection\b")


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[tex-ingest] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# .bib parsing


_BIB_ENTRY_HEAD = re.compile(r"@(\w+)\s*\{", re.IGNORECASE)


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Parse a BibTeX file's text into ``{key: {field: value, "_type": ...}}``.

    Tolerant of nested braces, quoted values, and trailing commas. Accent macros
    like ``{\\"o}`` and inline command braces are stripped from values.
    """
    entries: dict[str, dict[str, str]] = {}
    i = 0
    while i < len(text):
        m = _BIB_ENTRY_HEAD.search(text, i)
        if not m:
            break
        entry_type = m.group(1).lower()
        if entry_type in {"comment", "preamble", "string"}:
            i = m.end()
            continue
        # Find the matching close brace for the entry body.
        body_start = m.end()
        depth = 1
        j = body_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        body = text[body_start : j - 1] if j > body_start else ""
        i = j

        comma = body.find(",")
        if comma < 0:
            continue
        key = body[:comma].strip()
        if not key:
            continue
        fields = _parse_bib_fields(body[comma + 1 :])
        fields["_type"] = entry_type
        entries[key] = fields
    return entries


_FIELD_HEAD = re.compile(r"(\w+)\s*=\s*", re.IGNORECASE)


def _parse_bib_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(text):
        # Skip whitespace and commas.
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text):
            break
        m = _FIELD_HEAD.match(text, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i >= len(text):
            break

        if text[i] == "{":
            depth = 1
            i += 1
            start = i
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                if depth > 0:
                    i += 1
            value = text[start:i]
            i += 1  # consume closing brace
        elif text[i] == '"':
            i += 1
            start = i
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '"':
                    break
                i += 1
            value = text[start:i]
            i += 1  # consume closing quote
        else:
            # Bare value: identifier or number, possibly followed by '#' concatenations.
            m2 = re.match(r"[\w\-]+", text[i:])
            if not m2:
                break
            value = m2.group(0)
            i += m2.end()
        fields[name] = _clean_bib_value(value)
    return fields


def _clean_bib_value(s: str) -> str:
    """Strip TeX accent macros, simple wrapping commands, stray braces, ws."""
    # Drop common one-arg formatting commands but keep their argument.
    for _ in range(3):
        s = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|textsf|textrm)\{([^{}]*)\}", r"\1", s)
    # Accent macros: \"{o} → o, \'{e} → e, \"o → o, etc.
    s = re.sub(r"\\[`'\"^~=.]\{?(\w)\}?", r"\1", s)
    # \ss → ss, \aa → aa, etc.
    s = re.sub(r"\\(ss|aa|AA|oe|OE|ae|AE|o|O|l|L)\b", r"\1", s)
    # Drop remaining backslash-words and stray braces.
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_bib_authors(s: str) -> list[str]:
    """Split a BibTeX `author = { ... }` value into individual author full names."""
    if not s:
        return []
    out: list[str] = []
    for raw in re.split(r"\s+and\s+", s):
        raw = raw.strip().strip(",")
        if not raw:
            continue
        if "," in raw:
            last, first = raw.split(",", 1)
            full = f"{first.strip()} {last.strip()}".strip()
        else:
            full = raw
        full = re.sub(r"\s+", " ", full)
        if full:
            out.append(full)
    return out


_ARXIV_INLINE = re.compile(r"arXiv\s*[:.]?\s*(\d{4}\.\d{4,5})", re.IGNORECASE)


def bib_to_reference(key: str, fields: dict[str, str]) -> Reference:
    title = fields.get("title")
    year_str = fields.get("year") or ""
    year: Optional[int] = None
    m = re.search(r"\d{4}", year_str)
    if m:
        try:
            year = int(m.group(0))
        except ValueError:
            year = None
    venue = (
        fields.get("journal")
        or fields.get("booktitle")
        or fields.get("series")
        or fields.get("publisher")
    )
    arxiv_id: Optional[str] = None
    for k in ("eprint", "archiveprefix", "journal", "note", "url"):
        v = fields.get(k)
        if v:
            mm = _ARXIV_INLINE.search(v)
            if mm:
                arxiv_id = mm.group(1)
                break
            if k == "eprint" and re.match(r"^\d{4}\.\d{4,5}$", v):
                arxiv_id = v
                break

    raw_parts = [f"@{fields.get('_type', 'misc')}{{{key}"]
    for k in ("author", "title", "journal", "booktitle", "year", "publisher", "doi", "url"):
        if fields.get(k):
            raw_parts.append(f"  {k} = {{{fields[k]}}}")
    raw_text = ",\n".join(raw_parts) + "\n}"

    return Reference(
        ref_id=key,
        raw_text=raw_text,
        authors=parse_bib_authors(fields.get("author", "")),
        title=title,
        year=year,
        venue=venue,
        doi=fields.get("doi"),
        arxiv_id=arxiv_id,
        url=fields.get("url"),
    )


# ---------------------------------------------------------------------------
# .tex stripping


_CITE_CMDS = (
    "cite",
    "citep",
    "citet",
    "citealp",
    "citealt",
    "citeauthor",
    "citeyear",
    "citenum",
    "parencite",
    "textcite",
    "fullcite",
    "Citep",
    "Citet",
    "Cite",
)

_DROP_ENVS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "tikzpicture",
    "algorithm",
    "algorithmic",
    "verbatim",
    "lstlisting",
    "Verbatim",
    "tabular",
    "tabularx",
    "array",
    "matrix",
    "thebibliography",
)


def strip_tex_to_text(text: str) -> str:
    """Reduce .tex source to a plain-text body with ``[CITE:k1,k2]`` markers
    in place of citation commands. Lossy on math, environments, and exotic
    commands, but stable enough to find sentences around citations."""
    # Strip comments (% to end of line, but not \%).
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    # Cut to body only.
    m = re.search(r"\\begin\s*\{document\}", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\\end\s*\{document\}", text)
    if m:
        text = text[: m.start()]

    # Drop unwanted environments wholesale.
    for env in _DROP_ENVS:
        pat = r"\\begin\s*\{" + re.escape(env) + r"\}.*?\\end\s*\{" + re.escape(env) + r"\}"
        text = re.sub(pat, " ", text, flags=re.DOTALL)

    # Display math.
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    # Inline math: greedy single-$ would eat too much; bounded by next single-$.
    text = re.sub(r"(?<!\\)\$[^$\n]{0,200}?(?<!\\)\$", " ", text)

    # Citations → markers.
    cite_pat = (
        r"\\(?:" + "|".join(_CITE_CMDS) + r")\*?"
        r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?"
        r"\{([^{}]*)\}"
    )

    def _cite_repl(m: re.Match) -> str:
        keys = re.sub(r"\s+", "", m.group(1))
        return f"[CITE:{keys}]"

    text = re.sub(cite_pat, _cite_repl, text)

    # \label, \ref, \eqref, \cref, \pageref, \autoref → drop.
    text = re.sub(
        r"\\(?:label|ref|eqref|cref|Cref|pageref|autoref|nameref)\s*\{[^{}]*\}",
        "",
        text,
    )
    # \input, \include, \includegraphics, \usepackage → drop.
    text = re.sub(
        r"\\(?:input|include|includegraphics|usepackage|bibliography|bibliographystyle|addbibresource)"
        r"\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
        "",
        text,
    )

    # Section headings: keep the text content as a line break + the heading text.
    text = re.sub(
        r"\\(?:section|subsection|subsubsection|paragraph|subparagraph|chapter)\*?\s*\{([^{}]*)\}",
        r"\n\n\1\n",
        text,
    )

    # Strip text-formatting wrappers (preserve content). Iterate to handle nesting.
    formatters = (
        "textbf",
        "textit",
        "emph",
        "texttt",
        "textsc",
        "textrm",
        "textsf",
        "textnormal",
        "underline",
        "mathbf",
        "mathit",
        "mathrm",
        "mathsf",
    )
    fmt_pat = r"\\(?:" + "|".join(formatters) + r")\s*\{([^{}]*)\}"
    for _ in range(4):
        text = re.sub(fmt_pat, r"\1", text)

    # Strip generic one-arg commands: \cmd{X} → X. Iterate for nesting.
    for _ in range(4):
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r"\1", text)

    # Strip remaining no-arg commands.
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # LaTeX non-breaking spaces and stray braces.
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", "", text)
    # Collapse whitespace runs but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# title / abstract


def extract_title(text: str) -> Optional[str]:
    m = re.search(r"\\title\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    # Recursively strip braces and a few common commands.
    for _ in range(4):
        raw = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|textrm|textsf)\{([^{}]*)\}", r"\1", raw)
    raw = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", raw)
    raw = re.sub(r"\\[a-zA-Z]+\*?", " ", raw)
    raw = raw.replace("{", "").replace("}", "")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw or None


def extract_abstract(text: str) -> Optional[str]:
    m = re.search(r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}", text, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    body = strip_tex_to_text(r"\begin{document}" + body + r"\end{document}")
    body = re.sub(r"\[CITE:[^\]]*\]", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body or None


# ---------------------------------------------------------------------------
# manuscript structure: author / acknowledgments / sections (U1)
#
# These feed the deterministic desk-reject guards (U2 anonymization, U3
# submission-readiness). All three strip comments first so a commented-out
# ``% \author{...}`` cannot false-positive, and all three are brace-aware so a
# nested ``\thanks{Univ. of {X}}`` is captured whole. The comment-strip /
# brace-reader / fragment-flattener primitives live in :mod:`prereview.texutils`.


# ``\author`` but not ``\authors``/``\authorrunning`` (\b after the word).
_AUTHOR_RE = re.compile(r"\\author\b\s*\*?\s*(?:\[[^\]]*\])?\s*")
_ACKS_ENV_RE = re.compile(
    r"\\begin\s*\{(acks|acknowledgements?|acknowledgments?)\}(.*?)\\end\s*\{\1\}",
    re.DOTALL | re.IGNORECASE,
)
_ACKS_HEADING_RE = re.compile(
    r"\\(?:section|subsection|paragraph)\*?\s*\{\s*acknowledge?ments?\s*\}",
    re.IGNORECASE,
)
# Where an acknowledgments *section* (vs environment) ends: the next sectioning
# command or the bibliography / end of document.
_SECTION_BOUNDARY_RE = re.compile(
    r"\\(?:section|subsection|chapter|paragraph)\b"
    r"|\\(?:bibliography|printbibliography)\b"
    r"|\\begin\s*\{thebibliography\}"
    r"|\\end\s*\{document\}",
    re.IGNORECASE,
)
_SECTION_TITLE_RE = re.compile(r"\\section\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_author_block(text: str) -> Optional[str]:
    """Flattened content of the manuscript's ``\\author{...}`` block, or ``None``.

    Returns ``None`` when there is no ``\\author`` (anonymized submissions often
    have a blank or ``Anonymous`` one — that returns the literal text so U2 can
    judge it, not ``None``)."""
    text = _strip_tex_comments(text)
    m = _AUTHOR_RE.search(text)
    if not m:
        return None
    k = m.end()
    if k >= len(text) or text[k] != "{":
        return None
    inner, _ = _read_balanced(text, k)
    return _flatten_tex(inner) or None


def extract_acknowledgments(text: str) -> Optional[str]:
    """Flattened acknowledgments text, from either an ``\\begin{acks}`` /
    ``\\begin{acknowledgements}`` environment or a ``\\section*{Acknowledgements}``
    heading (US/UK spelling), or ``None`` when absent."""
    text = _strip_tex_comments(text)
    m = _ACKS_ENV_RE.search(text)
    if m:
        return _flatten_tex(m.group(2)) or None
    m = _ACKS_HEADING_RE.search(text)
    if not m:
        return None
    rest = text[m.end():]
    nxt = _SECTION_BOUNDARY_RE.search(rest)
    body = rest[: nxt.start()] if nxt else rest
    return _flatten_tex(body) or None


def extract_sections(text: str) -> list[str]:
    """Ordered list of top-level ``\\section{...}`` titles (numbered and starred)."""
    text = _strip_tex_comments(text)
    out: list[str] = []
    for m in _SECTION_TITLE_RE.finditer(text):
        title = _flatten_tex(m.group(1))
        if title:
            out.append(title)
    return out


# ---------------------------------------------------------------------------
# bib file discovery


def find_bib_file(tex_path: Path, tex_text: str) -> Optional[Path]:
    """Try, in order: ``\\bibliography{name}``, ``\\addbibresource{name.bib}``,
    sibling ``references.bib`` / ``main.bib`` / ``bibliography.bib``."""
    base = tex_path.parent

    for cmd in ("addbibresource", "bibliography"):
        for m in re.finditer(rf"\\{cmd}\s*\{{([^{{}}]+)\}}", tex_text):
            for chunk in m.group(1).split(","):
                name = chunk.strip()
                if not name:
                    continue
                cand = base / name
                if cand.suffix.lower() != ".bib":
                    cand = cand.with_suffix(".bib")
                if cand.exists():
                    return cand
    for default in ("references.bib", "main.bib", "bibliography.bib", "refs.bib"):
        cand = base / default
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# citations


_CITE_MARKER = re.compile(r"\[CITE:([^\]]+)\]")


def find_citations_tex(
    stripped_text: str,
    references: dict[str, Reference],
) -> tuple[list[Citation], dict[str, Reference]]:
    """Find ``[CITE:...]`` markers and link to references.

    Returns (citations, augmented_references). Citation keys that have no
    bib entry are surfaced as synthetic Reference records with raw_text set
    to a clear "(citation key not present in bibliography)" so resolution
    will (correctly) fail and the synthesizer flags them as ghost references.

    The ``[CITE:...]`` markers are stripped from the displayed surrounding
    sentence — readers should see prose, not internal markers.
    """
    citations: list[Citation] = []
    refs = dict(references)
    for m in _CITE_MARKER.finditer(stripped_text):
        raw_sentence = _surrounding_sentence(stripped_text, m.span())
        # Drop *all* CITE markers from the displayed sentence and tidy whitespace
        # and orphan commas/semicolons that may be left behind.
        clean = _CITE_MARKER.sub("", raw_sentence)
        clean = re.sub(r"\s*[;,]\s*(?=[;,)]|$)", "", clean)
        clean = re.sub(r"\(\s*\)", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        for key in (k.strip() for k in m.group(1).split(",")):
            if not key:
                continue
            if key not in refs:
                refs[key] = Reference(
                    ref_id=key,
                    raw_text=f"(citation key `{key}` not present in bibliography)",
                    authors=[],
                    title=None,
                )
            citations.append(Citation(ref_id=key, sentence=clean))
    return citations, refs


# ---------------------------------------------------------------------------
# hygiene checks


_LABEL_RE = re.compile(r"\\label\s*\{([^{}]+)\}")
_REF_CMDS = ("ref", "eqref", "cref", "Cref", "pageref", "autoref", "nameref")
_REF_RE = re.compile(r"\\(" + "|".join(_REF_CMDS) + r")\s*\{([^{}]+)\}")


def find_broken_refs(tex_text: str) -> list[BrokenRef]:
    """Return ``\\ref``-family commands whose target has no matching ``\\label``.

    Comments are stripped first so commented-out code doesn't false-positive.
    Surrounding context (~120 chars) is captured so the user can locate the
    issue in the source.
    """
    text = re.sub(r"(?<!\\)%[^\n]*", "", tex_text)
    labels = {m.group(1).strip() for m in _LABEL_RE.finditer(text)}

    out: list[BrokenRef] = []
    seen: set[tuple[str, str]] = set()
    for m in _REF_RE.finditer(text):
        cmd = m.group(1)
        target = m.group(2).strip()
        if not target or target in labels:
            continue
        # De-duplicate (cmd, target) pairs — the first occurrence carries
        # the most useful context, and a broken ref usually appears once
        # per logical site even if cited many places.
        sig = (cmd, target)
        if sig in seen:
            continue
        seen.add(sig)

        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        out.append(BrokenRef(command=cmd, target=target, surrounding=snippet))
    return out


_URL_CMD_RE = re.compile(r"\\(url|href)\s*\{([^{}]+)\}")


def _normalize_url(raw: str) -> Optional[str]:
    """Trim, prepend a scheme if missing, drop empty/javascript: junk.

    Returns None for values that don't look like a URL after cleaning.
    """
    s = raw.strip().rstrip(".,;)")
    if not s:
        return None
    low = s.lower()
    if low.startswith(("mailto:", "javascript:", "tel:", "data:")):
        return None
    if not low.startswith(("http://", "https://", "ftp://")):
        # Common case in .bib: url = {github.com/foo/bar} — assume https.
        if "." not in s.split("/", 1)[0]:
            return None  # not host-shaped
        s = "https://" + s
    return s


def extract_urls(
    tex_text: str,
    references: dict[str, Reference],
) -> list[LinkCheck]:
    """Pull URLs from a .tex body and its sibling .bib references.

    Returns a list of unconfirmed :class:`LinkCheck` entries (status fields
    blank). The actual reachability probing happens in :mod:`link_health`.

    Comments are stripped from the .tex first to avoid checking commented-out
    URLs. Duplicates across sources are kept (a URL appearing both in a
    ``\\url{}`` and a ``url = {...}`` bibliography entry should be flagged
    once per location, since the user may want to fix one and not the other).
    """
    text = re.sub(r"(?<!\\)%[^\n]*", "", tex_text)

    seen_pairs: set[tuple[str, str, Optional[str]]] = set()
    out: list[LinkCheck] = []

    for m in _URL_CMD_RE.finditer(text):
        cmd = m.group(1)
        url = _normalize_url(m.group(2))
        if not url:
            continue
        source = "tex_url" if cmd == "url" else "tex_href"
        sig = (source, url, None)
        if sig in seen_pairs:
            continue
        seen_pairs.add(sig)
        out.append(LinkCheck(url=url, source=source))

    for ref_id, ref in references.items():
        if not ref.url:
            continue
        url = _normalize_url(ref.url)
        if not url:
            continue
        sig = ("bib_url", url, ref_id)
        if sig in seen_pairs:
            continue
        seen_pairs.add(sig)
        out.append(LinkCheck(url=url, source="bib_url", bibkey=ref_id))

    return out


def find_unused_bibkeys(
    references: dict[str, Reference],
    citations: list[Citation],
) -> list[str]:
    """Return bibkeys present in the bibliography but never cited in the body.

    Synthetic ghost-reference entries (created by find_citations_tex when a
    ``\\cite{key}`` has no matching .bib entry) are excluded — they are by
    construction cited at least once.
    """
    cited = {c.ref_id for c in citations}
    return sorted(k for k in references if k not in cited)


# ---------------------------------------------------------------------------
# entry point


async def ingest_tex(
    tex_path: Path,
    *,
    model: str,
    verbose: bool = False,
    bib_path: Optional[Path] = None,
    checklist_path: Optional[Path] = None,
    run_checklist: bool = True,
    run_anonymize: bool = True,
    authors: Optional[str] = None,
    venue: str = DEFAULT_VENUE,
    abstract_baseline: Optional[Path] = None,
    run_numeric: bool = True,
) -> IngestedPaper:
    """Ingest a .tex file (and its sibling .bib) into an :class:`IngestedPaper`.

    The ``model`` argument is accepted for signature symmetry with
    :func:`prereview.ingest.ingest_pdf`; this path does not call the LLM
    because the source is structured.
    """
    tex_text = tex_path.read_text(encoding="utf-8", errors="replace")
    title = extract_title(tex_text)
    abstract = extract_abstract(tex_text)
    author_block = extract_author_block(tex_text)
    acknowledgments = extract_acknowledgments(tex_text)
    section_titles = extract_sections(tex_text)

    if bib_path is None:
        bib_path = find_bib_file(tex_path, tex_text)
    if bib_path is None:
        _log(verbose, f"no .bib file found alongside {tex_path}; bibliography will be empty")
        references: dict[str, Reference] = {}
    else:
        _log(verbose, f"reading bibliography from {bib_path}")
        bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
        bib_entries = parse_bib(bib_text)
        references = {k: bib_to_reference(k, v) for k, v in bib_entries.items()}
        _log(verbose, f"parsed {len(references)} bibliography entries")

    body = strip_tex_to_text(tex_text)
    citations, references_with_ghosts = find_citations_tex(body, references)
    _log(
        verbose,
        f"found {len(citations)} in-text citations across "
        f"{len({c.ref_id for c in citations})} unique keys",
    )

    broken_refs = find_broken_refs(tex_text)
    unused_bibkeys = find_unused_bibkeys(references, citations)
    link_checks = extract_urls(tex_text, references)
    if verbose:
        if broken_refs:
            _log(verbose, f"found {len(broken_refs)} broken \\ref/\\cref targets")
        if unused_bibkeys:
            _log(verbose, f"found {len(unused_bibkeys)} bib entries that are never cited")
        if link_checks:
            _log(verbose, f"found {len(link_checks)} URLs to probe in stage 1.5")

    checklist_found = False
    checklist_findings: list[ChecklistFinding] = []
    if run_checklist:
        resolved = find_checklist_file(tex_path, tex_text, explicit=checklist_path)
        checklist_text: Optional[str] = None
        if resolved is not None and resolved.exists():
            _log(verbose, f"reading reproducibility checklist from {resolved}")
            checklist_text = resolved.read_text(encoding="utf-8", errors="replace")
        elif _CHECKSUB_RE.search(tex_text):
            # The checklist was pasted / \input-ed inline into the main .tex.
            # Cross-checks run against ``body`` here, which includes the checklist
            # prose — that can only mask evidence (a false negative), never
            # manufacture a false flag, so it stays on the precision-safe side.
            _log(verbose, "reproducibility checklist found inline in the main .tex")
            checklist_text = tex_text
        if checklist_text is not None:
            checklist_found = True
            items = parse_checklist(checklist_text)
            checklist_findings = check_self_consistency(items) + check_claims_vs_paper(
                items, body, link_checks
            )
            _log(
                verbose,
                f"found {len(checklist_findings)} checklist issue(s) across {len(items)} item(s)",
            )

    anonymization_checked = False
    anonymization_findings: list[AnonymizationFinding] = []
    if run_anonymize:
        anonymization_checked = True
        anonymization_findings = audit_anonymization(
            tex_text=tex_text,
            body=body,
            author_block=author_block,
            acknowledgments=acknowledgments,
            link_checks=link_checks,
            authors=authors,
        )
        _log(verbose, f"anonymization audit: {len(anonymization_findings)} finding(s)")

    submission_findings: list[SubmissionFinding] = audit_submission_tex(
        get_rules(venue),
        title=title,
        abstract=abstract,
        tex_text=tex_text,
        checklist_found=checklist_found,
        checklist_findings=checklist_findings,
        abstract_baseline=abstract_baseline,
    )
    _log(verbose, f"submission-readiness guard: {len(submission_findings)} finding(s)")

    numeric_findings: list[NumericFinding] = []
    if run_numeric:
        numeric_findings = audit_numeric(body, tex_text, abstract)
        _log(verbose, f"numerical-sanity pack: {len(numeric_findings)} finding(s)")

    return IngestedPaper(
        title=title,
        abstract=abstract,
        sections=[("body", body)],
        references=references_with_ghosts,
        citations=citations,
        unused_bibkeys=unused_bibkeys,
        broken_refs=broken_refs,
        link_checks=link_checks,
        checklist_found=checklist_found,
        checklist_findings=checklist_findings,
        author_block=author_block,
        acknowledgments=acknowledgments,
        section_titles=section_titles,
        anonymization_checked=anonymization_checked,
        anonymization_findings=anonymization_findings,
        submission_checked=True,
        submission_findings=submission_findings,
        numeric_checked=run_numeric,
        numeric_findings=numeric_findings,
    )
