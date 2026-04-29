"""Stage 1: parse the input PDF into IngestedPaper.

Approach:

1. Extract raw text with pypdf, lightly cleaned (re-join hyphenated line breaks,
   collapse runs of whitespace).
2. Heuristically locate the title, abstract, and references section.
3. Use the LLM (Sonnet by default) to convert the references section into a
   list of structured Reference records. The LLM is doing parsing here, not
   inventing facts: every entry's ``raw_text`` must appear in the bibliography
   text we passed in, or we drop it.
4. Find in-text citations with regex (numeric ``[1]`` style or parenthesized
   ``(Smith, 2023)`` style) and map each to a Reference.
5. Capture the surrounding sentence for each citation.

Stage 2 (resolution) takes it from here. Canonical metadata is never trusted
from this stage; everything we record here is what the *author wrote*.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from .llm import acompletion_json
from .models import Citation, IngestedPaper, Reference


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[ingest] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# text extraction


def extract_text(pdf_path: Path) -> str:
    """Best-effort text extraction. Unwrap line-break hyphens and collapse newlines."""
    import logging as _logging

    from pypdf import PdfReader

    pypdf_logger = _logging.getLogger("pypdf")
    prev_level = pypdf_logger.level
    pypdf_logger.setLevel(_logging.ERROR)
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    finally:
        pypdf_logger.setLevel(prev_level)
    text = "\n".join(pages)
    # Join words split by line-break hyphens: "neural- \nnetwork" → "neuralnetwork".
    text = re.sub(r"-\s*\n\s*(\w)", r"\1", text)
    # Collapse multiple blank lines but keep paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# section discovery


_REFS_HEADERS = re.compile(
    r"^\s*(?:\d+\.?\s+)?(References|Bibliography|Works\s+Cited|Literature\s+Cited|R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_at_references(text: str) -> tuple[str, str]:
    """Return (body, refs_section). If no references heading is found, refs_section is ''."""
    matches = list(_REFS_HEADERS.finditer(text))
    if not matches:
        return text, ""
    # Use the *last* match: a paper may say "References" in the abstract and again
    # at the bibliography. The bibliography is almost always the final occurrence.
    m = matches[-1]
    return text[: m.start()], text[m.end() :]


def guess_title(text: str) -> Optional[str]:
    """First non-empty line is the canonical heuristic for a paper title."""
    for line in text.splitlines():
        s = line.strip()
        if 8 <= len(s) <= 250:
            return s
    return None


_ABSTRACT_RE = re.compile(
    r"(?:^|\n)\s*A\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*[\.\:\-]?\s*\n+(.+?)(?=\n\s*(?:1\.?\s+)?(?:Introduction|Background|Keywords)\b)",
    re.IGNORECASE | re.DOTALL,
)


def find_abstract(text: str) -> Optional[str]:
    m = _ABSTRACT_RE.search(text[:8000])  # Abstract is on page 1, no need to scan further.
    if not m:
        return None
    body = " ".join(m.group(1).split())
    return body or None


# ---------------------------------------------------------------------------
# bibliography parsing (LLM)


_BIB_SYSTEM = (
    "You parse academic bibliography sections into JSON. "
    "You never invent fields. If the source text doesn't tell you a value, leave it null. "
    "Output a single JSON object only — no prose, no fences."
)


def _bib_prompt(refs_section: str) -> str:
    # Trim very long sections — keep the head/tail so the LLM still sees the boundaries.
    if len(refs_section) > 60000:
        refs_section = refs_section[:30000] + "\n...\n" + refs_section[-30000:]
    return f"""Parse the bibliography below into a JSON object of the shape:

{{
  "format": "numeric" | "author-year",
  "refs": [
    {{
      "ref_id": "1" | "Smith2023" | ...,
      "raw_text": "<the entry, copied verbatim from the input>",
      "authors": ["Last, First", "Last, First"],
      "title": "...",
      "year": 2023,
      "venue": "...",
      "doi": "10.xxxx/xxxx" | null,
      "arxiv_id": "2401.12345" | null,
      "url": "..." | null
    }}
  ]
}}

Rules:
- "format" is "numeric" if the entries are numbered like "[1]" or "1.", otherwise "author-year".
- "ref_id" for numeric format is the number as a string ("1", "2", "12"). For author-year format use the first author's surname plus the year, e.g. "Smith2023". Disambiguate duplicates with letter suffixes: "Smith2023a", "Smith2023b".
- "raw_text" must be copied verbatim from the input — do not reformat or correct it.
- "authors" is a list of full names. Preserve the original order.
- "doi", "arxiv_id", "url" — only fill these in if they appear literally in the entry. Do not guess from the title.
- If an entry is unintelligible, skip it rather than making something up.

Bibliography:

```
{refs_section}
```
"""


async def parse_references(
    refs_section: str,
    *,
    model: str,
    verbose: bool = False,
) -> tuple[str, dict[str, Reference]]:
    """Returns (citation_format, references_by_ref_id).

    citation_format is "numeric" or "author-year"; this drives the in-text
    regex used in the next step.
    """
    if not refs_section.strip():
        return "author-year", {}

    data = await acompletion_json(
        model=model,
        system=_BIB_SYSTEM,
        user=_bib_prompt(refs_section),
        verbose=verbose,
    )
    fmt = data.get("format") or "author-year"
    refs: dict[str, Reference] = {}
    for item in data.get("refs", []) or []:
        try:
            ref_id = str(item.get("ref_id") or "").strip()
            raw_text = (item.get("raw_text") or "").strip()
            if not ref_id or not raw_text:
                continue
            # Sanity check: the raw_text the LLM gave us should appear in the input.
            # We do a loose check (collapse whitespace) rather than exact equality.
            if _normalize_ws(raw_text)[:80] not in _normalize_ws(refs_section):
                _log(verbose, f"dropping ref {ref_id}: raw_text not in source")
                continue
            year = item.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            refs[ref_id] = Reference(
                ref_id=ref_id,
                raw_text=raw_text,
                authors=[a for a in (item.get("authors") or []) if isinstance(a, str)],
                title=item.get("title"),
                year=year if isinstance(year, int) else None,
                venue=item.get("venue"),
                doi=item.get("doi"),
                arxiv_id=item.get("arxiv_id"),
                url=item.get("url"),
            )
        except Exception as e:
            _log(verbose, f"skipping malformed ref entry: {e!r}")
    _log(verbose, f"parsed {len(refs)} bibliography entries (format={fmt})")
    return fmt, refs


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# in-text citation detection


_NUMERIC_CITE = re.compile(r"\[(\d+(?:\s*[,–\-]\s*\d+)*)\]")
_PAREN_AY_CITE = re.compile(
    r"\(([A-Z][\w\-']+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][\w\-']+)?(?:\s+et\s+al\.?)?\s*[,;]?\s*\d{4}[a-z]?(?:\s*[;,]\s*[A-Z][\w\-']+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][\w\-']+)?(?:\s+et\s+al\.?)?\s*[,;]?\s*\d{4}[a-z]?)*)\)"
)
_INLINE_AY_CITE = re.compile(
    r"\b([A-Z][\w\-']+(?:\s+(?:and|&)\s+[A-Z][\w\-']+|\s+et\s+al\.?)?)\s*\((\d{4}[a-z]?)\)"
)
# Sentence boundary: terminator + capital, OR a paragraph break.
# The paragraph break catches section transitions left as ``\n\nHeading\n`` by
# the TeX stripper, so a sentence near the end of one section doesn't bleed
# into the heading text of the next.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])|\n\s*\n")


def _expand_numeric_range(s: str) -> list[str]:
    """[1, 2, 5–7] → ["1", "2", "5", "6", "7"]"""
    out: list[str] = []
    for chunk in re.split(r"\s*,\s*", s):
        m = re.match(r"\s*(\d+)\s*[–\-]\s*(\d+)\s*$", chunk)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a < 100:
                out.extend(str(i) for i in range(a, b + 1))
                continue
        m2 = re.match(r"\s*(\d+)\s*$", chunk)
        if m2:
            out.append(m2.group(1))
    return out


def _surrounding_sentence(text: str, span: tuple[int, int]) -> str:
    """Pick the sentence containing the span."""
    start, end = span
    # Find the start of the sentence.
    s = max(0, start - 600)
    chunk = text[s:end + 600]
    splits = list(_SENTENCE_SPLIT.finditer(chunk))
    sent_start = 0
    sent_end = len(chunk)
    rel_pos = start - s
    for m in splits:
        if m.end() <= rel_pos:
            sent_start = m.end()
        elif m.start() >= rel_pos and sent_end == len(chunk):
            sent_end = m.start()
            break
    return _normalize_ws(chunk[sent_start:sent_end])


def find_citations(
    body: str,
    references: dict[str, Reference],
    fmt: str,
) -> list[Citation]:
    citations: list[Citation] = []

    if fmt == "numeric" and references:
        for m in _NUMERIC_CITE.finditer(body):
            for num in _expand_numeric_range(m.group(1)):
                if num in references:
                    citations.append(
                        Citation(
                            ref_id=num,
                            sentence=_surrounding_sentence(body, m.span()),
                        )
                    )
        return citations

    # Author-year: build a lookup from (surname_lower, year) → ref_id.
    lookup: dict[tuple[str, int], str] = {}
    for ref_id, ref in references.items():
        if not ref.authors or ref.year is None:
            continue
        first = ref.authors[0]
        # Names can be "Last, First" or "First Last".
        if "," in first:
            surname = first.split(",", 1)[0].strip()
        else:
            surname = first.split()[-1].strip()
        if surname:
            lookup[(surname.lower(), ref.year)] = ref_id

    for m in _PAREN_AY_CITE.finditer(body):
        sentence = _surrounding_sentence(body, m.span())
        for cite in _split_ay_cluster(m.group(1)):
            ref_id = _match_ay_to_ref(cite, lookup)
            if ref_id:
                citations.append(Citation(ref_id=ref_id, sentence=sentence))

    for m in _INLINE_AY_CITE.finditer(body):
        sentence = _surrounding_sentence(body, m.span())
        ref_id = _match_ay_to_ref(f"{m.group(1)}, {m.group(2)}", lookup)
        if ref_id:
            citations.append(Citation(ref_id=ref_id, sentence=sentence))

    return citations


def _split_ay_cluster(s: str) -> list[str]:
    return [c.strip() for c in re.split(r"\s*;\s*", s) if c.strip()]


def _match_ay_to_ref(cite: str, lookup: dict[tuple[str, int], str]) -> Optional[str]:
    m = re.match(
        r"([A-Z][\w\-']+)(?:\s+(?:and|&)\s+[A-Z][\w\-']+|\s+et\s+al\.?)?\s*[,;]?\s*(\d{4})",
        cite,
    )
    if not m:
        return None
    surname = m.group(1).lower()
    year = int(m.group(2))
    return lookup.get((surname, year))


# ---------------------------------------------------------------------------
# entry point


async def ingest_pdf(
    pdf_path: Path,
    *,
    model: str,
    verbose: bool = False,
) -> IngestedPaper:
    text = extract_text(pdf_path)
    if not text.strip():
        raise RuntimeError(f"could not extract any text from {pdf_path}")

    title = guess_title(text)
    abstract = find_abstract(text)
    body, refs_section = split_at_references(text)
    fmt, references = await parse_references(refs_section, model=model, verbose=verbose)
    citations = find_citations(body, references, fmt)
    _log(verbose, f"found {len(citations)} in-text citations")
    return IngestedPaper(
        title=title,
        abstract=abstract,
        sections=[("body", body)],
        references=references,
        citations=citations,
    )
