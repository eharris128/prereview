"""Small shared TeX-source helpers.

A leaf utility module (imports only ``re``) so feature modules — ``tex_ingest``,
``anonymize``, ``venue_rules``, ``numeric_sanity`` — can share these without
coupling to each other. Extracted once the same comment-strip / brace-reader /
fragment-flattener appeared verbatim in three of them.

``checklist.py`` keeps its own ``_strip_latex`` deliberately: it does *not* keep
one-arg command arguments (it drops them), which is different behavior from
:func:`flatten_tex` here, so it is not the same helper.
"""

from __future__ import annotations

import re


def strip_comments(text: str) -> str:
    """Drop ``% ...`` line comments (but not escaped ``\\%``)."""
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def read_balanced(text: str, i: int) -> tuple[str, int]:
    """Given ``text[i] == '{'``, return ``(inner, index_after_close)``, brace-aware.

    Tolerates nested ``{...}`` so an ``\\author`` block holding
    ``\\thanks{Univ. of {X}}`` is not truncated. An unterminated brace returns
    everything to end-of-string rather than raising.
    """
    depth = 1
    i += 1
    start = i
    n = len(text)
    while i < n and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth > 0:
            i += 1
    return text[start:i], i + 1


# Table-family environments whose bodies survive prose stripping only in the raw
# source — used by the color-table guard (U3) and the table-aware numeric checks (U6).
TABLE_ENV_RE = re.compile(
    r"\\begin\s*\{(table\*?|tabular\*?|tabularx|longtable)\}(.*?)\\end\s*\{\1\}",
    re.DOTALL,
)


def table_bodies(tex_text: str) -> list[str]:
    """The inner body of every table/tabular environment, comments stripped."""
    text = strip_comments(tex_text)
    return [m.group(2) for m in TABLE_ENV_RE.finditer(text)]


def flatten_tex(s: str) -> str:
    """Reduce a TeX fragment to readable text, **keeping the argument** of one-arg
    commands (so ``\\thanks{a@b.edu}`` retains its identity-bearing content) while
    dropping command names and stray braces. Mirrors :func:`tex_ingest.extract_title`'s
    cleanup, reused for author/ack/section text and anonymization evidence."""
    for _ in range(4):
        s = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|textrm|textsf)\{([^{}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()
