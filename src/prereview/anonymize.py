"""Anonymization audit for double-blind submissions (deterministic, advisory).

Why this exists: AAAI (and most ML venues) summarily desk-reject papers that
break double-blind anonymization — a residual ``\\author`` block, a self-revealing
"in our previous work [4]" sentence, an acknowledgments section that names the
funder, or a ``github.com/jsmith`` link that identifies the author. These are
mechanically detectable in the ``.tex`` source, so this module flags them the
same precision-over-recall, advisory way the checklist linter does: every finding
quotes the exact offending fragment and is framed *"verify this does not
deanonymize you"*, never *"you leaked your identity"*.

Everything here is deterministic (no LLM), mirroring :mod:`prereview.checklist`.
Subtle / semantic self-revelation that genuinely needs judgement is deferred to a
follow-up LLM augmentation — the high-precision vectors ship first. A bare
common-surname grep is too noisy for every-save use, so the name-aware vector is
opt-in (``--authors``) and suppresses ``\\cite`` arguments and hyphenated
method/dataset n-grams.
"""

from __future__ import annotations

import re
from typing import Optional

from .ingest import _surrounding_sentence
from .models import AnonymizationFinding, AnonymizationFindingKind, LinkCheck

_Kind = AnonymizationFindingKind


# ---------------------------------------------------------------------------
# small shared helpers (kept local, like checklist.py, to stay a leaf module)


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _read_balanced(text: str, i: int) -> tuple[str, int]:
    """``text[i] == '{'`` → ``(inner, index_after_close)``, brace-aware."""
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


def _flatten(s: str) -> str:
    """Reduce a TeX fragment to readable text, keeping one-arg command content."""
    for _ in range(4):
        s = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|textrm|textsf)\{([^{}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def _truncate(s: str, n: int = 240) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# Markers that mean the field has been anonymized — these never flag.
_ANON_MARKERS = (
    "anonymous",
    "anonymized",
    "anonymised",
    "redacted",
    "double-blind",
    "blind review",
    "author names",
    "removed for review",
    "omitted for review",
    "withheld",
    "for review",
)


def _is_anonymized(s: Optional[str]) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True
    return any(marker in low for marker in _ANON_MARKERS)


# ---------------------------------------------------------------------------
# vector 1: residual identity blocks


_AUTHOR_HEAD_RE = re.compile(r"\\author\b\s*\*?\s*(?:\[[^\]]*\])?\s*")
_IDENTITY_CMDS = ("thanks", "affiliation", "affil", "institute", "email", "address")


def _author_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of every ``\\author{...}`` block (brace-aware)."""
    spans: list[tuple[int, int]] = []
    for m in _AUTHOR_HEAD_RE.finditer(text):
        k = m.end()
        if k < len(text) and text[k] == "{":
            _, end = _read_balanced(text, k)
            spans.append((m.start(), end))
    return spans


def check_residual_identity(
    tex_text: str, author_block: Optional[str]
) -> list[AnonymizationFinding]:
    """Flag an ``\\author`` block, or a standalone ``\\thanks`` / ``\\affiliation`` /
    ``\\email``, that still carries author-identifying content.

    ``author_block`` (U1, already brace-aware and flattened, including any nested
    ``\\thanks``) supplies the primary vector; the raw-source scan adds identity
    commands that sit *outside* the author block, deduped against it so a nested
    ``\\thanks`` is not double-reported.
    """
    findings: list[AnonymizationFinding] = []
    if author_block and not _is_anonymized(author_block):
        findings.append(
            AnonymizationFinding(
                kind=_Kind.RESIDUAL_IDENTITY,
                evidence=_truncate(author_block),
                detail="the \\author block still names the authors — verify the "
                "submission build is anonymized",
            )
        )

    text = _strip_comments(tex_text)
    spans = _author_spans(text)
    for cmd in _IDENTITY_CMDS:
        for m in re.finditer(r"\\" + cmd + r"\b\s*(?:\[[^\]]*\])?\s*\{", text):
            if any(s <= m.start() < e for s, e in spans):
                continue  # nested inside \author — already covered above
            inner, _ = _read_balanced(text, m.end() - 1)
            content = _flatten(inner)
            if content and not _is_anonymized(content):
                findings.append(
                    AnonymizationFinding(
                        kind=_Kind.RESIDUAL_IDENTITY,
                        evidence=_truncate(content),
                        detail=f"a \\{cmd}{{...}} outside the author block carries "
                        "identifying content — verify it is removed for review",
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# vector 2: self-revealing phrasing


_SELF_REVEAL_PATTERNS = (
    re.compile(r"\bin our (?:previous|prior|earlier|recent|past) work\b", re.IGNORECASE),
    re.compile(
        r"\bwe (?:previously|earlier|recently) (?:showed|proposed|introduced|developed|presented|demonstrated|published)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bour (?:previous|prior|earlier|recent) (?:paper|work|study|approach|method|model)\b", re.IGNORECASE),
    re.compile(r"\bas we (?:showed|proposed|argued|demonstrated) in\b", re.IGNORECASE),
    re.compile(r"\b(?:builds?|building) (?:up)?on our (?:previous|prior|earlier)\b", re.IGNORECASE),
    re.compile(r"\bextends? our (?:previous|prior|earlier)\b", re.IGNORECASE),
)

_CITE_MARKER_RE = re.compile(r"\[CITE:[^\]]*\]")


def _clean_sentence(s: str) -> str:
    s = _CITE_MARKER_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def check_self_revealing(body: str) -> list[AnonymizationFinding]:
    """Flag first-person references to prior work ("in our previous work …") that
    can deanonymize when the cited work is the authors' own. Quotes the sentence;
    third-person phrasing ("their previous work") never matches."""
    findings: list[AnonymizationFinding] = []
    seen: set[str] = set()
    for pat in _SELF_REVEAL_PATTERNS:
        for m in pat.finditer(body):
            sentence = _clean_sentence(_surrounding_sentence(body, m.span()))
            key = sentence.lower()
            if not sentence or key in seen:
                continue
            seen.add(key)
            findings.append(
                AnonymizationFinding(
                    kind=_Kind.SELF_REVEALING_PHRASE,
                    evidence=_truncate(sentence),
                    detail="a first-person reference to prior work can deanonymize — "
                    "verify it does not point to your own identity",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# vector 3: identity-revealing URLs


_IDENTITY_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")


def _is_identity_url(url: str) -> bool:
    low = (url or "").lower()
    host = low.split("//", 1)[-1].split("/", 1)[0]
    if any(h in host for h in _IDENTITY_HOSTS):
        return True
    if host.endswith(".github.io") or host.endswith(".gitlab.io"):
        return True
    if "/~" in low:  # classic ~user personal homepage
        return True
    return False


def check_identity_urls(link_checks: list[LinkCheck]) -> list[AnonymizationFinding]:
    """Flag code-host / personal-homepage URLs surfaced from the source. These
    both risk deanonymization and violate AAAI's no-web-links-to-supplementary
    rule. Anonymized mirrors (e.g. anonymous.4open.science) do not match."""
    findings: list[AnonymizationFinding] = []
    seen: set[str] = set()
    for lc in link_checks:
        url = lc.url or ""
        if url in seen or not _is_identity_url(url):
            continue
        seen.add(url)
        findings.append(
            AnonymizationFinding(
                kind=_Kind.IDENTITY_URL,
                evidence=url,
                detail="a code/personal URL can name the authors and also violates "
                "AAAI's no-links-to-supplementary rule — verify it is anonymized",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# vector 4: acknowledgments present


def check_acknowledgments(acknowledgments: Optional[str]) -> list[AnonymizationFinding]:
    """Flag a non-empty acknowledgments section in a submission build (funders /
    colleagues named there routinely deanonymize)."""
    if acknowledgments and not _is_anonymized(acknowledgments):
        return [
            AnonymizationFinding(
                kind=_Kind.ACKNOWLEDGMENTS_PRESENT,
                evidence=_truncate(acknowledgments),
                detail="acknowledgments often name funders/colleagues that deanonymize — "
                "verify this section is removed from the submission build",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# vector 5: name-aware surname grep (opt-in, suppressed)


def parse_authors(authors: Optional[str]) -> list[str]:
    """Split ``--authors "Smith, Jones"`` into title-cased surnames."""
    if not authors:
        return []
    out: list[str] = []
    for chunk in re.split(r"[,;]", authors):
        name = chunk.strip()
        if not name:
            continue
        # Take the last whitespace-token as the surname if a full name was given.
        surname = name.split()[-1]
        out.append(surname[:1].upper() + surname[1:])
    return out


def check_author_names(body: str, surnames: list[str]) -> list[AnonymizationFinding]:
    """Grep each surname against the (stripped) body, suppressing hits inside
    ``[CITE:...]`` markers and hyphenated method/dataset n-grams (Smith-Waterman).
    Running-prose hits are flagged with context — advisory, since the author asked
    for the name-aware pass explicitly."""
    findings: list[AnonymizationFinding] = []
    cite_spans = [m.span() for m in _CITE_MARKER_RE.finditer(body)]
    seen: set[str] = set()
    for surname in surnames:
        if not surname:
            continue
        for m in re.finditer(r"\b" + re.escape(surname) + r"\b", body):
            start, end = m.span()
            if any(cs <= start < ce for cs, ce in cite_spans):
                continue  # inside a citation key — not a prose mention
            after = body[end : end + 1]
            before = body[start - 1 : start]
            if (after == "-" and body[end + 1 : end + 2].isalpha()) or before == "-":
                continue  # hyphenated compound, e.g. Smith-Waterman
            sentence = _clean_sentence(_surrounding_sentence(body, m.span()))
            key = (surname, sentence.lower())
            sig = repr(key)
            if sig in seen:
                continue
            seen.add(sig)
            findings.append(
                AnonymizationFinding(
                    kind=_Kind.AUTHOR_NAME_IN_BODY,
                    evidence=_truncate(sentence),
                    detail=f'the author surname "{surname}" appears in running prose — '
                    "verify it does not deanonymize you",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# vector 6: dual-submission tell (advisory only — not real detection)


_DUAL_SUB_PATTERNS = (
    re.compile(r"\b(?:submitted|under review|under submission|in submission)\s+(?:to|at)\b", re.IGNORECASE),
    re.compile(r"\bcurrently under review\b", re.IGNORECASE),
    re.compile(r"\bsimultaneously submitted\b", re.IGNORECASE),
)


def check_dual_submission(body: str) -> list[AnonymizationFinding]:
    """Flag in-paper "submitted to / under review at X" phrasing. This cannot
    detect actual dual submission — it is a cheap advisory tell only."""
    findings: list[AnonymizationFinding] = []
    seen: set[str] = set()
    for pat in _DUAL_SUB_PATTERNS:
        for m in pat.finditer(body):
            sentence = _clean_sentence(_surrounding_sentence(body, m.span()))
            key = sentence.lower()
            if not sentence or key in seen:
                continue
            seen.add(key)
            findings.append(
                AnonymizationFinding(
                    kind=_Kind.DUAL_SUBMISSION_TELL,
                    evidence=_truncate(sentence),
                    detail="phrasing that names another venue can signal dual submission — "
                    "verify this does not breach the single-submission policy",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# entry point


def audit_anonymization(
    *,
    tex_text: str,
    body: str,
    author_block: Optional[str],
    acknowledgments: Optional[str],
    link_checks: list[LinkCheck],
    authors: Optional[str] = None,
) -> list[AnonymizationFinding]:
    """Run every name-free anonymization vector (plus the name-aware vector when
    ``authors`` is given) and return the combined advisory findings."""
    findings: list[AnonymizationFinding] = []
    findings += check_residual_identity(tex_text, author_block)
    findings += check_self_revealing(body)
    findings += check_identity_urls(link_checks)
    findings += check_acknowledgments(acknowledgments)
    findings += check_author_names(body, parse_authors(authors))
    findings += check_dual_submission(body)
    return findings
