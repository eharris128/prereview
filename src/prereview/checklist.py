"""Reproducibility-checklist linter (AAAI-27 first).

Why this exists: AAAI / NeurIPS / ACL-ARR require a reproducibility checklist,
and ACL/ARR has desk-rejected incorrect or incomplete checklists since Dec 2024.
The checklist is the same *class* of artifact ``prereview`` already lints — a
structured, source-level file that is mechanically checkable — but the existing
hygiene checks never looked at it. This module closes that gap.

Everything here is deterministic (no LLM), mirroring :mod:`prereview.tex_ingest`:
detector functions take raw text and return pydantic models. Two tiers of
finding are produced:

1. **Self-consistency** (:func:`check_self_consistency`) — unanswered items,
   responses outside the allowed option set, and gate-vs-subitem contradictions.
   Needs only the checklist file.
2. **Claim-vs-paper** (:func:`check_claims_vs_paper`) — a checklist answer of
   "yes"/"partial" whose supporting evidence is absent from the paper body
   (e.g. "code will be made publicly available" = yes, but no repository URL
   appears anywhere). Presence-based and advisory-framed — precision over recall.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from .models import ChecklistFinding, ChecklistFindingKind, ChecklistItem, LinkCheck

# ---------------------------------------------------------------------------
# small shared helpers


def _strip_comments(text: str) -> str:
    """Drop ``% ...`` comments (but not escaped ``\\%``), like tex_ingest does."""
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    return i


def _read_braced(text: str, i: int) -> tuple[str, int]:
    """Given ``text[i] == '{'``, return ``(inner_content, index_after_close)``.

    Brace-aware (handles nested ``{...}``), so a ``\\question`` whose text holds a
    nested brace is not truncated. If the braces never close, returns everything
    to end-of-string rather than raising.
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


def _strip_latex(s: str) -> str:
    """Reduce a fragment of LaTeX to readable text: keep the argument of common
    formatting commands, drop other commands and stray braces, collapse space."""
    for _ in range(3):
        s = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|textsf|textrm)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def _parse_options(raw: str) -> list[str]:
    """``(yes/partial/no/NA)`` -> ``["yes", "partial", "no", "na"]``."""
    return [w.lower() for w in re.findall(r"[A-Za-z]+", raw)]


# ---------------------------------------------------------------------------
# discovery


def find_checklist_file(
    tex_path: Path,
    tex_text: str,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """Locate the reproducibility checklist file (KTD-5 order).

    Priority: an ``explicit`` ``--checklist`` path (returned as-is so the caller
    can produce a clean error if it is missing) > an ``\\input{...}`` whose target
    basename contains "checklist", resolved relative to the main ``.tex`` >
    sibling ``ReproducibilityChecklist.tex``. Returns ``None`` when nothing
    matches (the author may not have started the checklist yet).
    """
    if explicit is not None:
        return explicit

    text = _strip_comments(tex_text)
    for m in re.finditer(r"\\input\s*\{([^{}]+)\}", text):
        target = m.group(1).strip()
        if "checklist" not in Path(target).name.lower():
            continue
        cand = tex_path.parent / target
        if cand.suffix.lower() != ".tex":
            cand = cand.with_suffix(".tex")
        if cand.exists():
            return cand

    sibling = tex_path.parent / "ReproducibilityChecklist.tex"
    if sibling.exists():
        return sibling
    return None


# ---------------------------------------------------------------------------
# parsing

# Tokens we care about, in source order. \question/\checksubsection/\ifyespoints
# are matched by name only (their braced args are read manually, brace-aware);
# itemize delimiters drive nesting depth for gate -> sub-item association.
_TOKEN_RE = re.compile(
    r"\\checksubsection\b"
    r"|\\question\b"
    r"|\\ifyespoints\b"
    r"|\\begin\s*\{\s*itemize\s*\}"
    r"|\\end\s*\{\s*itemize\s*\}"
)

_GATE_RE = re.compile(r"^\s*does this paper", re.IGNORECASE)


def parse_checklist(text: str) -> list[ChecklistItem]:
    """Parse a checklist's LaTeX into a list of :class:`ChecklistItem`.

    Generic over question strings — it extracts ``(section, question, options,
    response, gate context)`` structurally, so it survives AAAI-year tweaks and
    extends to other venues as data. A file with no ``\\question`` macros yields
    ``[]`` (callers treat that as "found but unparseable — skip") rather than
    raising.
    """
    text = _strip_comments(text)
    items: list[ChecklistItem] = []

    section: Optional[str] = None
    depth = 0
    pending_gate: Optional[str] = None  # a gate awaiting its sub-block's \begin{itemize}
    active_gate: Optional[str] = None  # the gate currently owning sub-items
    active_gate_depth: Optional[int] = None  # itemize depth at which sub-items live

    n = len(text)
    i = 0
    while i < n:
        m = _TOKEN_RE.search(text, i)
        if not m:
            break
        tok = m.group(0)
        i = m.end()

        if tok.startswith("\\checksubsection"):
            j = _skip_ws(text, i)
            if j < n and text[j] == "{":
                name, j = _read_braced(text, j)
                section = _strip_latex(name)
                i = j
            # A new section starts a fresh gate/nesting context.
            pending_gate = active_gate = active_gate_depth = None
            depth = 0

        elif tok.startswith("\\question"):
            j = _skip_ws(text, i)
            if j >= n or text[j] != "{":
                # Not a real call (e.g. the `\string\question` example in the kit).
                continue
            q_raw, j = _read_braced(text, j)
            j = _skip_ws(text, j)
            if j < n and text[j] == "{":
                opts_raw, j = _read_braced(text, j)
            else:
                opts_raw = ""
            # The response is everything up to the next structural token.
            nxt = _TOKEN_RE.search(text, j)
            response = _strip_latex(text[j : nxt.start() if nxt else n])

            options = _parse_options(opts_raw)
            question = _strip_latex(q_raw)
            is_gate = options == ["yes", "no"] and bool(_GATE_RE.match(question))
            in_block = active_gate is not None and active_gate_depth is not None and depth >= active_gate_depth
            items.append(
                ChecklistItem(
                    section=section,
                    question=question,
                    options=options,
                    response=response,
                    is_gate=is_gate,
                    gate_question=active_gate if in_block else None,
                )
            )
            if is_gate:
                pending_gate = question
            i = j

        elif tok.startswith("\\ifyespoints"):
            j = _skip_ws(text, i)
            if j < n and text[j] == "{":
                _, j = _read_braced(text, j)
                i = j

        elif tok.startswith("\\begin"):
            depth += 1
            if pending_gate is not None:
                active_gate = pending_gate
                active_gate_depth = depth
                pending_gate = None

        else:  # \end{itemize}
            if active_gate is not None and active_gate_depth is not None and depth <= active_gate_depth:
                active_gate = active_gate_depth = None
            depth = max(0, depth - 1)

    return items


# ---------------------------------------------------------------------------
# response normalization


_PLACEHOLDER = "type your response here"


def _is_blank(response: str) -> bool:
    r = response.strip().lower()
    return not r or _PLACEHOLDER in r


def _answer_token(response: str) -> Optional[str]:
    """The leading answer token of a response, lowercased, or ``None`` if blank.

    Tolerant of trailing punctuation, surrounding markup, and a trailing
    explanation: ``"Yes."`` -> ``"yes"``; ``"yes, see Sec. 3"`` -> ``"yes"``;
    ``"N/A"`` -> ``"na"``.
    """
    if _is_blank(response):
        return None
    words = re.findall(r"[a-zA-Z]+", response.lower())
    if not words:
        return None
    first = words[0]
    compact = "".join(words)
    # "n/a" tokenizes to ["n", "a"]; accept the compacted form when it is short
    # enough to be an option word rather than a sentence.
    if len(compact) <= 3:
        return compact
    return first


# ---------------------------------------------------------------------------
# tier 1: self-consistency


def check_self_consistency(items: list[ChecklistItem]) -> list[ChecklistFinding]:
    """Findings that need only the checklist itself.

    - **unanswered** — left blank or as the "Type your response here" placeholder.
      Sub-items of a gate answered "no" are skipped: their answer legitimately
      cascades from the gate, so flagging them would be noise. A blank *gate* is
      still flagged.
    - **invalid_response** — a non-blank answer whose leading token is not one of
      the item's allowed options.
    - **gate_inconsistency** — a gate answered "no" whose sub-items nonetheless
      carry substantive answers, or a gate answered "yes" all of whose sub-items
      are blank. One finding per gate (sub-items are not double-warned here).
    """
    gate_answer = {it.question: _answer_token(it.response) for it in items if it.is_gate}
    findings: list[ChecklistFinding] = []

    for it in items:
        # Skip sub-items whose gate is "no" — their blankness is expected.
        if it.gate_question is not None and gate_answer.get(it.gate_question) == "no":
            continue
        if _is_blank(it.response):
            findings.append(
                ChecklistFinding(
                    kind=ChecklistFindingKind.UNANSWERED,
                    section=it.section,
                    question=it.question,
                    response=it.response.strip(),
                )
            )
            continue
        token = _answer_token(it.response)
        if it.options and token is not None and token not in it.options:
            findings.append(
                ChecklistFinding(
                    kind=ChecklistFindingKind.INVALID_RESPONSE,
                    section=it.section,
                    question=it.question,
                    response=it.response.strip(),
                    detail="allowed: " + ", ".join(it.options),
                )
            )

    # Gate-vs-subitem consistency (one finding per gate).
    subitems_by_gate: dict[str, list[ChecklistItem]] = {}
    for it in items:
        if it.gate_question is not None:
            subitems_by_gate.setdefault(it.gate_question, []).append(it)

    for gate in (it for it in items if it.is_gate):
        ans = gate_answer.get(gate.question)
        subs = subitems_by_gate.get(gate.question, [])
        if not subs:
            continue
        if ans == "no":
            answered = [s for s in subs if not _is_blank(s.response) and _answer_token(s.response) != "na"]
            if answered:
                findings.append(
                    ChecklistFinding(
                        kind=ChecklistFindingKind.GATE_INCONSISTENCY,
                        section=gate.section,
                        question=gate.question,
                        response=gate.response.strip(),
                        detail=(
                            f'answered "no", but {len(answered)} of its sub-items '
                            f"carry substantive answers"
                        ),
                    )
                )
        elif ans == "yes":
            if all(_is_blank(s.response) for s in subs):
                findings.append(
                    ChecklistFinding(
                        kind=ChecklistFindingKind.GATE_INCONSISTENCY,
                        section=gate.section,
                        question=gate.question,
                        response=gate.response.strip(),
                        detail='answered "yes", but all of its sub-items are blank',
                    )
                )

    return findings


# ---------------------------------------------------------------------------
# tier 2: claim-vs-paper cross-checks

_REPO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "zenodo.org", "osf.io", "huggingface.co", "codeocean.com")
_DATASET_HOSTS = (
    "huggingface.co/datasets",
    "kaggle.com",
    "openml.org",
    "zenodo.org",
    "figshare.com",
    "data.mendeley.com",
    "paperswithcode.com/dataset",
)
_DATASET_NAMES = (
    "imagenet", "cifar", "mnist", "coco", "squad", "glue", "wikitext", "librispeech",
    "penn treebank", "ptb", "wmt", "conll", "snli", "mnli", "celeba", "svhn",
)
_HARDWARE_RE = re.compile(
    r"\b(gpu|cpu|tpu|a100|h100|v100|a6000|a40|rtx|gtx|titan|tesla|cuda|nvidia|"
    r"\d+\s*gb|cores?|ram|gpu[- ]hours?)\b",
    re.IGNORECASE,
)
_RUNS_RE = re.compile(
    r"(\b\d+\s+(?:runs|trials|seeds|repetitions|replicates)\b"
    r"|\brandom seeds?\b|\bseeds?\b|\baveraged over\b|\bindependent runs?\b"
    r"|\bstandard deviation\b|\bstd\.?\b|±)",
    re.IGNORECASE,
)


def _has_repo_url(body: str, links: list[LinkCheck]) -> bool:
    hay = body.lower()
    if any(host in hay for host in _REPO_HOSTS):
        return True
    return any(any(host in (lc.url or "").lower() for host in _REPO_HOSTS) for lc in links)


def _has_hardware(body: str, links: list[LinkCheck]) -> bool:
    return bool(_HARDWARE_RE.search(body))


def _has_runs(body: str, links: list[LinkCheck]) -> bool:
    return bool(_RUNS_RE.search(body))


def _has_dataset_evidence(body: str, links: list[LinkCheck]) -> bool:
    hay = body.lower()
    if any(host in hay for host in _DATASET_HOSTS):
        return True
    if any(any(host in (lc.url or "").lower() for host in _DATASET_HOSTS) for lc in links):
        return True
    return any(name in hay for name in _DATASET_NAMES)


# Each cross-check: every substring in ``match`` must appear (lowercased) in the
# question, ``detector(body, links)`` returns True when supporting evidence is
# present, and ``evidence`` names what was sought (for the advisory finding).
# Deliberately small and high-precision — a check that cannot be made precise is
# left out rather than shipped noisy.
_Detector = Callable[[str, list[LinkCheck]], bool]
CROSS_CHECKS: list[tuple[list[str], str, _Detector]] = [
    (["source code", "publicly available"], "a public repository URL (e.g. github.com / zenodo.org)", _has_repo_url),
    (["computing infrastructure"], "hardware details (GPU/CPU/TPU model, memory)", _has_hardware),
    (["number of algorithm runs"], "the number of runs / random seeds", _has_runs),
    (["datasets", "publicly available"], "a dataset URL or a recognizable public-dataset name", _has_dataset_evidence),
]


def check_claims_vs_paper(
    items: list[ChecklistItem],
    body_text: str,
    link_checks: list[LinkCheck],
) -> list[ChecklistFinding]:
    """Flag "yes"/"partial" answers whose supporting evidence is absent.

    Only affirmative answers are cross-checked (a "no"/"NA"/blank answer claims
    nothing to verify). Each finding quotes the checklist item and names the
    evidence that was sought, phrased as "verify" rather than an accusation —
    presence checks are advisory, not a judgement that the author was untruthful.
    """
    findings: list[ChecklistFinding] = []
    for it in items:
        if _answer_token(it.response) not in ("yes", "partial"):
            continue
        q = it.question.lower()
        for needles, evidence, detector in CROSS_CHECKS:
            if not all(needle in q for needle in needles):
                continue
            if detector(body_text, link_checks):
                continue
            findings.append(
                ChecklistFinding(
                    kind=ChecklistFindingKind.CLAIM_UNSUPPORTED,
                    section=it.section,
                    question=it.question,
                    response=it.response.strip(),
                    detail=f"no {evidence} found in the paper — verify",
                )
            )
    return findings
