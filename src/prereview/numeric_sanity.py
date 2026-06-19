"""ML numerical-sanity pack (deterministic, high-precision).

Catches the numerical errors reviewers pounce on — a bounded metric over its
ceiling, a train/val/test split that doesn't sum to the stated total, a mean±std
whose range escapes the metric's bounds, a hyperparameter that disagrees between
prose and table, and an abstract headline-delta that no results table backs up.

Precision over recall, hard: every detector is conservative and **skips when it
is unsure**, because the one thing worse than missing an error is crying wolf on
a benign number. The benign-corpus test (``tests/test_numeric_sanity.py``) is the
contract — detectors #3 (mean±std) and #4 (hyperparameter) are the touchy ones,
so the corpus targets them with scientific notation, ±-as-tolerance, ranges, and
non-metric percentages. Findings are advisory: a flag is a number to re-check.

The first three checks read the stripped prose ``body``; mean±std and the two
table-aware checks read the raw ``tex_text`` because ``strip_tex_to_text`` deletes
math and ``tabular`` environments.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .models import NumericFinding, NumericFindingKind
from .texutils import table_bodies

_Kind = NumericFindingKind


# ---------------------------------------------------------------------------
# externalized thresholds / vocab

_HYPERPARAM_RATIO_GATE = 2.0  # prose vs table must differ by more than this to flag
_DELTA_TOLERANCE = 0.20  # abstract delta vs table delta: within 20% counts as a match
_SPLIT_TOLERANCE = 0.01  # train+val+test must match total within 1%

# Metrics that are conventionally bounded. "fraction" ones live in [0, 1];
# "percent" ones are usually reported on a 0–100 scale.
_FRACTION_METRICS = (
    "f1", "f-1", "f score", "f-score", "f-measure", "auc", "auroc", "au-roc", "roc-auc",
    "precision", "recall", "ap", "map", "miou", "iou", "dice", "mcc", "bleu", "rouge",
    "ndcg", "mrr", "r-squared", "r2", "pearson", "spearman", "kappa", "jaccard",
)
_PERCENT_METRICS = ("accuracy", "acc", "error rate", "top-1", "top-5", "top 1", "top 5")
_ALL_METRICS = _FRACTION_METRICS + _PERCENT_METRICS

# A number is part of a delta/multiplier, not a metric value, when adjacent to one
# of these — keeps the bounded-metric check off "improved by 1.5 points" etc. ("of"
# is deliberately excluded: it is the connective in "F1 of 1.07", a real value.)
_DELTA_BEFORE = re.compile(r"(?:\bby\b|[+±]|\bgain\b|\bimprov\w*\b)\s*$", re.IGNORECASE)
_DELTA_AFTER = re.compile(r"^\s*(?:x\b|×|times|points?|pp\b|percentage points|fold\b)", re.IGNORECASE)


def _to_float(s: str) -> Optional[float]:
    """Parse a numeric token, honoring ``k``/``M`` suffixes and scientific notation."""
    s = s.strip().replace(",", "")
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([kKmM]?)$", s)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    suffix = m.group(2).lower()
    if suffix == "k":
        val *= 1e3
    elif suffix == "m":
        val *= 1e6
    return val


def _truncate(s: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _table_text(tex_text: str) -> str:
    """Raw table bodies joined (where math/numbers survive the prose stripper)."""
    return "\n".join(table_bodies(tex_text))


# ---------------------------------------------------------------------------
# detector 1: bounded-metric violation


_METRIC_ALT = "|".join(re.escape(m) for m in sorted(_ALL_METRICS, key=len, reverse=True))
_FRACTION_ALT = "|".join(re.escape(m) for m in sorted(_FRACTION_METRICS, key=len, reverse=True))
# metric <connective> NN.N%  (a tight 0-15 gap so an unrelated nearby "120% of
# baseline" is not attributed to the metric word).
_PCT_METRIC_RE = re.compile(rf"\b({_METRIC_ALT})\b[^.\n]{{0,15}}?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
# fraction metric <connective> 1.NN  — spaces-only between metric and value (no
# intervening words), so "recall computation took 1.5 seconds" is NOT matched.
_FRAC_METRIC_RE = re.compile(
    rf"\b({_FRACTION_ALT})\b[ ]*(?:score|value)?[ ]*"
    r"(?:of|is|was|are|were|=|:|reaches?|reached|at)?[ ]*(\d+\.\d+)",
    re.IGNORECASE,
)


def check_bounded_metrics(body: str) -> list[NumericFinding]:
    findings: list[NumericFinding] = []
    for m in _PCT_METRIC_RE.finditer(body):
        if _to_float(m.group(2)) and float(m.group(2)) > 100.0:
            findings.append(
                NumericFinding(
                    kind=_Kind.BOUNDED_METRIC,
                    detail=f"a {m.group(1).lower()} of {m.group(2)}% exceeds the 100% ceiling — verify the number",
                    evidence=_truncate(m.group(0)),
                )
            )
    for m in _FRAC_METRIC_RE.finditer(body):
        val = float(m.group(2))
        if not (1.0 < val < 2.0):
            continue
        # Skip delta/multiplier contexts ("improved by 1.5 points", "1.05x").
        before = body[max(0, m.start(2) - 6): m.start(2)]
        after = body[m.end(2): m.end(2) + 14]
        if _DELTA_BEFORE.search(before) or _DELTA_AFTER.search(after) or after.lstrip().startswith("%"):
            continue
        findings.append(
            NumericFinding(
                kind=_Kind.BOUNDED_METRIC,
                detail=f"a {m.group(1).lower()} of {m.group(2)} exceeds the 1.0 ceiling for this metric — verify the number",
                evidence=_truncate(m.group(0)),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# detector 2: train/val/test split arithmetic


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _labeled_number(sent: str, label: str) -> Optional[float]:
    """A number attached to ``label`` as "label N", "label of N", or "N label".

    ``label`` is wrapped in a non-capturing group so an alternation inside it
    (e.g. ``val(?:idation)?|dev``) does not split the whole pattern or shift the
    numeric capture group.
    """
    # Number-before-label FIRST ("1000 test"), so it wins over the label-of-number
    # form ("test of 10000 total") which would otherwise bind the grand total as the
    # split count. No ',' connective either — it let "8000 train, 1000 val" bind the
    # next clause's number to ``train``.
    for pat in (rf"(\d[\d.,]*\s*[kKmM]?)\s+(?:{label})\b",
                rf"(?:{label})\b\s*(?:set\s*)?(?:of|:|=|with)?\s*(\d[\d.,]*\s*[kKmM]?)\b"):
        m = re.search(pat, sent, re.IGNORECASE)
        if m:
            return _to_float(m.group(1))
    return None


# The grand total must be *explicitly signalled* — a bare "of N" or "N examples"
# matched the train count itself ("a training set of 8000 …"), fabricating a
# mismatch. Require "total" / "in total" / "out of" adjacency instead.
_TOTAL_RE = re.compile(
    r"(?:a\s+)?total(?:ing)?\s+(?:of\s+)?(\d[\d.,]*\s*[kKmM]?)"
    r"|(\d[\d.,]*\s*[kKmM]?)\s*(?:examples|samples|instances|images|sentences|documents|data\s*points)?\s*(?:in\s+total|total\b)"
    r"|\bout\s+of\s+(?:a\s+total\s+of\s+)?(\d[\d.,]*\s*[kKmM]?)",
    re.IGNORECASE,
)


def check_split_arithmetic(body: str) -> list[NumericFinding]:
    findings: list[NumericFinding] = []
    for sent in _SENT_SPLIT.split(body):
        train = _labeled_number(sent, r"train(?:ing)?")
        val = _labeled_number(sent, r"val(?:idation)?|dev(?:elopment)?")
        test = _labeled_number(sent, r"test(?:ing)?")
        if not (train and val and test):
            continue
        tm = _TOTAL_RE.search(sent)
        if not tm:
            continue
        total = _to_float(next(g for g in tm.groups() if g))
        if not total or total <= 0:
            continue
        parts = train + val + test
        if abs(parts - total) / total > _SPLIT_TOLERANCE:
            findings.append(
                NumericFinding(
                    kind=_Kind.SPLIT_MISMATCH,
                    detail=(
                        f"train+val+test = {int(train)}+{int(val)}+{int(test)} = {int(parts)}, "
                        f"but the stated total is {int(total)} — verify the split"
                    ),
                    evidence=_truncate(sent),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# detector 3: mean±std escapes the metric bound (over raw tex — math survives)


_MEAN_STD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:±|\\pm)\s*(\d+(?:\.\d+)?)\s*(%?)")


def check_mean_std(tex_text: str) -> list[NumericFinding]:
    """Flag a metric reported as ``mean ± std`` whose **central value** is itself
    impossible — a percent-scale mean above 100.

    Deliberately conservative: a near-ceiling mean whose ``± std`` merely *touches*
    the bound (e.g. ``F1 of 0.95 ± 0.08`` → 1.03) is the normal way strong results
    are reported and must NOT flag — that was a false-positive in an earlier design.
    A fraction-metric mean above 1.0 is caught by the bounded-metric check instead."""
    findings: list[NumericFinding] = []
    for m in _MEAN_STD_RE.finditer(tex_text):
        mean = float(m.group(1))
        pct = m.group(3) == "%"
        window = tex_text[max(0, m.start() - 60): m.start()].lower()
        has_metric = pct or any(metric in window for metric in _ALL_METRICS)
        if not has_metric:
            continue  # not clearly a bounded-metric measurement — skip
        # Only a percent-scale mean over 100 is unambiguously impossible.
        if (pct or mean > 1.0) and mean > 100.0 + 1e-9:
            findings.append(
                NumericFinding(
                    kind=_Kind.MEAN_STD_RANGE,
                    detail=(
                        f"a mean of {m.group(1)}{'%' if pct else ''} ± {m.group(2)} exceeds the "
                        "100 ceiling for this metric — verify the number"
                    ),
                    evidence=_truncate(m.group(0)),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# detector 4: prose vs table hyperparameter drift


# (display name, regex naming the hyperparameter before its value)
_HYPERPARAMS: tuple[tuple[str, str], ...] = (
    ("learning rate", r"(?:learning[ \-]?rate|lr)"),
    ("batch size", r"batch[ \-]?size"),
    ("epochs", r"(?:number of\s+)?epochs?"),
)
_NUM = r"(\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _hyperparam_values(text: str, name_pat: str) -> list[float]:
    # The separator spans connectives AND table-cell delimiters (& |) so a value
    # in a ``name & value`` table row is captured the same as "name of value".
    vals: list[float] = []
    pat = rf"{name_pat}[\s:=&|~≈]*(?:set\s+to|of|is|was|are|were|to)?[\s:=&|~≈]*{_NUM}"
    for m in re.finditer(pat, text, re.IGNORECASE):
        v = _to_float(m.group(1))
        if v is not None and v > 0:
            vals.append(v)
    return vals


def check_hyperparameter_drift(body: str, tex_text: str) -> list[NumericFinding]:
    tables = _table_text(tex_text)
    if not tables:
        return []
    findings: list[NumericFinding] = []
    for name, pat in _HYPERPARAMS:
        prose_vals = _hyperparam_values(body, pat)
        table_vals = _hyperparam_values(tables, pat)
        if not prose_vals or not table_vals:
            continue
        # Disagree only if NO prose value lands within the gate of ANY table value.
        min_ratio = min(
            max(pv, tv) / min(pv, tv) for pv in prose_vals for tv in table_vals
        )
        if min_ratio > _HYPERPARAM_RATIO_GATE:
            findings.append(
                NumericFinding(
                    kind=_Kind.HYPERPARAM_DRIFT,
                    detail=(
                        f"{name} differs between prose ({', '.join(_fmt(v) for v in sorted(set(prose_vals)))}) "
                        f"and table ({', '.join(_fmt(v) for v in sorted(set(table_vals)))}) by more than "
                        f"{_HYPERPARAM_RATIO_GATE:g}× — verify they match"
                    ),
                    evidence="",
                )
            )
    return findings


def _fmt(v: float) -> str:
    return f"{v:g}"


# ---------------------------------------------------------------------------
# detector 5: abstract headline-delta absent from the tables


# Abstract deltas require an explicit unit (conservative — prose is noisy).
_ABS_DELTA_RE = re.compile(
    r"\+\s*(\d+(?:\.\d+)?)\s*(?:%|points?|pp\b|percentage\s+points)"
    r"|\bby\s+(\d+(?:\.\d+)?)\s*(?:%|points?|pp\b|percentage\s+points)"
    r"|(?:gain|improvement|increase|boost)\s+of\s+(\d+(?:\.\d+)?)\s*(?:%|points?|pp\b|percentage\s+points)"
    r"|(\d+(?:\.\d+)?)[\s\-]point\b",
    re.IGNORECASE,
)
# Inside a results table a bare "+N.N" reads as an improvement delta.
_TABLE_DELTA_RE = re.compile(r"\+\s*(\d+(?:\.\d+)?)")


def _first_group(m: re.Match) -> Optional[str]:
    return next((g for g in m.groups() if g), None)


def _abs_deltas(text: str) -> list[float]:
    out: list[float] = []
    for m in _ABS_DELTA_RE.finditer(text):
        g = _first_group(m)
        v = _to_float(g) if g else None
        if v is not None:
            out.append(v)
    return out


def _table_deltas(text: str) -> list[float]:
    out: list[float] = []
    for m in _TABLE_DELTA_RE.finditer(text):
        v = _to_float(m.group(1))
        if v is not None:
            out.append(v)
    return out


def _delta_matches(abstract_delta: float, table_delta: float) -> bool:
    """Within the relative tolerance — with a guard so two zero deltas (a '+0'
    abstract claim paired with a '+0' table cell) compare equal instead of
    dividing by zero."""
    scale = max(abs(abstract_delta), abs(table_delta))
    if scale == 0:
        return True
    return abs(abstract_delta - table_delta) / scale <= _DELTA_TOLERANCE


def check_table_abstract_delta(abstract: Optional[str], tex_text: str) -> list[NumericFinding]:
    if not abstract:
        return []
    abs_deltas = _abs_deltas(abstract)
    if not abs_deltas:
        return []
    table_deltas = _table_deltas(_table_text(tex_text))
    if not table_deltas:
        return []  # tables state no deltas — nothing to reconcile against
    findings: list[NumericFinding] = []
    for d in abs_deltas:
        if not any(_delta_matches(d, td) for td in table_deltas):
            findings.append(
                NumericFinding(
                    kind=_Kind.ABSTRACT_TABLE_DELTA,
                    detail=(
                        f"the abstract claims a +{_fmt(d)} improvement, but no results table "
                        f"reports a matching delta (tables show {', '.join('+' + _fmt(t) for t in sorted(set(table_deltas)))}) "
                        "— verify the headline number"
                    ),
                    evidence="",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# registry + entry point


_Detector = Callable[..., list[NumericFinding]]
# Capped and precision-guarded (a test asserts this stays small) so the pack does
# not drift into a noisy pile of low-confidence heuristics.
DETECTORS: tuple[str, ...] = (
    "bounded_metric",
    "split_arithmetic",
    "mean_std",
    "hyperparameter_drift",
    "abstract_table_delta",
)


def audit_numeric(body: str, tex_text: str, abstract: Optional[str]) -> list[NumericFinding]:
    """Run every numerical-sanity detector and return the combined findings."""
    findings: list[NumericFinding] = []
    findings += check_bounded_metrics(body)
    findings += check_split_arithmetic(body)
    findings += check_mean_std(tex_text)
    findings += check_hyperparameter_drift(body, tex_text)
    findings += check_table_abstract_delta(abstract, tex_text)
    return findings
