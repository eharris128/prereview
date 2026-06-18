"""Tests for prereview.anonymize — deterministic double-blind audit.

Fully deterministic (no LLM, no network), so these assert concrete detector
outputs. The precision contract: name-free vectors must not false-positive on
third-person prose, and the name-aware vector must suppress citation keys and
hyphenated method names.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prereview.anonymize import (
    audit_anonymization,
    check_acknowledgments,
    check_author_names,
    check_dual_submission,
    check_identity_urls,
    check_residual_identity,
    check_self_revealing,
    parse_authors,
)
from prereview.models import AnonymizationFindingKind, LinkCheck

_Kind = AnonymizationFindingKind


def _kinds(findings) -> set:
    return {f.kind for f in findings}


# ---------------------------------------------------------------------------
# vector 1: residual identity


def test_residual_identity_flags_named_author_block():
    findings = check_residual_identity(r"\author{Jane Smith}", "Jane Smith")
    assert _Kind.RESIDUAL_IDENTITY in _kinds(findings)
    assert "Jane Smith" in findings[0].evidence


def test_residual_identity_suppressed_when_anonymous():
    assert check_residual_identity(r"\author{Anonymous Authors}", "Anonymous Authors") == []


def test_residual_identity_suppressed_when_blank():
    assert check_residual_identity(r"\author{}", None) == []


def test_residual_identity_flags_standalone_affiliation():
    tex = r"\affiliation{Massachusetts Institute of Technology}"
    findings = check_residual_identity(tex, None)
    assert _Kind.RESIDUAL_IDENTITY in _kinds(findings)
    assert "Massachusetts" in findings[0].evidence


def test_residual_identity_does_not_double_report_nested_thanks():
    # \thanks nested in \author must be reported once (via author_block), not twice.
    tex = r"\author{Jane Smith \thanks{State University}}"
    author_block = "Jane Smith State University"
    findings = check_residual_identity(tex, author_block)
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# vector 2: self-revealing phrasing


def test_self_revealing_flags_first_person_prior_work():
    body = "In our previous work [CITE:x] we showed that the method scales. Unrelated."
    findings = check_self_revealing(body)
    assert _Kind.SELF_REVEALING_PHRASE in _kinds(findings)
    # The CITE marker is stripped from the quoted sentence.
    assert "CITE" not in findings[0].evidence


def test_self_revealing_ignores_third_person():
    assert check_self_revealing("In their previous work they showed something.") == []


# ---------------------------------------------------------------------------
# vector 3: identity URLs


def test_identity_url_flags_github():
    links = [LinkCheck(url="https://github.com/jsmith/repo", source="tex_url")]
    findings = check_identity_urls(links)
    assert _Kind.IDENTITY_URL in _kinds(findings)
    assert "github.com/jsmith" in findings[0].evidence


def test_identity_url_ignores_anonymized_mirror():
    links = [LinkCheck(url="https://anonymous.4open.science/r/abc", source="tex_url")]
    assert check_identity_urls(links) == []


def test_identity_url_flags_personal_homepage():
    links = [LinkCheck(url="https://cs.example.edu/~jsmith/", source="tex_url")]
    assert _Kind.IDENTITY_URL in _kinds(check_identity_urls(links))


# ---------------------------------------------------------------------------
# vector 4: acknowledgments


def test_acknowledgments_present_flags_named_thanks():
    findings = check_acknowledgments("We thank the Acme Lab and NSF grant 12345.")
    assert _Kind.ACKNOWLEDGMENTS_PRESENT in _kinds(findings)


def test_acknowledgments_none_when_absent_or_anonymized():
    assert check_acknowledgments(None) == []
    assert check_acknowledgments("Acknowledgments removed for review.") == []


# ---------------------------------------------------------------------------
# vector 5: name-aware grep + suppression


def test_parse_authors_splits_and_titlecases():
    assert parse_authors("Smith, Jones") == ["Smith", "Jones"]
    assert parse_authors("Alice Smith; bob jones") == ["Smith", "Jones"]
    assert parse_authors(None) == []


def test_author_name_flagged_in_running_prose():
    findings = check_author_names("Smith proposed a faster solver in 2019.", ["Smith"])
    assert _Kind.AUTHOR_NAME_IN_BODY in _kinds(findings)


def test_author_name_suppressed_inside_citation_marker():
    # The surname appears only as a citation key, never in prose.
    assert check_author_names("We build on [CITE:Smith] heavily.", ["Smith"]) == []


def test_author_name_suppressed_in_hyphenated_method():
    assert check_author_names("We run the Smith-Waterman alignment.", ["Smith"]) == []


def test_author_name_common_word_surname_still_flagged_in_prose():
    # "Park" is a common word, but the user opted into --authors, so a prose hit
    # is surfaced (advisory). A citation-key hit would still be suppressed.
    findings = check_author_names("Park introduced the benchmark.", ["Park"])
    assert _Kind.AUTHOR_NAME_IN_BODY in _kinds(findings)
    assert check_author_names("See [CITE:Park] for details.", ["Park"]) == []


# ---------------------------------------------------------------------------
# vector 6: dual-submission tell


def test_dual_submission_tell_flags_under_review_phrasing():
    findings = check_dual_submission("This paper is under review at NeurIPS 2026.")
    assert _Kind.DUAL_SUBMISSION_TELL in _kinds(findings)


def test_dual_submission_quiet_on_normal_prose():
    assert check_dual_submission("We review the related literature in Section 2.") == []


# ---------------------------------------------------------------------------
# entry point + advisory framing


def test_audit_combines_vectors_and_name_aware_opt_in():
    tex = r"\author{Jane Smith}"
    body = "In our previous work we showed gains. Smith proposed the idea."
    links = [LinkCheck(url="https://github.com/jsmith/repo", source="tex_url")]

    # Without --authors: name-aware vector does not run.
    without = audit_anonymization(
        tex_text=tex, body=body, author_block="Jane Smith",
        acknowledgments=None, link_checks=links, authors=None,
    )
    assert _Kind.AUTHOR_NAME_IN_BODY not in _kinds(without)
    assert {_Kind.RESIDUAL_IDENTITY, _Kind.SELF_REVEALING_PHRASE, _Kind.IDENTITY_URL} <= _kinds(without)

    # With --authors: the surname grep activates.
    withnames = audit_anonymization(
        tex_text=tex, body=body, author_block="Jane Smith",
        acknowledgments=None, link_checks=links, authors="Smith",
    )
    assert _Kind.AUTHOR_NAME_IN_BODY in _kinds(withnames)


def test_findings_are_advisory_not_accusatory():
    tex = r"\author{Jane Smith}"
    body = "In our previous work we showed gains."
    findings = audit_anonymization(
        tex_text=tex, body=body, author_block="Jane Smith",
        acknowledgments="We thank the Acme Lab.", link_checks=[], authors=None,
    )
    blob = " ".join(f.detail.lower() for f in findings)
    assert "verify" in blob
    for bad in ("lied", "false", "dishonest", "untruthful"):
        assert bad not in blob


# ---------------------------------------------------------------------------
# integration: ingest_tex attaches findings; --no-anonymize disables


def _write_deanon_paper(tmp_path: Path) -> Path:
    from prereview.tex_ingest import ingest_tex  # noqa: F401 (import-time wiring check)

    (tmp_path / "references.bib").write_text("@article{x, author={A}, title={t}, year={2023}}\n")
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{article}
\title{Toy}
\author{Jane Smith \thanks{State University}}
\begin{document}
\section{Intro}
In our previous work we showed gains \citep{x}. See \url{https://github.com/jsmith/repo}.
\section*{Acknowledgements}
We thank the Acme Lab.
\end{document}
"""
    )
    return tex


def test_ingest_tex_attaches_anonymization_findings(tmp_path: Path):
    from prereview.tex_ingest import ingest_tex

    tex = _write_deanon_paper(tmp_path)
    paper = asyncio.run(ingest_tex(tex, model="ignored", verbose=False))
    assert paper.anonymization_checked is True
    kinds = {f.kind for f in paper.anonymization_findings}
    assert _Kind.RESIDUAL_IDENTITY in kinds
    assert _Kind.SELF_REVEALING_PHRASE in kinds
    assert _Kind.IDENTITY_URL in kinds
    assert _Kind.ACKNOWLEDGMENTS_PRESENT in kinds


def test_ingest_tex_no_anonymize_flag_skips_audit(tmp_path: Path):
    from prereview.tex_ingest import ingest_tex

    tex = _write_deanon_paper(tmp_path)
    paper = asyncio.run(ingest_tex(tex, model="ignored", verbose=False, run_anonymize=False))
    assert paper.anonymization_checked is False
    assert paper.anonymization_findings == []
