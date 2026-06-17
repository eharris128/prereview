"""Tests for prereview.checklist — deterministic checklist linting (no LLM).

The parser is tested against the genuine AAAI-27 kit file (golden fixture) so
regex drift is caught against the real artifact, plus small synthetic checklists
for the self-consistency and claim-vs-paper logic.
"""

from __future__ import annotations

from pathlib import Path

from prereview.checklist import (
    CROSS_CHECKS,
    check_claims_vs_paper,
    check_self_consistency,
    find_checklist_file,
    parse_checklist,
)
from prereview.models import ChecklistFindingKind, ChecklistItem, LinkCheck

FIXTURE = Path(__file__).parent / "fixtures" / "ReproducibilityChecklist.tex"


def _item(question, options, response="", *, section="Sec", is_gate=False, gate_question=None):
    return ChecklistItem(
        section=section,
        question=question,
        options=options,
        response=response,
        is_gate=is_gate,
        gate_question=gate_question,
    )


# ---------------------------------------------------------------------------
# U2: parsing the real fixture


def test_parses_real_aaai_fixture_structure():
    items = parse_checklist(FIXTURE.read_text())
    # 31 \question macro calls in the kit (the \string\question examples in the
    # instructions are not real calls and must not be counted).
    assert len(items) == 31
    assert {it.section for it in items} == {
        "General Paper Structure",
        "Theoretical Contributions",
        "Dataset Usage",
        "Computational Experiments",
    }
    # Three gate questions, one per conditional section.
    gates = [it for it in items if it.is_gate]
    assert len(gates) == 3
    assert any(g.question.startswith("Does this paper include computational experiments") for g in gates)


def test_pristine_fixture_is_all_unanswered():
    items = parse_checklist(FIXTURE.read_text())
    assert all("type your response here" in it.response.lower() for it in items)


def test_fixture_subitems_carry_gate_question():
    items = parse_checklist(FIXTURE.read_text())
    comp = [it for it in items if it.section == "Computational Experiments"]
    gate = next(it for it in comp if it.is_gate)
    subs = [it for it in comp if not it.is_gate]
    assert len(subs) == 12
    assert gate.gate_question is None
    assert all(s.gate_question == gate.question for s in subs)


def test_fixture_options_parsed_lowercased():
    items = parse_checklist(FIXTURE.read_text())
    # First question is "(yes/partial/no/NA)".
    assert items[0].options == ["yes", "partial", "no", "na"]


# ---------------------------------------------------------------------------
# U2: parsing synthetic inputs


def test_answered_and_untouched_responses():
    src = r"""
\checksubsection{Sec A}
\begin{itemize}
\question{Q one}{(yes/no)}
yes
\question{Q two}{(yes/no)}
Type your response here
\end{itemize}
"""
    items = parse_checklist(src)
    assert len(items) == 2
    assert items[0].response == "yes"
    assert items[0].options == ["yes", "no"]
    assert "type your response here" in items[1].response.lower()


def test_gate_and_subitems_associated():
    src = r"""
\checksubsection{Theory}
\begin{itemize}
\question{Does this paper make theoretical contributions?}{(yes/no)}
yes
\ifyespoints{If yes, address the following:}
\begin{itemize}
\question{All assumptions stated}{(yes/partial/no)}
partial
\question{Proofs included}{(yes/partial/no)}
no
\end{itemize}
\end{itemize}
"""
    items = parse_checklist(src)
    gate = [it for it in items if it.is_gate]
    assert len(gate) == 1
    assert gate[0].question.startswith("Does this paper make theoretical contributions")
    subs = [it for it in items if not it.is_gate]
    assert len(subs) == 2
    assert all(s.gate_question == gate[0].question for s in subs)


def test_plain_yes_no_question_is_not_a_gate():
    src = r"""
\checksubsection{Sec}
\begin{itemize}
\question{Clearly delineates opinions from objective facts}{(yes/no)}
yes
\end{itemize}
"""
    items = parse_checklist(src)
    assert items[0].is_gate is False


def test_nested_brace_in_question_text_not_truncated():
    src = r"""
\checksubsection{Sec}
\begin{itemize}
\question{Uses the \texttt{foo{bar}} tool correctly}{(yes/no)}
yes
\end{itemize}
"""
    items = parse_checklist(src)
    assert len(items) == 1
    assert "tool correctly" in items[0].question  # text after the nested brace survived
    assert items[0].options == ["yes", "no"]
    assert items[0].response == "yes"


def test_no_gate_subblock_does_not_latch_onto_sibling_itemize():
    """Regression: a gate whose sub-items were deleted (answered 'no') must not
    bind to a later unrelated itemize in the same section — that produced a
    spurious gate_inconsistency and could swallow a real unanswered finding."""
    src = r"""
\checksubsection{Computational Experiments}
\begin{itemize}
\question{Does this paper include computational experiments?}{(yes/no)}
no
\end{itemize}

\begin{itemize}
\question{This paper lists all final hyper-parameters}{(yes/partial/no/NA)}
yes
\end{itemize}
"""
    items = parse_checklist(src)
    # The second question is a sibling, not a sub-item of the gate.
    sibling = next(it for it in items if "hyper-parameters" in it.question)
    assert sibling.gate_question is None
    # No spurious gate inconsistency, and the answered sibling is not flagged.
    assert check_self_consistency(items) == []


def test_deleted_subblock_does_not_hide_unanswered_sibling():
    """The flip side: a blank sibling after a 'no' gate must still be flagged
    unanswered (it is not a cascaded sub-item)."""
    src = r"""
\checksubsection{Computational Experiments}
\begin{itemize}
\question{Does this paper include computational experiments?}{(yes/no)}
no
\end{itemize}

\begin{itemize}
\question{This paper lists all final hyper-parameters}{(yes/partial/no/NA)}
Type your response here
\end{itemize}
"""
    items = parse_checklist(src)
    findings = check_self_consistency(items)
    unanswered = [f for f in findings if f.kind == ChecklistFindingKind.UNANSWERED]
    assert len(unanswered) == 1
    assert "hyper-parameters" in unanswered[0].question


def test_empty_and_garbage_return_empty_list():
    assert parse_checklist("") == []
    assert parse_checklist("random text with no checklist macros at all") == []
    assert parse_checklist(r"\section{Intro} text \cite{x}") == []


# ---------------------------------------------------------------------------
# U2: discovery


def test_find_checklist_explicit_wins(tmp_path: Path):
    explicit = tmp_path / "my_checklist.tex"
    explicit.write_text("x")
    sibling = tmp_path / "ReproducibilityChecklist.tex"
    sibling.write_text("y")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\input{ReproducibilityChecklist}")
    assert find_checklist_file(tex, tex.read_text(), explicit=explicit) == explicit


def test_find_checklist_via_input_resolves_relative(tmp_path: Path):
    sub = tmp_path / "sections"
    sub.mkdir()
    chk = sub / "repro_checklist.tex"
    chk.write_text("x")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\input{sections/repro_checklist}")
    assert find_checklist_file(tex, tex.read_text()) == chk


def test_find_checklist_via_input_dotted_basename(tmp_path: Path):
    """A dotted basename must get .tex appended, not its 'extension' replaced."""
    chk = tmp_path / "repro.checklist.tex"
    chk.write_text("x")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\input{repro.checklist}")
    assert find_checklist_file(tex, tex.read_text()) == chk


def test_find_checklist_sibling(tmp_path: Path):
    chk = tmp_path / "ReproducibilityChecklist.tex"
    chk.write_text("x")
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\documentclass{article}")
    assert find_checklist_file(tex, tex.read_text()) == chk


def test_find_checklist_none_when_nothing_matches(tmp_path: Path):
    tex = tmp_path / "paper.tex"
    tex.write_text(r"\documentclass{article}\begin{document}\end{document}")
    assert find_checklist_file(tex, tex.read_text()) is None


# ---------------------------------------------------------------------------
# U3: tier-1 self-consistency


def test_all_blank_fixture_flags_every_question_unanswered():
    items = parse_checklist(FIXTURE.read_text())
    findings = check_self_consistency(items)
    unanswered = [f for f in findings if f.kind == ChecklistFindingKind.UNANSWERED]
    assert len(unanswered) == len(items) == 31
    # Nothing else fires on a blank template.
    assert all(f.kind == ChecklistFindingKind.UNANSWERED for f in findings)


def test_response_case_and_punctuation_tolerant():
    items = [_item("Q", ["yes", "no"], "Yes.")]
    assert check_self_consistency(items) == []


def test_invalid_response_quotes_allowed_options():
    items = [_item("Q", ["yes", "no"], "maybe")]
    findings = check_self_consistency(items)
    assert len(findings) == 1
    assert findings[0].kind == ChecklistFindingKind.INVALID_RESPONSE
    assert "yes" in findings[0].detail and "no" in findings[0].detail


def test_na_response_accepted():
    items = [_item("Q", ["yes", "no", "na"], "N/A")]
    assert check_self_consistency(items) == []


def test_multi_token_short_response_uses_leading_option():
    """'no x' must read as the leading option 'no' (valid), not 'nox' (invalid)."""
    items = [_item("Q", ["yes", "no"], "no x")]
    assert check_self_consistency(items) == []


def test_gate_no_with_answered_subitem_flags_inconsistency():
    gate = _item("Does this paper make theoretical contributions?", ["yes", "no"], "no", is_gate=True)
    sub = _item("Proofs included", ["yes", "partial", "no"], "yes", gate_question=gate.question)
    findings = check_self_consistency([gate, sub])
    gate_findings = [f for f in findings if f.kind == ChecklistFindingKind.GATE_INCONSISTENCY]
    assert len(gate_findings) == 1
    assert "no" in gate_findings[0].response


def test_gate_no_cascades_blank_subitems_are_not_flagged():
    gate = _item("Does this paper rely on one or more datasets?", ["yes", "no"], "no", is_gate=True)
    sub = _item("Datasets are public", ["yes", "no"], "Type your response here", gate_question=gate.question)
    findings = check_self_consistency([gate, sub])
    # The blank sub-item is expected (gate says no) — no unanswered, no inconsistency.
    assert findings == []


def test_gate_yes_with_all_subitems_blank_flags_once_plus_unanswered_subs():
    gate = _item("Does this paper include computational experiments?", ["yes", "no"], "yes", is_gate=True)
    sub1 = _item("Specifies infrastructure", ["yes", "no"], "Type your response here", gate_question=gate.question)
    sub2 = _item("States number of runs", ["yes", "no"], "", gate_question=gate.question)
    findings = check_self_consistency([gate, sub1, sub2])
    gate_findings = [f for f in findings if f.kind == ChecklistFindingKind.GATE_INCONSISTENCY]
    # Exactly one gate inconsistency — not one per blank sub-item.
    assert len(gate_findings) == 1
    # The blank sub-items still surface as unanswered (different kind).
    unanswered = [f for f in findings if f.kind == ChecklistFindingKind.UNANSWERED]
    assert len(unanswered) == 2


def test_fully_correct_checklist_has_no_findings():
    src = r"""
\checksubsection{Sec}
\begin{itemize}
\question{Includes a conceptual outline}{(yes/partial/no/NA)}
yes
\question{Does this paper make theoretical contributions?}{(yes/no)}
no
\ifyespoints{If yes:}
\begin{itemize}
\question{All assumptions stated}{(yes/partial/no)}
Type your response here
\end{itemize}
\end{itemize}
"""
    items = parse_checklist(src)
    assert check_self_consistency(items) == []


# ---------------------------------------------------------------------------
# U4: tier-2 claim-vs-paper cross-checks

_CODE_Q = "All source code required for conducting and analyzing the experiments will be made publicly available upon publication"
_INFRA_Q = "This paper specifies the computing infrastructure used for running experiments, including GPU/CPU models"
_RUNS_Q = "This paper states the number of algorithm runs used to compute each reported result"
_DATA_Q = "All datasets drawn from the existing literature are publicly available"


def test_code_availability_satisfied_by_repo_link():
    items = [_item(_CODE_Q, ["yes", "partial", "no"], "yes")]
    links = [LinkCheck(url="https://github.com/foo/bar", source="tex_url")]
    assert check_claims_vs_paper(items, "We describe our method.", links) == []


def test_code_availability_flagged_without_any_repo_url():
    items = [_item(_CODE_Q, ["yes", "partial", "no"], "yes")]
    findings = check_claims_vs_paper(items, "We describe our method.", [])
    assert len(findings) == 1
    assert findings[0].kind == ChecklistFindingKind.CLAIM_UNSUPPORTED
    assert "repository" in findings[0].detail


def test_code_availability_keys_on_url_not_the_word_code():
    """Precision guard: 'release code at <github url>' satisfies via the URL.
    A body that says 'code' with no repo URL anywhere is still flagged."""
    items = [_item(_CODE_Q, ["yes", "partial", "no"], "yes")]
    # URL in the body text (not as a LinkCheck) still counts.
    assert check_claims_vs_paper(items, "We will release code at https://github.com/x/y.", []) == []
    # The bare word "code" with no URL does not.
    assert len(check_claims_vs_paper(items, "Our code is great.", [])) == 1


def test_computing_infrastructure_positive_and_negative():
    items = [_item(_INFRA_Q, ["yes", "partial", "no"], "yes")]
    assert check_claims_vs_paper(items, "We trained on 4x A100 GPUs.", []) == []
    assert len(check_claims_vs_paper(items, "We ran several experiments.", [])) == 1


def test_number_of_runs_positive_and_negative():
    items = [_item(_RUNS_Q, ["yes", "no"], "yes")]
    assert check_claims_vs_paper(items, "Results are averaged over 5 runs.", []) == []
    assert len(check_claims_vs_paper(items, "We report final accuracy.", [])) == 1


def test_dataset_availability_positive_and_negative():
    items = [_item(_DATA_Q, ["yes", "partial", "no", "na"], "yes")]
    assert check_claims_vs_paper(items, "We evaluate on ImageNet and CIFAR-10.", []) == []
    assert len(check_claims_vs_paper(items, "We evaluate on a private corpus.", [])) == 1


def test_no_and_blank_answers_are_never_cross_checked():
    items = [
        _item(_CODE_Q, ["yes", "partial", "no"], "no"),
        _item(_INFRA_Q, ["yes", "partial", "no"], "Type your response here"),
        _item(_RUNS_Q, ["yes", "no"], ""),
    ]
    assert check_claims_vs_paper(items, "Nothing relevant here.", []) == []


def test_partial_answer_triggers_cross_check():
    items = [_item(_CODE_Q, ["yes", "partial", "no"], "partial")]
    assert len(check_claims_vs_paper(items, "No repo anywhere.", [])) == 1


def test_question_with_no_mapping_is_ignored():
    items = [_item("Provides pedagogical references for less-familiar readers", ["yes", "no"], "yes")]
    assert check_claims_vs_paper(items, "No evidence of anything.", []) == []


def test_non_public_dataset_question_is_not_cross_checked():
    """Precision guard: the AAAI item about datasets that are *not* publicly
    available makes no availability claim — answering 'yes' (we described them)
    must not be flagged for a missing dataset URL."""
    q = (
        "All datasets that are not publicly available are described in detail, "
        "with explanation why publicly available alternatives are not scientifically satisficing"
    )
    items = [_item(q, ["yes", "partial", "no", "na"], "yes")]
    assert check_claims_vs_paper(items, "We use a private corpus.", []) == []


def test_cross_checks_table_is_small_and_high_precision():
    # Guard against the table quietly growing into low-precision territory.
    assert len(CROSS_CHECKS) == 4
