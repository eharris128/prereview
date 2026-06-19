"""Tests for prereview.numeric_sanity — the ML numerical-sanity pack.

Two contracts: planted errors must fire at 100%, and a corpus of >=15-20 realistic
benign numeric sentences must produce ZERO findings (with an adversarial-benign
subset aimed at the touchy detectors #3 mean±std and #4 hyperparameter).
"""

from __future__ import annotations

from prereview.models import NumericFindingKind
from prereview.numeric_sanity import (
    DETECTORS,
    audit_numeric,
    check_bounded_metrics,
    check_hyperparameter_drift,
    check_mean_std,
    check_split_arithmetic,
    check_table_abstract_delta,
)

_K = NumericFindingKind


def _kinds(findings):
    return {f.kind for f in findings}


# ---------------------------------------------------------------------------
# planted errors — each detector fires


def test_bounded_metric_over_100_percent():
    findings = check_bounded_metrics("We report an accuracy of 102.3% on the held-out test set.")
    assert _K.BOUNDED_METRIC in _kinds(findings)


def test_bounded_fraction_metric_over_one():
    findings = check_bounded_metrics("The model achieves an F1 score of 1.07 on this benchmark.")
    assert _K.BOUNDED_METRIC in _kinds(findings)


def test_split_arithmetic_mismatch():
    findings = check_split_arithmetic("We use train 8k / val 1k / test 2k of 10k examples in total.")
    assert _K.SPLIT_MISMATCH in _kinds(findings)


def test_mean_std_exceeds_ceiling():
    findings = check_mean_std("The model reaches an accuracy of 99.5 ± 1.2 averaged over five seeds.")
    assert _K.MEAN_STD_RANGE in _kinds(findings)


def test_hyperparameter_prose_table_drift():
    body = "We train with a learning rate of 0.1 for all experiments."
    tex = r"\begin{tabular}{ll} learning rate & 0.001 \\ batch size & 32 \\ \end{tabular}"
    findings = check_hyperparameter_drift(body, tex)
    assert _K.HYPERPARAM_DRIFT in _kinds(findings)


def test_abstract_table_delta_mismatch():
    abstract = "Our method improves accuracy by 5.2 points over the strongest baseline."
    tex = r"\begin{tabular}{ll} ours & +2.1 \\ baseline & --- \\ \end{tabular}"
    findings = check_table_abstract_delta(abstract, tex)
    assert _K.ABSTRACT_TABLE_DELTA in _kinds(findings)


def test_audit_numeric_combines_all():
    body = "We set the learning rate to 0.1. We use train 8k / val 1k / test 2k of 10k examples."
    tex = r"\begin{tabular}{ll} learning rate & 0.001 \\ \end{tabular}"
    findings = audit_numeric(body, body + tex, abstract=None)
    assert {_K.SPLIT_MISMATCH, _K.HYPERPARAM_DRIFT} <= _kinds(findings)


# ---------------------------------------------------------------------------
# the precision contract: a benign corpus must produce ZERO findings


_BENIGN_SENTENCES = [
    # in-bounds metrics
    "The model attains an accuracy of 94.7% on the test set.",
    "We report an F1 of 0.87 and a precision of 0.91.",
    "Our approach reaches 88.2% top-1 accuracy on ImageNet.",
    "The AUC is 0.96, comparable to prior work.",
    "Recall improves from 0.71 to 0.79 across configurations.",
    # adversarial-benign for #1 / #3: percentages and ± as tolerance, not std
    "About 95% of users preferred the new interface.",
    "We sample within ± 0.5 of the target temperature.",
    "The annotation budget covers 30% of the corpus.",
    "Accuracy is 92.0 ± 2.0 over three runs.",  # 94 < 100, in bounds
    "An F1 of 0.95 ± 0.03 was observed.",  # 0.98 < 1.0, in bounds
    # adversarial-benign for #4: scientific notation equals decimal
    "We use a learning rate of 1e-3 throughout training.",
    "Each model trains for 100 epochs with a batch size of 64.",
    # ranges and non-metric numbers
    "We sweep the learning rate over the range 0.1 to 0.9.",
    "The dataset spans 1.2M sentences across 14 languages.",
    "Training took 1.5 hours on a single A100 GPU.",
    "The recall computation took 1.5 seconds per query.",  # 'recall' near 1.5 but not a metric value
    "We average results over 5 independent runs.",
    "The temperature was set to 0.7 for decoding.",
    "We report a speedup of 1.8x over the baseline.",
    "Coverage reached 99.5% of the held-out queries.",  # 99.5 < 100
]


def test_benign_corpus_is_clean():
    """No detector may fire on any benign sentence — the precision guarantee."""
    offenders = []
    for sent in _BENIGN_SENTENCES:
        findings = audit_numeric(sent, sent, abstract=None)
        if findings:
            offenders.append((sent, [f.kind.value for f in findings]))
    assert not offenders, f"false positives on benign prose: {offenders}"


def test_benign_corpus_has_enough_coverage():
    # The plan calls for >=15-20 realistic benign sentences.
    assert len(_BENIGN_SENTENCES) >= 15


def test_sci_notation_hyperparameter_not_flagged():
    # 1e-3 and 0.001 are the same magnitude → no drift.
    body = "We use a learning rate of 1e-3 in all runs."
    tex = r"\begin{tabular}{ll} learning rate & 0.001 \\ \end{tabular}"
    assert check_hyperparameter_drift(body, tex) == []


def test_mean_std_tolerance_not_flagged_without_metric_context():
    # A bare ± with no metric context is a tolerance, not a metric measurement.
    assert check_mean_std("The setpoint is maintained at 3.2 ± 0.5 throughout.") == []


# ---------------------------------------------------------------------------
# precision guard: the detector registry stays small


def test_detector_registry_is_small_and_stable():
    assert len(DETECTORS) == 5
