"""Tests for prereview.verify.

LLM and PDF fetching are mocked. We verify:
- target_unavailable when canonical is None
- abstract_too_thin when no abstract or full text
- abstract-only verdict path passes through verbatim
- full-text fetch path is tried only when fetch_cited=True
- malformed verdict from LLM is treated as abstract_too_thin
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prereview import verify as verify_mod
from prereview.cache import Cache
from prereview.models import (
    CanonicalRecord,
    Citation,
    CitationRole,
    Reference,
    Verdict,
)
from prereview.verify import Verifier


def _ref(ref_id: str) -> Reference:
    return Reference(
        ref_id=ref_id,
        raw_text=f"[{ref_id}] Smith, A. A toy paper. Toy J. 2023.",
        authors=["Alice Smith"],
        title="A Toy Paper",
        year=2023,
    )


def _canonical(*, abstract=None, pdf_url=None) -> CanonicalRecord:
    return CanonicalRecord(
        source="crossref",
        title="A Toy Paper",
        authors=["Alice Smith"],
        year=2023,
        venue="Toy J.",
        doi="10.1234/toy",
        url="https://doi.org/10.1234/toy",
        abstract=abstract,
        open_access_pdf_url=pdf_url,
    )


def _cite(ref_id: str = "1", sentence: str = "Toys are nice [1].") -> Citation:
    return Citation(ref_id=ref_id, sentence=sentence)


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache")


@pytest.mark.asyncio
async def test_target_unavailable_when_canonical_none(cache):
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), None, model="x")
    assert r.verdict == Verdict.TARGET_UNAVAILABLE
    assert r.canonical is None


@pytest.mark.asyncio
async def test_no_evidence_method_attribution_supports_from_metadata(cache, monkeypatch):
    """When no abstract or PDF is available but the cite is a method
    attribution (named tool, canonical record's title/authors match), the
    role-aware prompt can return `supports` from title+authors alone."""
    can = _canonical(abstract=None, pdf_url=None)

    captured = {}

    async def fake_json(*, user, **_):
        captured["user"] = user
        return {
            "role": "method_attribution",
            "verdict": "supports",
            "rationale": "Title and authors identify the canonical paper for the named tool.",
        }

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.SUPPORTS
    assert r.role == CitationRole.METHOD_ATTRIBUTION
    assert r.abstract_only is True
    # The placeholder makes the no-evidence state explicit to the model.
    assert "no abstract or full text was retrievable" in captured["user"].lower()


@pytest.mark.asyncio
async def test_no_evidence_claim_support_returns_abstract_too_thin(cache, monkeypatch):
    """When no evidence is available and the cite supports a specific factual
    claim, the model should still return abstract_too_thin — the body would
    be needed and we have nothing."""
    can = _canonical(abstract=None, pdf_url=None)

    async def fake_json(**_):
        return {
            "role": "claim_support",
            "verdict": "abstract_too_thin",
            "rationale": "Specific quantitative claim cannot be verified without evidence.",
        }

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.ABSTRACT_TOO_THIN
    assert r.role == CitationRole.CLAIM_SUPPORT
    assert r.abstract_only is True


@pytest.mark.asyncio
async def test_abstract_only_verdict_pass_through(cache, monkeypatch):
    can = _canonical(abstract="Toys are pleasant in mild weather.")
    captured = {}

    async def fake_json(*, model, user, system=None, **kw):
        captured["user"] = user
        return {"verdict": "supports", "rationale": "The abstract states toys are pleasant."}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="anthropic/claude-sonnet-4-6", fetch_cited=False)
    assert r.verdict == Verdict.SUPPORTS
    assert r.abstract_only is True
    assert "Toys are pleasant" in captured["user"]


@pytest.mark.asyncio
async def test_full_text_path_marks_not_abstract_only(cache, monkeypatch):
    can = _canonical(
        abstract="short abstract",
        pdf_url="https://example.org/paper.pdf",
    )

    async def fake_fetch(self, canonical, reference=None, *, verbose):
        return "FULL BODY TEXT WITH DETAILED EVIDENCE."

    async def fake_json(*, model, user, system=None, **kw):
        # We should be passed the full text, not just the abstract.
        assert "FULL BODY TEXT" in user
        return {"verdict": "partially_supports", "rationale": "Body text supports a narrower claim."}

    monkeypatch.setattr(Verifier, "_fetch_full_text", fake_fetch, raising=True)
    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)

    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=True)
    assert r.verdict == Verdict.PARTIALLY_SUPPORTS
    assert r.abstract_only is False


@pytest.mark.asyncio
async def test_no_fetch_cited_skips_pdf_path(cache, monkeypatch):
    can = _canonical(abstract="abstract here", pdf_url="https://example.org/paper.pdf")

    fetched = False

    async def fake_fetch(self, canonical, reference=None, *, verbose):
        nonlocal fetched
        fetched = True
        return "FULL TEXT"

    async def fake_json(**_):
        return {"verdict": "supports", "rationale": "Abstract suffices."}

    monkeypatch.setattr(Verifier, "_fetch_full_text", fake_fetch, raising=True)
    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)

    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert fetched is False
    assert r.abstract_only is True


@pytest.mark.asyncio
async def test_metadata_mismatch_real_world_guo2023pypi_case(cache, monkeypatch):
    """Real-world case from the Gitcha test: a .bib DOI that returned a
    completely different paper. Both title and authors disagree — flag it."""
    ref = Reference(
        ref_id="bad",
        raw_text="[bad] An Empirical Study of Malicious Code In PyPI Ecosystem",
        authors=["Wenbo Guo", "Zhengzi Xu"],
        title="An Empirical Study of Malicious Code In PyPI Ecosystem",
        year=2023,
    )
    can = CanonicalRecord(
        source="crossref",
        title="PreciseBugCollector: Extensible, Executable and Precise Bug-Fix Collection",
        authors=["Ye He", "Zimin Chen"],
        year=2023,
        doi="10.1109/ase56229.2023.00163",
        abstract="abstract about bug collection",
    )
    called = False

    async def fake_json(**_):
        nonlocal called
        called = True
        return {"verdict": "supports", "rationale": "should not be reached"}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), ref, can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.METADATA_MISMATCH
    assert "author-surname overlap" in r.rationale
    assert called is False


@pytest.mark.asyncio
async def test_metadata_mismatch_does_not_trigger_on_truncated_title(cache, monkeypatch):
    """Real-world case from Gitcha (akiba2019optuna): Crossref returned a
    truncated title "Optuna" but the authors match exactly. Should NOT flag —
    that's the same paper, just with a noisy Crossref record."""
    ref = Reference(
        ref_id="ok",
        raw_text="[ok] Optuna: A Next-generation Hyperparameter Optimization Framework",
        authors=["Takuya Akiba", "Shotaro Sano", "Toshihiko Yanase"],
        title="Optuna: A Next-generation Hyperparameter Optimization Framework",
        year=2019,
    )
    can = CanonicalRecord(
        source="crossref",
        title="Optuna",  # Crossref truncates
        authors=["Takuya Akiba", "Shotaro Sano", "Toshihiko Yanase"],
        year=2019,
        doi="10.1145/3292500.3330701",
        abstract="A hyperparameter framework.",
    )

    async def fake_json(**_):
        return {"verdict": "supports", "rationale": "framework abstract supports use."}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), ref, can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.SUPPORTS  # not METADATA_MISMATCH


@pytest.mark.asyncio
async def test_metadata_mismatch_when_authors_disagree(cache, monkeypatch):
    ref = Reference(
        ref_id="bad",
        raw_text="[bad] Same title here, different authors",
        authors=["Alice Foo", "Bob Bar"],
        title="A study of widgets",
        year=2023,
    )
    can = CanonicalRecord(
        source="crossref",
        title="A study of widgets",
        authors=["Carol Baz", "Dave Qux"],
        year=2023,
        doi="10.1/x",
        abstract="something",
    )

    async def fake_json(**_):
        return {"verdict": "supports", "rationale": "x"}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), ref, can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.METADATA_MISMATCH
    assert "author-surname overlap" in r.rationale or "No author" in r.rationale


@pytest.mark.asyncio
async def test_verify_cache_hit_skips_llm(cache, monkeypatch):
    can = _canonical(abstract="some abstract")
    calls = 0

    async def fake_json(**_):
        nonlocal calls
        calls += 1
        return {"verdict": "supports", "rationale": "abstract states it"}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r1 = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
        r2 = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert calls == 1  # second call hit the cache
    assert r1.verdict == r2.verdict == Verdict.SUPPORTS


@pytest.mark.asyncio
async def test_unknown_verdict_falls_back_to_abstract_too_thin(cache, monkeypatch):
    can = _canonical(abstract="abstract")

    async def fake_json(**_):
        return {"verdict": "yes_obviously", "rationale": "we believe so"}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert r.verdict == Verdict.ABSTRACT_TOO_THIN


@pytest.mark.asyncio
async def test_role_flows_through_from_llm_response(cache, monkeypatch):
    can = _canonical(abstract="Introduces the HST anomaly detector for streaming data.")

    async def fake_json(**_):
        return {
            "role": "method_attribution",
            "verdict": "supports",
            "rationale": "Title and abstract identify this as the canonical HST paper.",
        }

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(
            _cite(sentence="We use HST [tan2011hst]."),
            _ref("tan2011hst"),
            can,
            model="x",
            fetch_cited=False,
        )
    assert r.role == CitationRole.METHOD_ATTRIBUTION
    assert r.verdict == Verdict.SUPPORTS


@pytest.mark.asyncio
async def test_unknown_role_parses_as_none(cache, monkeypatch):
    """If the LLM omits or mangles the role field, role should be None,
    not crash. The verdict still flows through."""
    can = _canonical(abstract="Some content.")

    async def fake_json(**_):
        return {
            "role": "definitely_not_a_role",
            "verdict": "supports",
            "rationale": "ok",
        }

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        r = await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)
    assert r.role is None
    assert r.verdict == Verdict.SUPPORTS


@pytest.mark.asyncio
async def test_prompt_includes_role_classification_step(cache, monkeypatch):
    """The new prompt should be teaching the model to classify role first.
    This test guards against accidentally reverting to the v1 prompt."""
    can = _canonical(abstract="abstract")
    captured = {}

    async def fake_json(*, user, **_):
        captured["user"] = user
        return {"role": "claim_support", "verdict": "supports", "rationale": "ok"}

    monkeypatch.setattr(verify_mod, "acompletion_json", fake_json)
    async with Verifier(cache=cache) as v:
        await v.verify(_cite(), _ref("1"), can, model="x", fetch_cited=False)

    prompt = captured["user"]
    assert "method_attribution" in prompt
    assert "claim_support" in prompt
    assert "background" in prompt
    # The key methodological rule that we want preserved.
    assert "method paper" in prompt.lower() or "method attribution" in prompt.lower()
