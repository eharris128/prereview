"""Tests for prereview.artifacts — HF / GitHub existence checks.

HTTP is respx-mocked. The key precision points: HF's unauthenticated 401 (not
404) means a miss, a 200-with-unparseable-body is a transient (not a miss), and an
unreachable host degrades cleanly rather than crashing the pipeline.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from prereview.artifacts import _parse_artifact, check_artifacts, extract_artifacts
from prereview.models import ArtifactStatus, LinkCheck

_S = ArtifactStatus


def _lc(url: str) -> LinkCheck:
    return LinkCheck(url=url, source="tex_url")


# ---------------------------------------------------------------------------
# extraction (pure)


def test_parse_hf_model():
    assert _parse_artifact("https://huggingface.co/google/bert-base-uncased") == (
        "hf_model", "google/bert-base-uncased", "https://huggingface.co/google/bert-base-uncased",
    )


def test_parse_hf_dataset():
    kind, ident, _ = _parse_artifact("https://huggingface.co/datasets/rajpurkar/squad")
    assert kind == "hf_dataset" and ident == "rajpurkar/squad"


def test_parse_github_strips_subpath_and_git():
    kind, ident, _ = _parse_artifact("https://github.com/huggingface/transformers/tree/main")
    assert kind == "github_repo" and ident == "huggingface/transformers"
    _, ident2, _ = _parse_artifact("https://github.com/owner/repo.git")
    assert ident2 == "owner/repo"


def test_parse_ignores_non_artifacts_and_reserved_paths():
    assert _parse_artifact("https://example.com/foo/bar") is None
    assert _parse_artifact("https://github.com/about") is None  # reserved, single seg
    assert _parse_artifact("https://huggingface.co/docs/hub") is None  # reserved


def test_extract_dedups_by_kind_and_id():
    arts = extract_artifacts([_lc("https://github.com/a/b"), _lc("https://github.com/a/b")], "")
    assert len(arts) == 1


def test_extract_pulls_from_body_text():
    arts = extract_artifacts([], "Our code is at https://github.com/a/b and weights on HF.")
    assert ("github_repo", "a/b", "https://github.com/a/b") in arts


# ---------------------------------------------------------------------------
# probing (respx)


@pytest.mark.asyncio
@respx.mock
async def test_hf_model_exists_is_hit():
    respx.get("https://huggingface.co/api/models/google/bert").mock(
        return_value=httpx.Response(200, json={"id": "google/bert"})
    )
    checks = await check_artifacts([_lc("https://huggingface.co/google/bert")], "")
    assert checks[0].status == _S.HIT
    assert checks[0].detail == ""  # hits are silent


@pytest.mark.asyncio
@respx.mock
async def test_hf_401_is_terminal_miss_not_transient():
    # The crux: HF returns 401 for a missing/private model, NOT 404.
    respx.get("https://huggingface.co/api/models/nope/model").mock(
        return_value=httpx.Response(401, json={"error": "Invalid username or password."})
    )
    checks = await check_artifacts([_lc("https://huggingface.co/nope/model")], "")
    assert checks[0].status == _S.TERMINAL_MISS
    assert "verify" in checks[0].detail.lower()


@pytest.mark.asyncio
@respx.mock
async def test_github_404_is_terminal_miss():
    respx.get("https://api.github.com/repos/nope/repo").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    checks = await check_artifacts([_lc("https://github.com/nope/repo")], "")
    assert checks[0].status == _S.TERMINAL_MISS


@pytest.mark.asyncio
@respx.mock
async def test_200_unparseable_body_is_transient_not_miss():
    respx.get("https://huggingface.co/api/models/x/y").mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    checks = await check_artifacts([_lc("https://huggingface.co/x/y")], "")
    assert checks[0].status == _S.TRANSIENT_FAIL


@pytest.mark.asyncio
@respx.mock
async def test_github_rate_limit_403_is_transient_not_miss():
    respx.get("https://api.github.com/repos/a/b").mock(
        return_value=httpx.Response(403, json={"message": "API rate limit exceeded"})
    )
    checks = await check_artifacts([_lc("https://github.com/a/b")], "")
    assert checks[0].status == _S.TRANSIENT_FAIL


@pytest.mark.asyncio
@respx.mock
async def test_unreachable_host_degrades_cleanly():
    respx.get("https://huggingface.co/api/models/x/y").mock(side_effect=httpx.ConnectError("down"))
    checks = await check_artifacts([_lc("https://huggingface.co/x/y")], "")
    assert checks[0].status == _S.TRANSIENT_FAIL  # not a crash, not a miss


@pytest.mark.asyncio
async def test_no_artifacts_returns_empty_without_network():
    # First-class flag-off / nothing-to-check path: no network, empty result.
    assert await check_artifacts([], "this paper claims no models or repos") == []
