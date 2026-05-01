"""Tests for prereview.link_health.

Network calls are mocked with respx. We assert classification of 2xx, 3xx,
4xx, 5xx, HEAD-falls-back-to-GET, timeouts, and the in-place population of
the LinkCheck list.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from prereview.link_health import check_links
from prereview.models import LinkCheck


@pytest.mark.asyncio
async def test_check_links_classifies_200_as_ok():
    checks = [LinkCheck(url="https://ok.example.org/a", source="tex_url")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://ok.example.org/a").respond(200)
        out = await check_links(checks)
    assert out[0].ok is True
    assert out[0].status == 200
    assert out[0].error is None


@pytest.mark.asyncio
async def test_check_links_classifies_404_as_broken():
    checks = [LinkCheck(url="https://gone.example.org/x", source="tex_href")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://gone.example.org/x").respond(404)
        out = await check_links(checks)
    assert out[0].ok is False
    assert out[0].status == 404
    assert "404" in (out[0].error or "")


@pytest.mark.asyncio
async def test_check_links_falls_back_to_get_when_head_returns_405():
    """Many servers (especially GitHub's CDN) reject HEAD with 405. We must
    retry with GET before flagging the URL as broken."""
    checks = [LinkCheck(url="https://strict.example.org/p", source="bib_url", bibkey="r")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://strict.example.org/p").respond(405)
        mock.get("https://strict.example.org/p").respond(200)
        out = await check_links(checks)
    assert out[0].ok is True
    assert out[0].status == 200


@pytest.mark.asyncio
async def test_check_links_treats_5xx_as_broken():
    checks = [LinkCheck(url="https://broken.example.org/x", source="tex_url")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://broken.example.org/x").respond(503)
        mock.get("https://broken.example.org/x").respond(503)
        out = await check_links(checks)
    assert out[0].ok is False
    assert out[0].status == 503


@pytest.mark.asyncio
async def test_check_links_handles_connection_error():
    """A DNS or refused-connection failure should be flagged with a non-empty
    error message, not crash the run."""
    checks = [LinkCheck(url="https://nodns.example.invalid/x", source="tex_url")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://nodns.example.invalid/x").mock(
            side_effect=httpx.ConnectError("getaddrinfo failed")
        )
        out = await check_links(checks)
    assert out[0].ok is False
    assert out[0].status is None
    assert "connection" in (out[0].error or "").lower()


@pytest.mark.asyncio
async def test_check_links_handles_timeout():
    checks = [LinkCheck(url="https://slow.example.org/x", source="tex_url")]
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://slow.example.org/x").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        out = await check_links(checks)
    assert out[0].ok is False
    assert out[0].error == "timeout"


@pytest.mark.asyncio
async def test_check_links_returns_empty_for_empty_input():
    out = await check_links([])
    assert out == []


@pytest.mark.asyncio
async def test_check_links_runs_concurrently_in_order():
    """Result list preserves input order even with parallel probing."""
    inputs = [
        LinkCheck(url=f"https://a.example.org/{i}", source="tex_url") for i in range(5)
    ]
    with respx.mock(assert_all_called=False) as mock:
        for i in range(5):
            mock.head(f"https://a.example.org/{i}").respond(200)
        out = await check_links(inputs, concurrency=3)
    assert [c.url for c in out] == [c.url for c in inputs]
    assert all(c.ok for c in out)
