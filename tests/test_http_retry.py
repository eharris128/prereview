"""Tests for prereview.http_retry — the single outbound-HTTP retry layer.

No real network and no real sleeps: a zero-backoff RetryPolicy keeps the retry
loop instant, and respx drives the status/exception sequences.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
import respx

from prereview.http_retry import (
    RetryPolicy,
    TransientExhausted,
    classify_status,
    get_with_retry,
    retry_after_seconds,
)

FAST = RetryPolicy(base=0.0, cap=0.0, max_attempts=3)
URL = "https://api.example.org/works"


@respx.mock
async def test_retries_then_succeeds():
    retries: list[int] = []
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, policy=FAST, on_retry=lambda: retries.append(1))
    assert resp.status_code == 200
    assert len(retries) == 2  # one per retry, not per attempt


@respx.mock
async def test_exhausts_raises_transient():
    route = respx.get(URL).mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        with pytest.raises(TransientExhausted):
            await get_with_retry(client, URL, policy=FAST)
    assert route.call_count == FAST.max_attempts


@respx.mock
async def test_terminal_status_returned_not_retried():
    retries: list[int] = []
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, policy=FAST, on_retry=lambda: retries.append(1))
    assert resp.status_code == 404
    assert route.call_count == 1
    assert retries == []


@respx.mock
async def test_connect_error_then_success():
    respx.get(URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200)])
    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, policy=FAST)
    assert resp.status_code == 200


@respx.mock
async def test_read_timeout_exhausts():
    respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(TransientExhausted):
            await get_with_retry(client, URL, policy=FAST)


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_classify_retry(code: int):
    assert classify_status(code) == "retry"


@pytest.mark.parametrize("code", [200, 400, 401, 403, 404, 409])
def test_classify_terminal(code: int):
    # 409 is terminal here (OpenAlex credit-exhaustion); the resolver maps it to a
    # degraded outcome, but http_retry must not retry it.
    assert classify_status(code) == "terminal"


def test_retry_after_seconds_numeric():
    resp = httpx.Response(429, headers={"retry-after": "2"})
    assert retry_after_seconds(resp) == 2.0


def test_retry_after_seconds_clamped():
    resp = httpx.Response(429, headers={"retry-after": "9999"})
    assert retry_after_seconds(resp) == 60.0


def test_retry_after_seconds_absent():
    assert retry_after_seconds(httpx.Response(200)) is None


def test_retry_after_seconds_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    resp = httpx.Response(429, headers={"retry-after": format_datetime(future)})
    secs = retry_after_seconds(resp)
    assert secs is not None and 0 <= secs <= 60
