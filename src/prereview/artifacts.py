"""Artifact existence checks — Hugging Face models/datasets + GitHub repos (U7).

When a paper claims a model, dataset, or code repo, verify the artifact actually
resolves — CS/ML-native ghost-artifact catching the citation resolver cannot do.

Papers-With-Code is excluded (its API was wound down in mid-2025). This module
talks to the public HF and GitHub REST APIs directly through the existing
:mod:`prereview.http_retry` layer — no heavyweight SDK, one retry layer, fully
respx-testable. Each lookup classifies its own response into HIT / TERMINAL_MISS /
TRANSIENT_FAIL exactly like the scholarly resolver, so a transient outage degrades
cleanly instead of inventing a "you faked this" verdict, and a missing artifact is
advisory ("verify"), never a defect.

Note (confirmed by a live smoke probe, 2026-06-18): HF's *unauthenticated* API
returns **401** — not 404 — for a model/dataset that is missing or private, so
401/403/404 all map to TERMINAL_MISS for HF. GitHub returns a clean 404 for a
missing repo, and 403 for rate-limiting (a transient, not a miss).
"""

from __future__ import annotations

import re
import sys
from typing import Optional

import httpx

from .http_retry import TransientExhausted, get_with_retry
from .models import ArtifactCheck, ArtifactStatus, LinkCheck

_Status = ArtifactStatus


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[artifacts] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# extraction


# Path segments under huggingface.co that are not model namespaces.
_HF_RESERVED = {
    "datasets", "spaces", "models", "docs", "blog", "join", "pricing",
    "settings", "api", "search", "login", "new",
}
# github.com paths that are site chrome, not repos.
_GH_RESERVED = {
    "sponsors", "features", "about", "pricing", "marketplace", "topics",
    "collections", "orgs", "apps", "settings", "login", "join", "explore", "search",
}

_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")


def _parse_artifact(url: str) -> Optional[tuple[str, str, str]]:
    """``url`` → ``(kind, identifier, canonical_url)`` or None.

    Handles ``org/name`` and single-segment HF ids and strips ``.git`` / trailing
    path from GitHub repo URLs.
    """
    # Anchor the host to the scheme so an *embedded* host — e.g.
    # "https://example.com/redirect?to=github.com/a/b" — is not mis-extracted as a
    # claimed artifact. URLs reach here already scheme-prefixed (link_checks are
    # normalized; body URLs are matched by _URL_RE, which requires https?://).
    m = re.search(r"https?://(?:www\.)?(huggingface\.co|github\.com)/(.+)", url, re.IGNORECASE)
    if not m:
        return None
    host = m.group(1).lower()
    raw = m.group(2).split("?")[0].split("#")[0].split()  # split() drops any trailing prose
    if not raw:
        return None
    segs = [s for s in raw[0].strip("/").split("/") if s]
    if not segs:
        return None

    if "github.com" in host:
        if len(segs) < 2 or segs[0].lower() in _GH_RESERVED:
            return None
        owner, repo = segs[0], re.sub(r"\.git$", "", segs[1])
        return ("github_repo", f"{owner}/{repo}", f"https://github.com/{owner}/{repo}")

    # huggingface.co
    if segs[0].lower() == "datasets":
        rest = segs[1:]
        if not rest:
            return None
        ident = "/".join(rest[:2]) if len(rest) >= 2 else rest[0]
        return ("hf_dataset", ident, f"https://huggingface.co/datasets/{ident}")
    if segs[0].lower() in _HF_RESERVED:
        return None
    ident = "/".join(segs[:2]) if len(segs) >= 2 else segs[0]
    return ("hf_model", ident, f"https://huggingface.co/{ident}")


def extract_artifacts(link_checks: list[LinkCheck], body: str) -> list[tuple[str, str, str]]:
    """Claimed artifacts (HF model/dataset, GitHub repo) from the paper's URLs and
    body, de-duplicated by ``(kind, identifier)``."""
    urls = [lc.url for lc in link_checks if lc.url]
    urls += _URL_RE.findall(body or "")
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for url in urls:
        parsed = _parse_artifact(url)
        if parsed is None:
            continue
        kind, ident, canonical = parsed
        key = (kind, ident.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, ident, canonical))
    return out


# ---------------------------------------------------------------------------
# probing


# kind → (api url template, statuses that mean an authoritative miss)
_PROBE: dict[str, tuple[str, frozenset]] = {
    "hf_model": ("https://huggingface.co/api/models/{}", frozenset({401, 403, 404})),
    "hf_dataset": ("https://huggingface.co/api/datasets/{}", frozenset({401, 403, 404})),
    "github_repo": ("https://api.github.com/repos/{}", frozenset({404})),
}

_KIND_LABEL = {"hf_model": "Hugging Face model", "hf_dataset": "Hugging Face dataset", "github_repo": "GitHub repo"}


async def _probe(
    client: httpx.AsyncClient, kind: str, identifier: str, url: str, *, hf_token: Optional[str]
) -> ArtifactCheck:
    api_url_tmpl, miss_statuses = _PROBE[kind]
    api_url = api_url_tmpl.format(identifier)
    headers: dict[str, str] = {}
    if kind.startswith("hf") and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if kind == "github_repo":
        headers["Accept"] = "application/vnd.github+json"

    label = _KIND_LABEL[kind]

    def _mk(status: ArtifactStatus, detail: str) -> ArtifactCheck:
        return ArtifactCheck(artifact_kind=kind, identifier=identifier, url=url, status=status, detail=detail)

    try:
        resp = await get_with_retry(client, api_url, headers=headers or None, timeout=15.0)
    except TransientExhausted:
        return _mk(_Status.TRANSIENT_FAIL, f"{label} `{identifier}`: host did not respond after retries")
    except httpx.HTTPError as e:
        return _mk(_Status.TRANSIENT_FAIL, f"{label} `{identifier}`: request failed ({type(e).__name__})")

    if resp.status_code in miss_statuses:
        return _mk(
            _Status.TERMINAL_MISS,
            f"claimed {label} `{identifier}` did not resolve (not found, private, or gated) — "
            "verify the link",
        )
    if resp.status_code == 200:
        try:
            resp.json()
        except (ValueError, httpx.HTTPError):
            return _mk(_Status.TRANSIENT_FAIL, f"{label} `{identifier}`: 200 with an unparseable body")
        return _mk(_Status.HIT, "")
    return _mk(_Status.TRANSIENT_FAIL, f"{label} `{identifier}`: unexpected HTTP {resp.status_code}")


async def check_artifacts(
    link_checks: list[LinkCheck], body: str, *, hf_token: Optional[str] = None, verbose: bool = False
) -> list[ArtifactCheck]:
    """Probe every claimed artifact and return the classified results. A network
    layer that is wholly unreachable degrades each probe to TRANSIENT_FAIL rather
    than raising — the pipeline always completes."""
    artifacts = extract_artifacts(link_checks, body)
    if not artifacts:
        return []
    _log(verbose, f"probing {len(artifacts)} claimed artifact(s)")
    results: list[ArtifactCheck] = []
    headers = {"User-Agent": "prereview (artifact existence check; mailto via PREREVIEW_MAILTO)"}
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        for kind, ident, url in artifacts:
            check = await _probe(client, kind, ident, url, hf_token=hf_token)
            _log(verbose, f"  {ident}: {check.status.value}")
            results.append(check)
    return results
