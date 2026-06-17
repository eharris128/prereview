"""Stage 2: resolve bibliography references against external scholarly APIs.

Priority order, applied per reference:

1. Crossref — by DOI if present, else by bibliographic + author search.
2. Semantic Scholar — by DOI / arXiv ID, else by title search.
3. arXiv — by ID, else by title + author search.
4. OpenAlex — by DOI if present, else by search.

Rate limits and politeness:

- Crossref: include a `mailto:` in User-Agent for the polite pool. Cap calls at
  ~3 RPS to be a good citizen.
- Semantic Scholar: shared 1-RPS pool when unauthenticated. If `S2_API_KEY` is
  set, send `x-api-key` header.
- arXiv: their docs ask for a 3-second pause between requests; we honor that.
- OpenAlex: include `mailto=` query param for the polite pool. Cap at ~5 RPS.

The LLM is never used in this module. Canonical fields come only from the four
APIs above.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import quote

import httpx

from . import __version__
from .cache import Cache, cache_key
from .http_retry import RetryPolicy, TransientExhausted, get_with_retry
from .models import CanonicalRecord, Reference, Resolution, ResolutionStatus

log = logging.getLogger("prereview.resolve")


# ---------------------------------------------------------------------------
# helpers


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[resolve] {msg}", file=sys.stderr, flush=True)


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_ARXIV_NEW_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_ARXIV_OLD_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")


def normalize_doi(s: Optional[str]) -> Optional[str]:
    """Pull a bare DOI out of a URL or raw string. Returns lowercase DOI or None."""
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    m = _DOI_RE.search(s)
    return m.group(0).lower().rstrip(".,;)") if m else None


def normalize_arxiv_id(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^https?://arxiv\.org/abs/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^arxiv:", "", s, flags=re.IGNORECASE)
    m = _ARXIV_NEW_RE.search(s) or _ARXIV_OLD_RE.search(s)
    return m.group(1) if m else None


def _decode_openalex_abstract(inv_idx: dict | None) -> Optional[str]:
    """OpenAlex stores abstracts as {word: [positions]}. Reconstruct the string."""
    if not inv_idx:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_idx.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)


# ---------------------------------------------------------------------------
# rate limiting


class _MinIntervalGate:
    """Forces a minimum interval between calls. Async-safe."""

    def __init__(self, min_interval_s: float):
        self.min_interval = min_interval_s
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = (self._last + self.min_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Resolver


@dataclass
class _SourceConfig:
    crossref_min_interval: float = 1.0    # >=1 r/s — public-pool list-endpoint limit (Dec 2025)
    s2_min_interval: float = 1.1          # ~1 RPS shared
    arxiv_min_interval: float = 3.0       # arXiv asks for 3s
    openalex_min_interval: float = 0.2    # ~5 RPS


# Circuit-breaker thresholds, per source per run. Once a source trips we stop
# calling it for the rest of the run and emit TRANSIENT_FAIL for its refs — this
# bounds retry amplification against a systemic outage (AWS's "retries are selfish"
# guard). The consecutive counter resets on any definitive answer; the total-retry
# budget also bounds a half-working source that alternates fail/success.
_BREAKER_CONSECUTIVE = 5
_BREAKER_TOTAL_RETRIES = 8


class _Outcome(str, Enum):
    """Per-source resolution outcome, before cross-source aggregation."""

    HIT = "hit"  # the source returned a matching canonical record
    TERMINAL_MISS = "terminal_miss"  # authoritative "not here" (404 or parsed empty result)
    TRANSIENT_FAIL = "transient_fail"  # transient failure after retries, or breaker-tripped


class Resolver:
    """Resolve Reference records to canonical records via four scholarly APIs.

    Use as an async context manager or call ``aclose()`` when done.
    """

    def __init__(
        self,
        *,
        cache: Cache,
        s2_api_key: Optional[str] = None,
        openalex_api_key: Optional[str] = None,
        polite_mailto: Optional[str] = None,
        verbose: bool = False,
        timeout_s: float = 30.0,
    ):
        self.cache = cache
        self.s2_api_key = s2_api_key
        self.openalex_api_key = openalex_api_key
        self.polite_mailto = polite_mailto
        self.verbose = verbose
        ua = f"prereview/{__version__} (https://github.com/echarris/prereview"
        if polite_mailto:
            ua += f"; mailto:{polite_mailto}"
        ua += ")"
        self._ua = ua
        self.client = httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": ua, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._cfg = _SourceConfig()
        self._g_crossref = _MinIntervalGate(self._cfg.crossref_min_interval)
        self._g_s2 = _MinIntervalGate(self._cfg.s2_min_interval)
        self._g_arxiv = _MinIntervalGate(self._cfg.arxiv_min_interval)
        self._g_openalex = _MinIntervalGate(self._cfg.openalex_min_interval)
        # Retry + circuit-breaker state (per Resolver instance == per run).
        self._policy = RetryPolicy()
        self._consec_fail: dict[str, int] = {}
        self._retry_budget: dict[str, int] = {}
        self._tripped: set[str] = set()
        self.recovered_after_retry = 0

    async def __aenter__(self) -> "Resolver":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    # ----- internals -------------------------------------------------------

    def _record_transient(self, source: str, retries: int) -> None:
        self._consec_fail[source] = self._consec_fail.get(source, 0) + 1
        self._retry_budget[source] = self._retry_budget.get(source, 0) + retries
        if source not in self._tripped and (
            self._consec_fail[source] >= _BREAKER_CONSECUTIVE
            or self._retry_budget[source] >= _BREAKER_TOTAL_RETRIES
        ):
            self._tripped.add(source)
            _log(self.verbose, f"circuit breaker tripped for {source}; skipping it for the rest of the run")

    def _record_answer(self, source: str, retries: int) -> None:
        # A definitive HTTP answer (success or terminal status) clears the
        # consecutive-failure streak; if it needed retries, the retry layer recovered it.
        self._consec_fail[source] = 0
        if retries > 0:
            self.recovered_after_retry += 1
            self._retry_budget[source] = self._retry_budget.get(source, 0) + retries

    async def _request(
        self,
        source: str,
        gate: "_MinIntervalGate",
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
        parse: str = "json",
    ):
        """Gate + retry one GET, honoring the per-source circuit breaker.

        Returns ``("ok", payload)`` (parsed JSON or raw text for a 2xx),
        ``("terminal", status_code)`` for an authoritative non-2xx, or
        ``("transient", None)`` when the source is breaker-tripped, every retry was
        exhausted, or a 2xx body failed to parse.
        """
        if source in self._tripped:
            return "transient", None
        await gate.wait()
        retried = [0]
        try:
            r = await get_with_retry(
                self.client,
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                policy=self._policy,
                on_retry=lambda: retried.__setitem__(0, retried[0] + 1),
            )
        except TransientExhausted as e:
            self._record_transient(source, retried[0])
            _log(self.verbose, f"{source}: transient failure not recovered ({e})")
            return "transient", None
        self._record_answer(source, retried[0])
        if r.status_code == 200:
            if parse == "json":
                try:
                    return "ok", r.json()
                except Exception:
                    # A 2xx we can't parse is a transient symptom (truncated body,
                    # proxy error page), never an authoritative "not found".
                    self._record_transient(source, 0)
                    _log(self.verbose, f"{source}: 200 with unparseable body — treating as transient")
                    return "transient", None
            return "ok", r.text
        return "terminal", r.status_code

    # ----- public ----------------------------------------------------------

    async def resolve(self, ref: Reference) -> Resolution:
        """Resolve one reference across the four sources.

        RESOLVED with the record on the first hit; UNRESOLVED only when *every*
        source gave an authoritative terminal miss (a true ghost); DEGRADED when at
        least one source failed transiently after retries and none resolved
        (couldn't verify — never reported as a ghost).
        """
        doi = normalize_doi(ref.doi)
        arxiv_id = normalize_arxiv_id(ref.arxiv_id) or _guess_arxiv_id(ref)
        key = cache_key(
            doi=doi,
            arxiv_id=arxiv_id,
            title=ref.title,
            first_author=ref.authors[0] if ref.authors else None,
        )

        cached = self.cache.get_record(key)
        if cached is not None:
            _log(self.verbose, f"cache hit for {ref.ref_id} ({cached.source})")
            return Resolution(status=ResolutionStatus.RESOLVED, record=cached)

        any_transient = False
        for fn, label in [
            (lambda: self._crossref(ref, doi), "crossref"),
            (lambda: self._semanticscholar(ref, doi, arxiv_id), "semanticscholar"),
            (lambda: self._arxiv(ref, arxiv_id), "arxiv"),
            (lambda: self._openalex(ref, doi), "openalex"),
        ]:
            try:
                outcome, rec = await fn()
            except Exception as e:
                # An unexpected error in a source method is a transient symptom, not
                # an authoritative "not here" — never let it manufacture a ghost.
                _log(self.verbose, f"{ref.ref_id}: {label} errored: {e!r}")
                any_transient = True
                continue
            if outcome == _Outcome.HIT and rec is not None:
                # OpenAlex mirrors Retraction Watch via `is_retracted`. For hits from
                # other sources, do a follow-up OpenAlex DOI lookup so retraction is
                # known regardless of which source resolved. Cached with the record.
                if not rec.is_retracted and rec.source != "openalex" and rec.doi:
                    try:
                        if await self._openalex_is_retracted(rec.doi):
                            rec.is_retracted = True
                    except Exception as e:
                        _log(self.verbose, f"{ref.ref_id}: retraction check failed: {e!r}")
                self.cache.put_record(key, rec)  # cache only a RESOLVED record, never DEGRADED
                return Resolution(status=ResolutionStatus.RESOLVED, record=rec)
            if outcome == _Outcome.TRANSIENT_FAIL:
                any_transient = True
            # TERMINAL_MISS → try the next source
        if any_transient:
            return Resolution(status=ResolutionStatus.DEGRADED)
        return Resolution(status=ResolutionStatus.UNRESOLVED)

    # ----- per-source ------------------------------------------------------

    async def _crossref(self, ref: Reference, doi: Optional[str]) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        transient = False
        if doi:
            tag, payload = await self._request(
                "crossref",
                self._g_crossref,
                f"https://api.crossref.org/works/{quote(doi, safe='')}",
                params={"mailto": self.polite_mailto} if self.polite_mailto else None,
            )
            if tag == "transient":
                transient = True
            elif tag == "ok":
                msg = payload.get("message")
                if msg:
                    return _Outcome.HIT, _crossref_to_record(msg)
            # terminal (404) → fall through to the title search

        if ref.title:
            params = {"query.bibliographic": ref.title, "rows": "5"}
            if ref.authors:
                params["query.author"] = ref.authors[0]
            if self.polite_mailto:
                params["mailto"] = self.polite_mailto
            tag, payload = await self._request(
                "crossref", self._g_crossref, "https://api.crossref.org/works", params=params
            )
            if tag == "transient":
                transient = True
            elif tag == "ok":
                for item in payload.get("message", {}).get("items", []):
                    if _title_matches(ref.title, item.get("title", [])):
                        return _Outcome.HIT, _crossref_to_record(item)
        return (_Outcome.TRANSIENT_FAIL if transient else _Outcome.TERMINAL_MISS), None

    async def _semanticscholar(
        self, ref: Reference, doi: Optional[str], arxiv_id: Optional[str]
    ) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key
        fields = "title,authors,year,abstract,externalIds,openAccessPdf,venue"
        transient = False

        async def _by_id(ext_id: str):
            tag, payload = await self._request(
                "semanticscholar",
                self._g_s2,
                f"https://api.semanticscholar.org/graph/v1/paper/{ext_id}",
                headers=headers,
                params={"fields": fields},
            )
            return tag, (_s2_to_record(payload) if tag == "ok" else None)

        for ext_id in (f"DOI:{doi}" if doi else None, f"ARXIV:{arxiv_id}" if arxiv_id else None):
            if not ext_id:
                continue
            tag, rec = await _by_id(ext_id)
            if tag == "transient":
                transient = True
            elif rec is not None:
                return _Outcome.HIT, rec

        if ref.title:
            tag, payload = await self._request(
                "semanticscholar",
                self._g_s2,
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={"query": ref.title, "limit": "5", "fields": fields},
            )
            if tag == "transient":
                transient = True
            elif tag == "ok":
                for item in payload.get("data", []):
                    if _title_matches(ref.title, [item.get("title", "")]):
                        return _Outcome.HIT, _s2_to_record(item)
        return (_Outcome.TRANSIENT_FAIL if transient else _Outcome.TERMINAL_MISS), None

    async def _arxiv(self, ref: Reference, arxiv_id: Optional[str]) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        transient = False
        if arxiv_id:
            tag, payload = await self._request(
                "arxiv",
                self._g_arxiv,
                "http://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": "1"},
                parse="text",
            )
            if tag == "transient":
                transient = True
            elif tag == "ok":
                if not _xml_well_formed(payload):
                    transient = True  # 200 but corrupt XML — transient, not "not found"
                else:
                    rec = _arxiv_to_record(payload)
                    if rec:
                        return _Outcome.HIT, rec

        if ref.title:
            # Title + first-author search.
            q = f'ti:"{ref.title}"'
            if ref.authors:
                au = ref.authors[0].split()[-1]  # last name only
                q = f"{q}+AND+au:{au}"
            tag, payload = await self._request(
                "arxiv",
                self._g_arxiv,
                "http://export.arxiv.org/api/query",
                params={"search_query": q, "max_results": "5"},
                parse="text",
            )
            if tag == "transient":
                transient = True
            elif tag == "ok":
                if not _xml_well_formed(payload):
                    transient = True
                else:
                    for entry_xml in _split_arxiv_entries(payload):
                        cand = _arxiv_to_record(entry_xml)
                        if cand and _title_matches(ref.title, [cand.title]):
                            return _Outcome.HIT, cand
        return (_Outcome.TRANSIENT_FAIL if transient else _Outcome.TERMINAL_MISS), None

    async def _openalex_is_retracted(self, doi: str) -> bool:
        """Lightweight retraction-only OpenAlex check by DOI, via the retry helper.

        Used as a follow-up when the primary resolver hit was Crossref / S2 / arXiv
        (none of which expose retraction status reliably). Returns False on any
        non-OK outcome (transient, terminal, or unparseable) — a missed retraction is
        preferable to a false positive that would scare a user about a valid paper.
        Routed through ``_request`` so it honors the OpenAlex key, retry/backoff, and
        the circuit breaker like the resolver's own calls.
        """
        params = {}
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
        tag, payload = await self._request(
            "openalex",
            self._g_openalex,
            f"https://api.openalex.org/works/doi:{quote(doi, safe='')}",
            params=params or None,
        )
        if tag == "ok":
            return bool(payload.get("is_retracted"))
        return False

    async def _openalex(self, ref: Reference, doi: Optional[str]) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        # OpenAlex requires a free API key since 2026-02-13; the polite-pool mailto is
        # gone. Send the key when set, and treat a 409 (keyless / daily credits
        # exhausted) as infrastructure-degraded, never as an authoritative "not found".
        params: dict = {}
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
        transient = False

        if doi:
            tag, payload = await self._request(
                "openalex",
                self._g_openalex,
                f"https://api.openalex.org/works/doi:{quote(doi, safe='')}",
                params=params or None,
            )
            if tag == "transient" or (tag == "terminal" and payload == 409):
                transient = True
            elif tag == "ok":
                return _Outcome.HIT, _openalex_to_record(payload)
            # terminal 404 → fall through to the search

        if ref.title:
            tag, payload = await self._request(
                "openalex",
                self._g_openalex,
                "https://api.openalex.org/works",
                params={"search": ref.title, "per-page": "5", **params},
            )
            if tag == "transient" or (tag == "terminal" and payload == 409):
                transient = True
            elif tag == "ok":
                for item in payload.get("results", []):
                    if _title_matches(ref.title, [item.get("title", "")]):
                        return _Outcome.HIT, _openalex_to_record(item)
        return (_Outcome.TRANSIENT_FAIL if transient else _Outcome.TERMINAL_MISS), None


# ---------------------------------------------------------------------------
# convenience wrapper used by the pipeline


async def resolve_reference(
    ref: Reference,
    *,
    cache: Cache,
    s2_api_key: Optional[str] = None,
    openalex_api_key: Optional[str] = None,
    polite_mailto: Optional[str] = None,
    verbose: bool = False,
) -> Resolution:
    async with Resolver(
        cache=cache,
        s2_api_key=s2_api_key,
        openalex_api_key=openalex_api_key,
        polite_mailto=polite_mailto,
        verbose=verbose,
    ) as r:
        return await r.resolve(ref)


# ---------------------------------------------------------------------------
# format adapters


def _crossref_to_record(msg: dict) -> CanonicalRecord:
    titles = msg.get("title") or []
    title = titles[0] if titles else ""
    authors = []
    for a in msg.get("author", []) or []:
        given = a.get("given", "")
        family = a.get("family", "")
        full = (given + " " + family).strip()
        if full:
            authors.append(full)
    issued = msg.get("issued", {}).get("date-parts") or [[None]]
    year = None
    if issued and issued[0] and issued[0][0]:
        try:
            year = int(issued[0][0])
        except (TypeError, ValueError):
            year = None
    venue = None
    for k in ("container-title", "publisher"):
        v = msg.get(k)
        if isinstance(v, list) and v:
            venue = v[0]
            break
        if isinstance(v, str) and v:
            venue = v
            break
    doi = msg.get("DOI")
    url = f"https://doi.org/{doi}" if doi else msg.get("URL")
    abstract = msg.get("abstract")
    if abstract:
        # Crossref abstracts often arrive wrapped in <jats:p>...</jats:p>
        abstract = re.sub(r"<[^>]+>", "", abstract).strip() or None
    return CanonicalRecord(
        source="crossref",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi.lower() if doi else None,
        url=url,
        abstract=abstract,
    )


def _s2_to_record(item: dict) -> CanonicalRecord:
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI") or ext.get("doi")
    if doi:
        doi = doi.lower()
    arxiv = ext.get("ArXiv") or ext.get("arxiv")
    oa = item.get("openAccessPdf") or {}
    pdf_url = oa.get("url") if isinstance(oa, dict) else None
    return CanonicalRecord(
        source="semanticscholar",
        title=item.get("title") or "",
        authors=[a.get("name", "") for a in item.get("authors", []) or [] if a.get("name")],
        year=item.get("year"),
        venue=item.get("venue"),
        doi=doi,
        url=(f"https://doi.org/{doi}" if doi else (f"https://arxiv.org/abs/{arxiv}" if arxiv else None)),
        abstract=item.get("abstract"),
        open_access_pdf_url=pdf_url,
    )


def _arxiv_to_record(xml_text: str) -> Optional[CanonicalRecord]:
    """Parse a single arXiv Atom entry (or full feed and pick the first entry)."""
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    entry = root if root.tag.endswith("entry") else root.find("a:entry", ns)
    if entry is None:
        return None

    def text(el: ET.Element | None) -> str:
        return (el.text or "").strip() if el is not None else ""

    title = " ".join(text(entry.find("a:title", ns)).split())
    if not title:
        return None
    summary = " ".join(text(entry.find("a:summary", ns)).split())
    authors = [text(a.find("a:name", ns)) for a in entry.findall("a:author", ns)]
    authors = [a for a in authors if a]
    id_url = text(entry.find("a:id", ns))
    arxiv_id = None
    if id_url:
        m = re.search(r"arxiv\.org/abs/(.+)$", id_url)
        if m:
            arxiv_id = m.group(1).split("v")[0]
    published = text(entry.find("a:published", ns))
    year = None
    if published[:4].isdigit():
        year = int(published[:4])
    pdf_url = None
    for link in entry.findall("a:link", ns):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break
    doi_el = entry.find("arxiv:doi", ns)
    doi = (doi_el.text.lower() if doi_el is not None and doi_el.text else None)

    return CanonicalRecord(
        source="arxiv",
        title=title,
        authors=authors,
        year=year,
        venue="arXiv",
        doi=doi,
        url=id_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
        abstract=summary or None,
        open_access_pdf_url=pdf_url,
    )


def _xml_well_formed(text: str) -> bool:
    """True if ``text`` parses as XML. Distinguishes a corrupt 200 body (transient)
    from a genuinely empty arXiv feed (terminal not-found)."""
    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False


def _split_arxiv_entries(feed_xml: str) -> list[str]:
    """Yield each <entry> element from a feed as its own XML string."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError:
        return []
    out: list[str] = []
    for entry in root.findall("a:entry", ns):
        out.append(
            ET.tostring(entry, encoding="unicode")
            .replace("ns0:", "")
            .replace("xmlns:ns0", "xmlns")
        )
    # The simplest path is to wrap each entry in a feed root so namespaces resolve.
    return [
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">' + e + "</feed>"
        for e in out
    ]


def _openalex_to_record(item: dict) -> CanonicalRecord:
    title = item.get("title") or item.get("display_name") or ""
    authors = []
    for au in item.get("authorships", []) or []:
        a = au.get("author", {}) or {}
        n = a.get("display_name")
        if n:
            authors.append(n)
    year = item.get("publication_year")
    doi_url = item.get("doi") or ""
    doi = doi_url.replace("https://doi.org/", "").lower() if doi_url else None
    venue = None
    pv = item.get("primary_location") or {}
    src = pv.get("source") if isinstance(pv, dict) else None
    if isinstance(src, dict):
        venue = src.get("display_name")
    abstract = _decode_openalex_abstract(item.get("abstract_inverted_index"))
    oa = item.get("open_access") or {}
    return CanonicalRecord(
        source="openalex",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        url=doi_url or item.get("id"),
        abstract=abstract,
        open_access_pdf_url=oa.get("oa_url") if isinstance(oa, dict) else None,
        is_retracted=bool(item.get("is_retracted")),
    )


# ---------------------------------------------------------------------------
# heuristics


def _norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _title_matches(query: str, candidates: list[str]) -> bool:
    """Loose-equality title match. Accepts exact prefix / Jaccard >= 0.7 over tokens."""
    if not query:
        return False
    q = _norm_title(query)
    if not q:
        return False
    q_tokens = set(q.split())
    for c in candidates:
        if not c:
            continue
        c_norm = _norm_title(c)
        if not c_norm:
            continue
        if q == c_norm:
            return True
        c_tokens = set(c_norm.split())
        if not q_tokens or not c_tokens:
            continue
        inter = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        if union and inter / union >= 0.7:
            return True
    return False


def _guess_arxiv_id(ref: Reference) -> Optional[str]:
    """Sometimes the parsed Reference puts the arXiv ID in url or raw_text."""
    for s in (ref.url, ref.raw_text):
        if not s:
            continue
        a = normalize_arxiv_id(s)
        if a:
            return a
    return None
