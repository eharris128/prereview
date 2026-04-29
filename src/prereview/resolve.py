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
from typing import Optional
from urllib.parse import quote

import httpx

from . import __version__
from .cache import Cache, cache_key
from .models import CanonicalRecord, Reference

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
    crossref_min_interval: float = 0.35   # ~3 RPS
    s2_min_interval: float = 1.1          # ~1 RPS shared
    arxiv_min_interval: float = 3.0       # arXiv asks for 3s
    openalex_min_interval: float = 0.2    # ~5 RPS


class Resolver:
    """Resolve Reference records to canonical records via four scholarly APIs.

    Use as an async context manager or call ``aclose()`` when done.
    """

    def __init__(
        self,
        *,
        cache: Cache,
        s2_api_key: Optional[str] = None,
        polite_mailto: Optional[str] = None,
        verbose: bool = False,
        timeout_s: float = 30.0,
    ):
        self.cache = cache
        self.s2_api_key = s2_api_key
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

    async def __aenter__(self) -> "Resolver":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    # ----- public ----------------------------------------------------------

    async def resolve(self, ref: Reference) -> Optional[CanonicalRecord]:
        """Resolve one reference; return the first source that yields a hit."""
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
            return cached

        for fn, label in [
            (lambda: self._crossref(ref, doi), "crossref"),
            (lambda: self._semanticscholar(ref, doi, arxiv_id), "semanticscholar"),
            (lambda: self._arxiv(ref, arxiv_id), "arxiv"),
            (lambda: self._openalex(ref, doi), "openalex"),
        ]:
            try:
                rec = await fn()
            except Exception as e:
                _log(self.verbose, f"{ref.ref_id}: {label} errored: {e!r}")
                continue
            if rec is not None:
                self.cache.put_record(key, rec)
                return rec
        return None

    # ----- per-source ------------------------------------------------------

    async def _crossref(self, ref: Reference, doi: Optional[str]) -> Optional[CanonicalRecord]:
        if doi:
            await self._g_crossref.wait()
            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            params = {"mailto": self.polite_mailto} if self.polite_mailto else None
            r = await self.client.get(url, params=params)
            if r.status_code == 200:
                msg = r.json().get("message")
                if msg:
                    return _crossref_to_record(msg)
            elif r.status_code != 404:
                _log(self.verbose, f"crossref DOI {doi}: HTTP {r.status_code}")

        if not ref.title:
            return None

        await self._g_crossref.wait()
        params = {
            "query.bibliographic": ref.title,
            "rows": "5",
        }
        if ref.authors:
            params["query.author"] = ref.authors[0]
        if self.polite_mailto:
            params["mailto"] = self.polite_mailto
        r = await self.client.get("https://api.crossref.org/works", params=params)
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        for item in items:
            if _title_matches(ref.title, item.get("title", [])):
                return _crossref_to_record(item)
        return None

    async def _semanticscholar(
        self, ref: Reference, doi: Optional[str], arxiv_id: Optional[str]
    ) -> Optional[CanonicalRecord]:
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        fields = "title,authors,year,abstract,externalIds,openAccessPdf,venue"

        async def _by_id(ext_id: str) -> Optional[CanonicalRecord]:
            await self._g_s2.wait()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{ext_id}"
            r = await self.client.get(url, headers=headers, params={"fields": fields})
            if r.status_code == 200:
                return _s2_to_record(r.json())
            return None

        if doi:
            rec = await _by_id(f"DOI:{doi}")
            if rec:
                return rec
        if arxiv_id:
            rec = await _by_id(f"ARXIV:{arxiv_id}")
            if rec:
                return rec

        if not ref.title:
            return None

        await self._g_s2.wait()
        r = await self.client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={"query": ref.title, "limit": "5", "fields": fields},
        )
        if r.status_code != 200:
            return None
        for item in r.json().get("data", []):
            if _title_matches(ref.title, [item.get("title", "")]):
                return _s2_to_record(item)
        return None

    async def _arxiv(self, ref: Reference, arxiv_id: Optional[str]) -> Optional[CanonicalRecord]:
        if arxiv_id:
            await self._g_arxiv.wait()
            r = await self.client.get(
                "http://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": "1"},
            )
            if r.status_code == 200:
                rec = _arxiv_to_record(r.text)
                if rec:
                    return rec

        if not ref.title:
            return None

        # Title + first-author search.
        q = f'ti:"{ref.title}"'
        if ref.authors:
            au = ref.authors[0].split()[-1]  # last name only
            q = f"{q}+AND+au:{au}"
        await self._g_arxiv.wait()
        r = await self.client.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": q, "max_results": "5"},
        )
        if r.status_code != 200:
            return None
        # Walk entries; pick the first whose title fuzzily matches.
        for entry_xml in _split_arxiv_entries(r.text):
            cand = _arxiv_to_record(entry_xml)
            if cand and _title_matches(ref.title, [cand.title]):
                return cand
        return None

    async def _openalex(self, ref: Reference, doi: Optional[str]) -> Optional[CanonicalRecord]:
        params = {}
        if self.polite_mailto:
            params["mailto"] = self.polite_mailto

        if doi:
            await self._g_openalex.wait()
            url = f"https://api.openalex.org/works/doi:{quote(doi, safe='')}"
            r = await self.client.get(url, params=params)
            if r.status_code == 200:
                return _openalex_to_record(r.json())
            elif r.status_code != 404:
                _log(self.verbose, f"openalex DOI {doi}: HTTP {r.status_code}")

        if not ref.title:
            return None

        await self._g_openalex.wait()
        params2 = {"search": ref.title, "per-page": "5", **params}
        r = await self.client.get("https://api.openalex.org/works", params=params2)
        if r.status_code != 200:
            return None
        for item in r.json().get("results", []):
            if _title_matches(ref.title, [item.get("title", "")]):
                return _openalex_to_record(item)
        return None


# ---------------------------------------------------------------------------
# convenience wrapper used by the pipeline


async def resolve_reference(
    ref: Reference,
    *,
    cache: Cache,
    s2_api_key: Optional[str] = None,
    polite_mailto: Optional[str] = None,
    verbose: bool = False,
) -> Optional[CanonicalRecord]:
    async with Resolver(
        cache=cache,
        s2_api_key=s2_api_key,
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
