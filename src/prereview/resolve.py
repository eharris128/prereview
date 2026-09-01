"""Stage 2: resolve bibliography references against external scholarly APIs.

Two phases, applied per reference:

**Identifier lookups** (authoritative — never pre-empted by a title search):

1. Crossref — by DOI.
2. Semantic Scholar — by DOI, then by arXiv ID.
3. arXiv — by ID.
4. OpenAlex — by DOI.

**Title searches**, same source order, only when no identifier lookup hit. A
search hit must match on title *and* be year-compatible with the bibliography
entry (``_SourceConfig.year_tolerance``) to count. Same-title records with the
wrong year are common — reprints, mirror "journals", a later edition — and a
Crossref one was observed winning over the real NeurIPS paper. A title+author
match with the wrong year is kept as a *weak* candidate and accepted, with a
visible ``match_note``, only when no source has a strong hit and nothing failed
transiently; other rejected hits are reported as ``near_misses`` so a ghost
verdict can say what it did find.

Also on every resolved record, as follow-ups: an OpenAlex retraction lookup by
DOI, and — for entries that cite an arXiv preprint — a published-version check
(from the record itself when a source already told us, else Semantic Scholar by
arXiv ID).

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
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from urllib.parse import quote

import httpx

from . import __version__
from .cache import Cache, cache_key
from .http_retry import RetryPolicy, TransientExhausted, get_with_retry
from .models import CanonicalRecord, PublishedVersion, Reference, Resolution, ResolutionStatus

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
    dblp_min_interval: float = 1.0        # no published limit; be polite (published-version confirmer only)
    # A title-search hit whose year is further than this from the bibliography's
    # year is not a strong match. 2 covers preprint-vs-proceedings lag (a 2019
    # arXiv paper published in 2021) without admitting a 2025 reprint of a 2017
    # paper. Identifier lookups are never year-gated.
    year_tolerance: int = 2


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


@dataclass
class _Scratch:
    """Per-``resolve()`` call: what the title searches saw but did not accept."""

    weak: list[CanonicalRecord] = field(default_factory=list)  # title+author match, wrong year
    near_misses: list[str] = field(default_factory=list)  # title match only; described for the ghost rationale

    def best_weak(self, ref_year: Optional[int]) -> Optional[CanonicalRecord]:
        if not self.weak:
            return None

        def gap(rec: CanonicalRecord) -> int:
            if ref_year is None or rec.year is None:
                return 10_000
            return abs(rec.year - ref_year)

        return min(self.weak, key=gap)


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
        self._g_dblp = _MinIntervalGate(self._cfg.dblp_min_interval)
        # Retry + circuit-breaker state (per Resolver instance == per run).
        self._policy = RetryPolicy()
        self._consec_fail: dict[str, int] = {}
        self._retry_budget: dict[str, int] = {}
        self._tripped: set[str] = set()
        self.recovered_after_retry = 0
        # arXiv-cited entries whose published-version follow-up failed transiently
        # this run (left un-checked in the cache; disclosed in Coverage).
        self.published_version_unchecked = 0

    async def __aenter__(self) -> "Resolver":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    @property
    def circuit_broken_sources(self) -> list[str]:
        """Sources whose circuit breaker tripped this run (stopped being queried)."""
        return sorted(self._tripped)

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

        Identifier lookups first (see the module docstring), then title searches.
        RESOLVED with the record on the first strong hit — or, failing that, the
        best weak hit with a ``match_note``; UNRESOLVED only when *every* source
        gave an authoritative terminal miss (a true ghost, with the rejected
        near-misses attached); DEGRADED when at least one source failed transiently
        after retries and none resolved (couldn't verify — never reported as a
        ghost, and a weak hit is *not* accepted over a transient failure).
        """
        doi = normalize_doi(ref.doi)
        arxiv_id = arxiv_id_of(ref)
        key = cache_key(
            doi=doi,
            arxiv_id=arxiv_id,
            title=ref.title,
            first_author=ref.authors[0] if ref.authors else None,
        )

        cached = self.cache.get_record(key)
        if cached is not None:
            _log(self.verbose, f"cache hit for {ref.ref_id} ({cached.source})")
            # Records cached before the published-version check existed, or whose
            # follow-up failed transiently last run, get the check on the way out.
            if await self._ensure_published_version(ref, cached, arxiv_id):
                self.cache.put_record(key, cached)
            return Resolution(status=ResolutionStatus.RESOLVED, record=cached)

        scratch = _Scratch()
        any_transient = False
        for mode in ("id", "search"):
            sources = [
                (lambda m=mode: self._crossref(ref, doi, m, scratch), "crossref"),
                (lambda m=mode: self._semanticscholar(ref, doi, arxiv_id, m, scratch), "semanticscholar"),
                (lambda m=mode: self._arxiv(ref, arxiv_id, m, scratch), "arxiv"),
                (lambda m=mode: self._openalex(ref, doi, m, scratch), "openalex"),
            ]
            for fn, label in sources:
                try:
                    outcome, rec = await fn()
                except Exception as e:
                    # An unexpected error in a source method is a transient symptom, not
                    # an authoritative "not here" — never let it manufacture a ghost.
                    _log(self.verbose, f"{ref.ref_id}: {label} ({mode}) errored: {e!r}")
                    any_transient = True
                    continue
                if outcome == _Outcome.HIT and rec is not None:
                    return await self._finalize(ref, key, rec, arxiv_id)
                if outcome == _Outcome.TRANSIENT_FAIL:
                    any_transient = True
                # TERMINAL_MISS → try the next source

        if any_transient:
            return Resolution(status=ResolutionStatus.DEGRADED, near_misses=scratch.near_misses)

        weak = scratch.best_weak(ref.year)
        if weak is not None:
            weak.match_note = (
                "Matched on title and author surnames only: the bibliography gives "
                f"{ref.year if ref.year is not None else '(no year)'} but the resolved record is "
                f"dated {weak.year if weak.year is not None else '(no year)'}"
                f"{' (' + weak.venue + ')' if weak.venue else ''}. No source had a "
                "year-compatible record. Verify this is the intended paper — a different "
                "edition, or a reprint / mirror of the original, looks identical here."
            )
            _log(self.verbose, f"{ref.ref_id}: accepting weak {weak.source} match ({weak.year} vs bib {ref.year})")
            return await self._finalize(ref, key, weak, arxiv_id)

        return Resolution(status=ResolutionStatus.UNRESOLVED, near_misses=scratch.near_misses)

    async def _finalize(
        self, ref: Reference, key: str, rec: CanonicalRecord, arxiv_id: Optional[str]
    ) -> Resolution:
        """Follow-ups on a resolved record, then cache it (only RESOLVED records
        are ever cached, never DEGRADED)."""
        # OpenAlex mirrors Retraction Watch via `is_retracted`. For hits from other
        # sources, do a follow-up OpenAlex DOI lookup so retraction is known
        # regardless of which source resolved. Cached with the record.
        if not rec.is_retracted and rec.source != "openalex" and rec.doi:
            try:
                if await self._openalex_is_retracted(rec.doi):
                    rec.is_retracted = True
            except Exception as e:
                _log(self.verbose, f"{ref.ref_id}: retraction check failed: {e!r}")
        try:
            await self._ensure_published_version(ref, rec, arxiv_id)
        except Exception as e:
            _log(self.verbose, f"{ref.ref_id}: published-version check failed: {e!r}")
            self.published_version_unchecked += 1
        self.cache.put_record(key, rec)
        return Resolution(status=ResolutionStatus.RESOLVED, record=rec)

    # ----- match acceptance ------------------------------------------------

    def _accept_search_hit(self, ref: Reference, rec: CanonicalRecord, scratch: _Scratch) -> bool:
        """Judge a title-matched search hit. True (strong) when its year is
        compatible with the bibliography entry. Otherwise it is stashed — as a
        weak candidate when the author surnames overlap, else as a described
        near miss — and False is returned so the caller keeps looking."""
        if _year_compatible(ref.year, rec.year, self._cfg.year_tolerance):
            return True
        if _surnames_overlap(ref.authors, rec.authors):
            scratch.weak.append(rec)
            _log(
                self.verbose,
                f"{ref.ref_id}: {rec.source} title+author match kept as weak — "
                f"year {rec.year} vs bibliography {ref.year}",
            )
        else:
            scratch.near_misses.append(_describe_record(rec))
            _log(
                self.verbose,
                f"{ref.ref_id}: {rec.source} title match rejected — year {rec.year} vs "
                f"bibliography {ref.year}, no author overlap",
            )
        return False

    # ----- published-version follow-up (outdated arXiv citations) -----------

    async def _ensure_published_version(
        self, ref: Reference, rec: CanonicalRecord, arxiv_id: Optional[str]
    ) -> bool:
        """For an entry that cites an arXiv preprint, make sure ``rec`` carries a
        definitive published-version determination. Returns True when the record
        was changed (so a cached copy should be rewritten).

        Order: the record itself (a strong Crossref / OpenAlex venue hit, or a
        Semantic Scholar / arXiv record that already decided) → Semantic Scholar by
        arXiv ID. A transient failure leaves the record un-checked and is counted
        in ``published_version_unchecked`` — never recorded as "no published version".
        """
        if rec.published_version_checked or not cites_arxiv_version(ref):
            return False
        pv = _published_from_record(rec)
        if pv is not None:
            rec.published_version = pv
            rec.published_version_checked = True
            return True
        def found(pv: PublishedVersion) -> bool:
            rec.published_version = pv
            rec.published_version_checked = True
            _log(
                self.verbose,
                f"{ref.ref_id}: published version found via {pv.source} — {pv.venue or pv.doi} ({pv.year})",
            )
            return True

        # 1. Semantic Scholar by arXiv ID — the authority: its merged record either
        #    lists the venue version or says there is none. A record that *came
        #    from* S2 already is S2's answer ("none", or the adapter would have
        #    marked it checked) — don't ask again.
        aid = arxiv_id or normalize_arxiv_id(rec.url)
        s2_tag, pv = ("skip", None)
        if rec.source == "semanticscholar":
            s2_tag = "ok"
        elif aid:
            s2_tag, pv = await self._s2_published_version(aid)
            if pv is not None:
                return found(pv)
        # 2. DBLP title search — curated, keyless, CS-native; it knows the NeurIPS /
        #    ICLR / ICML versions that carry no DOI and is often ahead of S2 for
        #    fresh proceedings. Confirms only: a miss is not a determination.
        d_tag, pv = await self._dblp_published_version(ref, rec)
        if pv is not None:
            return found(pv)
        if s2_tag in ("ok", "terminal"):
            # S2 answered ("no venue version" / "unknown paper") and DBLP did not
            # disagree: definitive for this run.
            rec.published_version = None
            rec.published_version_checked = True
            return True
        # 3. S2 unavailable (the keyless pool rate-limits into the breaker quickly)
        #    or no arXiv ID to ask it by: Crossref title search, tightly gated. Used
        #    only here — Crossref is uncurated, and a same-year mirror reprint could
        #    pass the gates, so it is never allowed to overrule S2's "none".
        cr_tag, pv = await self._crossref_published_version(ref, rec)
        if pv is not None:
            return found(pv)
        if "transient" in (s2_tag, d_tag, cr_tag):
            self.published_version_unchecked += 1
            _log(
                self.verbose,
                f"{ref.ref_id}: published-version check unavailable (Semantic Scholar / DBLP / Crossref)",
            )
        return False

    async def _dblp_published_version(
        self, ref: Reference, rec: CanonicalRecord
    ) -> tuple[str, Optional[PublishedVersion]]:
        """DBLP publication search (``https://dblp.org/search/publ/api``), gated the
        same way as the Crossref fallback: a hit must be a conference / journal /
        collection record (never ``Informal and Other Publications`` — DBLP's bucket
        for arXiv / CoRR), title-match, sit within the year tolerance, share an
        author surname, and carry a non-CoRR venue. The query is the title plus the
        first author's surname, which DBLP ANDs — this is what lifts the real
        record above the many derivative titles for a well-known paper."""
        title = ref.title or rec.title
        if not title:
            return "skip", None
        authors = ref.authors or rec.authors
        q = title
        if authors:
            first = authors[0].strip()
            surname = first.split(",", 1)[0].strip() if "," in first else (first.split()[-1] if first else "")
            if surname:
                q = f"{title} {surname}"
        tag, payload = await self._request(
            "dblp",
            self._g_dblp,
            "https://dblp.org/search/publ/api",
            params={"q": q, "format": "json", "h": "10"},
        )
        if tag == "transient":
            return "transient", None
        if tag != "ok" or not isinstance(payload, dict):
            return "terminal", None
        hits = (((payload.get("result") or {}).get("hits") or {}).get("hit")) or []
        year = ref.year if ref.year is not None else rec.year
        for h in hits:
            info = (h or {}).get("info") or {}
            if info.get("type") not in _DBLP_PUBLISHED_TYPES:
                continue
            venue = info.get("venue")
            if isinstance(venue, list):
                venue = ", ".join(str(v) for v in venue if v)
            if not venue or _ARXIV_VENUE_RE.search(str(venue)):
                continue
            if not _title_matches(title, [info.get("title") or ""]):
                continue
            try:
                y: Optional[int] = int(str(info.get("year"))[:4])
            except (TypeError, ValueError):
                y = None
            if not _year_compatible(year, y, self._cfg.year_tolerance):
                continue
            if not _surnames_overlap(authors, _dblp_author_names(info)):
                continue
            doi = (info.get("doi") or "").lower() or None
            if doi and doi.startswith("10.48550/"):
                doi = None
            url = f"https://doi.org/{doi}" if doi else (info.get("ee") or info.get("url"))
            return "ok", PublishedVersion(venue=str(venue), year=y, doi=doi, url=url, source="dblp")
        return "terminal", None

    async def _crossref_published_version(
        self, ref: Reference, rec: CanonicalRecord
    ) -> tuple[str, Optional[PublishedVersion]]:
        """Fallback published-version lookup: a Crossref title search whose hit must
        be a journal / proceedings / chapter record (never ``posted-content``, i.e.
        a preprint server), title-match, sit within the year tolerance, share an
        author surname, and carry a non-arXiv container title and a publisher DOI."""
        title = ref.title or rec.title
        if not title:
            return "skip", None
        params = {"query.bibliographic": title, "rows": "5"}
        authors = ref.authors or rec.authors
        if authors:
            params["query.author"] = authors[0]
        if self.polite_mailto:
            params["mailto"] = self.polite_mailto
        tag, payload = await self._request(
            "crossref", self._g_crossref, "https://api.crossref.org/works", params=params
        )
        if tag == "transient":
            return "transient", None
        if tag != "ok":
            return "terminal", None
        year = ref.year if ref.year is not None else rec.year
        for item in payload.get("message", {}).get("items", []) or []:
            if item.get("type") not in _CROSSREF_PUBLISHED_TYPES:
                continue
            if not item.get("container-title"):
                continue
            if not _title_matches(title, item.get("title", [])):
                continue
            cand = _crossref_to_record(item)
            if not cand.doi or cand.doi.startswith("10.48550/"):
                continue
            if not cand.venue or _ARXIV_VENUE_RE.search(cand.venue):
                continue
            if not _year_compatible(year, cand.year, self._cfg.year_tolerance):
                continue
            if not _surnames_overlap(authors, cand.authors):
                continue
            return "ok", PublishedVersion(
                venue=cand.venue,
                year=cand.year,
                doi=cand.doi,
                url=f"https://doi.org/{cand.doi}",
                source="crossref",
            )
        return "terminal", None

    async def _s2_published_version(self, arxiv_id: str) -> tuple[str, Optional[PublishedVersion]]:
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else None
        tag, payload = await self._request(
            "semanticscholar",
            self._g_s2,
            f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}",
            headers=headers,
            params={"fields": "externalIds,venue,publicationVenue,year"},
        )
        if tag == "ok":
            return "ok", _s2_published_version(payload)
        if tag == "terminal":
            return "terminal", None
        return "transient", None

    # ----- per-source ------------------------------------------------------
    #
    # Each source method serves one *mode*: ``"id"`` (lookup by DOI / arXiv ID;
    # authoritative, never year-gated) or ``"search"`` (title search; a hit must
    # pass ``_accept_search_hit``). A mode with nothing to do — no identifier, no
    # title — returns TERMINAL_MISS without a request.

    async def _crossref(
        self, ref: Reference, doi: Optional[str], mode: str, scratch: _Scratch
    ) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        if mode == "id":
            if not doi:
                return _Outcome.TERMINAL_MISS, None
            tag, payload = await self._request(
                "crossref",
                self._g_crossref,
                f"https://api.crossref.org/works/{quote(doi, safe='')}",
                params={"mailto": self.polite_mailto} if self.polite_mailto else None,
            )
            if tag == "transient":
                return _Outcome.TRANSIENT_FAIL, None
            if tag == "ok":
                msg = payload.get("message")
                if msg:
                    return _Outcome.HIT, _crossref_to_record(msg)
            return _Outcome.TERMINAL_MISS, None  # 404 → the title search runs in the next phase

        if not ref.title:
            return _Outcome.TERMINAL_MISS, None
        params = {"query.bibliographic": ref.title, "rows": "5"}
        if ref.authors:
            params["query.author"] = ref.authors[0]
        if self.polite_mailto:
            params["mailto"] = self.polite_mailto
        tag, payload = await self._request(
            "crossref", self._g_crossref, "https://api.crossref.org/works", params=params
        )
        if tag == "transient":
            return _Outcome.TRANSIENT_FAIL, None
        if tag == "ok":
            for item in payload.get("message", {}).get("items", []) or []:
                if _title_matches(ref.title, item.get("title", [])):
                    rec = _crossref_to_record(item)
                    if self._accept_search_hit(ref, rec, scratch):
                        return _Outcome.HIT, rec
        return _Outcome.TERMINAL_MISS, None

    async def _semanticscholar(
        self,
        ref: Reference,
        doi: Optional[str],
        arxiv_id: Optional[str],
        mode: str,
        scratch: _Scratch,
    ) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key
        fields = "title,authors,year,abstract,externalIds,openAccessPdf,venue,publicationVenue"

        if mode == "id":
            transient = False
            for ext_id in (f"DOI:{doi}" if doi else None, f"ARXIV:{arxiv_id}" if arxiv_id else None):
                if not ext_id:
                    continue
                tag, payload = await self._request(
                    "semanticscholar",
                    self._g_s2,
                    f"https://api.semanticscholar.org/graph/v1/paper/{ext_id}",
                    headers=headers,
                    params={"fields": fields},
                )
                if tag == "transient":
                    transient = True
                elif tag == "ok" and isinstance(payload, dict) and payload.get("title"):
                    return _Outcome.HIT, _s2_to_record(payload)
            return (_Outcome.TRANSIENT_FAIL if transient else _Outcome.TERMINAL_MISS), None

        if not ref.title:
            return _Outcome.TERMINAL_MISS, None
        tag, payload = await self._request(
            "semanticscholar",
            self._g_s2,
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={"query": ref.title, "limit": "5", "fields": fields},
        )
        if tag == "transient":
            return _Outcome.TRANSIENT_FAIL, None
        if tag == "ok":
            for item in payload.get("data", []) or []:
                if _title_matches(ref.title, [item.get("title", "")]):
                    rec = _s2_to_record(item)
                    if self._accept_search_hit(ref, rec, scratch):
                        return _Outcome.HIT, rec
        return _Outcome.TERMINAL_MISS, None

    async def _arxiv(
        self, ref: Reference, arxiv_id: Optional[str], mode: str, scratch: _Scratch
    ) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        if mode == "id":
            if not arxiv_id:
                return _Outcome.TERMINAL_MISS, None
            tag, payload = await self._request(
                "arxiv",
                self._g_arxiv,
                "http://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": "1"},
                parse="text",
            )
            if tag == "transient":
                return _Outcome.TRANSIENT_FAIL, None
            if tag == "ok":
                if not _xml_well_formed(payload):
                    return _Outcome.TRANSIENT_FAIL, None  # 200 but corrupt XML — transient, not "not found"
                rec = _arxiv_to_record(payload)
                if rec:
                    return _Outcome.HIT, rec
            return _Outcome.TERMINAL_MISS, None

        if not ref.title:
            return _Outcome.TERMINAL_MISS, None
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
            return _Outcome.TRANSIENT_FAIL, None
        if tag == "ok":
            if not _xml_well_formed(payload):
                return _Outcome.TRANSIENT_FAIL, None
            for entry_xml in _split_arxiv_entries(payload):
                cand = _arxiv_to_record(entry_xml)
                if cand and _title_matches(ref.title, [cand.title]) and self._accept_search_hit(ref, cand, scratch):
                    return _Outcome.HIT, cand
        return _Outcome.TERMINAL_MISS, None

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

    async def _openalex(
        self, ref: Reference, doi: Optional[str], mode: str, scratch: _Scratch
    ) -> tuple[_Outcome, Optional[CanonicalRecord]]:
        # OpenAlex requires a free API key since 2026-02-13; the polite-pool mailto is
        # gone. Send the key when set, and treat a 409 (keyless / daily credits
        # exhausted) as infrastructure-degraded, never as an authoritative "not found".
        params: dict = {}
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key

        if mode == "id":
            if not doi:
                return _Outcome.TERMINAL_MISS, None
            tag, payload = await self._request(
                "openalex",
                self._g_openalex,
                f"https://api.openalex.org/works/doi:{quote(doi, safe='')}",
                params=params or None,
            )
            if tag == "transient" or (tag == "terminal" and payload == 409):
                return _Outcome.TRANSIENT_FAIL, None
            if tag == "ok":
                return _Outcome.HIT, _openalex_to_record(payload)
            return _Outcome.TERMINAL_MISS, None  # 404 → the search runs in the next phase

        if not ref.title:
            return _Outcome.TERMINAL_MISS, None
        tag, payload = await self._request(
            "openalex",
            self._g_openalex,
            "https://api.openalex.org/works",
            params={"search": ref.title, "per-page": "5", **params},
        )
        if tag == "transient" or (tag == "terminal" and payload == 409):
            return _Outcome.TRANSIENT_FAIL, None
        if tag == "ok":
            for item in payload.get("results", []) or []:
                if _title_matches(ref.title, [item.get("title", "")]):
                    rec = _openalex_to_record(item)
                    if self._accept_search_hit(ref, rec, scratch):
                        return _Outcome.HIT, rec
        return _Outcome.TERMINAL_MISS, None


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
        # S2 merges the preprint and the venue version into one record. A found
        # venue version is definitive; "none" is left un-checked so the resolver's
        # follow-up can still ask DBLP (which is often ahead of S2 for fresh
        # proceedings) without re-asking S2 — see ``_ensure_published_version``.
        published_version=_s2_published_version(item),
        published_version_checked=_s2_published_version(item) is not None,
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

    # An unknown ID does not 404: arXiv answers 200 with an <entry> whose <id> is
    # an errors URL and whose <title> is "Error". That is a terminal miss.
    if "/api/errors" in text(entry.find("a:id", ns)):
        return None

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
    journal_ref = " ".join(text(entry.find("arxiv:journal_ref", ns)).split()) or None

    # Authors may register the venue DOI (and a journal_ref) on the arXiv record
    # itself; when present that is a definitive published version. Absent, arXiv
    # simply doesn't know — leave the record un-checked for the S2 follow-up.
    published = None
    if doi and not doi.startswith("10.48550/"):
        published = PublishedVersion(
            venue=journal_ref, doi=doi, url=f"https://doi.org/{doi}", source="arxiv"
        )

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
        published_version=published,
        published_version_checked=published is not None,
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


_ARXIV_CONTEXT_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\s*[:\s]\s*|arxiv\.)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)
_ARXIV_VENUE_RE = re.compile(r"\barxiv\b|\bcorr\b", re.IGNORECASE)
# Crossref work types that count as a published version. ``posted-content`` is the
# preprint-server type (arXiv via DataCite, bioRxiv, SSRN …) and is excluded.
_CROSSREF_PUBLISHED_TYPES = frozenset({"journal-article", "proceedings-article", "book-chapter"})
# DBLP record types that count as a published version. "Informal and Other
# Publications" is DBLP's bucket for arXiv / CoRR and is excluded.
_DBLP_PUBLISHED_TYPES = frozenset(
    {"Conference and Workshop Papers", "Journal Articles", "Parts in Books or Collections"}
)
_DBLP_DISAMBIGUATOR_RE = re.compile(r"\s+\d{4}$")


def _dblp_author_names(info: dict) -> list[str]:
    """DBLP's ``authors.author`` is a dict for one author, a list otherwise, and
    names carry a numeric disambiguator ("Zhe Li 0030") that must not become
    the surname."""
    au = (info.get("authors") or {}).get("author") or []
    if isinstance(au, dict):
        au = [au]
    out: list[str] = []
    for a in au:
        name = a.get("text") if isinstance(a, dict) else (a if isinstance(a, str) else None)
        if name:
            out.append(_DBLP_DISAMBIGUATOR_RE.sub("", name.strip()))
    return out


def _guess_arxiv_id(ref: Reference) -> Optional[str]:
    """Recover an arXiv ID the parser did not put in ``Reference.arxiv_id``.

    Looks at a DataCite arXiv DOI (``10.48550/arXiv.XXXX``), then the URL, then
    ``raw_text`` — the last only where the number sits in an arXiv context
    (``arXiv:2401.12345``, ``arxiv.org/abs/…``), since a bare ``NNNN.NNNNN`` number also
    matches the tail of many DOIs.
    """
    doi = normalize_doi(ref.doi)
    if doi and doi.startswith("10.48550/arxiv."):
        a = normalize_arxiv_id(doi[len("10.48550/arxiv.") :])
        if a:
            return a
    if ref.url:
        a = normalize_arxiv_id(ref.url)
        if a:
            return a
    if ref.raw_text:
        m = _ARXIV_CONTEXT_RE.search(ref.raw_text)
        if m:
            return normalize_arxiv_id(m.group(1))
    return None


def arxiv_id_of(ref: Reference) -> Optional[str]:
    """The arXiv ID a bibliography entry points at, explicit or recovered."""
    return normalize_arxiv_id(ref.arxiv_id) or _guess_arxiv_id(ref)


def cites_arxiv_version(ref: Reference) -> bool:
    """True when the bibliography entry cites the arXiv preprint (an arXiv ID, a
    DataCite arXiv DOI, or an arXiv / CoRR venue string) rather than a venue
    version. A real publisher DOI in the entry means they cite the published
    version, whatever else the entry says."""
    doi = normalize_doi(ref.doi)
    if doi and not doi.startswith("10.48550/"):
        return False
    if arxiv_id_of(ref):
        return True
    hay = " ".join(x for x in (ref.venue, ref.url) if x)
    return bool(_ARXIV_VENUE_RE.search(hay))


def _year_compatible(ref_year: Optional[int], rec_year: Optional[int], tolerance: int) -> bool:
    """Year gate for title-search hits. Unknown on either side → cannot judge →
    compatible (the author-overlap check in ``detect_metadata_mismatch`` still
    applies downstream)."""
    if ref_year is None or rec_year is None:
        return True
    return abs(int(ref_year) - int(rec_year)) <= tolerance


def _surname(author: str) -> str:
    a = author.strip()
    if "," in a:
        return a.split(",", 1)[0].strip().lower()
    return a.split()[-1].lower() if a else ""


def _surnames_overlap(a: list[str], b: list[str]) -> bool:
    """Same signal ``verify.detect_metadata_mismatch`` uses: any shared surname
    among the first five authors on each side. Empty on either side → False (no
    evidence is not evidence of a match)."""
    sa = {s for s in (_surname(x) for x in a[:5]) if s}
    sb = {s for s in (_surname(x) for x in b[:5]) if s}
    return bool(sa and sb and (sa & sb))


def _describe_record(rec: CanonicalRecord) -> str:
    where = rec.venue or rec.source
    year = rec.year if rec.year is not None else "no year"
    doi = f", doi:{rec.doi}" if rec.doi else ""
    return f'"{rec.title}" ({year}, {where}{doi})'


def _published_from_record(rec: CanonicalRecord) -> Optional[PublishedVersion]:
    """A strong Crossref / OpenAlex hit that carries a real publisher DOI *and* a
    non-arXiv container is itself the published version. Weak matches never
    qualify (the record may be the very reprint the year gate distrusted)."""
    if rec.match_note or rec.source not in ("crossref", "openalex"):
        return None
    if not rec.doi or rec.doi.startswith("10.48550/"):
        return None
    if not rec.venue or _ARXIV_VENUE_RE.search(rec.venue):
        return None
    return PublishedVersion(
        venue=rec.venue,
        year=rec.year,
        doi=rec.doi,
        url=f"https://doi.org/{rec.doi}",
        source=rec.source,
    )


def _s2_published_version(item: dict) -> Optional[PublishedVersion]:
    """Read a published version off a Semantic Scholar paper payload.

    Signals, any one sufficient: a non-DataCite DOI; a DBLP key outside
    ``journals/corr/`` (DBLP's bucket for arXiv-only records); or a
    ``publicationVenue`` typed conference / journal whose name is not arXiv.
    Precision over recall — an arXiv-only paper must never be called published.
    """
    if not isinstance(item, dict):
        return None
    ext = item.get("externalIds") or {}
    doi = (ext.get("DOI") or ext.get("doi") or "").lower() or None
    if doi and doi.startswith("10.48550/"):
        doi = None
    dblp = ext.get("DBLP") or ""
    dblp_published = bool(dblp) and not dblp.startswith("journals/corr/")
    pv = item.get("publicationVenue")
    pv = pv if isinstance(pv, dict) else {}
    pv_type = (pv.get("type") or "").lower()
    venue = pv.get("name") or item.get("venue") or None
    if venue and _ARXIV_VENUE_RE.search(venue):
        venue = None
    if doi or dblp_published or (venue and pv_type in ("conference", "journal")):
        return PublishedVersion(
            venue=venue,
            year=item.get("year"),
            doi=doi,
            url=f"https://doi.org/{doi}" if doi else None,
            source="semanticscholar",
        )
    return None
