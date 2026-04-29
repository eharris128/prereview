"""On-disk JSON cache for resolved references and fetched PDFs.

The cache is keyed by a normalized lookup key (DOI, arXiv ID, or title+author hash)
and holds CanonicalRecord JSON plus, optionally, a path to a downloaded PDF.

The cache is best-effort: a missing or unreadable cache file is treated as a miss
rather than an error. Concurrent writes can race; the worst outcome is that one
of two simultaneous writes overwrites the other, which is fine for this use case.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from .models import CanonicalRecord


DEFAULT_CACHE_DIR = Path.home() / ".prereview" / "cache"


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80]


def cache_key(*, doi: Optional[str], arxiv_id: Optional[str], title: Optional[str], first_author: Optional[str]) -> str:
    """Stable, filesystem-safe key for a reference."""
    if doi:
        return "doi-" + _slug(doi)
    if arxiv_id:
        return "arxiv-" + _slug(arxiv_id)
    seed = (title or "") + "|" + (first_author or "")
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return "ta-" + _slug(title or "untitled") + "-" + h


def verification_key(*, model: str, ref_id: str, sentence: str, evidence_marker: str = "") -> str:
    """Stable key for a (model, citation, reference, evidence-mode) tuple."""
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(ref_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(sentence.encode("utf-8"))
    h.update(b"\x00")
    h.update(evidence_marker.encode("utf-8"))
    return "v-" + h.hexdigest()[:24]


class Cache:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.refs_dir = self.cache_dir / "refs"
        self.pdfs_dir = self.cache_dir / "pdfs"
        self.verify_dir = self.cache_dir / "verify"
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.verify_dir.mkdir(parents=True, exist_ok=True)

    def get_record(self, key: str) -> Optional[CanonicalRecord]:
        path = self.refs_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return CanonicalRecord.model_validate_json(path.read_text())
        except Exception:
            return None

    def put_record(self, key: str, record: CanonicalRecord) -> None:
        path = self.refs_dir / f"{key}.json"
        path.write_text(record.model_dump_json(indent=2))

    def pdf_path(self, key: str) -> Path:
        return self.pdfs_dir / f"{key}.pdf"

    def has_pdf(self, key: str) -> bool:
        p = self.pdf_path(key)
        return p.exists() and p.stat().st_size > 0

    def get_verification_json(self, key: str) -> Optional[str]:
        path = self.verify_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def put_verification_json(self, key: str, payload: str) -> None:
        path = self.verify_dir / f"{key}.json"
        path.write_text(payload)
