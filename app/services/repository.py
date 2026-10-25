"""File-based persistence: one folder per document, keyed by a hash of the PDF bytes.

data/documents/<doc_id>/
    source.pdf        the uploaded file (used for page previews)
    document.json     title + section tree + text (written last => marks "complete")
    summaries.json    cache of generated summaries
Same PDF uploaded twice -> same id -> no re-parse, no re-embed, cached summaries.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.domain import ParsedDocument, Section

_DOC_ID = re.compile(r"^[a-f0-9]{16}$")


class DocumentNotFound(KeyError):
  pass


class DocumentRepository:
  def __init__(self, root: Path) -> None:
    self.root = root / "documents"
    self.root.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()

  @staticmethod
  def make_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

  def _dir(self, doc_id: str) -> Path:
    if not _DOC_ID.match(doc_id):
      raise DocumentNotFound(doc_id)
    return self.root / doc_id

  def exists(self, doc_id: str) -> bool:
    return (self._dir(doc_id) / "document.json").exists()

  def save(self, doc_id: str, parsed: ParsedDocument, pdf_bytes: bytes) -> None:
    folder = self._dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source.pdf").write_bytes(pdf_bytes)
    self._write_json(
      folder / "document.json",
      {"title": parsed.title, "page_count": parsed.page_count, "sections": [asdict(s) for s in parsed.sections]}
    )

  @staticmethod
  def _write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@lru_cache
def get_repository() -> DocumentRepository:
  return DocumentRepository(get_settings().data_dir)