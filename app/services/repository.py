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
  def make_id(data: bytes, source_type: str = "pdf") -> str:
    payload = data if source_type == "pdf" else source_type.encode("utf-8") + b"\0" + data
    return hashlib.sha256(payload).hexdigest()[:16]

  def _dir(self, doc_id: str) -> Path:
    if not _DOC_ID.match(doc_id):
      raise DocumentNotFound(doc_id)
    return self.root / doc_id

  def exists(self, doc_id: str) -> bool:
    return (self._dir(doc_id) / "document.json").exists()

  def save(self, doc_id: str, parsed: ParsedDocument, data: bytes, extension: str) -> None:
    folder = self._dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"source{extension}").write_bytes(data)
    self._write_json(
      folder / "document.json",
      {
        "title": parsed.title, 
        "page_count": parsed.page_count,
        "sections": [asdict(s) for s in parsed.sections],
        "source_type": parsed.source_type,
        "unit_count": parsed.unit_count,
        "unit_label": parsed.unit_label
      },
    )

  def load(self, doc_id: str) -> ParsedDocument:
    path = self._dir(doc_id) / "document.json"
    if not path.exists():
      raise DocumentNotFound(doc_id)
    raw = json.loads(path.read_text(encoding = "utf-8"))
    return ParsedDocument(
      raw["title"], 
      raw["page_count"], 
      [Section(**s) for s in raw["sections"]],
      raw.get("source_type", "pdf"),
      raw.get("unit_count", raw["page_count"]),
      raw.get("unit_label", "pages"),
    )

  def pdf_path(self, doc_id: str) -> Path:
    path = self._dir(doc_id) / "source.pdf"
    if not path.exists():
      raise DocumentNotFound(doc_id)
    return path

  def get_summary(self, doc_id: str, section_id: str) -> str | None:
    path = self._dir(doc_id) / "summaries.json"
    if not path.exists():
      return None
    cache = json.loads(path.read_text(encoding="utf-8"))
    return cache.get(section_id)

  def save_summary(self, doc_id: str, section_id: str, summary: str) -> None:
    path = self._dir(doc_id) / "summaries.json"
    with self._lock:
      cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
      cache[section_id] = summary
      self._write_json(path, cache)

  @staticmethod
  def _write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@lru_cache
def get_repository() -> DocumentRepository:
  return DocumentRepository(get_settings().data_dir)