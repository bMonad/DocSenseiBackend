"""Data structures shared by the services for faster validation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Section:
  id: str
  title: str
  parent_id: str | None
  page_start: int
  page_end: int
  own_text: str = ""

@dataclass
class ParsedDocument:
  title: str
  page_count: int
  sections: list[Section]