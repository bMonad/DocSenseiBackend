"""Data structures shared by the services for faster validation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Section:
  id: str
  title: str
  level: int
  parent_id: str | None
  page_start: int
  page_end: int
  own_text: str = ""

@dataclass
class ParsedDocument:
  title: str
  page_count: int
  sections: list[Section]


def subtree(sections: list[Section], section_id: str) -> list[Section]:
  """The section plus all of its descendants. Raises KeyError if unknown."""
  start = next((i for i, s in enumerate(sections) if s.id == section_id), None)
  if start is None:
    raise KeyError(section_id)
  level = sections[start].level
  end = start + 1
  while end < len(sections) and sections[end].level > level:
    end += 1
  return sections[start : end]

def subtree_text(scope: list[Section]) -> str:
  return "\n\n".join(s.own_text for s in scope if s.own_text).strip()