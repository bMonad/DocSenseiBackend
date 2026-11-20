"""Data structures shared by the services for faster validation."""
from __future__ import annotations

from dataclasses import dataclass
from app.schemas import SectionNode

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
  source_type: str = "pdf"
  unit_count: int = 0
  unit_label: str = "pages"


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

def to_outline(sections: list[Section]) -> list[SectionNode]:
  """Flat, ordered sections -> nested tree for the sidebar."""
  roots: list[SectionNode] = []
  stack: list[SectionNode] = []
  for s in sections:
    node: SectionNode = {
      "id": s.id,
      "title": s.title,
      "level": s.level,
      "page_start": s.page_start,
      "page_end": s.page_end,
      "children": []
    }

    while stack and stack[-1]["level"] >= s.level:
      stack.pop()
    (stack[-1]["children"] if stack else roots).append(node)
    stack.append(node)
  return roots