"""PDF -> topic tree.

Strategy (first one that yields headings wins):
  1. The PDF's embedded outline / bookmarks   (most reliable)
  2. Font-size / numbering heuristics          (works on most papers, books, reports)
  3. Fixed page windows                        (last resort, e.g. slide decks)

Everything is first flattened into a stream of `Line`s in reading order. Headings are
positions in that stream, so a section is simply "the lines between two headings".
That also lets two sections share a page without duplicating text.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
import pymupdf

from app.domain import ParsedDocument, Section

log = logging.getLogger(__name__)

MAX_LEVEL = 3                   # deeper headings fold into their level-3 parent
BOLD_FLAG = 16                  # PyMuPDF span flag for bold
EDGE_BAND = 0.09                # top/bottom 9% of a page = header/footer zone
MIN_TOTAL_CHARS = 200           # below this we assume a scanned PDF
MIN_PREAMBLE_CHARS = 300        # text before the first heading becomes "Front matter"
FALLBACK_PAGES_PER_SECTION = 5    

NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")


class ParseError(ValueError):
  """Raised for PDFs we cannot turn into sections."""


@dataclass(slots=True)
class Line:
  page: int
  y: float
  text: str
  size: float
  bold: bool


@dataclass(slots=True)
class Heading:
  level: int
  title: str
  page: int
  start: int
  end: int


# ------ Extraction ------ 

def extract_lines(doc: pymupdf.Document) -> list[Line]:
  """Return lines in form of list from document."""
  lines: list[Line] = []
  for index in range(doc.page_count):
    page = doc[index]
    height = page.rect.height or 1.0
    for block in page.get_text("dict", sort="True")["blocks"]:
      if block.get("type") != 0:
        continue
      for raw in block["lines"]:
        spans = [s for s in raw["spans"] if s["text"].strip()]
        if not spans:
          continue
        text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
        dominant = max(spans, key=lambda s: len(s["text"]))
        bold = bool(dominant["flags"] & BOLD_FLAG) or "bold" in dominant["font"].lower()
        lines.append(Line(index + 1, raw["bbox"][1] / height, text, float(dominant["size"]), bold))
  return lines


def drop_page_furniture(lines: list[Line], page_count: int) -> list[Line]:
  """Remove running headers/footers and page numbers."""
  if page_count < 4:
    return lines

  def in_edge(line: Line) -> bool:
    return line.y < EDGE_BAND or line.y > 1-EDGE_BAND

  def key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()

  pages_by_key: dict[str, set[int]] = defaultdict(set)
  for line in lines:
    if in_edge(line):
      pages_by_key[key(line.text)].add(line.page)

  threshold = max(3, int(page_count * 0.4))
  furniture = {k for k, pages in pages_by_key.items() if len(pages) >= threshold}
  return [ln for ln in lines if not (in_edge(ln) and key(ln.text) in furniture)]


def lines_to_text(lines: list[Line]) -> str:
  """Re-flow visual lines into paragraphs: join wrapped lines, repair hyphenation."""
  out = ""
  prev = ""
  for line in lines:
    text = line.text
    if not out:
      out = text
    elif prev.endswith("-") and text[:1].islower():
      out = out[:-1] + text
    elif len(prev) < 45:
      out += "\n" + text
    else:
      out += " " + text
    prev = text
  return out.strip()


# ------ Headings ------

def _norm(text: str) -> str:
  """Normalize text to lower form."""
  return  re.sub(r"\W+", " ", text.lower()).strip()


def _find_title(lines: list[Line], title: str, begin:int, page: int) -> tuple[int, int] | None:
  """Locate an outline Title on its page."""
  target = _norm(title)
  i = begin
  while i < len(lines) and lines[i].page <= page:
    acc = ""
    for j in range(i, min(i + 3, len(lines))):
      if lines[j].page != lines[i].page:
        break
      acc = _norm(f"{acc} {lines[j].text}")
      if acc == target:
        return i, j + 1
      if not target.startswith(acc):
        break
    i += 1
  return None


def _normalize_levels(headings: list[Heading]) -> None:
  """Clamp levels so the tree never skips a depth (1 -> 3 becomes 1 -> 2)."""
  prev = 0
  for h in headings:
    h.level = max(1, min(h.level, prev + 1, MAX_LEVEL))
    prev = h.level


def headings_from_outline(doc: pymupdf.Document, lines: list[Line]) -> list[Heading]:
  """Return normalized headings extracted from outline."""
  entries = [
    (lvl, title.strip(), page) 
    for lvl, title, page in doc.get_toc(simple=True)
    if title.strip() and 1 <= page <= doc.page_count and lvl <= MAX_LEVEL
  ]
  if len(entries) < 2:
    return []

  first_line_of_page: dict[int, int] = {}
  for i, line in enumerate(lines):
    first_line_of_page.setdefault(line.page, i)

  def start_for_page(page: int) -> int:
    later = [p for p in first_line_of_page if p >= page]
    return first_line_of_page[min(later)] if later else len(lines)

  headings: list[Heading] = []
  cursor = 0
  for level, title, page in entries:
    begin = max(cursor, start_for_page(page))
    match = _find_title(lines, title, begin, page)
    start, end = match if match else (begin, begin)
    headings.append(Heading(level, title, page, start, end))
    cursor = end
  _normalize_levels(headings)
  return headings


def detect_font_headings(lines: list[Line]) -> list[Heading]:
  """Return normalized headings detected by font size and numbering."""
  if not lines:
    return []

  sizes: Counter[float] = Counter()
  for line in lines:
    sizes[round(line.size * 2) / 2] += len(line.text)
  body = sizes.most_common(1)[0][0]

  candidates: list[int] = []
  for i, line in enumerate(lines):
    text = line.text
    if not 3 <= len(text) <= 120 or len(text.split()) > 16:
      continue
    if text.endswith((",", ";")) or re.fullmatch(r"[\d\W_]+", text):
      continue
    is_big = line.size >= body * 1.15 and line.size >= body + 1.0
    is_numbered_bold = line.bold and line.size >= body * 0.98 and bool(NUMBERED.match(text)) and len(text) <= 80
    if is_big or is_numbered_bold:
      candidates.append(i)

  # merge heading of same size wrapped across lines
  groups: list[list[int]] = []
  for i in candidates:
    if groups: 
      last = groups[-1][-1]
      if i == last + 1 and lines[i].page == lines[last].page and abs(lines[i].size - lines[last].size) < 0.6:
        groups[-1].append(i)
        continue
    groups.append([i])

  if len(groups) < 2 or len(groups) > max(8, len(lines) // 6):
    return []

  ranked = sorted({round(lines[g[0]].size) for g in groups}, reverse=True)
  headings = [
    Heading(
      level=min(ranked.index(round(lines[g[0]].size)) + 1, MAX_LEVEL),
      title=" ".join(lines[k].text for k in g),
      page=lines[g[0]].page,
      start=g[0],
      end=g[-1] + 1
    )
    for g in groups
  ]
  _normalize_levels(headings)
  return headings


# ------ Section ------

def build_sections(lines: list[Line], headings: list[Heading]) -> list[Section]:
  """Return sections from lines and headings."""
  sections: list[Section] = []
  stack: list[Section] = []

  def next_id() -> str:
    return f"s{len(sections):04d}"

  preamble = lines[: headings[0].start]
  preamble_text = lines_to_text(preamble)
  if len(preamble_text) >= MIN_PREAMBLE_CHARS:
    sections.append(
      Section(next_id(), "Front matter", 1, None, preamble[0].page, preamble[-1].page, preamble_text)
    )

  for i, h in enumerate(headings):
    stop = headings[i + 1].start if i + 1 < len(headings) else len(lines)
    own = lines[h.end : max(h.end, stop)]
    while stack and stack[-1].level >= h.level:
      stack.pop()
    section = Section(
      id = next_id(),
      title = h.title,
      level = h.level,
      parent_id = stack[-1].id if stack else None,
      page_start = h.page,
      page_end = max([h.page] + [ln.page for ln in own]),
      own_text = lines_to_text(own)
    )
    sections.append(section)
    stack.append(section)

  # Cover descendants 
  by_id = {s.id: s for s in sections}
  for s in reversed(sections):
    if s.parent_id:
      parent = by_id[s.parent_id]
      parent.page_end = max(parent.page_end, s.page_end)
  return sections


def page_window_sections(lines: list[Line], page_count: int) -> list[Section]:
  """Return sections by fixed page windows."""
  sections: list[Section] = []
  by_page: dict[int, list[Line]] = defaultdict(list)
  for line in lines:
    by_page[line.page].append(line)

  for first in range(1, page_count + 1, FALLBACK_PAGES_PER_SECTION):
    last = min(first + FALLBACK_PAGES_PER_SECTION - 1, page_count)
    window = [ln for p in range(first, last + 1) for ln in by_page.get(p,[])]
    if not window:
      continue
    title = f"Page {first}" if first == last else f"Pages {first}–{last}"
    sections.append(Section(f"s{len(sections):04d}", title, 1, None, first, last, lines_to_text(window)))
  return sections


# ------ Parsing PDF ------

def parse_pdf(data: bytes, fallback_title: str, max_pages: int = 400) -> ParsedDocument:
  """Return data into ParsedDocument format after parsing PDF."""
  try:
    doc = pymupdf.open(stream=data, filetype="pdf")
  except (RuntimeError, ValueError) as exc:
    raise ParseError("This file could not be read as a PDF.") from exc

  with doc:
    if doc.needs_pass:
      raise ParseError("This PDF is password-protected. Remove the password and try again.")
    if doc.page_count == 0:
      raise ParseError("This PDF has no page.")
    if doc.page_count > max_pages:
      raise ParseError(f"This PDF has {doc.page_count} pages; the limit is {max_pages}.")

    lines = drop_page_furniture(extract_lines(doc), doc.page_count)
    if sum(len(ln.text) for ln in lines) < MIN_TOTAL_CHARS:
      raise ParseError("No selectable text found. Scanned PDFs need OCR, which isn't supported yet.")

    headings = headings_from_outline(doc, lines) or detect_font_headings(lines)
    if headings:
      sections = build_sections(lines, headings)
    else:
      log.info("No headings found; falling back to page windows")
      sections = page_window_sections(lines, doc.page_count)

    title = ((doc.metadata or {}).get("title") or "").strip() or fallback_title
    return ParsedDocument(title=title, page_count=doc.page_count, sections=sections)
